#!/usr/bin/env python3
# Simple ROS1 logger for:
# 1) Your full-state message -> state CSV
# 2) geometry_msgs/PoseStamped (mocap) -> mocap CSV
#
# Params (set via rosparam or _private params):
#   ~state_topic:      default "/full_state"
#   ~mocap_topic:      default "/mocap/pose"
#   ~state_csv:        default "~/ros_logs/state_log.csv"
#   ~mocap_csv:        default "~/ros_logs/mocap_log.csv"
#   ~flush_every_n:    default 200

from dataclasses import dataclass
import datetime, time
import csv, signal
from pathlib import Path
from typing import Dict, List, Tuple
import rospy
from geometry_msgs.msg import Pose, Twist, Vector3, PoseStamped
from std_msgs.msg import Header
import re, json
import genpy
from agiros_msgs.msg import QuadState as StateMsg, Command 

# ---- Message flattening Utils ---
_ARRAY_RE = re.compile(r"(.+?)\[(\d*)\]$")  # e.g. "float64[9]" or "geometry_msgs/Vector3[]"
_BUILTINS = {
    "bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64",
    "float32","float64","string","time","duration"
}


def _parse_slot_type(t: str):
    """
    Returns (base_type, is_array, array_len or None).
    """
    m = _ARRAY_RE.match(t)
    if not m:
        return t, False, None
    base, n = m.group(1), m.group(2)
    return base, True, (int(n) if n else None)

def _is_time_like(x):
    return isinstance(x, (genpy.Time, genpy.Duration))

def _time_to_sec(x):
    # genpy Time/Duration have to_sec(); rospy.Time/Duration duck-type similarly
    return x.to_sec()

def flatten_msg(msg, prefix=""):
    """
    Flatten any ROS message (including nested) to a flat dict.
    - fixed-size arrays -> expand to field_0, field_1, ...
    - variable-size arrays -> field_len + field (comma-joined if primitive; JSON if nested)
    - time/duration -> seconds (float)
    """
    out = {}
    slots = getattr(msg, "__slots__", [])
    types = getattr(msg, "_slot_types", [])

    for field, ftype in zip(slots, types):
        name = f"{prefix}{field}"
        val  = getattr(msg, field)
        base, is_arr, arr_len = _parse_slot_type(ftype)

        # Handle arrays
        if is_arr:
            if arr_len:  # fixed-size
                for i in range(arr_len):
                    v = val[i] if i < len(val) else None
                    if v is None:
                        out[f"{name}_{i}"] = ""
                    elif base in _BUILTINS:
                        out[f"{name}_{i}"] = _time_to_sec(v) if _is_time_like(v) else v
                    else:
                        # nested fixed-size element
                        out.update(flatten_msg(v, f"{name}_{i}_"))
            else:  # variable-size
                out[f"{name}_len"] = len(val)
                if base in _BUILTINS:
                    out[name] = ",".join(str(_time_to_sec(x) if _is_time_like(x) else x) for x in val)
                else:
                    # nested array -> JSON string of flattened dicts
                    out[name] = json.dumps([flatten_msg(x) for x in val], separators=(",", ":"))
            continue

        # Non-array field
        if base in _BUILTINS:
            if _is_time_like(val):
                out[name] = _time_to_sec(val)
            else:
                out[name] = val
        else:
            # Nested message
            out.update(flatten_msg(val, name + "_"))

    return out



def ensure_dir(path: Path):
    d = path.parent
    if d:
        d.mkdir(parents=True, exist_ok=True)
        

class SimpleCsvWriter:
    """CSV writer that infers header from the first row, then appends."""
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(self.path)
        self.header_written = self.path.exists() and self.path.stat().st_size > 0
        self.buffer = []
        self.fieldnames = None

    def add(self, row: dict):
        if self.fieldnames is None:
            # Lock schema from first row
            self.fieldnames = list(row.keys())
        self.buffer.append(row)

    def flush(self, force=False):
        if not self.buffer and not force:
            return
        
        if self.fieldnames is None:
            return
        
        mode = "a" if self.header_written else "w"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self.header_written:
                w.writeheader()
                self.header_written = True
            w.writerows(self.buffer)
        self.buffer = []
        
@dataclass
class TopicCsvType:
    topic: str
    csv_name: str
    msg_type: type       
        
class LoggerNode:
    def __init__(self, entries: List[TopicCsvType], log_path: Path) -> None:
        self.flush_every_n = 400
        self.msgs_since_flush = 0
        rospy.loginfo("Starting logger node…")
        
        # Validate entries
        self._validate_entries(entries)        
        
        # Create a new timestamped directory 
        t = time.time()
        readable_time = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = log_path / f"recording_{readable_time}"

        # Create writers for each topic
        self.writers = []
        self.subscribers = []
        for entry in entries:
            writer, sub = self._process_topic_csv_msg_entry(entry, log_dir)
            self.writers.append(writer)
            self.subscribers.append(sub)
        
        
        
        # Clean shutdown
        signal.signal(signal.SIGINT, self._sigint)
        rospy.on_shutdown(self._on_shutdown)

    def _process_topic_csv_msg_entry(self, entry: TopicCsvType, directory: Path):
        # Create writer
        writer = SimpleCsvWriter(directory / (entry.csv_name + ".csv"))
        
        # Create subscriber callback
        callback = self._create_call_back(entry.msg_type, writer)
        subscriber = rospy.Subscriber(entry.topic, entry.msg_type, callback, queue_size=100)

        rospy.loginfo(f"Logging {entry.topic} -> {entry.csv_name + '.csv'}")

        return writer, subscriber
      
    def _validate_entries(self, entries: List[TopicCsvType]):
        if not entries:
            raise ValueError("At least one topic-csv-msg entry is required.")
        
        csv_names = [entry.csv_name for entry in entries]
        if len(csv_names) != len(set(csv_names)):
            raise ValueError("CSV names must be unique across entries.")
        
        for entry in entries:
            if not isinstance(entry, TopicCsvType):
                raise TypeError(f"Invalid entry type: {type(entry)}. Expected TopicCsvType.")
            if not entry.topic or not entry.csv_name or not entry.msg_type:
                raise ValueError("Each entry must have a valid topic, csv_name, and msg_type.")
        

    def _create_call_back(self, msg_type: type, writer: SimpleCsvWriter):
        def callback(msg):
            row = flatten_msg(msg)
            writer.add(row)
            self._maybe_flush()
        return callback
    
    def _maybe_flush(self):
        self.msgs_since_flush += 1
        if self.flush_every_n > 0 and self.msgs_since_flush >= self.flush_every_n:
            for writer in self.writers:
                writer.flush()
            self.msgs_since_flush = 0

    def _sigint(self, *_):
        rospy.loginfo("SIGINT: flushing logs…")
        for writer in self.writers:
            writer.flush(force=True)
        rospy.signal_shutdown("SIGINT")

    def _on_shutdown(self):
        # Final flush on normal shutdown
        for writer in self.writers:
            writer.flush(force=True)

def main():
    # Create entries
    log_dir = Path("~/catkin_ws/ros_logs").expanduser().resolve()
    entries = [
        TopicCsvType(topic="/volley_drone/agiros_pilot/state", csv_name="quad_state", msg_type=StateMsg),
        TopicCsvType(topic="/mocap/volley_drone/pose", csv_name="mocap", msg_type=PoseStamped),
        TopicCsvType(topic="/volley_drone/agiros_pilot/command", csv_name="command", msg_type=Command),
    ]
    
    rospy.init_node("state_mocap_logger", anonymous=False)
    LoggerNode(entries, log_dir)
    rospy.spin()

if __name__ == "__main__":
    main()