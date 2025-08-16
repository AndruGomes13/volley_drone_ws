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

import csv, signal
from pathlib import Path
from typing import Tuple
import rospy
from geometry_msgs.msg import Pose, Twist, Vector3, PoseStamped
from std_msgs.msg import Header

# --- CHANGE THIS import to your actual package/msg name ---
# Example: from my_robot_msgs.msg import FullState as StateMsg
from agiros_msgs.msg import QuadState as StateMsg  # <-- EDIT THIS

def ensure_dir(path: Path):
    d = path.parent
    if d:
        d.mkdir(parents=True, exist_ok=True)
        
def replace_stem(p: Path, new_stem: str) -> Path:
    # 3.8-safe "with_stem"
    return p.with_name(new_stem + p.suffix)

def make_unique_pair(base1: Path, base2: Path) -> Tuple[Path, Path]:
    base1 = base1.expanduser().resolve()
    base2 = base2.expanduser().resolve()

    i = 0
    while True:
        if i == 0:
            cand1, cand2 = base1, base2
        else:
            cand1 = replace_stem(base1, f"{base1.stem}_{i}")
            cand2 = replace_stem(base2, f"{base2.stem}_{i}")
        if not cand1.exists() and not cand2.exists():
            return cand1, cand2
        i += 1

def stamp_to_sec(stamp):
    return stamp.secs + stamp.nsecs * 1e-9

def msg_time(header):
    if isinstance(header, Header):
        return stamp_to_sec(header.stamp)
    return rospy.get_time()

def flatten_pose(pose: Pose, prefix=""):
    return {
        f"{prefix}pos_x": pose.position.x,
        f"{prefix}pos_y": pose.position.y,
        f"{prefix}pos_z": pose.position.z,
        f"{prefix}quat_x": pose.orientation.x,
        f"{prefix}quat_y": pose.orientation.y,
        f"{prefix}quat_z": pose.orientation.z,
        f"{prefix}quat_w": pose.orientation.w,
    }

def flatten_twist(tw: Twist, prefix=""):
    return {
        f"{prefix}lin_x": tw.linear.x,
        f"{prefix}lin_y": tw.linear.y,
        f"{prefix}lin_z": tw.linear.z,
        f"{prefix}ang_x": tw.angular.x,
        f"{prefix}ang_y": tw.angular.y,
        f"{prefix}ang_z": tw.angular.z,
    }

def flatten_vec3(v: Vector3, prefix=""):
    return {f"{prefix}x": v.x, f"{prefix}y": v.y, f"{prefix}z": v.z}

class SimpleCsvWriter:
    """Tiny CSV writer that writes header once and appends rows."""
    def __init__(self, path: Path, field_order):
        self.path = path
        ensure_dir(self.path)
        self.field_order = field_order[:]  # fixed order
        self.header_written = self.path.exists() and self.path.stat().st_size > 0
        self.buffer = []
        self.n_since_flush = 0

    def add(self, row):
        # Fill missing keys with empty string for stability
        out = {k: row.get(k, "") for k in self.field_order}
        self.buffer.append(out)

    def flush(self, force=False):
        if not self.buffer and not force:
            return
        mode = "w" if not self.header_written else "a"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.field_order)
            if not self.header_written:
                w.writeheader()
                self.header_written = True
            w.writerows(self.buffer)
        self.buffer = []

class LoggerNode:
    def __init__(self):
        rospy.loginfo("Starting state+mocap logger node…")
        # Params
        self.state_topic   = rospy.get_param("~state_topic", "/volley_drone/agiros_pilot/state")
        self.mocap_topic   = rospy.get_param("~mocap_topic", "/mocap/volley_drone/pose")
        self.state_csv     = rospy.get_param("~state_csv", "./ros_logs/state_log.csv")
        self.mocap_csv     = rospy.get_param("~mocap_csv", "./ros_logs/mocap_log.csv")
        self.flush_every_n = 200

        # Fixed schemas (simple, readable)
        state_fields = [
            # timing
            "header_stamp", "t", "header_seq", "header_frame",
            # pose
            "pose_pos_x","pose_pos_y","pose_pos_z",
            "pose_quat_x","pose_quat_y","pose_quat_z","pose_quat_w",
            # velocity
            "vel_lin_x","vel_lin_y","vel_lin_z","vel_ang_x","vel_ang_y","vel_ang_z",
            # acceleration
            "acc_lin_x","acc_lin_y","acc_lin_z","acc_ang_x","acc_ang_y","acc_ang_z",
            # biases
            "acc_bias_x","acc_bias_y","acc_bias_z",
            "gyr_bias_x","gyr_bias_y","gyr_bias_z",
            # jerk/snap
            "jerk_x","jerk_y","jerk_z",
            "snap_x","snap_y","snap_z",
            # motors
            "motors_len","motors",  # motors is a comma-separated string
        ]
        mocap_fields = [
            "header_stamp","header_seq","header_frame",
            "pos_x","pos_y","pos_z",
            "quat_x","quat_y","quat_z","quat_w",
        ]
        
        # Check if file already exist and add int to filename if so
        self.state_csv = Path(self.state_csv).expanduser().resolve()
        self.mocap_csv = Path(self.mocap_csv).expanduser().resolve()

        if self.state_csv.exists() or self.mocap_csv.exists():
            self.state_csv, self.mocap_csv = make_unique_pair(self.state_csv, self.mocap_csv)
            rospy.loginfo("Renamed existing log files to avoid overwriting.")

        rospy.loginfo("Logging state -> %s", self.state_csv)
        rospy.loginfo("Logging mocap -> %s", self.mocap_csv)

        self.state_writer = SimpleCsvWriter(self.state_csv, state_fields)
        self.mocap_writer = SimpleCsvWriter(self.mocap_csv, mocap_fields)
        self.msgs_since_flush = 0
        
        # Subs
        rospy.Subscriber(self.state_topic, StateMsg, self._state_cb, queue_size=100)
        rospy.Subscriber(self.mocap_topic, PoseStamped, self._mocap_cb, queue_size=100)
        
        # Clean shutdown
        signal.signal(signal.SIGINT, self._sigint)
        rospy.on_shutdown(self._on_shutdown)


    def _state_cb(self, msg: StateMsg):
        row = {
            "header_stamp": msg_time(msg.header),
            "t": float(getattr(msg, "t", float("nan"))),
            "header_seq": getattr(msg.header, "seq", -1),
            "header_frame": getattr(msg.header, "frame_id", ""),
        }
        row.update(flatten_pose(msg.pose, "pose_"))
        row.update(flatten_twist(msg.velocity, "vel_"))
        row.update(flatten_twist(msg.acceleration, "acc_"))
        row.update(flatten_vec3(msg.acc_bias, "acc_bias_"))
        row.update(flatten_vec3(msg.gyr_bias, "gyr_bias_"))
        row.update(flatten_vec3(msg.jerk, "jerk_"))
        row.update(flatten_vec3(msg.snap, "snap_"))

        motors = list(getattr(msg, "motors", []))
        row["motors_len"] = len(motors)
        # Store as comma-separated string; easy to parse later
        row["motors"] = ",".join(f"{m:.9g}" for m in motors)

        self.state_writer.add(row)
        self._maybe_flush()

    def _mocap_cb(self, msg: PoseStamped):
        row = {
            "header_stamp": msg_time(msg.header),
            "header_seq": msg.header.seq,
            "header_frame": msg.header.frame_id,
        }
        row.update(flatten_pose(msg.pose, ""))  # pos_x, quat_x, ...
        self.mocap_writer.add(row)
        self._maybe_flush()

    def _maybe_flush(self):
        self.msgs_since_flush += 1
        if self.flush_every_n > 0 and self.msgs_since_flush >= self.flush_every_n:
            self.state_writer.flush()
            self.mocap_writer.flush()
            self.msgs_since_flush = 0

    def _sigint(self, *_):
        rospy.loginfo("SIGINT: flushing logs…")
        self.state_writer.flush(force=True)
        self.mocap_writer.flush(force=True)
        rospy.signal_shutdown("SIGINT")

    def _on_shutdown(self):
        # Final flush on normal shutdown
        self.state_writer.flush(force=True)
        self.mocap_writer.flush(force=True)

def main():
    rospy.init_node("state_mocap_logger", anonymous=False)
    LoggerNode()
    rospy.spin()

if __name__ == "__main__":
    main()