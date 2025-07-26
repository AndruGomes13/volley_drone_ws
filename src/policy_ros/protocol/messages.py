import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Type

# === Base class for messages ===
class Message:
    type: str  # must be set in subclasses

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, s: str) -> 'Message':
        data = json.loads(s)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Message':
        raise NotImplementedError("Must implement in subclass")

# === Specific message types ===
@dataclass
class InferenceMsg(Message):
    type: str = "inference"
    obs: List[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InferenceMsg':
        return cls(obs=d["obs"])

@dataclass
class InferenceReply(Message):
    type: str = "reply"
    status: str = "ok"
    result: Optional[float] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InferenceReply':
        return cls(
            status=d["status"],
            result=d.get("result"),
            error=d.get("error"),
            traceback=d.get("traceback"),
        )

# === Dispatching ===

def parse_message(s: str) -> Message:
    d = json.loads(s)
    msg_type = d.get("type")

    if msg_type == "inference":
        return InferenceMsg.from_dict(d)
    elif msg_type == "reply":
        return InferenceReply.from_dict(d)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")