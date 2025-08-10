from __future__ import annotations
from typing import Deque, List, TypeVar, Generic
import bounce_policy_package.observation.observation_backend as bc
T = TypeVar("T")                              # payload PyTree (arrays only)

@bc.dataclass
class RingBuf(Generic[T]):
    """Immutable ring buffer whose leaves all have the same leading axis (=capacity)."""
    data: T             # PyTree with shape (capacity, …) at every leaf
    idx:  bc.xp.int32     # slot where the *next* push will go (0 … capacity-1)

    # ————— helpers ————————————————————————
    @property
    def capacity(self) -> int:
        return bc.tree_leaves(self.data)[0].shape[0]

    # ----------- construction -----------------
    @classmethod
    def init(cls, example: T, capacity: int):
        """Allocate a buffer able to hold `example` for `capacity` steps."""
        def alloc(x): return bc.full((capacity, *x.shape), x, x.dtype)
        return cls(bc.tree_map(alloc, example), bc.xp.int32(0))

    # ----------- write ------------------------
    def push(self, x: T) -> "RingBuf[T]":
        """Insert `x` and return a *new* RingBuf (pure)."""
        assert x.shape == self.data.shape[1:], f"Shape mismatch: {x.shape} vs {self.data.shape}"
        def write(buf_leaf, val_leaf):
            start = (bc.squeeze(self.idx),) + (0,) * (buf_leaf.ndim - 1)
            # return buf_leaf.at[start].set(val_leaf)
            return bc.dynamic_update_slice(buf_leaf, val_leaf[None, ...], start)

        new_data = bc.tree_map(write, self.data, x)
        return RingBuf(new_data, (self.idx + 1) % self.capacity)

    # ----------- read (delay k) ---------------
    def read(self, k: int = 0) -> T:
        """
        Return payload written k steps ago.
        k = 0 → most recent, k = delay → delayed sample.
        """
        pos = bc.squeeze((self.idx - 1 - k) % self.capacity) 

        def take(buf_leaf):
            return bc.dynamic_slice(buf_leaf,
                                     (pos,) + (0,) * (buf_leaf.ndim - 1),
                                     (1,)  + buf_leaf.shape[1:])[0]

        return bc.tree_map(take, self.data)

    # ----------- read window ------------------
    def window(self, length: int, delay: int = 0) -> T:
        """
        Return a window of `length` consecutive entries ending `delay` steps ago.
        Shapes: (length, …) per leaf.
        """
        assert length + delay <= self.capacity
        start = (self.idx - length - delay) % self.capacity
        idxs = (start + bc.arange(length)) % self.capacity

        def gather(buf_leaf):
            return buf_leaf[idxs]

        return bc.tree_map(gather, self.data)
    
    def window_rev(self, length: int, delay: int = 0) -> T:
        """
        Return a window where element 0 is the freshest sample and
        element (length-1) is the oldest.  Shapes: (length, …).
        """
        # assert length + delay <= self.capacity
        # length = jnp.minimum(length, self.capacity - delay)
        start = bc.squeeze((self.idx - 1 - delay) % self.capacity)        # newest
        idxs  = (start - bc.arange(length)) % self.capacity  # descending
        def gather(buf_leaf):
            return buf_leaf[idxs]

        return bc.tree_map(gather, self.data)

    def __repr__(self):
        """ Print in order"""
        data = self.window_rev(self.capacity)
        return f"RingBuf(data={data}, idx={self.idx}, capacity={self.capacity})"
        

@bc.dataclass
class HistBuf(Generic[T]):
    """
    User-facing buffer with
      • `history`      – #steps fed to the agent
      • `delay`        – sensor latency in steps
      • internal `ring` sized at history+delay
    """
    ring:    RingBuf[T]
    history:  int = bc.field(pytree_node=False)
    delay:    int = bc.field(pytree_node=False)

    # -------- factory ---------------------------------
    @classmethod
    def init(cls, example: T, history: int, delay: int = 0):
        cap = history + delay
        ring = RingBuf.init(example, cap)
        return cls(ring, history, delay)

    # -------- API -------------------------------------
    def push(self, x: T) -> "HistBuf[T]":
        return self.replace(ring=self.ring.push(x))

    def latest(self) -> T:
        return self.ring.read(k=self.delay)          # delayed “present”
    
    def latest_no_delay(self) -> T:
        """Return the latest value without delay."""
        return self.ring.read(k=0)                   # most recent, no delay

    def stack(self) -> T:
        """Return `(history,) + shape` window ending at `delay`."""
        return self.ring.window_rev(length=self.history, delay=self.delay)
    
    def stack_no_delay(self) -> T:
        """Return `(history,) + shape` window ending at `delay=0`."""
        return self.ring.window_rev(length=self.history, delay=0)
    
# @_struct
# class RingBufDeque(Generic[T]):
#     """
#     A deque-like structure that allows pushing and popping elements
#     while maintaining a fixed capacity.
#     """
#     data: Deque[T]
    
#     @classmethod
#     def init(cls, example: T, capacity: int):
#         """Allocate a buffer able to hold `example` for `capacity` steps."""
#         d = deque(maxlen=capacity)
#         for _ in range(capacity):
#             d.append(example)
#         return cls(d)
    
#     def push(self, x: T) -> "RingBufDeque[T]":
#         """Insert `x` and return a *new* RingBufDeque (pure)."""
#         new_data = self.data.copy()
#         new_data.append(x)
#         return RingBufDeque(new_data)
    
#     def read(self, k: int = 0) -> T:
#         """
#         Return payload written k steps ago.
#         k = 0 → most recent, k = delay → delayed sample.
#         """
#         if k < 0 or k >= len(self.data):
#             raise IndexError("Index out of range")
#         return self.data[-(k + 1)]
    
#     def window(self, length: int, delay: int = 0) -> List[T]:
#         """
#         Return a window of `length` consecutive entries ending `delay` steps ago.
#         Shapes: (length, …) per leaf.
#         """
#         if length + delay > len(self.data):
#             raise ValueError("Requested length exceeds buffer capacity")
        
#         start = len(self.data) - length - delay
#         return list(self.data)[start:start + length]

#     def window_rev(self, length: int, delay: int = 0) -> List[T]:
#         """
#         Return a window where element 0 is the freshest sample and
#         element (length-1) is the oldest.  Shapes: (length, …).
#         """
#         if length + delay > len(self.data):
#             raise ValueError("Requested length exceeds buffer capacity")
        
#         start = len(self.data) - 1 - delay
#         return list(self.data)[start - length + 1:start + 1][::-1]
    
#     def __repr__(self):
#         """ Print in order"""
#         return f"RingBufDeque(data={list(self.data)})"
    
# @_struct
# class HistBufDeque(Generic[T]):
#     """
#     User-facing buffer with
#       • `history`      – #steps fed to the agent
#       • `delay`        – sensor latency in steps
#       • internal `ring` sized at history+delay
#     """
#     ring:    RingBufDeque[T]
#     history:  int = field(pytree_node=False)
#     delay:    int = field(pytree_node=False)

#     # -------- factory ---------------------------------
#     @classmethod
#     def init(cls, example: T, history: int, delay: int = 0):
#         cap: int = history + delay
#         ring = RingBufDeque.init(example, cap)
#         return cls(ring, history, delay)

#     # -------- API -------------------------------------
#     def push(self, x: T) -> "HistBufDeque[T]":
#         return self.replace(ring=self.ring.push(x))

#     def latest(self) -> T:
#         return self.ring.read(k=self.delay)          # delayed “present”
    
#     def latest_no_delay(self) -> T:
#         """Return the latest value without delay."""
#         return self.ring.read(k=0)                   # most recent, no delay

#     def stack(self) -> List[T]:
#         """Return `(history,) + shape` window ending at `delay`."""
#         return self.ring.window_rev(length=self.history, delay=self.delay)
    
#     def stack_no_delay(self) -> List[T]:
#         """Return `(history,) + shape` window ending at `delay=0`."""
#         return self.ring.window_rev(length=self.history, delay=0)
    

# ---------------------------------------------------------------------------
# OPTIONAL: convenience helper for “push every RingBuf field in a big state”
# ---------------------------------------------------------------------------
# def push_all(history_state, live_snapshot):
#     """
#     Given two *structurally identical* PyTrees where the first one’s leaves
#     may be RingBufs, push the corresponding live value into each RingBuf.
#     Non-RingBuf leaves are left untouched.
#     """
#     def maybe_push(buf, val):
#         return buf.push(val) if isinstance(buf, (HistBuf, RingBuf)) else buf

#     return tree_util.tree_map(maybe_push, history_state, live_snapshot)



if __name__ == "__main__":

    example = bc.xp.array([1, 0, 3])
    buf = RingBuf.init(example, capacity=5)
    # print(buf)

    buf = buf.push(bc.xp.array([4, 5, 6]))
    # print(buf)

    # print("Latest:", buf.read())
    # print("Window:", buf.window(length=3))
    
    hist_buf = HistBuf.init(example, history=4, delay=2)
    for i in range(5):
        hist_buf = hist_buf.push(bc.xp.array([i, i, i]))
        # print(f"After push {i}: {hist_buf.stack()}")

    # print(hist_buf.stack()[3][5])
    # print(hist_buf.stack())
    v = lambda x: bc.xp.ones((3,)) * x
    t = HistBuf.init(v(1), history=5, delay=2)
    for i in range(9):
        t = t.push(v(i))
        
    print("RingBuf:", t)
    # print("RingBuf:", t.latest_no_delay())
    # print("RingBuf:", t.ring.read(0))
    
    
        

        
    