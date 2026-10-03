"""列车状态汇总。"""

from __future__ import annotations

from .align import AlignRegistry
from .door import DoorStateRegistry
from .position import PositionTracker


def train_state(
    train_id: str,
    align: AlignRegistry,
    doors: DoorStateRegistry,
    positions: PositionTracker,
) -> dict:
    """汇总一列车的分段对位、车门与位置状态。"""

    alignment = align.session_view(train_id)
    door_open = doors.is_open(train_id)
    position_m = positions.get(train_id)
    return {
        "train_id": train_id,
        "aligned": bool(alignment and alignment["durable"]),
        "alignment_suspended": bool(alignment and alignment["suspended"]),
        "alignment_m": alignment["segments"][-1]["position_m"]
        if alignment and alignment["segments"]
        else 0,
        "alignment_attempt": alignment["attempt"] if alignment else 0,
        "segment_count": alignment["segment_count"] if alignment else 0,
        "confirmed_segments": alignment["confirmed_count"] if alignment else 0,
        "rejected_segments": list(alignment["rejected_segments"]) if alignment else [],
        "segments": list(alignment["segments"]) if alignment else [],
        "doors_open": door_open,
        "position_m": position_m,
    }
