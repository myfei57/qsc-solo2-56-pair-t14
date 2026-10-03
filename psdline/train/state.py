"""列车状态汇总。"""

from __future__ import annotations

from .align import HELD, AlignRegistry
from .door import DoorStateRegistry
from .position import PositionTracker


def train_state(
    train_id: str,
    align: AlignRegistry,
    doors: DoorStateRegistry,
    positions: PositionTracker,
) -> dict:
    """汇总一列车的对位、车门与位置状态。

    旧单段数据保持原有五个键；分段会话额外附带会话状态与各段明细，供控制台
    判断"哪些段已确认、是否挂起、还差几段"。
    """

    alignment = align.get(train_id)
    session = align.get_session(train_id)
    door_open = doors.is_open(train_id)
    position_m = positions.get(train_id)
    state = {
        "train_id": train_id,
        "aligned": bool(alignment and alignment.durable),
        "alignment_m": alignment.position_m if alignment else 0,
        "doors_open": door_open,
        "position_m": position_m,
    }
    if session is not None:
        state["alignment_state"] = session.state
        state["alignment_held"] = session.state == HELD
        state["alignment_attempt"] = session.attempt
        state["segment_count"] = session.segment_count
        state["confirmed_segments"] = session.confirmed_indices()
        state["held_segments"] = session.held_indices()
        state["pending_segments"] = session.pending_indices()
        state["segments"] = {
            index: segment.as_dict() for index, segment in sorted(session.segments.items())
        }
    return state
