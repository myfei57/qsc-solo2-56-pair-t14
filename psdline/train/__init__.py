"""列车到发。

列车保存线路、站台与当前位置；对位记录按 attempt/segment 分文档保存每一段的停稳位置，
只有全段 confirmed 后会话才 completed。车门状态保存"车门已确认打开"。控制链要求
"全分段对位先落盘、再开门"，因此对位与车门分成两份文档。旧单位置文档仍可读取，
新到站自动升级为单段 v2 会话。
"""

from __future__ import annotations

from .align import (
    COMPLETED,
    CONFIRMED,
    DEPARTED,
    OPENED,
    PENDING,
    REJECTED,
    SUSPENDED,
    AlignRegistry,
    Alignment,
    SegmentAlignment,
    SegmentMeasurement,
)
from .door import DoorState, DoorStateRegistry
from .dwell import Dwell, DwellRegistry
from .position import PositionTracker
from .route import Route, RouteRegistry
from .state import train_state
from .train import Train, TrainRegistry

__all__ = [
    "COMPLETED",
    "CONFIRMED",
    "DEPARTED",
    "OPENED",
    "PENDING",
    "REJECTED",
    "SUSPENDED",
    "AlignRegistry",
    "Alignment",
    "SegmentAlignment",
    "SegmentMeasurement",
    "DoorState",
    "DoorStateRegistry",
    "Dwell",
    "DwellRegistry",
    "PositionTracker",
    "Route",
    "RouteRegistry",
    "Train",
    "TrainRegistry",
    "train_state",
]
