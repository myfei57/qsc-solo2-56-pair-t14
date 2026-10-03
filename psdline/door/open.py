"""开门动作。

开门是"全部对位段落盘之后"的第一个动作：读不到对位记录、对位记录没落盘、
分段会话仍在收集或处于挂起，都直接拒绝。长编组任一段超差都会把整车置为
挂起，因此这里不可能读到"半个对位"就放行。开门事实写成
``door/<train>/open`` 文档，发车前由开门状态推导。
"""

from __future__ import annotations

from ..errors import InterlockError
from .door import DoorController


def open_doc_id(train_id: str) -> str:
    return f"door/{train_id}/open"


def open_doors(controller: DoorController, train_id: str, at: float) -> dict:
    """按对位记录打开一列车的站台门。"""

    alignment = controller.align.get(train_id)
    if alignment is None:
        raise InterlockError("列车没有对位记录", component="door")
    if not alignment.durable:
        if controller.align.held(train_id):
            raise InterlockError(
                "列车分段对位挂起：有段超差，禁止开门，需重新对位", component="door"
            )
        session = controller.align.get_session(train_id)
        if session is not None:
            pending = session.pending_indices()
            raise InterlockError(
                f"分段对位尚未全部落盘（待确认段 {pending}），禁止开门",
                component="door",
            )
        raise InterlockError("列车对位尚未落盘", component="door")
    controller.store.write(
        open_doc_id(train_id),
        {"train_id": train_id, "position_m": alignment.position_m, "at": at},
    )
    controller.note_open()
    return {
        "ok": True,
        "train_id": train_id,
        "position_m": alignment.position_m,
        "segment_count": alignment.segment_count,
        "violations": [],
    }


def opened(controller: DoorController, train_id: str) -> bool:
    """判断一列车的站台门是否已打开。"""

    return controller.store.exists(open_doc_id(train_id))
