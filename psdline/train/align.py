"""停稳对位记录（分段）。

长编组列车只认一个整车位置是不安全的：车头到位时车尾未必到位。对位因此按
"段"组织：一次停车（stop）开一个对位会话，会话内每一段各自落盘，全部段确认
后整车对位才算落盘，门控只认这个结论。

状态机
------

会话状态（session.state）：

* ``seeking``   正在收集各段对位，尚无段超差；
* ``held``      至少一段超差，整车挂起，拒绝继续收报与开门，必须重新对位；
* ``confirmed`` 全部段已确认并各自落盘，允许开门；
* ``cleared``   本次停车结束（发车/清扫），结论归档，不再放行。

段状态只有 ``confirmed`` / ``held`` 两种；挂起是整车级的。

重新对位（续作）
----------------

``held`` 后调用 :meth:`AlignRegistry.realign` 开始新的一次对位尝试（attempt
递增）。原超差段作为"锚点"：司机重新对位后锚点的新位置与旧位置之差即整车
平移量。上一尝试中已确认的段不强制重新测量，而是按平移量推算新位置——推算
后仍在公差内就继承确认（续作从已确认段接着来），推算后超差则作废，退回待
报。锚点本身仍超差时整车立即重新挂起。

时序、幂等与部分失败
--------------------

* 每段先写段文档（attempt 级路径），再更新会话头；掉电后段文档在而头没更新
  时，下一次收报会从盘上补齐，头里声称确认但段文档缺失则读时降级，门控永远
  不可能基于"半个落盘"放行。
* 同一段在同一尝试内重复上报：位置一致（或 request_id 一致）视为重放直接
  返回；位置不一致拒绝。重新对位以 request_id 去重。
* 段文档按 stop/attempt 路径永久保留，审计流水只追加，历史不可变。
* 旧数据兼容：只报一个位置的旧落盘文档（``train/<id>/align``）仍被识别为
  单段会话；旧的 :meth:`align` 调用路径与旧文档格式保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import AlignmentError
from ..store import JsonStore

SEEKING = "seeking"
HELD = "held"
CONFIRMED = "confirmed"
CLEARED = "cleared"

SEG_CONFIRMED = "confirmed"
SEG_HELD = "held"

# 浮点比较容差（毫米以下视为同值），避免 0.7999999 这类舍入把刚好合格的段挂起。
_EPS = 1e-9


def _within(offset: float, tolerance: float) -> bool:
    return offset <= tolerance + _EPS

# 段文档路径前缀，恢复扫描时按它过滤。
def _seg_prefix(train_id: str, stop_id: str) -> str:
    return f"train/{train_id}/align/{stop_id}/"


@dataclass(frozen=True)
class SegmentReport:
    """一段的对位记录（位置可亚米精度，公差是亚米级的）。"""

    index: int
    target_m: int
    position_m: float
    offset_m: float
    status: str
    attempt: int
    at: float
    durable: bool = True
    carried: bool = False
    request_id: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "target_m": self.target_m,
            "position_m": self.position_m,
            "offset_m": self.offset_m,
            "status": self.status,
            "attempt": self.attempt,
            "at": self.at,
            "durable": self.durable,
            "carried": self.carried,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SegmentReport":
        return cls(
            index=int(payload.get("index", 0)),
            target_m=int(payload.get("target_m", 0)),
            position_m=float(payload.get("position_m", 0.0)),
            offset_m=float(payload.get("offset_m", 0.0)),
            status=str(payload.get("status", SEG_CONFIRMED)),
            attempt=int(payload.get("attempt", 1)),
            at=float(payload.get("at", 0.0)),
            durable=bool(payload.get("durable", True)),
            carried=bool(payload.get("carried", False)),
            request_id=str(payload.get("request_id", "")),
        )


@dataclass
class AlignmentSession:
    """一次停车的分段对位会话。"""

    train_id: str
    stop_id: str
    station_id: str
    attempt: int
    targets: list[int]
    state: str = SEEKING
    tolerance_m: float = 0.8
    segments: dict[int, SegmentReport] = field(default_factory=dict)
    anchor: dict = field(default_factory=dict)
    begun_at: float = 0.0
    updated_at: float = 0.0
    last_realign_request: str = ""

    @property
    def segment_count(self) -> int:
        return len(self.targets)

    def confirmed_indices(self) -> list[int]:
        return sorted(
            index
            for index, segment in self.segments.items()
            if segment.status == SEG_CONFIRMED and segment.attempt == self.attempt
        )

    def held_indices(self) -> list[int]:
        return sorted(
            index
            for index, segment in self.segments.items()
            if segment.status == SEG_HELD and segment.attempt == self.attempt
        )

    def pending_indices(self) -> list[int]:
        confirmed = set(self.confirmed_indices())
        held = set(self.held_indices())
        return [index for index in range(self.segment_count) if index not in confirmed | held]

    def all_confirmed(self) -> bool:
        return len(self.confirmed_indices()) == self.segment_count and self.segment_count > 0

    def anchor_position(self) -> float:
        """整车基准位置：优先锚点段，没有则取第一段。"""

        if self.anchor:
            return float(self.anchor.get("position_m", 0.0))
        first = self.segments.get(0)
        return first.position_m if first else 0.0

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "stop_id": self.stop_id,
            "station_id": self.station_id,
            "attempt": self.attempt,
            "state": self.state,
            "tolerance_m": self.tolerance_m,
            "targets": list(self.targets),
            "segments": {str(index): segment.as_dict() for index, segment in self.segments.items()},
            "anchor": dict(self.anchor),
            "begun_at": self.begun_at,
            "updated_at": self.updated_at,
            "last_realign_request": self.last_realign_request,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AlignmentSession":
        raw_segments = payload.get("segments", {})
        segments: dict[int, SegmentReport] = {}
        if isinstance(raw_segments, dict):
            for key, value in raw_segments.items():
                if isinstance(value, dict):
                    segments[int(key)] = SegmentReport.from_dict(value)
        return cls(
            train_id=str(payload.get("train_id", "")),
            stop_id=str(payload.get("stop_id", "")),
            station_id=str(payload.get("station_id", "")),
            attempt=int(payload.get("attempt", 1)),
            targets=[int(item) for item in payload.get("targets", [])],
            state=str(payload.get("state", SEEKING)),
            tolerance_m=float(payload.get("tolerance_m", 0.8)),
            segments=segments,
            anchor=dict(payload.get("anchor", {})) if isinstance(payload.get("anchor"), dict) else {},
            begun_at=float(payload.get("begun_at", 0.0)),
            updated_at=float(payload.get("updated_at", 0.0)),
            last_realign_request=str(payload.get("last_realign_request", "")),
        )


@dataclass(frozen=True)
class Alignment:
    """一次停稳对位（旧单段视图，兼容只认一个位置的调用方）。"""

    train_id: str
    position_m: int
    durable: bool
    at: float
    state: str = ""
    segment_count: int = 1

    def as_dict(self) -> dict:
        # 旧落盘格式保持四个键，旧数据原样可读。
        return {
            "train_id": self.train_id,
            "position_m": self.position_m,
            "durable": self.durable,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Alignment":
        return cls(
            train_id=str(payload.get("train_id", "")),
            position_m=int(payload.get("position_m", 0)),
            durable=bool(payload.get("durable", False)),
            at=float(payload.get("at", 0.0)),
            state=str(payload.get("state", CONFIRMED if payload.get("durable") else "")),
            segment_count=int(payload.get("segment_count", 1)),
        )


class AlignRegistry:
    """对位记录读写：分段会话 + 旧单段文档。"""

    def __init__(self, store: JsonStore, tolerance_m: float = 0.8) -> None:
        self._store = store
        self._tolerance_m = tolerance_m

    @property
    def tolerance_m(self) -> float:
        return self._tolerance_m

    @staticmethod
    def _flat_doc_id(train_id: str) -> str:
        return f"train/{train_id}/align"

    @staticmethod
    def _head_doc_id(train_id: str) -> str:
        return f"train/{train_id}/align/session"

    @staticmethod
    def _segment_doc_id(
        train_id: str, stop_id: str, attempt: int, index: int
    ) -> str:
        return f"train/{train_id}/align/{stop_id}/{attempt}/{index}"

    # ---- 旧单段路径（兼容只报一个位置的数据与调用） ------------------

    def align(self, train_id: str, position_m: int, stop_position_m: int, at: float) -> Alignment:
        """登记一次单段对位；偏差超公差时拒绝，不覆盖上一次有效记录。"""

        if not train_id:
            raise AlignmentError("对位的列车标识为空", component="train")
        offset = abs(position_m - stop_position_m)
        if not _within(offset, self._tolerance_m):
            raise AlignmentError(
                f"对位偏差 {offset:.3f} m 超出公差 {self._tolerance_m:.3f} m",
                component="train",
            )
        record = Alignment(
            train_id=train_id, position_m=position_m, durable=True, at=at
        )
        # 旧单段上报只能取代已经结束（cleared/seeking 残留）的分段会话；挂起中的
        # 会话受保护，旧路径不能把它覆盖掉。注意顺序：先处理头再写单段文档，
        # 因为头文档是单段文档路径的子节点。
        existing = self._load_head(train_id)
        if existing is not None:
            if existing.state == HELD:
                raise AlignmentError(
                    "列车分段对位挂起中，单位置上报不能取代挂起会话",
                    component="train",
                )
            self._store.remove(self._head_doc_id(train_id))
        self._store.write(self._flat_doc_id(train_id), record.as_dict())
        return record

    # ---- 分段会话 -----------------------------------------------------

    def begin(
        self,
        train_id: str,
        stop_id: str,
        station_id: str,
        targets: list[int],
        at: float,
    ) -> AlignmentSession:
        """开始一次停车的分段对位会话。"""

        if not train_id:
            raise AlignmentError("对位的列车标识为空", component="train")
        if not stop_id:
            raise AlignmentError("对位停车标识为空", component="train")
        if not targets:
            raise AlignmentError("对位段目标里程为空", component="train")

        current = self._load_head(train_id)
        if current is not None and current.state == HELD:
            raise AlignmentError(
                f"列车对位挂起中（停车 {current.stop_id} 第 {current.attempt} 次尝试），"
                "必须先重新对位",
                component="train",
            )

        session = AlignmentSession(
            train_id=train_id,
            stop_id=stop_id,
            station_id=station_id,
            attempt=1,
            targets=[int(item) for item in targets],
            state=SEEKING,
            tolerance_m=self._tolerance_m,
            begun_at=at,
            updated_at=at,
        )
        # 新会话取代旧的单段文档与旧头，旧段文档按路径保留不动。
        self._store.remove(self._flat_doc_id(train_id))
        self._store.write(self._head_doc_id(train_id), session.as_dict())
        return session

    def report(
        self,
        train_id: str,
        index: int,
        position_m: float,
        at: float,
        request_id: str = "",
        attempt: int | None = None,
    ) -> AlignmentSession:
        """上报一段的停稳位置；该段超差时整车挂起并抛出异常。

        ``attempt`` 给出报文所属的对位尝试号（现场报文在会话开始时拿到）。
        传入时必须与当前尝试一致；重新对位后迟到的旧尝试报文直接拒绝，避免
        旧时序的位置污染新一轮对位。
        """

        session = self.require_session(train_id)
        session = self._recover_durable_writes(session)
        if attempt is not None and int(attempt) != session.attempt:
            raise AlignmentError(
                f"报文属于第 {attempt} 次对位尝试，当前已是第 {session.attempt} 次，"
                "拒绝过期报文",
                component="train",
            )
        if session.state == HELD:
            raise AlignmentError(
                f"列车对位挂起（第 {session.held_indices()} 段超差），"
                "本停车次不再接收分段上报，必须重新对位",
                component="train",
            )
        if not 0 <= index < session.segment_count:
            raise AlignmentError(
                f"段号 {index} 超出 0..{session.segment_count - 1}",
                component="train",
            )
        if session.state == CLEARED:
            raise AlignmentError("本次停车对位已结束", component="train")

        existing = session.segments.get(index)
        if existing is not None and existing.attempt == session.attempt:
            if existing.status == SEG_CONFIRMED:
                # 幂等：同一位置（或同一请求号）的重放直接返回，不重复落盘。
                if existing.position_m == position_m and (
                    not request_id or existing.request_id == request_id
                ):
                    return session
                raise AlignmentError(
                    f"第 {index} 段已确认（{existing.position_m} m），"
                    f"不能改写为 {position_m} m",
                    component="train",
                )

        target = session.targets[index]
        offset = abs(float(position_m) - target)
        segment = SegmentReport(
            index=index,
            target_m=target,
            position_m=float(position_m),
            offset_m=float(offset),
            status=SEG_CONFIRMED,
            attempt=session.attempt,
            at=at,
            request_id=request_id,
        )
        # 段先落盘，再更新头。
        self._store.write(
            self._segment_doc_id(train_id, session.stop_id, session.attempt, index),
            segment.as_dict(),
        )
        session.segments[index] = segment

        if not _within(offset, session.tolerance_m):
            held = SegmentReport(
                index=index,
                target_m=target,
                position_m=float(position_m),
                offset_m=float(offset),
                status=SEG_HELD,
                attempt=session.attempt,
                at=at,
                request_id=request_id,
            )
            self._store.write(
                self._segment_doc_id(train_id, session.stop_id, session.attempt, index),
                held.as_dict(),
            )
            session.segments[index] = held
            session.state = HELD
            session.anchor = {
                "index": index,
                "target_m": target,
                "position_m": float(position_m),
                "offset_m": float(offset),
                "attempt": session.attempt,
            }
            session.updated_at = at
            self._store.write(self._head_doc_id(train_id), session.as_dict())
            error = AlignmentError(
                f"第 {index} 段对位偏差 {offset:.3f} m 超出公差 "
                f"{session.tolerance_m:.3f} m，整车挂起",
                component="train",
            )
            error.session = session  # type: ignore[attr-defined]
            raise error

        if session.all_confirmed():
            session.state = CONFIRMED
        session.updated_at = at
        self._store.write(self._head_doc_id(train_id), session.as_dict())
        return session

    def realign(
        self,
        train_id: str,
        anchor_position_m: int,
        at: float,
        request_id: str = "",
    ) -> AlignmentSession:
        """挂起后重新对位：锚点平移复核已确认段，开启新一次尝试。"""

        session = self.require_session(train_id)
        # 幂等优先于状态检查：网络重传同一个重新对位请求时，即便会话已被前一次
        # 请求推进出挂起态，也应原样返回，不能报错让上层误以为失败。
        if request_id and request_id == session.last_realign_request:
            return session
        if session.state != HELD:
            raise AlignmentError(
                "只有挂起中的对位会话才能重新对位", component="train"
            )
        if not session.anchor:
            raise AlignmentError("挂起会话缺少锚点段，无法重新对位", component="train")

        new_attempt = session.attempt + 1
        shift = float(anchor_position_m) - float(session.anchor["position_m"])
        anchor_index = int(session.anchor["index"])
        anchor_target = int(session.anchor["target_m"])

        carried: dict[int, SegmentReport] = {}
        for index in session.confirmed_indices():
            previous = session.segments[index]
            projected = previous.position_m + shift
            projected_offset = abs(projected - previous.target_m)
            if _within(projected_offset, session.tolerance_m):
                inherited = SegmentReport(
                    index=index,
                    target_m=previous.target_m,
                    position_m=projected,
                    offset_m=float(projected_offset),
                    status=SEG_CONFIRMED,
                    attempt=new_attempt,
                    at=at,
                    carried=True,
                )
                self._store.write(
                    self._segment_doc_id(train_id, session.stop_id, new_attempt, index),
                    inherited.as_dict(),
                )
                carried[index] = inherited
            # 推算后超差的已确认段：不继承，退回待报，旧段文档保留。

        anchor_offset = abs(float(anchor_position_m) - anchor_target)
        anchor_status = SEG_CONFIRMED if _within(anchor_offset, session.tolerance_m) else SEG_HELD
        anchor_segment = SegmentReport(
            index=anchor_index,
            target_m=anchor_target,
            position_m=float(anchor_position_m),
            offset_m=float(anchor_offset),
            status=anchor_status,
            attempt=new_attempt,
            at=at,
            request_id=request_id,
        )
        self._store.write(
            self._segment_doc_id(train_id, session.stop_id, new_attempt, anchor_index),
            anchor_segment.as_dict(),
        )
        carried[anchor_index] = anchor_segment

        session.attempt = new_attempt
        session.segments = carried
        session.updated_at = at
        session.last_realign_request = request_id
        if anchor_status == SEG_HELD:
            session.state = HELD
            session.anchor = {
                "index": anchor_index,
                "target_m": anchor_target,
                "position_m": float(anchor_position_m),
                "offset_m": float(anchor_offset),
                "attempt": new_attempt,
            }
        else:
            session.anchor = {}
            session.state = CONFIRMED if len(carried) == session.segment_count else SEEKING
        self._store.write(self._head_doc_id(train_id), session.as_dict())
        return session

    def clear(self, train_id: str) -> bool:
        """结束本次停车对位：会话置 cleared（段文档保留），旧文档则删除。"""

        session = self._load_head(train_id)
        if session is not None:
            if session.state == CLEARED:
                return False
            session.state = CLEARED
            self._store.write(self._head_doc_id(train_id), session.as_dict())
            return True
        return self._store.remove(self._flat_doc_id(train_id))

    # ---- 读取 ---------------------------------------------------------

    def get(self, train_id: str) -> Alignment | None:
        """对位视图：优先分段会话，其次旧单段文档，都没有返回 ``None``。"""

        session = self.get_session(train_id)
        if session is not None:
            return Alignment(
                train_id=train_id,
                position_m=session.anchor_position(),
                durable=session.state == CONFIRMED and self._all_durable(session),
                at=session.updated_at,
                state=session.state,
                segment_count=session.segment_count,
            )
        raw = self._store.read(self._flat_doc_id(train_id))
        if not isinstance(raw, dict):
            return None
        return Alignment.from_dict(raw)

    def get_session(self, train_id: str) -> AlignmentSession | None:
        """读取分段对位会话；缺失段文档时按盘上事实降级。"""

        session = self._load_head(train_id)
        if session is None:
            return None
        return self._downgrade_missing(session)

    def require_session(self, train_id: str) -> AlignmentSession:
        """读取会话，不存在时抛异常。"""

        session = self.get_session(train_id)
        if session is None:
            raise AlignmentError("列车没有分段对位会话", component="train")
        return session

    def durable(self, train_id: str) -> bool:
        """整车对位是否已全部落盘（门控唯一放行依据）。"""

        record = self.get(train_id)
        return bool(record and record.durable)

    def held(self, train_id: str) -> bool:
        """列车对位是否处于挂起。"""

        session = self.get_session(train_id)
        return bool(session and session.state == HELD)

    # ---- 内部 ---------------------------------------------------------

    def _load_head(self, train_id: str) -> AlignmentSession | None:
        raw = self._store.read(self._head_doc_id(train_id))
        if not isinstance(raw, dict):
            return None
        return AlignmentSession.from_dict(raw)

    def _segment_durable(
        self, train_id: str, stop_id: str, attempt: int, index: int
    ) -> bool:
        raw = self._store.read(self._segment_doc_id(train_id, stop_id, attempt, index))
        return (
            isinstance(raw, dict)
            and raw.get("status") == SEG_CONFIRMED
            and bool(raw.get("durable", True))
        )

    def _all_durable(self, session: AlignmentSession) -> bool:
        """会话头声称确认的每一段，盘上都必须有确认文档，缺一不可放行。"""

        for index in range(session.segment_count):
            if not self._segment_durable(
                session.train_id, session.stop_id, session.attempt, index
            ):
                return False
        return True

    def _downgrade_missing(self, session: AlignmentSession) -> AlignmentSession:
        """头里确认但段文档缺失（掉电/部分写入）的段降级为待报，重算状态。"""

        changed = False
        for index, segment in list(session.segments.items()):
            if segment.attempt != session.attempt:
                continue
            raw = self._store.read(
                self._segment_doc_id(
                    session.train_id, session.stop_id, session.attempt, index
                )
            )
            if not isinstance(raw, dict):
                del session.segments[index]
                changed = True
            elif segment.status == SEG_CONFIRMED and raw.get("status") != SEG_CONFIRMED:
                session.segments[index] = SegmentReport.from_dict(raw)
                changed = True
        if not changed:
            return session
        if session.state == HELD:
            # 挂起事实以盘上的 held 段文档为准；锚点段丢了说明写入未完成，
            # 退回报文收集，等待重新对位流程处理。
            if not session.held_indices():
                session.state = CONFIRMED if session.all_confirmed() else SEEKING
                session.anchor = {}
        elif session.state == CONFIRMED:
            session.state = CONFIRMED if session.all_confirmed() else SEEKING
        return session

    def _recover_durable_writes(self, session: AlignmentSession) -> AlignmentSession:
        """段文档已写、头没更新（写头前掉电）时，从盘上补齐当前尝试。"""

        prefix = _seg_prefix(session.train_id, session.stop_id)
        attempt_prefix = f"{prefix}{session.attempt}/"
        for doc_id in self._store.list_ids():
            if not doc_id.startswith(attempt_prefix):
                continue
            index = int(doc_id.rsplit("/", 1)[-1])
            if index in session.segments and session.segments[index].attempt == session.attempt:
                continue
            raw = self._store.read(doc_id)
            if isinstance(raw, dict):
                segment = SegmentReport.from_dict(raw)
                session.segments[index] = segment
                if segment.status == SEG_HELD and session.state != HELD:
                    session.state = HELD
                    session.anchor = {
                        "index": segment.index,
                        "target_m": segment.target_m,
                        "position_m": segment.position_m,
                        "offset_m": segment.offset_m,
                        "attempt": session.attempt,
                    }
        if session.state != HELD and session.all_confirmed():
            session.state = CONFIRMED
        return session
