"""停稳对位记录（分段落盘）。

一次对位会话由若干段组成：每一段单独写入一个不可变 JSON 文档；只有所有段都为
``confirmed``，会话总文档才会进入 ``completed``，门控才允许开门。任一段超出公差时，
该段写成 ``rejected`` 并把整列车挂起；重新对位时开启新的 attempt，并重新核对此前
已确认前缀，随后逐段重新落盘。旧 attempt 只追加、不覆盖。

旧版本只有 ``train/<id>/align`` 一个文档。读取时仍把它识别成单段会话；新的到站会
清理旧门控记录并升级为 v2 文档，历史 attempt 分段文档不覆盖、不删除。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from ..errors import AlignmentError
from ..store import JsonStore

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
COMPLETED = "completed"
OPENED = "opened"
DEPARTED = "departed"
SUSPENDED = "suspended"


@dataclass(frozen=True)
class SegmentMeasurement:
    """上位机上报的一段实测位置。"""

    segment_id: str
    position_m: float
    at: float = 0.0


@dataclass(frozen=True)
class SegmentAlignment:
    """一段对位结果。"""

    train_id: str
    attempt: int
    index: int
    segment_id: str
    target_m: float
    position_m: float
    offset_m: float
    status: str
    at: float
    measured_at: float = 0.0
    durable: bool = True
    carried_from: int = 0
    request_id: str = ""

    def as_dict(self) -> dict:
        payload = {
            "train_id": self.train_id,
            "attempt": self.attempt,
            "index": self.index,
            "segment_id": self.segment_id,
            "target_m": self.target_m,
            "position_m": self.position_m,
            "offset_m": self.offset_m,
            "status": self.status,
            "durable": self.durable,
            "at": self.at,
            "measured_at": self.measured_at,
        }
        if self.carried_from:
            payload["carried_from"] = self.carried_from
        if self.request_id:
            payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SegmentAlignment":
        return cls(
            train_id=str(payload.get("train_id", "")),
            attempt=int(payload.get("attempt", 1)),
            index=int(payload.get("index", 0)),
            segment_id=str(payload.get("segment_id", "")),
            target_m=float(payload.get("target_m", 0.0)),
            position_m=float(payload.get("position_m", 0.0)),
            offset_m=float(payload.get("offset_m", 0.0)),
            status=str(payload.get("status", CONFIRMED)),
            at=float(payload.get("at", 0.0)),
            measured_at=float(payload.get("measured_at", 0.0)),
            durable=bool(payload.get("durable", True)),
            carried_from=int(payload.get("carried_from", 0)),
            request_id=str(payload.get("request_id", "")),
        )


@dataclass(frozen=True)
class Alignment:
    """兼容旧门控接口的整列对位摘要。"""

    train_id: str
    position_m: float
    durable: bool
    at: float
    station_id: str = ""
    attempt: int = 1
    state: str = COMPLETED

    def as_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "station_id": self.station_id,
            "position_m": self.position_m,
            "durable": self.durable,
            "at": self.at,
            "attempt": self.attempt,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Alignment":
        return cls(
            train_id=str(payload.get("train_id", "")),
            station_id=str(payload.get("station_id", "")),
            position_m=float(payload.get("position_m", 0.0)),
            durable=bool(payload.get("durable", False)),
            at=float(payload.get("at", 0.0)),
            attempt=int(payload.get("attempt", 1)),
            state=str(payload.get("state", COMPLETED)),
        )


class AlignRegistry:
    """分段对位会话读写。"""

    def __init__(
        self,
        store: JsonStore,
        tolerance_m: float = 0.8,
        max_segments: int = 16,
        stale_after_s: float = 8.0,
        future_skew_s: float = 2.0,
    ) -> None:
        self._store = store
        self._tolerance_m = tolerance_m
        self._max_segments = max_segments
        self._stale_after_s = stale_after_s
        self._future_skew_s = future_skew_s

    @property
    def tolerance_m(self) -> float:
        return self._tolerance_m

    @property
    def max_segments(self) -> int:
        return self._max_segments

    @staticmethod
    def _doc_id(train_id: str) -> str:
        return f"train/{train_id}/align"

    @staticmethod
    def _attempt_doc_id(train_id: str, attempt: int) -> str:
        return f"train/{train_id}/align/attempt/{attempt:04d}"

    @staticmethod
    def _segment_doc_id(train_id: str, attempt: int, index: int) -> str:
        return f"train/{train_id}/align/attempt/{attempt:04d}/segment/{index:04d}"

    @staticmethod
    def _request_doc_id(train_id: str, request_id: str) -> str:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return f"train/{train_id}/align/request/{digest}"

    @staticmethod
    def _fingerprint(targets: dict[str, float], tolerance_m: float) -> str:
        encoded = ",".join(f"{key}={targets[key]:.6f}" for key in targets)
        raw = f"{tolerance_m:.6f}|{encoded}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ---- 旧接口兼容 ---------------------------------------------------

    def align(self, train_id: str, position_m: float, stop_position_m: float, at: float) -> Alignment:
        """登记旧格式的单段对位；偏差超公差时拒绝，不覆盖已有记录。"""

        if not train_id:
            raise AlignmentError("对位的列车标识为空", component="train")
        offset = abs(float(position_m) - float(stop_position_m))
        if offset > self._tolerance_m:
            raise AlignmentError(
                f"对位偏差 {offset:.3f} m 超出公差 {self._tolerance_m:.3f} m",
                component="train",
            )
        record = Alignment(
            train_id=train_id,
            station_id="",
            position_m=float(position_m),
            durable=True,
            at=at,
        )
        self._store.write(self._doc_id(train_id), record.as_dict())
        return record

    def get(self, train_id: str) -> Alignment | None:
        """读取当前可用于门控的整列对位摘要；挂起、发车或不存在时返回 ``None``。"""

        session = self.active_session(train_id)
        if session is None or session.get("state") not in (COMPLETED, OPENED):
            return None
        segments = self._session_segments(session)
        confirmed = [item for item in segments if item["status"] == CONFIRMED]
        if len(confirmed) != int(session.get("segment_count", 0)):
            return None
        return Alignment(
            train_id=train_id,
            station_id=str(session.get("station_id", "")),
            position_m=float(confirmed[-1].get("position_m", 0.0)),
            durable=True,
            at=float(session.get("updated_at", session.get("created_at", 0.0))),
            attempt=int(session.get("attempt", 1)),
            state=str(session.get("state", COMPLETED)),
        )

    def durable(self, train_id: str) -> bool:
        """判断整列对位是否已全部落盘。"""

        record = self.get(train_id)
        return bool(record and record.durable)

    def clear(self, train_id: str) -> bool:
        """移走当前指针；历史 attempt 文档保留，供审计追踪。"""

        existed = self._store.exists(self._doc_id(train_id))
        self._store.remove(self._doc_id(train_id))
        return existed

    # ---- 会话 ---------------------------------------------------------

    def active_session(self, train_id: str) -> dict | None:
        """读取当前对位会话，兼容 v1 单文档。"""

        raw = self._store.read(self._doc_id(train_id))
        if not isinstance(raw, dict):
            return None
        if int(raw.get("schema_version", 1)) < 2:
            if not raw.get("durable", False):
                return None
            position = float(raw.get("position_m", 0.0))
            return {
                "schema_version": 1,
                "train_id": train_id,
                "station_id": str(raw.get("station_id", "")),
                "attempt": 1,
                "attempts_total": 1,
                "state": COMPLETED,
                "tolerance_m": self._tolerance_m,
                "layout_revision": int(raw.get("layout_revision", 0)),
                "targets": {"S1": position},
                "segment_count": 1,
                "fingerprint": self._fingerprint({"S1": position}, self._tolerance_m),
                "created_at": float(raw.get("at", 0.0)),
                "updated_at": float(raw.get("at", 0.0)),
                "legacy": True,
            }
        session = dict(raw)
        self._recover_current_attempt(session)
        return session

    def session_view(self, train_id: str) -> dict | None:
        """返回带逐段状态的只读视图。"""

        session = self.active_session(train_id)
        if session is None:
            return None
        segments = self._session_segments(session)
        state = str(session.get("state", PENDING))
        confirmed = sum(1 for item in segments if item.get("status") == CONFIRMED)
        rejected = [item["segment_id"] for item in segments if item.get("status") == REJECTED]
        view = {
            "train_id": train_id,
            "station_id": str(session.get("station_id", "")),
            "state": state,
            "durable": state in (COMPLETED, OPENED),
            "suspended": state == SUSPENDED,
            "attempt": int(session.get("attempt", 1)),
            "attempts_total": int(session.get("attempts_total", 1)),
            "carried_prefix": int(session.get("carried_prefix", 0)),
            "resume_from_attempt": int(session.get("resume_from_attempt", 0)),
            "segment_count": int(session.get("segment_count", 0)),
            "confirmed_count": confirmed,
            "rejected_segments": rejected,
            "tolerance_m": float(session.get("tolerance_m", self._tolerance_m)),
            "layout_revision": int(session.get("layout_revision", 0)),
            "targets": dict(session.get("targets", {})),
            "created_at": float(session.get("created_at", 0.0)),
            "updated_at": float(session.get("updated_at", 0.0)),
            "opened_at": float(session.get("opened_at", 0.0)),
            "departed_at": float(session.get("departed_at", 0.0)),
            "segments": segments,
        }
        if session.get("legacy"):
            view["legacy"] = True
        return view

    def start_session(
        self,
        train_id: str,
        station_id: str,
        targets: dict[str, float],
        at: float,
        layout_revision: int = 0,
        request_id: str = "",
    ) -> dict:
        """创建一段新的对位会话；同站台、同目标的待确认会话可幂等返回。"""

        if not train_id:
            raise AlignmentError("对位的列车标识为空", component="train")
        normalized = self._normalize_targets(targets)
        active = self.active_session(train_id)
        fingerprint = self._fingerprint(normalized, self._tolerance_m)
        if active is not None and active.get("state") != DEPARTED:
            if str(active.get("station_id", "")) != station_id:
                raise AlignmentError(
                    f"列车 {train_id} 仍有 {active.get('station_id')} 站未结束的对位会话",
                    component="train",
                )
            if active.get("state") == SUSPENDED:
                raise AlignmentError("对位已挂起，请从未确认段重新对位", component="train")
            if active.get("fingerprint") != fingerprint:
                raise AlignmentError("对位目标已变化，不能复用旧会话", component="train")
            return self.session_view(train_id)  # type: ignore[return-value]

        next_attempt = 1
        if active is not None and not active.get("legacy"):
            next_attempt = self._next_attempt(train_id)

        session = {
            "schema_version": 2,
            "train_id": train_id,
            "station_id": station_id,
            "attempt": next_attempt,
            "attempts_total": next_attempt,
            "state": PENDING,
            "tolerance_m": self._tolerance_m,
            "layout_revision": int(layout_revision),
            "targets": normalized,
            "segment_count": len(normalized),
            "fingerprint": fingerprint,
            "request_id": request_id,
            "created_at": at,
            "updated_at": at,
            "opened_at": 0.0,
            "departed_at": 0.0,
            "segments": [],
        }
        self._save_session(session)
        return self.session_view(train_id)  # type: ignore[return-value]

    def resume_session(
        self,
        train_id: str,
        station_id: str,
        at: float,
        targets: dict[str, float] | None = None,
        request_id: str = "",
    ) -> dict:
        """从挂起会话开启新一轮对位，并重新核对此前已确认的前缀段。"""

        session = self._require_active(train_id)
        if str(session.get("station_id", "")) != station_id:
            raise AlignmentError("重新对位的站台与挂起会话不一致", component="train")
        if session.get("state") != SUSPENDED:
            raise AlignmentError("只有挂起的对位会话可以重新对位", component="train")
        normalized = self._normalize_targets(targets or dict(session.get("targets", {})))
        fingerprint = self._fingerprint(normalized, self._tolerance_m)
        if fingerprint != session.get("fingerprint"):
            raise AlignmentError("对位目标已变化，不能续作旧会话", component="train")

        next_attempt = self._next_attempt(train_id)
        prior_attempt = int(session.get("attempt", 1))
        prior_segments = self._read_attempt_segments(train_id, prior_attempt)
        carried_prefix = 0
        for item in prior_segments:
            if item.get("status") != CONFIRMED:
                break
            carried_prefix += 1
        self._write_attempt(train_id, next_attempt, [])

        session["attempt"] = next_attempt
        session["attempts_total"] = next_attempt
        session["state"] = PENDING
        session["updated_at"] = at
        session["carried_prefix"] = carried_prefix
        session["resume_from_attempt"] = prior_attempt
        if request_id:
            session["resume_request_id"] = request_id
        self._save_session(session)
        return self.session_view(train_id)  # type: ignore[return-value]

    def report_segments(
        self,
        train_id: str,
        measurements: list[SegmentMeasurement | dict],
        at: float,
        request_id: str = "",
    ) -> dict:
        """逐段上报并落盘；任一段超差即整体挂起，后续段不再处理。"""

        session = self._require_active(train_id)
        state = str(session.get("state", PENDING))
        if state == SUSPENDED:
            raise AlignmentError("对位已挂起，请先重新对位", component="train")
        if state in (COMPLETED, OPENED, DEPARTED):
            return self.session_view(train_id)  # type: ignore[return-value]

        attempt = int(session.get("attempt", 1))
        targets = {str(key): float(value) for key, value in dict(session.get("targets", {})).items()}
        normalized = [self._measurement(item) for item in measurements]
        seen: set[str] = set()
        for measurement in normalized:
            if measurement.segment_id in seen:
                raise AlignmentError(f"分段 {measurement.segment_id} 重复上报", component="train")
            seen.add(measurement.segment_id)
            if measurement.segment_id not in targets:
                raise AlignmentError(f"未知对位分段: {measurement.segment_id}", component="train")

        ordered = sorted(normalized, key=lambda item: self._target_index(targets, item.segment_id))
        attempt_segments = self._read_attempt_segments(train_id, attempt)
        by_id = {str(item["segment_id"]): item for item in attempt_segments}
        for measurement in ordered:
            index = self._target_index(targets, measurement.segment_id)
            existing = by_id.get(measurement.segment_id)
            if existing is not None:
                if existing.get("status") == CONFIRMED and self._same_position(
                    float(existing.get("position_m", 0.0)), measurement.position_m
                ):
                    continue
                raise AlignmentError(
                    f"分段 {measurement.segment_id} 已落盘，不能改写历史记录",
                    component="train",
                )

            target_m = targets[measurement.segment_id]
            if index != len(attempt_segments):
                raise AlignmentError(
                    f"必须从下一个未确认分段续报：当前应报索引 {len(attempt_segments)}",
                    component="train",
                )
            self._check_measurement_time(measurement, at)
            offset = abs(measurement.position_m - target_m)
            carried_prefix = int(session.get("carried_prefix", 0))
            segment = SegmentAlignment(
                train_id=train_id,
                attempt=attempt,
                index=index,
                segment_id=measurement.segment_id,
                target_m=target_m,
                position_m=measurement.position_m,
                offset_m=offset,
                status=CONFIRMED if offset <= self._tolerance_m else REJECTED,
                at=at,
                measured_at=measurement.at,
                carried_from=int(session.get("resume_from_attempt", 0)) if index < carried_prefix else 0,
                request_id=request_id,
            ).as_dict()
            # 单文档先原子落盘，再更新会话指针；崩溃后可由 attempt 文档恢复。
            self._store.write(self._segment_doc_id(train_id, attempt, index), segment)
            attempt_segments.append(segment)
            attempt_segments.sort(key=lambda item: int(item["index"]))
            self._write_attempt(train_id, attempt, attempt_segments)
            if segment["status"] == REJECTED:
                session["state"] = SUSPENDED
                session["updated_at"] = at
                session["rejected_segment"] = measurement.segment_id
                session["rejected_offset_m"] = offset
                self._save_session(session)
                raise AlignmentError(
                    f"分段 {measurement.segment_id} 对位偏差 {offset:.3f} m 超出公差 "
                    f"{self._tolerance_m:.3f} m，整列已挂起",
                    component="train",
                )

        attempt_segments.sort(key=lambda item: int(item["index"]))
        self._write_attempt(train_id, attempt, attempt_segments)
        self._recover_current_attempt(session)
        confirmed = sum(
            1 for item in self._read_attempt_segments(train_id, attempt)
            if item["status"] == CONFIRMED
        )
        if confirmed == len(targets):
            session["state"] = COMPLETED
        session["updated_at"] = at
        self._save_session(session)
        return self.session_view(train_id)  # type: ignore[return-value]

    def mark_opened(self, train_id: str, at: float) -> dict:
        """全段落盘后登记开门事实。"""

        session = self._require_active(train_id)
        if session.get("state") not in (COMPLETED, OPENED):
            raise AlignmentError("全段落盘前不能开门", component="train")
        if session.get("state") == COMPLETED:
            session["state"] = OPENED
            session["opened_at"] = at
            session["updated_at"] = at
            self._save_session(session)
        return self.session_view(train_id)  # type: ignore[return-value]

    def mark_departed(self, train_id: str, at: float) -> dict:
        """发车后关闭当前对位会话；历史分段仍可查。"""

        session = self._require_active(train_id)
        if session.get("state") != DEPARTED:
            session["state"] = DEPARTED
            session["departed_at"] = at
            session["updated_at"] = at
            self._save_session(session)
        return self.session_view(train_id)  # type: ignore[return-value]  # type: ignore[return-value]

    def history(self, train_id: str) -> list[dict]:
        """返回某列车全部 attempt 的分段落盘记录。"""

        found: list[dict] = []
        prefix = f"train/{train_id}/align/attempt/"
        for doc_id in self._store.list_ids():
            if not doc_id.startswith(prefix) or "/segment/" in doc_id:
                continue
            raw = self._store.read(doc_id)
            if isinstance(raw, dict):
                found.append(raw)
        found.sort(key=lambda item: int(item.get("attempt", 0)))
        return found

    # ---- 幂等请求 -----------------------------------------------------

    def load_idempotent(self, train_id: str, station_id: str, request_id: str) -> dict | None:
        if not request_id:
            return None
        if len(request_id) > 200:
            raise AlignmentError("幂等标识过长（最多 200 字符）", component="train")
        raw = self._store.read(self._request_doc_id(train_id, request_id))
        if not isinstance(raw, dict):
            return None
        if str(raw.get("station_id", "")) != station_id:
            raise AlignmentError("幂等标识属于另一个站台", component="train")
        result = raw.get("response")
        return dict(result) if isinstance(result, dict) else None

    def save_idempotent(
        self,
        train_id: str,
        station_id: str,
        request_id: str,
        response: dict,
        at: float,
    ) -> None:
        if not request_id:
            return
        if len(request_id) > 200:
            raise AlignmentError("幂等标识过长（最多 200 字符）", component="train")
        self._store.write(
            self._request_doc_id(train_id, request_id),
            {
                "train_id": train_id,
                "station_id": station_id,
                "request_id": request_id,
                "at": at,
                "response": response,
            },
        )

    # ---- 内部 ---------------------------------------------------------

    def _next_attempt(self, train_id: str) -> int:
        attempts = []
        prefix = f"train/{train_id}/align/attempt/"
        for doc_id in self._store.list_ids():
            if not doc_id.startswith(prefix) or "/segment/" in doc_id:
                continue
            raw = self._store.read(doc_id)
            if isinstance(raw, dict):
                attempts.append(int(raw.get("attempt", 0)))
        return (max(attempts) if attempts else 0) + 1

    def _recover_current_attempt(self, session: dict) -> None:
        """按不可变分段文档重建 attempt 状态，处理写指针前掉电的部分失败。"""

        train_id = str(session.get("train_id", ""))
        attempt = int(session.get("attempt", 1))
        segments = self._rebuild_attempt(train_id, attempt)
        state = str(session.get("state", PENDING))
        if state not in (PENDING, COMPLETED):
            return
        if any(item.get("status") == REJECTED for item in segments):
            session["state"] = SUSPENDED
            rejected = next(item for item in segments if item.get("status") == REJECTED)
            session["rejected_segment"] = rejected.get("segment_id", "")
            session["rejected_offset_m"] = rejected.get("offset_m", 0.0)
        elif len(segments) == int(session.get("segment_count", 0)) and all(
            item.get("status") == CONFIRMED for item in segments
        ):
            session["state"] = COMPLETED
        if session.get("state") != state:
            self._save_session(session)

    def _rebuild_attempt(self, train_id: str, attempt: int) -> list[dict]:
        prefix = f"train/{train_id}/align/attempt/{attempt:04d}/segment/"
        raw = self._store.read(self._attempt_doc_id(train_id, attempt))
        if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
            return [dict(item) for item in raw["segments"] if isinstance(item, dict)]
        segments: list[dict] = []
        for doc_id in self._store.list_ids():
            if not doc_id.startswith(prefix):
                continue
            item = self._store.read(doc_id)
            if isinstance(item, dict):
                segments.append(dict(item))
        segments.sort(key=lambda value: int(value.get("index", 0)))
        if segments:
            self._write_attempt(train_id, attempt, segments)
        return segments

    def _require_active(self, train_id: str) -> dict:
        session = self.active_session(train_id)
        if session is None:
            raise AlignmentError(f"列车 {train_id} 没有对位会话", component="train")
        if session.get("legacy"):
            raise AlignmentError("旧版单段对位不能续报，请重新到站", component="train")
        return session

    def _check_measurement_time(self, measurement: SegmentMeasurement, at: float) -> None:
        """拒绝过期或来自未来的分段报文，防止旧位置把门打开。"""

        if not measurement.at:
            return
        age = at - measurement.at
        if age > self._stale_after_s:
            raise AlignmentError(
                f"分段 {measurement.segment_id} 位置报文已过期 {age:.1f}s",
                component="train",
            )
        if measurement.at - at > self._future_skew_s:
            raise AlignmentError(
                f"分段 {measurement.segment_id} 位置报文时间超过未来偏差 {self._future_skew_s:.1f}s",
                component="train",
            )

    def _normalize_targets(self, targets: dict[str, float]) -> dict[str, float]:
        if not isinstance(targets, dict) or not targets:
            raise AlignmentError("对位分段不能为空", component="train")
        if len(targets) > self._max_segments:
            raise AlignmentError(
                f"对位分段数 {len(targets)} 超过上限 {self._max_segments}",
                component="train",
            )
        normalized: dict[str, float] = {}
        for key, value in targets.items():
            segment_id = str(key).strip()
            if not segment_id or "/" in segment_id:
                raise AlignmentError(f"非法分段标识: {key}", component="train")
            value = float(value)
            if not math.isfinite(value):
                raise AlignmentError(f"分段 {key} 目标位置必须有限", component="train")
            normalized[segment_id] = value
        if len(normalized) != len(targets):
            raise AlignmentError("对位分段标识重复", component="train")
        # 保留调用方给出的物理顺序，不能按标识字典序（S10 会排到 S2 前）。
        return dict(normalized.items())

    @staticmethod
    def _measurement(item: SegmentMeasurement | dict) -> SegmentMeasurement:
        if isinstance(item, SegmentMeasurement):
            return item
        if not isinstance(item, dict):
            raise AlignmentError("分段对位数据必须是对象", component="train")
        segment_id = str(item.get("segment_id", "")).strip()
        if not segment_id:
            raise AlignmentError("缺少分段标识", component="train")
        try:
            position = float(item.get("position_m", 0.0))
        except (TypeError, ValueError) as error:
            raise AlignmentError(f"分段 {segment_id} 位置不是数值", component="train") from error
        if not math.isfinite(position):
            raise AlignmentError(f"分段 {segment_id} 位置必须有限", component="train")
        try:
            measured_at = float(item.get("at", item.get("measured_at", 0.0)))
        except (TypeError, ValueError) as error:
            raise AlignmentError(f"分段 {segment_id} 的报文时间不是数值", component="train") from error
        if not math.isfinite(measured_at):
            raise AlignmentError(f"分段 {segment_id} 的报文时间必须有限", component="train")
        return SegmentMeasurement(
            segment_id=segment_id, position_m=position, at=measured_at
        )

    @staticmethod
    def _target_index(targets: dict[str, float], segment_id: str) -> int:
        return list(targets.keys()).index(segment_id)

    @staticmethod
    def _same_position(left: float, right: float) -> bool:
        return abs(left - right) < 1e-9

    def _save_session(self, session: dict) -> None:
        self._store.write(self._doc_id(str(session["train_id"])), session)

    def _attempt_key(self, train_id: str, attempt: int) -> str:
        return self._attempt_doc_id(train_id, attempt)

    def _write_attempt(self, train_id: str, attempt: int, segments: list[dict]) -> None:
        self._store.write(
            self._attempt_key(train_id, attempt),
            {
                "train_id": train_id,
                "attempt": attempt,
                "segments": segments,
            },
        )

    def _read_attempt_segments(self, train_id: str, attempt: int) -> list[dict]:
        return self._rebuild_attempt(train_id, attempt)

    def _session_segments(self, session: dict) -> list[dict]:
        if session.get("legacy"):
            target = float(next(iter(session.get("targets", {}).values()), 0.0))
            return [
                {
                    "attempt": 1,
                    "index": 0,
                    "segment_id": next(iter(session.get("targets", {}).keys()), "S1"),
                    "target_m": target,
                    "position_m": target,
                    "offset_m": 0.0,
                    "status": CONFIRMED,
                    "durable": True,
                    "at": float(session.get("updated_at", 0.0)),
                    "legacy": True,
                }
            ]
        return self._read_attempt_segments(
            str(session.get("train_id", "")), int(session.get("attempt", 1))
        )
