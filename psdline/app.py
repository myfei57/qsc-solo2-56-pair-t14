"""平台装配与控制循环。

``Platform`` 把命名空间、站台、列车、站台门、信号、联锁、门控、门区映射、间隙
探测、配额与审计按真实调用链装到一起，并暴露四件事：到发编排（到站、发车、
门区联动、批量开门、清扫）、控制循环、状态快照与手动检修动作。

控制链固定为「信号 → 分段停稳对位 → 全段落盘 → 开门 → 间隙探测 → 关门 → 发车 →
审计」，任何一步错位都会被对应组件拒绝，并把拒绝原因落进审计。
"""

from __future__ import annotations

import json
import math

from .audit import AuditLog, build_timeline
from .clock import Clock, SystemClock
from .config import PlatformConfig, default_config, with_storage_root
from .door import (
    DoorController,
    active_group,
    batch_open,
    close_and_audit,
    depart,
    depart_policy,
    door_report,
    door_state,
    enqueue,
    open_pressure,
    open_doors,
    open_group,
    opened,
    process_queue,
    queued_total,
)
from .errors import RailError
from .eventbus import Event, EventBus
from .gap import GapRegistry, gap_summary
from .ids import IdFactory
from .interlock import (
    FaultRegistry,
    InterlockArmer,
    interlock_status,
    next_sequence,
    releasable,
    restore_lockouts,
    verdict_of,
)
from .ns import Namespace, NamespaceRegistry
from .platform import MappingRegistry, adjacent, group_keys, physical_groups, resolve
from .psd import LockoutRegistry, PSD, PSDRegistry, sweep_station
from .quota import QuotaRegistry
from .sig import SignalController, aspect_of, depart_checked, refresh_position, signal_summary
from .station import (
    DoorZone,
    Layout,
    LayoutRegistry,
    Schedule,
    ScheduleRegistry,
    Station,
    StationRegistry,
    nearest_zone,
    spread_m,
)
from .store import CommandLog, JsonStore, JsonlJournal, StoreMetrics
from .telemetry import TrendBuffer
from .train import (
    DEPARTED,
    AlignRegistry,
    DoorStateRegistry,
    Dwell,
    DwellRegistry,
    PositionTracker,
    Route,
    RouteRegistry,
    SegmentMeasurement,
    SUSPENDED,
    Train,
    TrainRegistry,
    train_state,
)

ARRIVAL = "arrival"
DEPART = "depart"


class Platform:
    """站台门区顺序与联锁控制平台。"""

    def __init__(
        self,
        config: PlatformConfig | None = None,
        clock: Clock | None = None,
        unit: str | None = None,
    ) -> None:
        self.config = (config or default_config()).validate()
        self.clock = clock or SystemClock()
        self.bus = EventBus()
        self.unit = unit or self.config.platform.unit
        self.ids = IdFactory("op")

        store = JsonStore(self.config.storage_root)
        self.store = store
        self.metrics = StoreMetrics(store.root)
        self.audit = AuditLog(
            self.clock,
            self.bus,
            JsonlJournal(store.root, "audit", segment_limit=self.config.audit.journal_segment),
        )
        self.namespace = Namespace(
            self.config.namespace.line, list(self.config.namespace.stations)
        )
        self.namespaces = NamespaceRegistry(self.unit)
        self.trend = TrendBuffer(period_s=self.config.signal.refresh_period_s)

        self.stations = StationRegistry(store)
        self.layouts = LayoutRegistry(store)
        self.schedules = ScheduleRegistry(store)
        self.trains = TrainRegistry(store)
        self.align = AlignRegistry(
            store,
            tolerance_m=self.config.station.stop_tolerance_m,
            max_segments=self.config.station.alignment_max_segments,
            stale_after_s=self.config.signal.stale_after_s,
        )
        self.train_doors = DoorStateRegistry(store)
        self.dwells = DwellRegistry(store, max_seconds=self.config.train.dwell_max_s)
        self.positions = PositionTracker(
            store,
            max_position_m=self.config.train.max_position_m,
            stale_after_s=self.config.signal.stale_after_s,
        )
        self.routes = RouteRegistry(store)

        self.gates = PSDRegistry(store)
        self.lockouts = LockoutRegistry(self.gates)
        self.signals = SignalController(store)
        self.faults = FaultRegistry(store)
        self.armers = InterlockArmer(store)
        self.mappings = MappingRegistry(store)
        self.gaps = GapRegistry(store)
        self.quotas = QuotaRegistry(store)
        self.commands = CommandLog(store)
        self.doors = DoorController(
            store, self.align, self.commands, self.gates, self.lockouts
        )
        self.doors.set_batch_limit(self.config.door.batch_limit)

        self._cycles = 0
        self._operations = 0
        self._event_counts: dict[str, int] = {}
        self.bus.subscribe("*", self._on_event)
        self._register_namespaces()
        self._seed(store)
        self._restore()

    def _on_event(self, event: Event) -> None:
        """统计总线事件，供状态接口判断动作有没有真的流过组件。"""

        self._event_counts[event.topic] = self._event_counts.get(event.topic, 0) + 1

    # ---- 装配 ---------------------------------------------------------

    def _register_namespaces(self) -> None:
        """登记现场设备点位（控制台与审计共用同一套名字）。"""

        for area, device, kind in (
            ("line", "signal", "signal-controller"),
            ("line", "sequence", "sequence-source"),
            ("station", "platform", "platform-screen"),
            ("station", "gap", "gap-detector"),
            ("station", "door", "platform-door"),
            ("station", "interlock", "door-interlock"),
            ("station", "quota", "quota-counter"),
            ("station", "audit", "audit-sink"),
        ):
            self.namespaces.register(area, device, kind)

    def _seed(self, store: JsonStore) -> None:
        """按线路命名空间布置站台、布局、映射与站台门。"""

        seed_doc = "platform/seed"
        if store.exists(seed_doc):
            return
        groups_per_station = self.config.station.groups_per_station
        doors_per_group = self.config.station.doors_per_group
        span = self.config.station.group_span_m
        for station_index, station_id in enumerate(self.namespace.station_ids(), start=1):
            groups = [f"G{index:02d}" for index in range(1, groups_per_station + 1)]
            self.stations.upsert(
                Station(
                    id=station_id,
                    name=f"站台 {station_index}",
                    line=self.namespace.line_id(),
                    door_groups=groups,
                )
            )
            base = (station_index - 1) * span
            zones = [
                DoorZone(id=group, distance_m=int(base + (index - 1) * span / groups_per_station))
                for index, group in enumerate(groups, start=1)
            ]
            self.layouts.save(Layout(station_id=station_id, zones=zones))
            self.mappings.renumber(station_id, {group: group for group in groups})
            self.schedules.save(
                Schedule(station_id=station_id, headways=[120, 150, 180, 240])
            )
            for group in groups:
                for door_index in range(1, doors_per_group + 1):
                    self.gates.upsert(
                        PSD(
                            id=f"{station_id}-{group}-{door_index:02d}",
                            station=station_id,
                            group=group,
                        )
                    )
        for name, limit in (
            (ARRIVAL, self.config.quota.arrival_limit),
            (DEPART, self.config.quota.depart_limit),
        ):
            self.quotas.set_limit(name, limit)
        store.write(
            seed_doc,
            {"line": self.namespace.line_id(), "stations": self.namespace.station_ids()},
        )

    def _restore(self) -> None:
        """从落盘状态恢复操作计数与锁定状态。"""

        state = self.store.read("platform/state")
        if isinstance(state, dict):
            self._operations = int(state.get("operations", 0))
            self._cycles = int(state.get("cycles", 0))
        restore_lockouts(self.faults, self.gates, self.lockouts, not self.gaps.blocked_ids())

    def _persist(self) -> None:
        """落盘平台计数，供重启后对照。"""

        self.store.write(
            "platform/state",
            {
                "operations": self._operations,
                "cycles": self._cycles,
                "unit": self.unit,
                "line": self.namespace.line_id(),
            },
        )

    def _record_operation(self, op_id: str, operation: str, report: dict) -> dict:
        """把编排步骤逐条写事件，并记一条汇总审计。"""

        train_id = str(report.get("train_id", ""))
        station_id = str(report.get("station_id", ""))
        reserved = {
            "train_id",
            "station_id",
            "op_id",
            "severity",
            "message",
            "component",
            "action",
            "detail",
        }
        for step in report.get("steps", []):
            detail = {
                key: value
                for key, value in dict(step.get("detail", {})).items()
                if not isinstance(value, (dict, list)) and key not in reserved
            }
            self.bus.publish(
                Event.of(
                    "door",
                    str(step.get("component", "door")),
                    str(step.get("action", "step")),
                    op_id=op_id,
                    train_id=train_id,
                    severity="notice",
                    **detail,
                )
            )
        for violation in report.get("violations", []):
            self.bus.publish(
                Event.of(
                    "audit",
                    "interlock",
                    "interlock_rejected",
                    op_id=op_id,
                    train_id=train_id,
                    station_id=station_id,
                    severity="alarm",
                    message=f"{operation} 被拒绝：{violation}",
                    violation=violation,
                )
            )
        entry = self.audit.record(
            f"{operation}_complete",
            train_id,
            station_id,
            op_id=op_id,
            ok=bool(report.get("ok")),
            violations=list(report.get("violations", [])),
        )
        return {
            "op_id": op_id,
            "operation": operation,
            "ok": bool(report.get("ok")),
            "violations": list(report.get("violations", [])),
            "audit_id": entry.id,
        }

    # ---- 生命周期 -----------------------------------------------------

    def start(self) -> dict:
        """启动平台：核对命名空间、站台、布局与映射是否一致。"""

        missing: dict[str, list[str]] = {}
        for station_id in self.namespace.station_ids():
            if self.stations.get(station_id) is None:
                missing.setdefault("station", []).append(station_id)
            if self.layouts.get(station_id) is None:
                missing.setdefault("layout", []).append(station_id)
            if self.mappings.current(station_id) is None:
                missing.setdefault("mapping", []).append(station_id)
        return {
            "unit": self.unit,
            "line": self.namespace.label(),
            "missing": missing,
            "stations": len(self.namespace.station_ids()),
            "gates": len(self.gates.list()),
            "operations": self._operations,
            "ready": not missing,
        }

    def stop(self) -> dict:
        """停运平台：整站清扫后收起全部灯位。"""

        op_id = self.ids.labeled("stop")
        steps: list[dict] = []
        closed = 0
        for station_id in self.namespace.station_ids():
            report = self.sweep(station_id)
            closed += report["closed"]
            steps.append(
                {
                    "component": "psd",
                    "action": "sweep",
                    "detail": {"station_id": station_id, "closed": report["closed"]},
                }
            )
        for train in self.trains.list():
            self.signals.clear(train.id, ARRIVAL)
            self.signals.clear(train.id, DEPART)
        report = {"ok": True, "train_id": "", "closed": closed, "steps": steps, "violations": []}
        summary = self._record_operation(op_id, "stop", report)
        self._operations += 1
        self._persist()
        return {**report, **summary}

    # ---- 到发编排 -----------------------------------------------------

    def arrive(
        self,
        train_id: str,
        station_id: str,
        position_m: int | None = None,
        segments: list | dict | None = None,
        request_id: str = "",
    ) -> dict:
        """到站到开门：信号到站 → 分段对位逐段落盘 → 全段确认后开门。

        ``position_m`` 是旧上位机的单位置报文，按一个分段处理；新上位机可传
        ``segments``。同一列车同一站台的重复到站请求会复用未结束会话；显式
        ``request_id`` 的响应按标识幂等返回。
        """

        op_id = self.ids.labeled("arrive")
        steps: list[dict] = []
        violations: list[str] = []

        try:
            targets, measurements, legacy_scalar = self._parse_alignment_input(
                station_id, position_m, segments
            )
        except RailError as error:
            return self._arrive_rejected(
                op_id, train_id, station_id, error.message, steps, violations, request_id
            )

        request_fingerprint = self._request_fingerprint(position_m, targets, measurements)
        cached = self.align.load_idempotent(train_id, station_id, request_id)
        if cached is not None:
            if cached.get("request_fingerprint") != request_fingerprint:
                return self._arrive_rejected(
                    op_id,
                    train_id,
                    station_id,
                    "同一 request_id 的请求内容不一致",
                    steps,
                    violations,
                    request_id,
                )
            return {**cached, "op_id": op_id, "replayed": True}

        def reject(reason: str) -> dict:
            return self._arrive_rejected(
                op_id, train_id, station_id, reason, steps, violations, request_id
            )

        if not self.quotas.consume(ARRIVAL):
            return reject("到站配额已用完")
        try:
            self.namespace.require(station_id)
        except RailError as error:
            return reject(error.message)
        if self.stations.get(station_id) is None:
            return reject(f"站台未登记: {station_id}")

        active = self.align.active_session(train_id)
        if (
            active is not None
            and active.get("state") == SUSPENDED
            and legacy_scalar
            and active.get("segment_count") == 1
        ):
            resumed = self.align.resume_session(train_id, station_id, self.clock.now())
            self.audit.record(
                "alignment_resumed",
                train_id,
                station_id,
                attempt=resumed["attempt"],
                carried_segments=resumed["carried_prefix"],
                legacy=True,
            )
            active = self.align.active_session(train_id)
        fresh = active is None or active.get("state") == DEPARTED or bool(active.get("legacy"))
        if active is not None and active.get("state") == SUSPENDED:
            return reject("对位已挂起，请调用分段重新对位接口从未确认段续作")
        if (
            active is not None
            and not fresh
            and str(active.get("station_id", "")) != station_id
        ):
            return reject(f"列车仍在 {active.get('station_id')} 站对位，不能切换站台")

        if fresh:
            self._clear_arrival_products(train_id)
        self.signals.raise_signal(train_id, ARRIVAL, self.clock.now())
        steps.append(
            {"component": "sig", "action": "arrival_signal", "detail": {"train_id": train_id}}
        )
        if self.trains.get(train_id) is None:
            self.trains.upsert(Train(id=train_id, line=self.namespace.line_id()))
        self.trains.assign_station(train_id, station_id)

        try:
            if fresh:
                session = self.align.start_session(
                    train_id,
                    station_id,
                    targets,
                    self.clock.now(),
                    layout_revision=self.mappings.revision(station_id),
                    request_id=request_id,
                )
                self.audit.record(
                    "alignment_started",
                    train_id,
                    station_id,
                    attempt=session["attempt"],
                    segment_count=session["segment_count"],
                    legacy=legacy_scalar,
                )
            elif active.get("state") == SUSPENDED:
                session = self.align.session_view(train_id)
            else:
                session = self.align.session_view(train_id)
            session = self._report_alignment(
                train_id, station_id, measurements, request_id, steps, legacy_scalar
            )
        except RailError as error:
            return reject(error.message)

        if not session["durable"]:
            return reject(
                f"仅有 {session['confirmed_count']}/{session['segment_count']} 个分段落盘，禁止开门"
            )
        try:
            door_report = self.doors_open(train_id, station_id)
        except RailError as error:
            return reject(error.message)
        entry = self.audit.record(
            ARRIVAL,
            train_id,
            station_id,
            position_m=measurements[-1].position_m if measurements else 0,
            segment_count=session["segment_count"],
            attempt=session["attempt"],
        )
        report = {
            "ok": True,
            "train_id": train_id,
            "station_id": station_id,
            "position_m": session["segments"][-1]["position_m"] if session["segments"] else 0,
            "alignment": session,
            "segment_count": session["segment_count"],
            "confirmed_segments": session["confirmed_count"],
            "attempt": session["attempt"],
            "reused": not fresh,
            "group": door_report.get("group", ""),
            "steps": steps,
            "violations": [],
        }
        summary = self._record_operation(op_id, "arrive", report)
        result = {**report, **summary, "arrival_audit": entry.id, "request_fingerprint": request_fingerprint}
        self._operations += 1
        self._persist()
        self.align.save_idempotent(train_id, station_id, request_id, result, self.clock.now())
        return result

    def align_segments(
        self,
        train_id: str,
        station_id: str,
        segments: list | dict,
        *,
        resume: bool = False,
        request_id: str = "",
    ) -> dict:
        """分段上报对位；``resume=True`` 时从挂起 attempt 的已确认前缀续作。"""

        op_id = self.ids.labeled("align")
        active = self.align.active_session(train_id)
        if active is None:
            raise RailError(f"列车 {train_id} 没有对位会话", component="train")
        if str(active.get("station_id", "")) != station_id:
            raise RailError("分段上报的站台与当前会话不一致", component="train")
        if active.get("state") == SUSPENDED and not resume:
            raise RailError("对位已挂起，必须带 resume=true 重新对位", component="train")
        if resume and active.get("state") == SUSPENDED:
            session = self.align.resume_session(
                train_id, station_id, self.clock.now(), request_id=request_id
            )
            self.audit.record(
                "alignment_resumed",
                train_id,
                station_id,
                attempt=session["attempt"],
                carried_segments=session["carried_prefix"],
            )
        active_targets = dict(active.get("targets", {}))
        known_segments = list(active_targets.keys())
        targets, measurements, _legacy = self._parse_alignment_input(
            station_id,
            None,
            segments,
            fallback_targets=active_targets,
        )
        if resume:
            expected = known_segments
            provided = list(targets.keys())
            unknown = [key for key in provided if key not in active_targets]
            if unknown or provided != expected:
                raise RailError(
                    "续作必须从下一个未确认段开始，并重新核对全部后续分段",
                    component="train",
                )
            if any(targets[key] != active_targets[key] for key in targets):
                raise RailError("续作分段的目标位置与挂起会话不一致", component="train")
        elif list(targets.keys()) != list(active_targets.keys()):
            raise RailError("分段集合与当前对位会话不一致", component="train")
        request_fingerprint = self._request_fingerprint(None, targets, measurements)
        cached = self.align.load_idempotent(train_id, station_id, request_id)
        if cached is not None:
            if cached.get("request_fingerprint") != request_fingerprint:
                raise RailError("同一 request_id 的请求内容不一致", component="console")
            return {**cached, "op_id": op_id, "replayed": True}
        if not measurements:
            raise RailError("对位分段不能为空", component="console")
        steps: list[dict] = []
        session = self._report_alignment(
            train_id, station_id, measurements, request_id, steps, False
        )
        result = {
            "ok": True,
            "train_id": train_id,
            "station_id": station_id,
            "alignment": session,
            "state": session["state"],
            "suspended": session["suspended"],
            "all_confirmed": session["durable"],
            "confirmed_segments": session["confirmed_count"],
            "segment_count": session["segment_count"],
            "attempt": session["attempt"],
            "steps": steps,
            "violations": [],
            "request_fingerprint": request_fingerprint,
        }
        self._record_operation(op_id, "align_segments", result)
        self.align.save_idempotent(train_id, station_id, request_id, result, self.clock.now())
        return result

    def doors_open(self, train_id: str, station_id: str) -> dict:
        """开门并确认车门：所有分段对位必须先落盘。"""

        session = self.align.session_view(train_id)
        if session is None or not session["durable"]:
            raise RailError("全部分段落盘前不能开门", component="door")
        group = self._primary_group(station_id)
        already_open = opened(self.doors, train_id)
        opened_gate_ids: list[str] = []
        if not already_open and group:
            opened_gate_ids = open_group(self.doors, self.mappings, station_id, group)["opened"]
        try:
            if already_open:
                report = {
                    "ok": True,
                    "train_id": train_id,
                    "position_m": session["segments"][-1]["position_m"] if session["segments"] else 0,
                    "violations": [],
                }
            else:
                report = open_doors(self.doors, train_id, self.clock.now())
            self.align.mark_opened(train_id, self.clock.now())
            self.train_doors.confirm_open(train_id, self.clock.now())
        except RailError:
            if not already_open:
                for gate_id in opened_gate_ids:
                    self.gates.set_open(gate_id, False)
            raise
        self.bus.publish(
            Event.of(
                "door",
                "door",
                "opened",
                train_id=train_id,
                station_id=station_id,
                severity="notice",
                group=group,
            )
        )
        return {**report, "group": group}

    def depart(self, train_id: str, station_id: str) -> dict:
        """关门到发车：关门指令 → 间隙探测 → 关门落盘 → 发车信号。"""

        op_id = self.ids.labeled("depart")
        steps: list[dict] = []
        if not self.quotas.consume(DEPART):
            report = {
                "ok": False,
                "train_id": train_id,
                "station_id": station_id,
                "steps": steps,
                "violations": ["发车配额已用完"],
            }
            summary = self._record_operation(op_id, "depart", report)
            return {**report, **summary}

        gate = self._primary_gate(station_id)
        if gate is not None and not self.lockouts.is_locked(gate.id):
            self.lockouts.set(gate.id, True)
            steps.append(
                {"component": "interlock", "action": "lock", "detail": {"psd_id": gate.id}}
            )
        try:
            closed = close_and_audit(
                self.doors, self.audit, train_id, station_id, DEPART, self.clock.now()
            )
        except RailError as error:
            report = {
                "ok": False,
                "train_id": train_id,
                "station_id": station_id,
                "steps": steps,
                "violations": [error.message],
            }
            summary = self._record_operation(op_id, "depart", report)
            return {**report, **summary}
        steps.append(
            {"component": "door", "action": "close_durable", "detail": {"train_id": train_id}}
        )
        gap_state = self.gaps.state(train_id)
        steps.append(
            {
                "component": "gap",
                "action": "check",
                "detail": {"train_id": train_id, "state": gap_state},
            }
        )
        verdict = depart_checked(self.signals, self.store, self.gaps, train_id, self.clock.now())
        if not verdict["ok"]:
            report = {
                "ok": False,
                "train_id": train_id,
                "station_id": station_id,
                "steps": steps,
                "violations": list(verdict["violations"]),
            }
            summary = self._record_operation(op_id, "depart", report)
            return {**report, **summary}
        steps.append(
            {"component": "sig", "action": "depart_signal", "detail": {"train_id": train_id}}
        )
        departure = depart(self.doors, train_id, self.clock.now())
        self.align.mark_departed(train_id, self.clock.now())
        if gate is not None:
            self.lockouts.clear(gate.id)
        report = {
            "ok": True,
            "train_id": train_id,
            "station_id": station_id,
            "close_durable": bool(closed["close_durable"]),
            "aspect": verdict["aspect"],
            "cleared_commands": departure["cleared_commands"],
            "steps": steps,
            "violations": [],
        }
        summary = self._record_operation(op_id, "depart", report)
        self._operations += 1
        self._persist()
        return {**report, **summary}

    def check_gap(self, train_id: str, clear: bool = True) -> dict:
        """确认间隙无异常，或登记一次可疑探测结果。"""

        if clear:
            result = self.gaps.check(train_id, self.clock.now())
        else:
            result = self.gaps.probe(train_id, False, self.clock.now())
        self.audit.record("gap_probe", train_id, "", clear=clear)
        return {"ok": True, "train_id": train_id, "clear": result.clear, "violations": []}

    def renumber(self, station_id: str, groups: dict[str, str]) -> dict:
        """按最新布局重排门区映射，并按新映射联动一次。"""

        mapping = self.mappings.renumber(station_id, groups)
        first_key = sorted(groups)[0]
        linked = open_group(self.doors, self.mappings, station_id, first_key)
        self.audit.record("mapping_renumber", "", station_id, revision=mapping.revision)
        return {
            "ok": True,
            "station_id": station_id,
            "revision": mapping.revision,
            "mapping": mapping.as_dict(),
            "active_group": linked["group"],
            "violations": [],
        }

    def enqueue(self, train_id: str, command_id: str, kind: str, station_id: str = "") -> dict:
        """把一条门控指令排进队列。"""

        command = enqueue(self.doors, train_id, command_id, kind, station_id)
        self.audit.record(
            "door_enqueue", train_id, station_id, command_id=command.id, command_kind=kind
        )
        return {"ok": True, "train_id": train_id, "command": command.as_dict(), "violations": []}

    def process_queue(self, train_id: str) -> dict:
        """返回列车队列里待执行的门控指令，并带上发车自检结果。"""

        policy = depart_policy(self.doors, train_id)
        ids = process_queue(self.doors, train_id)
        self.audit.record("door_process", train_id, "", queued=len(ids))
        return {
            "ok": True,
            "train_id": train_id,
            "ids": ids,
            "ready": policy["allowed"],
            "violations": list(policy["failures"]),
        }

    def clear_fault(self, psd_id: str) -> dict:
        """清除门区故障并按条件解锁。"""

        gate = self.gates.require(psd_id)
        allowed, reason = releasable(self.gates, psd_id, True)
        if not allowed:
            return {"ok": False, "psd_id": psd_id, "locked": True, "violations": [reason]}
        self.faults.clear_fault(self.lockouts, psd_id, self.clock.now())
        self.audit.record("interlock_clear", "", gate.station, psd_id=psd_id)
        return {"ok": True, "psd_id": psd_id, "locked": False, "violations": []}

    def set_fault(self, psd_id: str) -> dict:
        """置位门区故障并锁定。"""

        gate = self.gates.require(psd_id)
        self.faults.set_fault(self.lockouts, psd_id, self.clock.now())
        self.audit.record("interlock_fault", "", gate.station, psd_id=psd_id)
        return {"ok": True, "psd_id": psd_id, "locked": True, "violations": []}

    def arm(self, train_id: str, armed: bool) -> dict:
        """启用或退出一列车的联锁。"""

        state = self.armers.arm(train_id, armed)
        self.audit.record("interlock_arm", train_id, "", armed=state)
        return {"ok": True, "train_id": train_id, "armed": state, "violations": []}

    def refresh_position(self, train_id: str, position_m: int) -> dict:
        """按信号刷新列车位置。"""

        outcome = refresh_position(self.positions, train_id, position_m, self.clock.now())
        if outcome["ok"]:
            self.trend.append("train_position_m", self.clock.now(), position_m)
        else:
            self.audit.record("position_rejected", train_id, "", reason=outcome["reason"])
        return {**outcome, "violations": [] if outcome["ok"] else [outcome["reason"]]}

    def sequence_verdict(self, previous: int, step: int) -> dict:
        """推进门控事件序号并给出回绕判定。"""

        current = next_sequence(previous, step)
        verdict = verdict_of(previous, current)
        self.audit.record(
            "sequence_verdict",
            "",
            "",
            previous=verdict.previous,
            current=verdict.current,
            advanced=verdict.advanced,
        )
        return {"ok": verdict.advanced, "verdict": verdict.as_dict(), "violations": [] if verdict.advanced else [verdict.reason]}

    def batch_open(self, station_id: str) -> dict:
        """按布局里程顺序批量开门。"""

        layout = self.layouts.get(station_id)
        zones = list(layout.zones) if layout else []
        outcome = batch_open(self.doors, station_id, zones)
        nearest = nearest_zone(zones)
        self.audit.record(
            "batch_open",
            "",
            station_id,
            opened=len(outcome["opened"]),
            skipped=len(outcome["skipped"]),
        )
        return {
            "ok": not outcome["skipped"],
            "station_id": station_id,
            "order": outcome["order"],
            "opened": outcome["opened"],
            "deferred": outcome["deferred"],
            "skipped": outcome["skipped"],
            "nearest": nearest.id if nearest else "",
            "violations": [],
        }

    def sweep(self, station_id: str) -> dict:
        """整站清扫。"""

        return sweep_station(self.gates, station_id)

    def register_dwell(self, train_id: str, seconds: float, station_id: str = "") -> dict:
        """登记停站时长。"""

        dwell = self.dwells.set(Dwell(train_id=train_id, seconds=seconds, station_id=station_id))
        return {"ok": True, "dwell": dwell.as_dict(), "violations": []}

    def assign_route(self, train_id: str, origin: str, destination: str, distance_m: int) -> dict:
        """登记运行交路。"""

        route = self.routes.assign(
            Route(train_id=train_id, origin=origin, destination=destination, distance_m=distance_m)
        )
        return {"ok": True, "route": route.__dict__, "violations": []}

    # ---- 控制循环 -----------------------------------------------------

    def cycle(self, dt: float = 1.0) -> dict:
        """推进一个控制周期，返回本周期摘要。"""

        if dt <= 0:
            return {"ok": False, "violations": ["步长必须为正"], "cycle": self._cycles}
        self._cycles += 1
        moment = self.clock.now()
        stale = [
            train.id for train in self.trains.list() if self.positions.stale(train.id, moment)
        ]
        interlock = interlock_status(self.faults, self.gates, self.config.interlock.max_locked_ratio)
        holding = [
            train.id
            for train in self.trains.list()
            if not self.armers.armed(train.id) and depart_policy(self.doors, train.id)["allowed"]
        ]
        self.trend.append("locked_count", moment, interlock["locked_count"])
        self.trend.append("queued_commands", moment, queued_total(self.doors))
        self.trend.append("audit_entries", moment, len(self.audit.list()))
        self._persist()
        return {
            "ok": True,
            "cycle": self._cycles,
            "dt": dt,
            "stale_positions": stale,
            "locked_count": interlock["locked_count"],
            "can_release_depart": interlock["can_release_depart"],
            "ready_trains": holding,
            "gaps": gap_summary(self.gaps, [train.id for train in self.trains.list()]),
            "violations": [],
        }

    # ---- 查询 ---------------------------------------------------------

    def status(self) -> dict:
        """汇总状态快照，供控制台与命令行读取。"""

        trains = self.trains.list()
        ids = [train.id for train in trains]
        return {
            "platform": self.config.name,
            "unit": self.unit,
            "line": self.namespace.label(),
            "operations": self._operations,
            "cycles": self._cycles,
            "utc": self.clock.utc_iso(),
            "namespace": self.namespace.as_dict(),
            "namespaces": self.namespaces.describe(),
            "open_gates": self.gates.count_open(),
            "lockout_changes": self.lockouts.changes,
            "signals_raised": self.signals.raises,
            "signals_cleared": self.signals.clears,
            "interlock_counters": {
                "faults_set": self.faults.sets,
                "faults_cleared": self.faults.clears,
                "armed": self.armers.arms,
                "disarmed": self.armers.disarms,
                "armed_trains": self.armers.armed_trains(),
            },
            "gap_engine": {"checks": self.gaps.checks, "probes": self.gaps.probes},
            "event_counts": dict(sorted(self._event_counts.items())),
            "console_hits": len(self.bus.history("console")),
            "stations": [station.as_dict() for station in self.stations.list()],
            "trains": {train.id: train_state(train.id, self.align, self.train_doors, self.positions) for train in trains},
            "doors": [
                {**gate.as_dict(), "open": gate.open, "locked": gate.locked}
                for gate in self.gates.list()
            ],
            "door_report": door_report(self.doors, self.namespace.station_ids()),
            "signals": signal_summary(self.signals, ids),
            "interlock": interlock_status(
                self.faults, self.gates, self.config.interlock.max_locked_ratio
            ),
            "gaps": gap_summary(self.gaps, ids),
            "quota": {
                ARRIVAL: self.quotas.status(ARRIVAL),
                DEPART: self.quotas.status(DEPART),
            },
            "store": self.metrics.summary(),
            "audit": self.audit.summary(),
            "trend": self.trend.snapshot(limit=4),
            "ids_issued": self.ids.issued(),
        }

    def snapshot(self) -> dict:
        """返回只读的整体快照（状态 + 命名空间 + 门区映射 + 审计时间线）。"""

        return {
            "status": self.status(),
            "groups": {
                station_id: group_keys(self.mappings, station_id)
                for station_id in self.namespace.station_ids()
            },
            "physical_groups": {
                station_id: physical_groups(self.mappings, station_id)
                for station_id in self.namespace.station_ids()
            },
            "mapping_revisions": {
                station_id: {
                    "latest": self.mappings.history(station_id).latest(),
                    "log": self.mappings.history(station_id).as_dict()["revisions"],
                }
                for station_id in self.namespace.station_ids()
            },
            "adjacent": self._adjacent_sample(),
            "timeline": build_timeline(self.audit.list(), limit=20),
        }

    def train_view(self, train_id: str) -> dict:
        """返回一列车的综合视图。"""

        return {
            "train": self.trains.get(train_id).as_dict() if self.trains.get(train_id) else None,
            "state": train_state(train_id, self.align, self.train_doors, self.positions),
            "doors": door_state(self.doors, train_id),
            "aspect": aspect_of(self.signals, train_id),
            "gap": self.gaps.state(train_id),
        }

    def station_view(self, station_id: str) -> dict:
        """返回一个站台的综合视图。"""

        layout = self.layouts.get(station_id)
        mapping = self.mappings.current(station_id)
        zones = list(layout.zones) if layout else []
        schedule = self.schedules.get(station_id)
        return {
            "station": self.stations.get(station_id).as_dict() if self.stations.get(station_id) else None,
            "group_count": self.stations.group_count(station_id),
            "zones": [zone.as_dict() for zone in zones],
            "order": [zone.id for zone in zones],
            "zone_distances": {
                zone.id: self.layouts.zone_distance(station_id, zone.id) for zone in zones
            },
            "span_m": spread_m(zones),
            "minimum_headway_s": self.schedules.minimum_headway(station_id) if schedule else 0,
            "revision": mapping.revision if mapping else 0,
            "groups": group_keys(self.mappings, station_id),
            "active_group": active_group(self.doors, station_id),
            "pressure": open_pressure(self.doors, station_id),
        }

    def resolve_group(self, station_id: str, group_key: str) -> dict:
        """解析一个逻辑门区。"""

        group = resolve(self.mappings, station_id, group_key)
        return {
            "station_id": station_id,
            "group_key": group_key,
            "group": group,
            "neighbours": adjacent(self.mappings, station_id, group_key),
        }

    def note_error(self, error: RailError) -> dict:
        """把平台异常写入审计，返回落盘记录。"""

        entry = self.audit.record(
            "error",
            "",
            "",
            component=error.component,
            message=error.message,
            error_type=type(error).__name__,
        )
        return entry.as_dict()

    def reset_state(self) -> bool:
        """清掉落盘的平台计数（交付前重新标定用）。"""

        if not self.store.exists("platform/state"):
            return False
        return self.store.remove("platform/state")

    def _arrive_rejected(
        self,
        op_id: str,
        train_id: str,
        station_id: str,
        reason: str,
        steps: list[dict],
        violations: list[str],
        request_id: str,
        request_fingerprint: str = "",
    ) -> dict:
        """登记到站拒绝，并让相同 request_id 的重试拿到一致失败结果。"""

        violations.append(reason)
        self.bus.publish(
            Event.of(
                "audit",
                "interlock",
                "arrive_rejected",
                op_id=op_id,
                train_id=train_id,
                station_id=station_id,
                severity="alarm",
                message=reason,
            )
        )
        self.audit.record("alignment_rejected", train_id, station_id, reason=reason)
        report = {
            "ok": False,
            "train_id": train_id,
            "station_id": station_id,
            "steps": steps,
            "violations": list(violations),
        }
        result = {
            **report,
            **self._record_operation(op_id, "arrive", report),
            "request_fingerprint": request_fingerprint,
        }
        self.align.save_idempotent(train_id, station_id, request_id, result, self.clock.now())
        return result

    @staticmethod
    def _request_fingerprint(
        position_m: float | None,
        targets: dict[str, float],
        measurements: list[SegmentMeasurement],
    ) -> str:
        """给幂等请求的业务输入定指纹；op_id、时间不参与。"""

        payload = {
            "position_m": None if position_m is None else float(position_m),
            "targets": targets,
            "segments": [
                {"segment_id": item.segment_id, "position_m": item.position_m, "at": item.at}
                for item in measurements
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _parse_alignment_input(
        self,
        station_id: str,
        position_m: int | float | None,
        segments: list | dict | None,
        fallback_targets: dict[str, float] | None = None,
        zone_offset: int = 0,
    ) -> tuple[dict[str, float], list[SegmentMeasurement], bool]:
        """把旧单位置和新分段报文归一成目标位置与实测位置。"""

        if segments is None:
            if position_m is None:
                raise RailError("缺少对位位置 position_m 或 segments", component="console")
            target = float(self._stop_position(station_id))
            return {"S1": target}, [SegmentMeasurement("S1", float(position_m))], True

        layout = self.layouts.get(station_id)
        zones = list(layout.zones) if layout else []
        if isinstance(segments, dict):
            raw_items = list(segments.get("segments", []))
        else:
            raw_items = list(segments)
        if not raw_items:
            raise RailError("对位分段不能为空", component="console")
        if len(raw_items) > self.config.station.alignment_max_segments:
            raise RailError(
                f"对位分段数超过上限 {self.config.station.alignment_max_segments}",
                component="console",
            )
        targets: dict[str, float] = {}
        measurements: list[SegmentMeasurement] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise RailError("每个对位分段必须是对象", component="console")
            segment_id = str(item.get("segment_id") or item.get("id") or f"S{index + 1}")
            if segment_id in seen:
                raise RailError(f"对位分段 {segment_id} 重复", component="console")
            seen.add(segment_id)
            if "position_m" not in item:
                raise RailError(f"分段 {segment_id} 缺少 position_m", component="console")
            try:
                measured = float(item["position_m"])
            except (TypeError, ValueError) as error:
                raise RailError(f"分段 {segment_id} 的 position_m 不是数值", component="console") from error
            if not math.isfinite(measured):
                raise RailError(f"分段 {segment_id} 的 position_m 必须有限", component="console")
            zone_index = index + zone_offset
            if item.get("target_m") is not None:
                target = float(item["target_m"])
            elif fallback_targets is not None and segment_id in fallback_targets:
                target = float(fallback_targets[segment_id])
            elif zone_index < len(zones):
                target = float(zones[zone_index].distance_m)
            else:
                raise RailError(f"分段 {segment_id} 缺少目标位置 target_m", component="console")
            if not math.isfinite(target):
                raise RailError(f"分段 {segment_id} 的 target_m 必须有限", component="console")
            targets[segment_id] = target
            measurements.append(SegmentMeasurement(segment_id, measured))
        if fallback_targets is not None and list(targets.keys()) != list(fallback_targets.keys()):
            raise RailError("分段标识顺序必须与已创建会话一致", component="train")
        return targets, measurements, False

    def _report_alignment(
        self,
        train_id: str,
        station_id: str,
        measurements: list[SegmentMeasurement],
        request_id: str,
        steps: list[dict],
        legacy_scalar: bool,
    ) -> dict:
        """上报分段并把逐段结果写入操作步骤；超差异常交调用方挂起编排。"""

        if not measurements:
            return self.align.session_view(train_id)
        before_session = self.align.session_view(train_id)
        before_count = before_session["confirmed_count"] if before_session else 0
        before_attempt = before_session["attempt"] if before_session else 0
        try:
            session = self.align.report_segments(
                train_id, measurements, self.clock.now(), request_id=request_id
            )
        except RailError:
            failed = self.align.session_view(train_id)
            self.audit.record(
                "alignment_segment_rejected",
                train_id,
                station_id,
                attempt=failed["attempt"] if failed else 0,
                rejected_segments=",".join(failed["rejected_segments"]) if failed else "",
            )
            raise
        new_segments = (
            session["segments"][before_count:]
            if before_attempt == session["attempt"]
            else session["segments"]
        )
        for segment in new_segments:
            self.audit.record(
                "alignment_segment_confirmed",
                train_id,
                station_id,
                attempt=session["attempt"],
                segment_id=segment["segment_id"],
                position_m=segment["position_m"],
                target_m=segment["target_m"],
                offset_m=segment["offset_m"],
                carried=bool(segment.get("carried_from")),
                legacy=legacy_scalar,
            )
            steps.append(
                {
                    "component": "train",
                    "action": "align_segment",
                    "detail": {
                        "train_id": train_id,
                        "segment_id": segment["segment_id"],
                        "position_m": segment["position_m"],
                        "offset_m": segment["offset_m"],
                    },
                }
            )
        if session["durable"]:
            self.audit.record(
                "alignment_completed",
                train_id,
                station_id,
                attempt=session["attempt"],
                segment_count=session["segment_count"],
            )
        return session

    def _clear_arrival_products(self, train_id: str) -> None:
        """新一轮到站前清门控/间隙派生态；历史审计与分段 attempt 不清。"""

        for suffix in ("open", "close", "departed"):
            self.store.remove(f"door/{train_id}/{suffix}")
        self.store.remove(f"gap/{train_id}")
        self.commands.clear(train_id)

    # ---- 内部 ---------------------------------------------------------

    def _stop_position(self, station_id: str) -> int:
        """返回站台的停稳位置（取最近门区里程）。"""

        layout = self.layouts.get(station_id)
        zones = list(layout.zones) if layout else []
        nearest = nearest_zone(zones)
        return nearest.distance_m if nearest else 0

    def _primary_group(self, station_id: str) -> str:
        """返回站台的首个门区。"""

        keys = group_keys(self.mappings, station_id)
        return keys[0] if keys else ""

    def _primary_gate(self, station_id: str) -> PSD | None:
        """返回站台的首樘站台门。"""

        gates = self.gates.by_station(station_id)
        return gates[0] if gates else None

    def _adjacent_sample(self) -> dict:
        """给快照提供一个相邻关系样例。"""

        station_ids = self.namespace.station_ids()
        if not station_ids:
            return {}
        station_id = station_ids[0]
        keys = group_keys(self.mappings, station_id)
        if not keys:
            return {}
        return {keys[0]: adjacent(self.mappings, station_id, keys[0])}


def build_platform(
    config: PlatformConfig | None = None,
    clock: Clock | None = None,
    storage_root: str | None = None,
    unit: str | None = None,
) -> Platform:
    """构造一个平台；``storage_root`` 可覆盖配置里的存储目录。"""

    base = config or default_config()
    if storage_root is not None:
        base = with_storage_root(base, storage_root)
    return Platform(base, clock=clock, unit=unit)
