"""分段对位：按段落盘、整车挂起、重新对位续作、门控一致性与旧数据兼容。"""

from __future__ import annotations

import pytest

from psdline.config import config_from_mapping, config_to_mapping, default_config
from psdline.door import open_doors
from psdline.errors import AlignmentError, InterlockError
from psdline.store import JsonStore
from psdline.train import (
    AlignRegistry,
    CONFIRMED,
    HELD,
    SEEKING,
    train_state,
)


def _seg_config(segments: int = 4):
    mapping = config_to_mapping(default_config())
    mapping["train"] = {**mapping["train"], "alignment_segments": segments}
    return config_from_mapping(mapping)


def _begin(platform, train_id="T01", station_id="ST1", segments=None):
    platform.start()
    return platform.begin_alignment(train_id, station_id, segment_count=segments)


def _report_all(platform, targets, train_id="T01"):
    result = None
    for index, target in enumerate(targets):
        result = platform.report_alignment(train_id, index, float(target))
        assert result["ok"], result
    return result


# ---- 基本分段落盘 ----------------------------------------------------


def test_begin_creates_segment_targets_spanning_train(platform):
    platform.start()
    begun = platform.begin_alignment("T01", "ST1", segment_count=4)
    assert begun["ok"]
    assert begun["state"] == SEEKING
    assert begun["targets"] == [0, 6, 12, 18]
    assert begun["pending_segments"] == [0, 1, 2, 3]


def test_doors_blocked_until_every_segment_durable(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    # 一段都没报：禁开。
    blocked = platform.open_aligned_doors("T01", "ST1")
    assert blocked["ok"] is False
    with pytest.raises(InterlockError):
        open_doors(platform.doors, "T01", at=1.0)
    # 只报一部分：仍然禁开。
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, float(targets[1]))
    with pytest.raises(InterlockError):
        open_doors(platform.doors, "T01", at=2.0)
    # 全部落盘：放行。
    final = platform.report_alignment("T01", 2, float(targets[2]))
    assert final["state"] == CONFIRMED
    assert platform.align.durable("T01") is True
    opened = platform.open_aligned_doors("T01", "ST1")
    assert opened["ok"] and opened["segment_count"] == 3


def test_segment_out_of_tolerance_holds_whole_train(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, float(targets[1]))
    held = platform.report_alignment("T01", 2, targets[2] + 5)
    assert held["ok"] is False
    assert held["state"] == HELD
    assert held["held_segments"] == [2]
    assert platform.align.held("T01") is True
    # 挂起期间：继续收报拒绝、开门拒绝、另开新会话拒绝。
    assert platform.report_alignment("T01", 2, float(targets[2]))["ok"] is False
    assert platform.open_aligned_doors("T01", "ST1")["ok"] is False
    assert platform.begin_alignment("T01", "ST1", segment_count=3)["ok"] is False


def test_held_train_shows_up_in_cycle_and_status(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, targets[1] + 3)
    cycle = platform.cycle()
    assert "T01" in cycle["held_alignments"]
    assert "T01" in platform.status()["held_alignments"]


# ---- 重新对位续作 ----------------------------------------------------


def test_realign_carries_confirmed_segments_within_projection(platform):
    begun = _begin(platform, segments=4)
    targets = begun["targets"]
    for index in range(3):
        platform.report_alignment("T01", index, float(targets[index]))
    platform.report_alignment("T01", 3, targets[3] + 1.2)
    # 司机回拉 0.8 m：锚点合格，前三段按 -0.8 平移后仍在 0.8 公差内，全部继承。
    realigned = platform.realign_alignment("T01", targets[3] + 0.4)
    assert realigned["ok"]
    assert realigned["attempt"] == 2
    assert realigned["state"] == CONFIRMED
    assert realigned["confirmed_segments"] == [0, 1, 2, 3]
    session = platform.align.get_session("T01")
    assert all(session.segments[i].carried for i in range(3))
    assert session.segments[3].carried is False


def test_realign_drops_segments_whose_projection_fails(platform):
    begun = _begin(platform, segments=4)
    targets = begun["targets"]
    for index in range(3):
        platform.report_alignment("T01", index, float(targets[index]))
    platform.report_alignment("T01", 3, targets[3] + 1.2)
    # 回拉 1.2 m：锚点合格，但已确认段平移后偏差 1.2 > 0.8，全部退回待报。
    realigned = platform.realign_alignment("T01", float(targets[3]))
    assert realigned["state"] == SEEKING
    assert realigned["confirmed_segments"] == [3]
    assert realigned["pending_segments"] == [0, 1, 2]
    # 从已确认的锚点段接着把剩余段补齐即可开门。
    for index in range(3):
        platform.report_alignment("T01", index, float(targets[index]))
    assert platform.align.durable("T01") is True


def test_realign_anchor_still_bad_reholds(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, targets[1] + 1.2)
    again = platform.realign_alignment("T01", targets[1] + 2.0, request_id="r-bad")
    assert again["ok"] is False and again["state"] == HELD
    # 回拉到目标 +0.3（相对原始测量 -0.9）：第 0 段投影 -0.9 超差退回待报，
    # 锚点合格，会话脱离挂起。
    proper = platform.realign_alignment("T01", targets[1] + 0.3, request_id="r-fix")
    assert proper["state"] == SEEKING
    assert proper["confirmed_segments"] == [1]
    # 从已确认的锚点段接着把其余段补齐。
    for index in (0, 2):
        platform.report_alignment("T01", index, float(targets[index]))
    assert platform.align.durable("T01") is True


def test_realign_is_idempotent_by_request_id(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, targets[1] + 1.2)
    first = platform.realign_alignment("T01", targets[1] + 0.4, request_id="r-1")
    second = platform.realign_alignment("T01", targets[1] + 0.4, request_id="r-1")
    # 同 request_id 的重复调用返回同一尝试，不递增。
    assert first["attempt"] == second["attempt"] == 2


def test_segment_report_replay_is_idempotent_but_position_immutable(platform):
    begun = _begin(platform, segments=2)
    targets = begun["targets"]
    first = platform.report_alignment("T01", 0, float(targets[0]), request_id="q-1")
    replay = platform.report_alignment("T01", 0, float(targets[0]), request_id="q-1")
    assert first["ok"] and replay["ok"]
    changed = platform.report_alignment("T01", 0, targets[0] + 0.3)
    assert changed["ok"] is False


def test_stale_attempt_report_rejected_after_realign(platform):
    begun = _begin(platform, segments=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, targets[1] + 1.2)
    platform.realign_alignment("T01", targets[1] + 0.4, request_id="r-1")
    assert platform.align.require_session("T01").attempt == 2
    # 重新对位后，属于第 1 次尝试的迟到报文必须被拒。
    stale = platform.report_alignment(
        "T01", 2, float(targets[2]), attempt=1
    )
    assert stale["ok"] is False
    assert "过期" in stale["violations"][0]
    # 当前尝试的报文正常接受。
    fresh = platform.report_alignment("T01", 2, float(targets[2]), attempt=2)
    assert fresh["ok"]


# ---- 重启恢复与部分失败 ----------------------------------------------


def test_held_session_survives_restart_and_resumes(tmp_path):
    config = _seg_config(3)
    platform = __build(config, tmp_path, "a")
    platform.start()
    begun = platform.begin_alignment("T01", "ST1", segment_count=3)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, float(targets[1]))
    platform.report_alignment("T01", 2, targets[2] + 1.2)
    assert platform.align.held("T01")

    restarted = __build(config, tmp_path, "a")
    assert restarted.align.held("T01")
    realigned = restarted.realign_alignment("T01", targets[2] + 0.4)
    assert realigned["state"] == CONFIRMED
    assert realigned["confirmed_segments"] == [0, 1, 2]


def test_missing_segment_doc_degrades_and_blocks_door(tmp_path):
    """头里写了确认、段文档丢了（部分失败）时读时降级，绝不放行。"""

    config = _seg_config(2)
    platform = __build(config, tmp_path, "b")
    platform.start()
    begun = platform.begin_alignment("T01", "ST1", segment_count=2)
    targets = begun["targets"]
    platform.report_alignment("T01", 0, float(targets[0]))
    platform.report_alignment("T01", 1, float(targets[1]))
    assert platform.align.durable("T01") is True
    session = platform.align.require_session("T01")
    # 模拟第 0 段落盘文档丢失。
    platform.store.remove(
        f"train/T01/align/{session.stop_id}/1/0"
    )
    assert platform.align.durable("T01") is False
    with pytest.raises(InterlockError):
        open_doors(platform.doors, "T01", at=9.0)


# ---- 旧数据兼容 ------------------------------------------------------


def test_legacy_single_position_align_still_durable(tmp_path):
    store = JsonStore(str(tmp_path))
    store.write("train/OLD/align", {"train_id": "OLD", "position_m": 3, "durable": True, "at": 1.0})
    registry = AlignRegistry(store, tolerance_m=0.8)
    assert registry.durable("OLD") is True
    assert registry.get_session("OLD") is None
    assert registry.get("OLD").position_m == 3


def test_legacy_align_method_path_unchanged(tmp_path):
    registry = AlignRegistry(JsonStore(str(tmp_path)), tolerance_m=0.5)
    record = registry.align("T01", 10, 10, at=100.0)
    assert record.durable is True
    with pytest.raises(AlignmentError):
        registry.align("T01", 12, 10, at=101.0)
    assert registry.get("T01").position_m == 10


def test_train_state_extra_fields_only_for_segmented_sessions(tmp_path):
    store = JsonStore(str(tmp_path / "x"))
    registry = AlignRegistry(store, tolerance_m=0.8)
    registry.begin("T01", "stop-1", "ST1", [0, 6, 12], at=1.0)
    from psdline.train import DoorStateRegistry, PositionTracker

    doors = DoorStateRegistry(store)
    positions = PositionTracker(store, max_position_m=100)
    state = train_state("T01", registry, doors, positions)
    assert state["alignment_state"] == SEEKING
    assert state["segment_count"] == 3
    assert state["pending_segments"] == [0, 1, 2]
    # 旧单段数据保持原有键集合。
    registry.align("T02", 5, 5, at=2.0)
    legacy = train_state("T02", registry, doors, positions)
    assert set(legacy) == {"train_id", "aligned", "alignment_m", "doors_open", "position_m"}
    assert legacy["aligned"] is True


# ---- 分段会话与发车 --------------------------------------------------


def test_segmented_stop_full_chain_then_depart_clears(tmp_path):
    config = _seg_config(3)
    platform = __build(config, tmp_path, "c")
    platform.start()
    begun = platform.begin_alignment("T01", "ST1", segment_count=3)
    targets = begun["targets"]
    _report_all(platform, targets)
    assert platform.open_aligned_doors("T01", "ST1")["ok"]
    assert platform.check_gap("T01")["ok"]
    departure = platform.depart("T01", "ST1")
    assert departure["ok"], departure
    session = platform.align.get_session("T01")
    assert session.state == "cleared"
    # 同一车下一次停车：新会话、尝试计数复位。
    again = platform.begin_alignment("T01", "ST1", segment_count=3, stop_id="stop-2")
    assert again["ok"] and again["attempt"] == 1


def __build(config, tmp_path, name):
    from psdline.app import build_platform

    root = tmp_path / name
    root.mkdir(exist_ok=True)
    return build_platform(config, storage_root=str(root))
