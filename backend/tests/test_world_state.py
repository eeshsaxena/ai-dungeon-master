"""Tests for world_state_store.py — quest/location/world persistence."""
import pytest
from world_state_store import WorldStateStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    import world_state_store
    monkeypatch.setattr(world_state_store, "STATE_DIR", tmp_path)
    return WorldStateStore(session_id="test_ws")


def test_initial_state_empty(store):
    assert store.quest_statuses == {}
    assert store.visited_locations == set()


def test_set_quest_status(store):
    store.set_quest_status("quest_001", "active")
    assert store.get_quest_status("quest_001") == "active"


def test_visit_location_returns_true_first_time(store):
    assert store.visit_location("loc_001") is True
    assert store.visit_location("loc_001") is False


def test_completed_objectives(store):
    store.complete_objective("quest_001", "obj_01")
    store.complete_objective("quest_001", "obj_02")
    store.complete_objective("quest_001", "obj_01")   # dupe
    assert len(store.objectives_completed["quest_001"]) == 2


def test_world_events_cap(store):
    for i in range(250):
        store.add_event(f"event_{i}")
    assert len(store.world_events) == 200   # capped


def test_reputation_delta(store):
    store.adjust_reputation("Ministry of Magic", 10)
    store.adjust_reputation("Ministry of Magic", -3)
    assert store.get_reputation_deltas()["Ministry of Magic"] == 7


def test_flags(store):
    assert store.get_flag("never_set") is False
    store.set_flag("hollow_mage_revealed")
    assert store.get_flag("hollow_mage_revealed") is True


def test_npc_state_per_field(store):
    store.set_npc_state("npc_001", "alive", False)
    store.set_npc_state("npc_001", "mood", "grieving")
    state = store.get_npc_state("npc_001")
    assert state["alive"] is False
    assert state["mood"] == "grieving"


def test_to_dict_from_dict_roundtrip(store):
    store.set_quest_status("quest_001", "completed")
    store.visit_location("loc_002")
    store.set_flag("test_flag")

    data = store.to_dict()
    new_store = WorldStateStore("test_ws_2")
    new_store.from_dict(data)

    assert new_store.get_quest_status("quest_001") == "completed"
    assert new_store.has_visited("loc_002")
    assert new_store.get_flag("test_flag") is True


def test_summary(store):
    store.set_quest_status("q1", "active")
    store.set_quest_status("q2", "completed")
    store.visit_location("loc_001")
    s = store.summary()
    assert s["quests_active"] == 1
    assert s["quests_completed"] == 1
    assert s["locations_visited"] == 1
