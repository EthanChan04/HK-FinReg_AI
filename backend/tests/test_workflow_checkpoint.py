from types import SimpleNamespace


def _make_state_with_interrupt(interrupt_value=None):
    task = SimpleNamespace(
        interrupt=SimpleNamespace(value=interrupt_value) if interrupt_value is not None else None
    )
    return SimpleNamespace(tasks=[task], values={"original_input": "demo"})


def test_rebuild_review_queue_prunes_stale_registry_without_mutation_error(monkeypatch):
    from app.services.workflow_checkpoint import CheckpointManager
    import app.services.workflow_checkpoint as checkpoint

    manager = CheckpointManager()
    manager._review_queue = {}

    monkeypatch.setattr(
        checkpoint,
        "_load_registry",
        lambda: {
            "wf-stale-1": {"checkpoint_created_at": 1},
            "wf-stale-2": {"checkpoint_created_at": 2},
        },
    )

    saved_registry = {}

    def fake_save_registry(registry):
        saved_registry.clear()
        saved_registry.update(registry)

    monkeypatch.setattr(checkpoint, "_save_registry", fake_save_registry)

    class FakeGraph:
        def get_state(self, config):
            return _make_state_with_interrupt(None)

    rebuilt = manager.rebuild_review_queue(FakeGraph())

    assert rebuilt == 0
    assert saved_registry == {}


def test_rebuild_review_queue_rebuilds_valid_and_prunes_stale(monkeypatch):
    from app.services.workflow_checkpoint import CheckpointManager
    import app.services.workflow_checkpoint as checkpoint

    manager = CheckpointManager()
    manager._review_queue = {}

    monkeypatch.setattr(
        checkpoint,
        "_load_registry",
        lambda: {
            "wf-valid": {"checkpoint_created_at": 10},
            "wf-stale": {"checkpoint_created_at": 20},
        },
    )

    saved_registry = {}

    def fake_save_registry(registry):
        saved_registry.clear()
        saved_registry.update(registry)

    monkeypatch.setattr(checkpoint, "_save_registry", fake_save_registry)

    class FakeGraph:
        def get_state(self, config):
            thread_id = config["configurable"]["thread_id"]
            if thread_id == "wf-valid":
                return _make_state_with_interrupt(
                    {
                        "gate_type": "low_confidence_gate",
                        "evidence_snapshot": {"k": "v"},
                        "latest_draft_report": "draft",
                        "confidence_data": {"retrieval": 0.5},
                    }
                )
            return _make_state_with_interrupt(None)

    rebuilt = manager.rebuild_review_queue(FakeGraph())

    assert rebuilt == 1
    assert "wf-valid" in manager._review_queue
    assert "wf-stale" not in saved_registry
