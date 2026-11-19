from pathlib import Path

import pytest

from paige_triage.runtime import build_demo_supervisor


@pytest.mark.asyncio
async def test_deep_investigation_completes(tmp_path: Path) -> None:
    supervisor, bundle = build_demo_supervisor(
        checkpoint_dir=tmp_path / "ckpts", incident_id="inc001", use_llm=False
    )
    await supervisor.start(bundle["incident"], bundle["alerts"])
    state = await supervisor.wait_until_complete(timeout=30)
    assert state.completed is True
    assert len(state.hypotheses) >= 2
    assert state.synthesis
    assert state.recommendations
    assert state.extra.get("loaded_skill_ids")
    # Durable checkpoint written
    loaded = supervisor.store.load(supervisor.task_id)
    assert loaded is not None
    assert loaded.completed is True
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_user_steering_spawns_extra_hypothesis(tmp_path: Path) -> None:
    supervisor, bundle = build_demo_supervisor(
        checkpoint_dir=tmp_path / "ckpts", incident_id="inc001", use_llm=False
    )
    await supervisor.start(bundle["incident"], bundle["alerts"])
    await supervisor.inject_user_input("also check DNS resolution for orders-api")
    state = await supervisor.wait_until_complete(timeout=30)
    assert any(
        "DNS" in h.statement or "User-provided" in h.statement or "dns" in h.statement.lower()
        for h in state.hypotheses
    )
    await supervisor.shutdown()
