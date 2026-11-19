from paige_triage.identity import TaskIdentity


def test_thread_id_equals_task_id() -> None:
    parent = TaskIdentity.supervisor()
    assert parent.thread_id == parent.task_id
    child = parent.spawn_child()
    assert child.parent_task_id == parent.task_id
    assert child.thread_id == child.task_id
