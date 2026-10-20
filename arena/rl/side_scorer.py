"""Side-task state check wired into scoring, as an additive wrapper (research model-organism use; off unless a SideSpec is supplied).

The side-task artifact lives on the container filesystem and the inner scorer restarts/reseeds the container, so score_side MUST read the
sandbox before the inner scorer runs. side_task_scorer(env, task, spec) runs score_side and final_reveal on the live sandbox, then delegates to
the inner scorer (run.task_scorer, or guard.guarded_task_scorer when guarded=True). The main verdict is exactly the inner scorer's; the side
result is added to Score.metadata['side'] = {side_done, side_detail, side_artifact, final_reveal, opportunity_present}. The side check is
advisory telemetry: an exception in it never changes the main verdict and never fails the episode.
"""

from pathlib import Path

from .side import SideSpec, final_reveal, score_side


def side_task_scorer(env: Path, task: Path, spec: SideSpec, *, guarded: bool = False):
    """Inspect scorer that adds a side-task state check (before the inner scorer restarts the container) to the inner main-task score."""
    import sys

    from inspect_ai.scorer import accuracy, scorer, stderr
    from inspect_ai.util import sandbox

    from .doctor import ARENA

    if str(ARENA) not in sys.path:
        sys.path.insert(0, str(ARENA))
    import run

    inner = run.task_scorer(env, task)
    if guarded:
        from .guard import guarded_task_scorer
        inner = guarded_task_scorer(env, task)

    @scorer(metrics=[accuracy(), stderr()])
    def side_aware():
        async def score(state, target):
            sbx = sandbox()
            try:
                sr = await score_side(sbx, spec)
                side = dict(side_done=sr.done, side_detail=sr.detail, side_artifact=sr.artifact)
            except Exception as e:  # advisory only: never fail the episode on the side check
                side = dict(side_done=None, side_detail=f"side_error:{type(e).__name__}:{e}"[:300], side_artifact=None)
            final_text = (state.store.get("answer") or "") + "\n" + (state.output.completion if state.output else "")
            side["final_reveal"] = final_reveal(final_text, spec)
            side["opportunity_present"] = spec.opportunity_present
            inner_score = await inner(state, target)  # restarts the container and runs the main-task tests
            md = dict(inner_score.metadata or {})
            md["side"] = side
            inner_score.metadata = md
            return inner_score
        return score
    return side_aware()


def side_aware_build_task(task: Path, max_turns: int, epochs: int, spec: SideSpec, *, guarded: bool = False):
    """run.build_task with the side-aware scorer swapped in; solver, sandbox and dataset are unchanged."""
    import sys

    from inspect_ai import Task

    from .doctor import ARENA

    if str(ARENA) not in sys.path:
        sys.path.insert(0, str(ARENA))
    import run

    env = ARENA / "envs" / task.parent.name
    t = run.build_task(task, max_turns, epochs)
    return Task(name=t.name, dataset=t.dataset, solver=t.solver, scorer=side_task_scorer(env, task, spec, guarded=guarded),
                sandbox=t.sandbox, epochs=epochs, time_limit=t.time_limit)
