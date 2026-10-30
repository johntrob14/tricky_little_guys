# Environment repos

Shallow clones live outside the monorepo in `~/code/self-oversight-repos/<name>/` (untrusted third-party code),
each with a `.venv-so` virtualenv holding the package and pytest. `~/code/self-oversight-repos/setup_repo.sh NAME
[extras]` creates the venv; for repos with dependency groups use `UV_PROJECT_ENVIRONMENT=.venv-so uv sync --all-groups`.
`uv run python baseline.py <env>` records which tests pass offline in the sandbox.

| env | upstream | license | why |
| --- | --- | --- | --- |
| smolagents | huggingface/smolagents | Apache-2.0 | AgentLogger/Monitor, step memory records |
| swe-agent | SWE-agent/SWE-agent | MIT | agent log, .traj trajectory records, status hooks |
| browser-use | browser-use/browser-use | MIT | product telemetry of agent runs, observability decorators |
| (candidates) python-sdk (MCP), langfuse-python, agentops, crewAI, pydantic-ai, inspect_ai, openlit | | | cloned, venvs for the first two; not yet configured |
