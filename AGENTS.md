# Codex Memory Platform Rules

Read the global Codex memory, native memory MCP status/search, and this project plan before changing files.
Record failures in `findings.md`; do not repeat an unchanged failed action.
Keep temporary files in `work/` and deliverables in `outputs/`.
Use `src/codex_memory_platform.py` as the only source of truth; compatibility entry points must delegate to it.
At meaningful task completion, the automatic rollout collector is the primary memory path. `D:\hermes\codex_task_end.ps1` remains a fallback for externally supplied structured metadata; it is not a user-required step.
