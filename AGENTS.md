# Agent entrypoint

This repository's canonical session handoff and progress log is **[agent.md](./agent.md)**.

**MANDATORY for every agent / every new conversation:**
1. Read `agent.md` and `README.md` before inspecting or editing code.
2. Confirm HEAD, last recorded completed stage, test evidence and remaining tasks.
3. After each implementation/test stage, update `agent.md` in the same commit/PR.
4. Never report untested functionality as PASS; never commit secrets.

Only `agent.md` contains the progress log. Keep this `AGENTS.md` wrapper stable to help agent tooling discover it.
