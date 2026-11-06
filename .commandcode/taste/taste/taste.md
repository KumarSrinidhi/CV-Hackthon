# Taste
- Use UV for Python environment and dependency management (uv init / uv add / uv pin), not pip or conda. Confidence: 0.9
- Confine generated files and scratch work to a designated tmp/ directory inside the repo; don't create or modify anything elsewhere unless asked. Confidence: 0.6
- Push code to GitHub continuously and quickly (small, frequent pushes) when others are working on the repo; open a PR for each new piece of work or change. Confidence: 0.9
- Before starting execution, produce an upfront full plan with a timeline and phase breakdown, structured so teammates can work different phases in parallel and finish faster. Confidence: 0.9
- Materialize team plans/task boards as markdown files inside the repo (e.g., TEAM_PLAN.md) and push them to GitHub, so teammates can pick up different tasks themselves without going through the assistant. Confidence: 0.8
- Never add "Co-authored-by: CommandCodeBot <noreply@commandcode.ai>" (or any co-author trailer) to git commits. Confidence: 0.95
