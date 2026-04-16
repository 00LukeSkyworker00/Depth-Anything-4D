# Devlog

## Project summary
- This project aims to achieve feed-forward 4D Gaussian splatting through 3D canonical scene tokens.

## 2026-04-16

Baseline: branch state before this day's updates (`5e6654c`). Linked commits: `d201597`, `1f4e85a`, `8458ffb`.

### Technical changes
- [inference_4d.sh](../inference_4d.sh), [src/depth_anything_3/configs/da3-large-4d.yaml](../src/depth_anything_3/configs/da3-large-4d.yaml), and [train.py](../train.py) were updated to use newer 4D inference artifacts, simplify `scene_head`, and adjust `compute_loss()` depth supervision by removing confidence-mask/log-depth preprocessing.
- Added [.github/agents/research-review-improvements.agent.md](../.github/agents/research-review-improvements.agent.md), a workspace custom agent focused on repository-level review with web-backed improvement proposals for stability, speed, memory, and maintainability.
- Added [.github/prompts/devlog.prompt.md](../.github/prompts/devlog.prompt.md) to standardize devlog generation against a baseline (previous entry or latest tag), with daily summarized entries.
- Refined the devlog prompt workflow to enforce initialization checks for new projects, daily-entry behavior, no-change early exit, and concise objective-focused summaries.

### Bug fixes
- Removed a brittle depth confidence-mask/log-depth path in `compute_loss()` that could destabilize supervision under unfavorable confidence or depth-value conditions.
- Reduced ambiguity in devlog generation by preventing goal inference on new projects without user-provided direction.
- Fixed a prior workflow gap where unchanged repositories could still trigger unnecessary devlog updates.

### Risks / regressions to watch
- The stricter clarification-first behavior can slow first-entry generation when user direction is incomplete.
- The no-change early-exit depends on accurate baseline detection; incorrect baseline selection may suppress valid entries.
