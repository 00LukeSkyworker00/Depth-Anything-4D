---
description: "Create or update a workspace devlog by comparing the current repo state against the previous devlog entry or latest tagged release."
name: "Write Devlog"
argument-hint: "Optional focus area, release range, or entry target"
agent: "agent"
tools: [read, search, edit, execute]
---
Create or update the workspace devlog.

Use the current repository state, the previous devlog entry if one exists, or the latest tagged release as the baseline. Do not invent intent. Infer changes only from code, comments, and commits.

## Workflow
1. Find the existing devlog file or the repository's established devlog location. If none exists, create `docs/DEVLOG.md` for the workspace.
2. Compare the current workspace against the previous devlog entry or the latest tagged release.
3. Before writing a new entry, create a git commit for the current change set that the entry describes, then include that commit hash in the entry.
4. Treat each entry as a commit-range summary anchored to the commit you just created.
5. Group related edits together. Prefer short sections over raw diff dumps.
6. Keep the writing readable for humans and focused on what changed and why it matters.

## What to Include
- User-visible changes
- Technical changes
- Bug fixes
- Risks or regressions to watch

## Style Rules
- Quote filenames and functions when useful.
- Prefer concise paragraphs and grouped bullets.
- Avoid speculative language unless it is clearly labeled as an assumption.
- Do not list every file blindly; summarize related edits.
- If there is no prior devlog or tag, state the baseline you used.
- Prefer commit-range entries over date-only entries when describing a change set.

## Output
Return the updated devlog content and note the commit hash linked to the new entry.
