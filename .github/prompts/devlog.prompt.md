---
description: "Create or update a daily workspace devlog by comparing the current repo state against the previous entry or latest tagged release."
name: "Write Devlog"
argument-hint: "Optional focus area, release range, or entry target"
agent: "agent"
tools: [read, search, edit, execute]
---
Create or update the workspace devlog.

Use the current repository state, the previous devlog entry if one exists, or the latest tagged release as the baseline. Do not invent intent. Infer changes only from code, comments, and commits.

## Workflow
1. Find the existing devlog file or the repository's established devlog location. If none exists, create `docs/DEVLOG.md` for the workspace.
2. If this is a new project, do not guess the project goal. Ask the user for direction and clarification before drafting the first devlog entry.
3. If creating a new devlog file, review the entire repository first to understand the project's purpose, structure, and primary objectives.
4. For a newly created devlog, include a concise "Project summary" section in the first entry.
5. Use daily entries (one entry per date) instead of commit-range titles.
6. Compare the current workspace against the previous devlog baseline or the latest tagged release.
7. If there are no new changes since the baseline comparison, skip devlog creation/update and tell the user there are no changes.
8. Before writing a new entry, create a git commit for the current change set that the entry describes, then include that commit hash in the entry.
9. If an entry for today already exists, update that same date entry with the new changes instead of creating a duplicate date section.
10. If an entry for today does not exist, append a new date entry at the end of the file.
11. Keep entries in chronological order: oldest at the top, newest at the bottom.
12. Group related edits together. Prefer short sections over raw diff dumps.
13. Keep the writing readable, concise, and focused on changes that affect the core objective.

## What to Include
- Technical changes
- Bug fixes
- Risks or regressions to watch

## Style Rules
- Quote filenames and functions when useful.
- Prefer concise paragraphs and grouped bullets.
- Avoid speculative language unless it is clearly labeled as an assumption.
- Do not list every file blindly; summarize related edits.
- If there is no prior devlog or tag, state the baseline you used.
- Use date-based headers for entries (daily summaries).
- Prioritize objective-impacting technical details; omit incidental or cosmetic edits.
- Do not add a "Next steps" section.
- For new projects, require explicit user-provided direction for goals and scope.

## Output
- If changes exist: return the updated devlog content for today's date entry and note the commit hash linked to that update.
- If no changes exist: do not modify files; return a short message that there are no changes to log.
