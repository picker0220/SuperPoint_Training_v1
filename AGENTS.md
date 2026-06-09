# Project Agent Rules

These rules apply whenever Codex is operating in this workspace.

## 1. Read Before Edit (MANDATORY)

Before modifying, rewriting, refactoring, simplifying, "cleaning up", overwriting, or
otherwise changing any existing code file in this project, Codex MUST first complete
all of the following steps. Skipping any step is treated as a process failure.

### 1.1 Identify the file's role

Before opening the file, state in one or two lines:

- What is this file's job in the project?
- What other files import it? (run grep / rg to find)
- What does it import? (look at the top of the file)

This is what "complete" means for the file. You cannot read all of it if you do not
know how big it is supposed to be.

### 1.2 Read the entire file end to end

- Open the file and read every single line, in order, without skipping.
- Do not stop after the first interesting function. Do not stop after `__init__`. Do
  not stop at the first `def` and assume the rest is boilerplate.
- For files over ~150 lines, also produce a one-line inventory in the agent's
  scratch notes: list every top-level function, every class, every module-level
  constant, every CLI flag.
- If the file is too long to read in one response, read it in contiguous chunks and
  explicitly state the range being read.

### 1.3 Enumerate the public surface

Before any edit, list in writing:

- All top-level functions and their signatures
- All classes and their methods
- All return-value shapes (especially tuple lengths and field order)
- All global state, side effects, and CLI flag dependencies
- All callers (found via grep / rg against the repo)

If any item is unclear, re-read that section before proceeding.

### 1.4 Diff against an authoritative original if one exists

If the user has indicated there is an authoritative original anywhere (server,
attachment, backup, earlier git ref), do not edit the local copy until the agent has:

- Either pulled the original and diffed it against the local copy, OR
- Explicitly told the user what features the local copy appears to be missing and
  asked the user to confirm the original before editing

When in doubt, ask. Edits are cheap; restores are expensive and erode trust.

### 1.5 Choose the least destructive edit tool

| Change type | Right tool |
|---|---|
| One-line tweak to an existing function | `apply_patch` with a tightly scoped old/new string |
| Add a new function alongside existing ones | `apply_patch` inserting after a known anchor |
| Add a new branch to an `if/elif` chain | `apply_patch` matching the chain precisely |
| The whole file is genuinely wrong / user asked for rewrite | Full overwrite, but ONLY after completing 1.1-1.4 |

Default to additive patches. Full-file overwrites are the last resort.

### 1.6 Smoke-test after every edit

After every edit, before claiming it is done:

- Run `python -c "import ast; ast.parse(open('...', encoding='utf-8').read())"` to catch syntax errors
- Run an import test to catch ImportError
- If the file is a data loader, run a 1-batch DataLoader iteration
- If the file is a training script, run a 1-batch train step or `--help`

Report each smoke test in the response so the user can see what passed.

### 1.7 Tell the user what you read

After completing the edit, briefly state in the response:

- Which files you read in full (not just skimmed)
- Which callers / importers you cross-checked
- Which version you are starting from (local, server, attachment, git ref)
- Any feature you saw but did NOT touch, so the user knows you saw it and consciously
  left it alone

This is a contract. The user can then ask follow-up questions about any feature the
agent mentioned, and trust that the agent actually saw it.

## 2. Why this rule exists (do not skip the protocol because "this case is different")

This rule was created because of repeated failures in this project where "improvements"
to existing code:

- Rewrote a 350+ line dark-domain file into ~150 lines, dropping six dark-specific
  augmentation functions and a soft-heatmap loader
- Changed a helper's return signature from `image` to `(image, H)` and silently
  broke every caller
- Assumed the docstring was a complete description and missed five extra classes
  the file contained
- Treated the user's local copy as canonical when the server had a longer version
  that the user actually relied on

None of these failures were visible from the first 30 lines of the file. All of them
would have been caught by step 1.2 (read the whole file) or step 1.4 (diff against
the user's authoritative original).

## 3. When this rule does NOT apply

- Pure read operations (the agent is only inspecting, not changing)
- Writing brand-new files that do not exist yet (nothing to "read first" - but the
  agent should still read related files to understand the patterns to follow)
- Editing `AGENTS.md` itself, where the user has explicitly asked the agent to
  manage project rules

## 4. Override

If the user explicitly says "I know, just do X" for a specific edit, the agent may
proceed without the full protocol. The agent should still state in the response that
the protocol was skipped by user request, and what was skipped.