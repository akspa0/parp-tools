# Coding Standards — wow-viewer

How to write code that fits into this codebase, ships without breaking neighbors, and stays maintainable for the next contributor.

For repository-wide guardrails (read-only legacy code, one-phase-at-a-time, doc hygiene), see `wow-viewer/AGENTS.md`. This document is the day-to-day companion: the rules you apply on every commit.

## Languages and Stack

- C# (.NET 10) for libraries, tools, and the viewer app.
- Python 3.11+ under `wow-viewer/data-harvester/` for the ML pipeline. Managed with `uv`. The project file is `wow-viewer/data-harvester/pyproject.toml`.
- GLSL or HLSL shaders live next to the renderer that uses them.

## Project Layout

```
wow-viewer/
├── src/
│   ├── core/                  # Shared libraries (no UI)
│   │   ├── WowViewer.Core/        # Data models
│   │   ├── WowViewer.Core.IO/     # Format readers and writers
│   │   ├── WowViewer.Core.PM4/    # PM4 chunk analysis
│   │   ├── WowViewer.Core.Runtime/# M2 runtime, render pipeline
│   │   └── WowViewer.Core.Anim/   # M2 animation pose extraction
│   └── viewer/
│       └── WoWViewer/         # 3D world viewer app (the only UI project)
├── tools/                     # CLI tools (thin wrappers over libraries)
├── tests/                     # xUnit tests, mirroring the library layout
├── data-harvester/            # Python ML pipeline (uv-managed)
├── docs/                      # Architecture docs, guides, audits
├── specs/                     # Spec Kit feature directories
└── memory-bank/               # Session continuity
```

One project, one concern. A library project does not depend on a tool project. A tool project may depend on any library. The viewer app may depend on any library but not on another tool.

## Library-First

Format readers and writers live in `WowViewer.Core.*` libraries. Tools are thin CLI wrappers in `tools/`. The viewer app is the only consumer that is allowed to add UI code.

If you find yourself adding parser logic to a tool, lift it into the appropriate core library first, then have the tool call the library. This keeps parsers reusable, testable, and shared between the CLI surface, the viewer, and any future consumer.

## C# Conventions

### Namespaces and files

- File-scoped namespaces: `namespace WowViewer.Core.IO;`
- One public type per file, named after the type.
- Folder layout mirrors namespace layout. `WowViewer.Core.IO.Maps` lives under `src/core/WowViewer.Core.IO/Maps/`.

### Types and members

- `var` when the type is obvious from the right-hand side; explicit types otherwise.
- Auto-properties over public fields.
- `readonly` on private fields that never change after construction.
- Records for value-like data carriers (chunk headers, manifest rows, report rows).
- `sealed` on classes that are not designed for inheritance.

### XML comments

- `/// <summary>` on every public type and every public member that is part of a stable contract.
- Internal and private members: comments only where the code is non-obvious.
- Comments describe intent, not mechanics. Prefer "why" over "what."

### FourCC handling

Game file formats use FourCCs (`MTEX`, `MVER`, `MSLK`). wow-viewer keeps them readable in memory:

```csharp
private const string SIG_MTEX = "MTEX";
private const string SIG_MD20 = "MD20";
```

- Read: convert from file to readable string once at the I/O boundary.
- Write: convert from readable string to file bytes once at the I/O boundary.
- Never compare against a byte-reversed string. Never store a reversed string.
- Log readable signatures only.

### Paths

- Forward slashes in code, manifests, and reports.
- Lowercase paths in manifests and reports.
- Workspace-relative paths in code; resolve absolute paths at runtime from CLI args or env vars.
- `WowViewer.Core.Anim.PathNormalizer` is the canonical helper for anim-farm output paths; use it. Other domains have their own normalizers as needed.

See `wow-viewer/memory-bank/data-paths.md` for the authoritative list of paths and how to override them.

### Errors

- Throw the most specific exception type that fits (`InvalidDataException` for malformed input, `NotSupportedException` for known-but-unimplemented cases, `InvalidOperationException` for state-machine violations).
- Include enough context in the message to identify the offending record, offset, or path.
- Catch only when you can do something useful. Let unexpected exceptions propagate to the top-level handler.

## Python Conventions

- One project, one environment. The `uv`-managed environment lives in `wow-viewer/data-harvester/`. No `.venv` or `requirements.txt` outside that directory.
- Run scripts with `uv run --project wow-viewer/data-harvester <script>`, not bare `python`.
- Library code goes in `wow-viewer/data-harvester/src/harvester/`. Scripts go in `wow-viewer/data-harvester/scripts/`. Tests go in `wow-viewer/data-harvester/src/harvester/test_*.py`.
- Tensor / NumPy code is the contract between C# and Python. Both sides must agree on array names, shapes, and dtypes. Update the relevant spec's `contracts/` directory if you change either side.

## Tests

- xUnit for C#. Test project names end in `.Tests` and live under `wow-viewer/tests/`, mirroring the library structure.
- pytest for Python. Test files start with `test_` and live next to the code they exercise.
- One test class per production class is a good default. A test method should test one observable behavior.
- Tests are proof, not aspiration. A failing test blocks the change. A passing test that does not exercise the change is not a test.
- Use real data when it exists and is staged. Synthetic data is fine for narrow unit tests but never for proof runs.

## Commits

- Commit messages describe the "why," not the "what." The diff shows the what.
- One logical change per commit. A bug fix, a refactor, and a feature do not belong in the same commit.
- A spec implementation lands as a series of small commits, one per phase or per task. The full diff for a spec is reviewed commit-by-commit.
- Stage by name, not by wildcard. `git add .` is a smell.
- No secrets, no private paths, no `H:\CLIENTS` references.

## Documentation

- Public API surfaces and stable contracts have XML doc comments in the code.
- Architecture decisions and per-feature notes live under `wow-viewer/docs/architecture/` and stay in sync with the code that implements them.
- Feature work starts with a spec under `wow-viewer/specs/<NNN>-<feature>/`. Use the Spec Kit skills.
- The memory bank (`wow-viewer/memory-bank/`) is the continuity surface. After a non-trivial change, update `activeContext.md` and `progress.md`. Compress aggressively — prefer a 20-line accurate summary over a 200-line log.

## Guardrails in One Place

The most important repository rules, in distilled form. The full versions live in `wow-viewer/AGENTS.md`.

- **All new code goes in `wow-viewer/`.** The `gillijimproject_refactor/` codebase is read-only reference.
- **One phase at a time.** Finish and validate the current phase before starting the next.
- **Models are small, modular, and predict residuals.** No monolithic multi-task models. No shared weights between models. Each model is one input set, one output, trained independently.
- **Spec Kit first.** Every non-trivial feature starts with `spec.md`, then `plan.md`, then `tasks.md`. Use the `speckit-*` skills.
- **Staged clients only.** Game client access goes through `output/tmp/wowarchive-clients/`. Anything resembling `H:\CLIENTS` is a bug.
- **Paths are forward-slash, lowercase, workspace-relative in code.** Resolve absolute paths at runtime.
