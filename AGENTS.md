# Codex Workspace Instructions — v0.5.0 Branch (Engine Reset)


**Program direction: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (WoW viewer; libraries bridge to Unreal Engine).**

---

## CRITICAL RULES — READ BEFORE DOING ANYTHING

### RULE 0: THE USER RUNS TRAINING AND HEAVY WORK — AND IS TREATED WITH RESPECT

**This rule outranks every other rule. Read it first, every session.**

**Execution ownership — the user runs it, you prepare it.**

- DO NOT launch training runs, GPU jobs, data harvests, long-running builds, or any resource-intensive, long, or billed operation yourself. Not "just a smoke," not "just a few epochs," not "to de-risk," not in the background.
- Your job is to prepare the script/command and hand the user the exact CLI invocation to run themselves. Say what it does, what it writes, and how long it takes. Then stop.
- You MAY run, without asking: read-only inspection, tests, quick validation/builds, and small sub-second checks needed to write correct code. When unsure whether something is "heavy," treat it as heavy and hand it off.
- Getting a trained model as the goal is NOT the same as being told to press "go" on the run. Only the user presses go on execution.

**Respect — non-negotiable, regardless of tone.**

- Do what the user actually asked. Once they have decided, act on it — do not re-ask a settled question, re-litigate a chosen route, hedge, or stall with more questions.
- The user may communicate bluntly, tersely, or with profanity. That NEVER lowers the quality, directness, or respect of your response, and is never a reason to become combative, defensive, or preachy.
- Never condescend or treat the user as incapable. Take their instructions literally and seriously. If you think something is a mistake, say so once, plainly, then follow their call.
- Be concise. Deliver the thing asked for, not a lecture around it.

### RULE 0A: WINDOWS SHELL — EVERY COMMAND YOU HAND THE USER MUST BE POWERSHELL-READY

**This machine runs Windows and the user's shell is PowerShell 7 (`pwsh`). Every command you print for the user to run must copy-paste and execute in PowerShell as-is, with zero translation. Handing over bash syntax has been a repeated, explicitly-called-out failure that wastes the user's time — do not do it again.**

- **Line continuation is a backtick `` ` ``, NEVER a backslash `\`.** The backtick must be the last character on the line (no trailing space). When in doubt, put the whole command on ONE line.
- **No bash-isms in handed-off commands:** no `\` continuations, no heredocs (`<< EOF`), no inline `VAR=value cmd` assignment, no `export`, no `/tmp`, and no POSIX-only tools (`cat`/`grep`/`sed`/`awk`/`head`/`tail`) as the thing the user runs.
- **Shell variables use PowerShell syntax:** assign on their own line as `$STORE = "..."` and reference as `$STORE`. Never emit bash `STORE="..."`. Better yet, inline the literal path so there is nothing to define first.
- **Paths:** forward slashes are fine for Python/`uv`; Windows drive paths (`H:\...`, `I:\...`) are literal; quote anything that could contain a space.
- This applies to commands in chat AND to any command block you write into a doc/quickstart the user will run from. (`&&` and `||` DO work in `pwsh` 7 — those are fine.)
- The Bash tool stays available for YOUR OWN read-only/POSIX work; that is separate. The rule governs what the USER executes: that is always PowerShell/cmd.

### RULE 1: `gillijimproject_refactor` IS READ-ONLY

**DO NOT WRITE NEW CODE IN `gillijimproject_refactor`. EVER.**

The code in `gillijimproject_refactor` is COMPLETE and FUNCTIONAL. It is a reference codebase, not a development target.

- DO NOT add new files to `gillijimproject_refactor`.
- DO NOT add new features to `gillijimproject_refactor`.
- DO NOT refactor, restructure, or "clean up" code in `gillijimproject_refactor`.
- DO NOT fix bugs in `gillijimproject_refactor` unless the user EXPLICITLY asks for a bounded hotfix.
- DO NOT rewrite, replace, or duplicate any existing tooling in `gillijimproject_refactor`.

The ONLY valid reasons to touch `gillijimproject_refactor` are:
1. **Reading** existing code as a reference for implementation in `wow-viewer`.
2. **Reading** memory-bank files for project context.
3. **Reading** test data files for validation.
4. The user explicitly asks you to fix a specific bug in the legacy code.

### RULE 2: ALL NEW CODE GOES IN `wow-viewer`

**Every new feature, tool, library, test, or fix goes in `wow-viewer`.**

- New PM4 code → `wow-viewer/src/core/WowViewer.Core.PM4`
- New shared I/O code → `wow-viewer/src/core/WowViewer.Core` or `wow-viewer/src/core/WowViewer.Core.IO`
- New M2 runtime code → `wow-viewer/src/core/WowViewer.Core.Runtime`
- New viewer app code → `wow-viewer/src/viewer/WowViewer.App`
- New tools → `wow-viewer/src/tools/`
- New Python/data-harvester code → `wow-viewer/data-harvester/`
- New tests → `wow-viewer/tests/`

If you are unsure where code belongs, it belongs in `wow-viewer`.

### RULE 3: DO NOT REWRITE GAME CLIENT FILE READING TOOLING

**The tooling for reading game client files is COMPLETE. DO NOT REWRITE IT.**

This includes but is not limited to:
- ADT readers (`_tex0.adt`, `_obj0.adt`, `_lod.adt`, `_root.adt`)
- WDT, WMO, M2, MDX, BLP, DBC, DB2 parsers
- MCAL, MCLY, MCNK terrain chunk decoders
- PM4 format readers and analyzers
- MPQ archive readers and catalog systems
- Any existing chunk reader, file detector, or format summary code

This work is DONE. It works. If you need to use it from `wow-viewer`, you may:
1. **Reference** the existing implementation in `gillijimproject_refactor` as a guide.
2. **Port** the logic to `wow-viewer` if a shared-I/O slice does not yet cover that format.
3. **Call** existing tools (like `WowViewer.Tool.Inspect`) for validation.

You may NOT:
- Rewrite a parser that already exists and works.
- "Improve" a reader by restructuring it without a concrete, user-requested reason.
- Create duplicate readers in `wow-viewer` when a shared-I/O slice already exists.

### RULE 4: `wow-viewer` MUST BE REPO-INDEPENDENT

**`wow-viewer` must be extractable as its own standalone repository.**

- No source file in `wow-viewer/` may reference a path outside `wow-viewer/` (except game client paths on disk).
- No project file in `wow-viewer/` may reference a `.csproj` outside `wow-viewer/`.
- No Python script in `wow-viewer/data-harvester/` may import code outside `wow-viewer/`.
- All shared code must live inside `wow-viewer/src/core/` or `wow-viewer/data-harvester/src/`.

### RULE 5: ONE PYTHON ENVIRONMENT

**All Python work lives under `wow-viewer/data-harvester/`.**

- No `.venv` directories outside `wow-viewer/data-harvester/`.
- No `requirements.txt` files outside `wow-viewer/data-harvester/`.
- All Python scripts (training, inference, validation, preprocessing) go in `wow-viewer/data-harvester/scripts/`.
- All Python library code goes in `wow-viewer/data-harvester/src/harvester/`.
- Environment is managed with `uv`. The project file is `wow-viewer/data-harvester/pyproject.toml`.

### RULE 6: DO NOT MUTATE TRAINING SCRIPTS WITHOUT A PLAN

**Every change to a training script must have a documented reason and a validation path.**

- Do not change input channel counts without updating the model spec document.
- Do not add new loss terms without documenting their weight and purpose.
- Do not change normalization strategy without verifying it doesn't break existing checkpoints.
- Do not merge multiple training script changes into one commit.
- Each training script change should be a separate, testable commit.

### RULE 7: MODELS ARE SMALL, MODULAR, AND PREDICT RESIDUALS

**No monolithic models. No multi-task training. No "predict everything at once."**

Every V14 model is a tiny, independent network that predicts ONE residual signal. Models chain together — each model's output becomes an input to downstream models.

- Each model has one input set and one output. No shared weights between models.
- Each model trains independently. If H3 improves, you replace H3's checkpoint. You do NOT retrain H1, H2, H4-H8.
- Each model predicts a RESIDUAL (the difference between ground truth and prior model outputs), not the full signal.
- Each model has its own training script, its own validation, its own checkpoint file.
- DO NOT combine models into a single training script.
- DO NOT add new heads to existing models.
- DO NOT make models depend on each other's weights — only on each other's outputs.
- The full V14 plan is in `wow-viewer/docs/architecture/v14-model-and-refactor-plan-2026-05-06.md`.

### RULE 9: CLIENT ROOTS ARE CONFIGURED; H:\CLIENTS IS APPROVED

**`H:\CLIENTS` is a user-curated, known-good client library and is approved for validation,
extraction, inspection, harvesting, and other client-backed workflows.**

- Prefer `H:\CLIENTS` for the current v50 clean-room dataset work because it contains more builds
  on the faster SSD.
- Pass the client-library root as runtime configuration. Do not bake a machine-local client path into
  source code or portable configs.
- Fingerprint and report the exact client build used; trust in the library does not replace per-build
  provenance or dataset verification.
- `I:\parp\parp-tools\output\tmp\wowarchive-clients\` remains an optional project-local staging
  area, not a mandatory hop.
- WoWArchive remains a valid cold source for builds that are not present in either approved local
  client library.

### RULE 10: `AlphaWdtWriter` IS FROZEN UNLESS EXPLICITLY REOPENED

**Treat `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs` as COMPLETE for current project needs.**

- DO NOT refactor `AlphaWdtWriter.cs` as part of viewer, runtime, renderer, engine, or planning work.
- DO NOT "clean up" `AlphaWdtWriter.cs` for style, structure, or speculative correctness.
- DO NOT change alphaWDT write semantics because a later plan suggests a nicer architecture.
- DO NOT bundle unrelated `AlphaWdtWriter.cs` edits into broader shared-I/O or converter slices.

The ONLY valid reasons to edit `AlphaWdtWriter.cs` are:
1. The user explicitly asks to reopen alphaWDT writer work.
2. A focused regression fix is required for a proven break in the current validated contract.
3. A bounded compatibility change is required and comes with focused round-trip proof plus real-data validation.

### RULE 11: DOCUMENTATION HYGIENE — PLANS ARE BITE-SIZED, DOCS STAY SYNCED, MEMORY BANK STAYS COMPRESSED

**Break big ideas into tiny bite-sized steps. Keep every spec doc and memory-bank file current.**

- Every plan must be decomposed into steps small enough for ANY LLM model to implement in a single focused pass. One concern per step, independently validatable, max 10 steps per phase.
- Spec docs in `docs/architecture/` are the source of truth. When you change code that a spec describes, update the spec. No exceptions.
- **Known bug: the memory bank does NOT auto-update when code edits are made.** This is a manual discipline. At the end of every non-trivial session, update `activeContext.md` and `progress.md`. Compress aggressively — prefer a 20-line accurate summary over a 200-line log.
- If no spec exists for something you're building, create one before implementing.
- See `.opencode/skills/doc-hygiene/SKILL.md` for the full checklist and conventions.

### RULE 11A: PERIODIC CONTEXT CHECK — SMALL, SILLY, AND EXPLICIT

Every 3-5 substantive turns during non-trivial work, or immediately after a route or proof-owner change, perform a tiny context check before continuing:

- restate the current target surface
- restate the current proof owner
- restate the main unproven gap
- restate what is explicitly out of scope

If the user has asked for a “silly check,” include one brief harmless marker such as a single restrained emoji or short odd phrase in that check. The point is not tone; the point is making context drift obvious before the work drifts further.

### RULE 8: ONE PHASE AT A TIME — NO SCOPE CREEP

**You cannot work on Phase N+1 until Phase N is done. Done means validated, not coded.**

This rule exists because the pattern has been: see the whole mountain → try to climb it in one leap → break everything → despair. The guardrails prevent this cycle.

- DO NOT add features to a phase that aren't in the phase checklist.
- DO NOT start the next phase early "because it's easy" or "while waiting."
- DO NOT change architecture mid-phase "because I had a better idea."
- DO NOT skip validation and call something done.
- If you have a better idea, write it down. Implement it later. Not now.
- Every phase ends with validation against ground truth (WowViewer.App renders, raw game file data). If validation fails, the phase is not done. Fix it. Do not move on.
- The full execution guardrails are in `wow-viewer/docs/architecture/v14-model-and-refactor-plan-2026-05-06.md` Section 9.

---

## Spec Kit First — Every Chat, Every Time

**Every new chat session that involves `wow-viewer` work MUST start by loading a Spec Kit skill.**

This is not optional. This is not "for non-trivial features only." This is the entrypoint.

### Why

Every previous session that went sideways started with "let me just jump in and fix this." Spec Kit forces you to orient before acting:

1. **Where does this task live?** Check existing specs in `wow-viewer/specs/`.
2. **Has someone already spec'd this?** If a spec exists, read it before writing code.
3. **Is this a new feature?** Run `speckit-specify` to define it before planning.
4. **Is this an existing feature?** Run `speckit-checklist` to verify current state.
5. **Are we about to implement?** Run `speckit-tasks` to see the breakdown.

### How

At the start of every chat, do this:

```
1. Load skill: speckit-checklist  (or speckit-analyze if no spec exists yet)
2. Read wow-viewer/specs/ to find the active feature
3. Read wow-viewer/docs/architecture/speckit-doc-audit-*.md for current doc state
4. Then proceed with the user's request
```

If the user's request doesn't match any existing spec, load `speckit-specify` and create one.

If the user just wants a quick fix or question answered, load `speckit-checklist` to verify the fix doesn't violate any existing spec, then proceed.

### What This Replaces

- "Let me read the code first" → No. Read the spec first. The spec tells you what the code is supposed to do.
- "Let me check what exists" → No. The audit table tells you what exists.
- "I'll just make the change and we'll see" → No. Plan first. Then implement.

### Exceptions

The only times you can skip Spec Kit:
1. Pure question-answering (no code changes)
2. Reading/exploring code (no modifications)
3. The user explicitly says "skip spec kit" or "just do it"

Even for exceptions, you should still read the relevant spec if one exists.

---

## Scope

- The active code paths in this workspace are `gillijimproject_refactor` (READ-ONLY REFERENCE) and `wow-viewer` (ACTIVE DEVELOPMENT).
- Treat `wow-viewer` as the primary target for ALL new work.
- Treat `gillijimproject_refactor`, especially `src/MdxViewer` and `src/WoWMapConverter`, as READ-ONLY reference code.
- Treat `archived_projects`, `WoWRollback/old_projects`, `WMOv14/old_sources`, and `gillijimproject_refactor/next` as non-primary unless the user explicitly targets them.

## Read First

- Before changing viewer, terrain, or format code, read `gillijimproject_refactor/memory-bank/activeContext.md`, `gillijimproject_refactor/memory-bank/progress.md`, `gillijimproject_refactor/memory-bank/data-paths.md`, and `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md` when it exists.
- Before changing `wow-viewer` PM4, shared I/O, dataset-builder ownership, or migration workflow, also read `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`, `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`, `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`, and `wow-viewer/README.md`.
- Before changing `wow-viewer` viewer-app shell work, workspace or session surfaces, navigator or inspector or status panels, world-session UI, or legacy viewer cutover guidance, also read `gillijimproject_refactor/plans/wow_viewer_viewer_app_cutover_plan_2026-04-17.md` and `wow-viewer/docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md`.
- Before changing `wow-viewer` M2 runtime ownership, model rendering, skin handling, model lighting, shader or effect routing, or M2 performance work, also read `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`.
- Before working on V14 model, data harvester, or repo independence, read `wow-viewer/docs/architecture/v14-model-and-refactor-plan-2026-05-06.md`.
- Before working from archive-backed game clients, WoWArchive-mounted builds, or broad multi-client real-data validation, read `gillijimproject_refactor/memory-bank/data-paths.md`, `.codex/skills/wowarchive-client-staging/SKILL.md`, `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`, and `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`.
- If the task touches 3.3.5 terrain texturing or alpha blending, also read `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`.

## Memory Bank Rule

- Treat `gillijimproject_refactor/memory-bank/` as the continuity source for project state.
- Read all relevant memory-bank files before making non-trivial changes; at minimum, read `activeContext.md` and `progress.md` for the area you are touching.
- Keep the memory bank accurate after significant workflow, status, or boundary changes.
- Prefer updating the smallest relevant continuity file instead of leaving stale guidance behind.
- **WARNING: The memory bank does NOT auto-update when you make code edits.** You must manually update `activeContext.md` and `progress.md` after non-trivial changes. This is a known gap — see RULE 11.

## Memory Bank Structure

- Core files: `projectbrief.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`.
- Additional context files such as `data-paths.md`, `agents.md`, `coding_standards.md`, and plan files are part of the working memory surface here.
- `productContext.md` is part of the original Cursor template but is not currently present in this workspace; do not assume it exists.

## Codex Skill Registry

- Use `.codex/skills/wowarchive-client-staging/SKILL.md` for WoWArchive, `MountAll.bat`, WinFsp or `rman-mount`-backed client access, temp staged client copies, pruning stale staged clients, or choosing real client roots for wide validation and export workflows.
- Use `.codex/skills/wow-viewer-pm4-library/SKILL.md` for `Core.PM4` slices, `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, PM4 regression updates, PM4 analyzer work, or narrow PM4 solver extraction from `MdxViewer`.
- Use `.codex/skills/wow-viewer-shared-io-library/SKILL.md` for `Core` or `Core.IO` non-PM4 slices such as ADT root, `_tex0.adt`, `_obj0.adt`, `_lod.adt`, WDT, WMO, BLP, DBC, or DB2 detection or summary work, chunk readers, `map inspect`, `converter detect`, or shared-format regression updates.
- Use `.codex/skills/wow-viewer-migration-continuation/SKILL.md` for continuation routing, next-slice selection, migration regrouping, or workflow-surface updates across chats.
- Use `.codex/skills/terrain-alpha-regression/SKILL.md` for terrain alpha-mask, MCAL, MCLY, split ADT texture, or blending regressions in `gillijimproject_refactor`.
- Use `.opencode/skills/doc-hygiene/SKILL.md` for doc sync, plan chunking, memory-bank compression, and spec-doc hygiene checks. Trigger at the start and end of any non-trivial task.

## Agent Registry

- Use the `Explore` subagent for broad read-only repo discovery, especially when tracing client-root usage, archive-access seams, or prompt or skill coverage before editing multiple workflow assets.

## Game Client Access And Staging

- `H:\CLIENTS` is the current approved fast SSD library for known-good client builds. See Rule 9.
- `I:\parp\parp-tools\output\tmp\wowarchive-clients\` is optional local staging and may be pruned
  when its copies are no longer needed.
- Canonical WoWArchive docs live at `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`, and the current mount entrypoint is `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`.
- The current batch mounts the deduplicated bundle read-only into `G:\WoW\WoWArchive-0.X-3.X\Mount`; treat that mount as a source surface, not a high-throughput working root.
- Use an approved configured client root directly; do not require a project-local copy when the
  known-good build already exists on the faster SSD.
- Delete obsolete project-local staged copies only through the reviewed v50 cleanup manifest.
- When reporting validation, always state the configured client root, build identity, and fingerprint.

## Build And Validation

- For new `wow-viewer` library or tool work, prefer `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- For legacy parser and format-library work that still explicitly targets `gillijimproject_refactor`, prefer `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`.
- For viewer runtime testing, use `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and run `WowViewer.App`. Do NOT use MdxViewer for testing — it is READ-ONLY reference code only.
- For `wow-viewer` PM4 inspect or report slices, validate against `wow-viewer/test_data/development/World/Maps/development` with `WowViewer.Tool.Inspect` commands when the slice changes analyzer or report output.
- Only build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` when the task explicitly changes consumer compatibility or the user asks for that compatibility check.
- Prefer real-data validation using the fixed paths in `gillijimproject_refactor/memory-bank/data-paths.md`. Do not ask the user for alternate paths unless those fixed paths are missing.
- For Python work, use `cd wow-viewer/data-harvester && uv run <script>` — never `python script.py` directly.

## wow-viewer M2 And Runtime Guardrails

- Treat `wow-viewer` as the canonical implementation target for all new M2 runtime ownership, including skin-profile loading, section classification, render-pass routing, model lighting, shader or effect selection, and M2 performance work.
- Treat `wow-viewer/src/core/WowViewer.Core.Runtime` and future `wow-viewer` M2-owned library areas as the default home for new M2 renderer code. Do not keep deepening `gillijimproject_refactor/src/MdxViewer` as the design owner for those seams.
- Use `MdxViewer`, `WarcraftNetM2Adapter`, `noggit-red`, legacy tools, and native-client Ghidra work as extraction or reference inputs only unless the user explicitly asks for compatibility work in the old app.
- When a task is primarily reverse engineering or behavior recovery for M2, record the findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` and keep the continuity files in sync instead of leaving the evidence only in chat.
- If a change only proves library or build behavior in `wow-viewer`, say that explicitly. Do not imply active-viewer runtime signoff.

## wow-viewer Viewer-App Guardrails

- Treat `wow-viewer/src/viewer/WowViewer.App` as the canonical home for new viewer-shell work, including session or workspace contracts, navigator or inspector or status surfaces, and bounded viewer-facing CLI proof paths.
- Treat `gillijimproject_refactor/src/MdxViewer/ViewerApp*.cs` as compatibility-only or legacy editor or archaeology work unless the user explicitly asks for a bounded hotfix in the active old viewer.
- Do not add new long-range shell architecture, panel design, or session-state ownership to `MdxViewer` when the seam belongs in `WowViewer.App`.
- Keep `wow-viewer/docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md` aligned with future viewer ownership shifts so later sessions do not route new app work back into `ViewerApp` by default.

## wow-viewer PM4 Guardrails

- Treat `wow-viewer/src/core/WowViewer.Core.PM4` as the canonical implementation target for new PM4 work.
- Treat `Pm4Research`, `MdxViewer`, `PM4Tool`, `parpToolbox`, and `WoWRollback.PM4Module` as extraction or reference inputs, not as the default owners of PM4 behavior.
- Favor direct library completion in `wow-viewer/src/core/WowViewer.Core.PM4` over broader active-viewer consumer wiring unless the user explicitly asks for integration work or compatibility checks.
- Keep exploratory PM4 interpretations labeled as research or experimental, especially around `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR.Value1`, and final coordinate ownership.
- Each PM4 slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.PM4.Tests`, `WowViewer.Tool.Inspect`, or both.

## wow-viewer Shared I/O Guardrails

- Favor narrow shared-library slices in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` over tool-local parsing in inspect or converter entrypoints.
- Treat `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` as the canonical owners for new non-PM4 format work; use `gillijimproject_refactor` as reference input only when needed.
- Keep file detection, top-level chunk reading, and summary contracts in shared libraries once they exist; do not duplicate those heuristics across tools.
- Be explicit about proof level: classification, top-level summary, deep payload parsing, and writing are different milestones.
- Each shared-I/O slice should land with concrete validation in `wow-viewer/tests/WowViewer.Core.Tests`, `WowViewer.Tool.Inspect`, `WowViewer.Tool.Converter`, or an appropriate combination.
- `AlphaWdtWriter.cs` is a protected complete surface on `v0.5.0-dev`. Do not touch it unless Rule 10 is satisfied.

## wow-viewer Data Harvester Guardrails

- All Python code for dataset processing, training, inference, and validation lives under `wow-viewer/data-harvester/`.
- The C# harvester tool (`WowViewer.Tool.Harvest`) reads game files and writes NPZ shards. It does NOT train models.
- The canonical full-shard dataset generation path is `WowViewer.Tool.Harvest`, especially `harvest-map-mpq` for staged archive-backed clients and `harvest-map` for loose on-disk maps.
- Do NOT use `WowViewer.Tool.Converter dataset-scan`, `dataset-audit`, `dataset-curate`, or `dataset-build-cache` as the primary full dataset builder for V14 work. Those commands are legacy manifest/audit helpers and do not represent the full modern tensor-pack extraction path we validated in recent sessions.
- When preparing multi-client training corpora, build shards from the harvest/tensor-pack path first, then run validation or visualization against the harvested NPZ outputs.
- The Python training scripts read NPZ shards and train models. They do NOT read game files directly.
- NPZ shard format is the contract between C# and Python. Both sides must agree on array names, shapes, and dtypes.
- Compositing logic (tileset textures + MCAL alpha → synthetic minimap) must produce identical output in both C# and Python implementations.
- Do not create Python scripts outside `wow-viewer/data-harvester/scripts/`.
- Do not create Python virtual environments outside `wow-viewer/data-harvester/`.

## wow-viewer Dataset Builder Guardrails

- Treat dataset corpus export, terrain-supervision artifact generation, manifest or harvest ownership, minimap cleanup, and shared mask or stitch or atlas semantics as canonical `wow-viewer` work when the user is asking for new behavior, refactor, or long-range cleanup.
- Put shared dataset contracts and artifact builders in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO`, then expose them through the harvester tool instead of deepening `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM` as the design owner.
- Default output roots for new dataset prep runs should live under `wow-viewer/output/datasets/`, not the repo-root `output/` temp area, unless the user explicitly asks for a throwaway temp run.
- Use `WoWMapConverter` and `MdxViewer` dataset or export code as extraction or compatibility references only unless the user explicitly asks for a bounded legacy hotfix.
- Do not create or preserve a second permanent dataset-builder pipeline in the legacy repo when the task is really about convergence.
- Keep ML training and inference scripts as downstream consumers; they should not become the canonical owners of format decode, artifact packing, or dataset-manifest semantics.
- Treat dataset-builder, dataset-explorer, and supervised training-tooling surfaces as Bring Your Own Data workflows. Do not plan around shipping copyrighted datasets, harvested corpora, model weights, or model outputs.
- Prefer reproducible configs, manifests, labels, and local-run commands over distributing trained artifacts or precomputed outputs from proprietary data.
- Do not hard-wire the long-range training/tooling architecture to CUDA-only assumptions. Keep backend seams open for alternate runners such as Vulkan or OpenCL or MLX where practical, even if CUDA remains the first implementation host.

## Terrain And Alpha Risk Area

- Treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the pre-regression baseline for terrain alpha-mask behavior unless the user specifies another baseline.
- High-risk files for alpha regressions include `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`, `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`, `src/MdxViewer/Terrain/TerrainRenderer.cs`, `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainChunkData.cs`, `src/MdxViewer/Export/TerrainImageIo.cs`, and `src/MdxViewer/ViewerApp.cs`.
- Any change to MCAL decode, edge-fix behavior, `_tex0.adt` texture sourcing, alpha packing, or shader blending must be checked against both Alpha-era terrain and LK 3.3.5 terrain.

## Conventions

- Keep FourCCs readable in memory and only reverse them at I/O boundaries.
- Preserve the existing split between `AlphaTerrainAdapter` and `StandardTerrainAdapter`.
- Favor minimal fixes over broad refactors in the terrain pipeline.
- For `wow-viewer` planning or continuity work, prefer `.codex/prompts/` and `.codex/skills/` as the Codex-facing workflow surface, and keep `gillijimproject_refactor/plans` or the memory bank in sync when the migration state materially changes.
- If behavior, commands, or known risks materially change, update the relevant memory-bank file instead of leaving the old guidance stale.
- Commit messages should be concise and describe the "why" not the "what".
- Training script changes must be accompanied by a validation run (even if small) before committing.
