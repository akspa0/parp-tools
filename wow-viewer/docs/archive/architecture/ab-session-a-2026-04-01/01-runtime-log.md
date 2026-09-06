# Runtime Log

## 2026-04-01

### Initial debugger state

- x64dbg-mcp status reported:
  - is_debugging = true
  - is_running = false
- Baseline paused EIP snapshots observed in system DLL space, not WoW module anchors.

### M2 anchor breakpoint setup

Breakpoints set successfully at static image addresses:

- 0x0083cc80 (skin choose/load)
- 0x00838490 (skin init/rebuild)
- 0x00836600 (combiner builder)
- 0x00835a80 (format %02d.skin)
- 0x00835a20 (format %04d-%02d.anim)

Additional controllable sanity-check breakpoint:

- 0x0077f600 (console command handler for maxLOD)

### Post-restart live captures (x64dbg)

#### Hit A: 0x0083cc80 (M2_ChooseAndLoadSkinProfile)

- EIP: 0x0083cc80
- Registers:
  - EAX=0x00000001
  - EBX=0x19502B78
  - ECX=0x19502B78
  - EDX=0x0014B940
  - ESI=0x05B99030
  - EDI=0x0014B940
  - ESP=0x047FFBAC
  - EBP=0x047FFBD4
- Entry disassembly confirmed expected skin-profile choose/load prologue and quality-tier branch.
- Memory at ECX/related model structure contained path bytes for:
  - interface\glues\models\ui_mainmenu_northrend\ui_mainmenu_northrend.m2
- Interpretation:
  - anchor is active and reachable in this runtime
  - first captured model was UI path, not yet world doodad path

#### Hit B: 0x00835a80 (M2_FormatSkinFilename_02d)

- EIP: 0x00835a80
- Stack decode at entry:
  - return = 0x0083CB60
  - arg1 (src model path ptr) = 0x19502BB4
  - arg2 (skin profile id) = 0x00000001
  - arg3 (dest buffer ptr) = 0x047FFA84
- Source path bytes at arg1 confirmed same UI model path:
  - interface\glues\models\ui_mainmenu_northrend\ui_mainmenu_northrend.m2
- Interpretation:
  - exact `%02d.skin` formatter path is active and matches expected call shape
  - current hit stream still not yet world-M2 specific

#### Hit C: 0x00838490 (M2_InitializeSkinProfileAndRebuildInstances)

- EIP: 0x00838490
- Registers at hit:
  - ECX=0x19502B78 (same active model object family as Hit A)
  - EAX=0x1C7AED68
  - EDX=0x00030010
  - ESP=0x047FFBAC
- Entry disassembly confirmed expected guard and setup sequence:
  - tests `byte ptr [edi+0x08], 0x02`
  - references `edi+0x150` and `edi+0x170` as profile/runtime payload pointers
- Associated model path memory remained in UI glue family (`ui_mainmenu_northrend.m2`).

#### Hit D: 0x00836600 (M2_BuildCombinerEffectName)

- EIP: 0x00836600
- Registers at hit:
  - ECX=0x19502B78
  - ESP=0x047FFA38
  - EAX=0x1
  - EDX=0x0
- Stack decode at entry:
  - return = 0x00836DAB
  - arg1 = 1
  - arg2 (flag payload) = 0x10
- Entry disassembly confirmed bitfield split/combiner routing behavior (mask/shift of packed mode bits) matching static decompilation.
- Model context remained UI glue path during this capture.

### Runtime stepping behavior

- Repeated Continue/Run calls landed in system DLL frames (EIP values in 0x71xx/0x73xx/0x74xx/0x75xx ranges).
- Confirmed stops were captured at 0x0083cc80 and 0x00835a80.
- Immediate next runtime task is to capture equivalent hits for world-path M2s plus combiner and skin-init anchors (0x00836600 and 0x00838490).

### Control-session interruption

- After clearing noisy breakpoints and issuing resume, MCP run call timed out and debugger state transitioned to not debugging.
- Treat this as a control/session interruption; reconnect or reattach is required before the next live capture pass.

### Process memory sanity check

- PE header signature found at 0x6EF20000: bytes start with MZ.
- This address is a mapped module base candidate observed from stack context.
- Static M2 anchor disassembly at low addresses still resolves to valid code and matches Ghidra decompilation structure.

### Deep continuation pass (x64dbg + Ghidra correlation)

Additional captures in the same window confirmed a contiguous native choose/load/init/rebuild/effect chain:

- `0x0083cd2a` (inside `M2_ChooseAndLoadSkinProfile`)
  - `EAX=1` just before `FUN_0083cb40` call, so profile index `1` is selected.
  - `EDI=0x40` in the same frame, matching the selected quality bucket threshold path.
- `0x0083cb60` (post `M2_FormatSkinFilename_02d` call inside `FUN_0083cb40`)
  - stack-local buffer built as:
    - `interface\\glues\\models\\ui_mainmenu_northrend\\ui_mainmenu_northrend01.skin`
- `0x0083cd32` (post skin-load call site)
  - `EAX=1` confirms load success path continuation.
- `0x0083cd50` and `0x0083cd6f`
  - profile and texture-array setup path continues with non-null allocation result (`EAX=0x15C61BA8` at `0x0083cd6f`).
- `0x00838561` (inside `M2_InitializeSkinProfileAndRebuildInstances` completion path)
  - model state flags at `this+0x8` transitioned from `0x52415701` to `0x52415703` after the load bit set.
- callback rebuild loop confirmation
  - hit `0x00824510` and `0x00832ea0` with `ECX=0x1457A250`, proving live instance reset/init callbacks executed.
- combiner return confirmation
  - hit `0x00836dab` after `M2_BuildCombinerEffectName` with non-null handle (`EAX=0x15A042C8`).

Notes:

- all confirmed live captures in this pass were still in UI model context (`ui_mainmenu_northrend.m2`), not yet world-path M2 content.
- after later continue-run attempts, debugger control timed out and MCP reported `is_debugging=false`; world-path capture now requires reattach before the next pass.

### Static hidden-path continuation (same pass)

Ghidra-side continuation after debugger control timeout recovered additional hidden-path details:

- startup callsite around `0x004048b8` calls `M2_RegisterRuntimeFlags` and then `FUN_0081c6e0` (M2 cache/runtime init wrapper)
- `M2_RegisterRuntimeFlags` forces bit `0x8` in its return (`uVar1 | 8`)
- `FUN_0081c0d0` has a fallback branch that sets bit `0x40` only when `(flags & 0x8) == 0`, so this branch is likely not taken in normal startup-driven init
- callback-owned flag writes are explicit:
  - doodad batching (`0x20`)
  - particle batching (`0x80`)
  - additive particle sorting (`0x100`)
- `M2Faster` and `M2FasterDebug` callbacks flow through `FUN_00402100`, which can generate high optimization masks (`0x2000`, `0x6000`, `0xe000`) with non-obvious parser gating
- repeated probe path is real and called through a prewarm chain:
  - `FUN_0053c520`, `FUN_0053e810`, `FUN_0053e930`, `FUN_0053eaa0`
  - this chain repeatedly invokes `M2_NormalizeModelPathAndProbeSkins` and appears tied to player-object update flows (`FUN_006e7d60`, `FUN_006e7e00`)
- a secondary portrait-specific M2 path (`FUN_00619580`) was also recovered; it uses M2 runtime/cache state but renders through a dedicated portrait texture pipeline rather than world-scene submission
- strict cache-open rejection path (`FUN_0081c390`) was confirmed:
  - unsupported extension -> `Model2: Invalid file extension: %s`
  - open failure -> `Model2: File not found: %s`
  - `.mdl`/`.mdx` are still normalized to `.m2` before open

### Static subsystem sweep continuation (Ghidra only)

After the debugger detached, a static-only subsystem pass was added for renderer architecture continuity:

- rendering/effect ownership
  - `FUN_00780f50` reloads `MapObj.wfx`, `MapObjU.wfx`, `Model2.wfx`, `Particle.wfx`, `ShadowMap.wfx`
  - `FUN_00876d90` + `FUN_00876be0` + `FUN_00872d30` + `FUN_008728c0` confirm explicit effect-load/cache/bind seams
- shader capability gating
  - `FUN_0068a9a0` and `FUN_00684c40` log chosen shader targets
  - `FUN_0078de60` shows specular enable depends on pixel shader support
- liquids
  - `FUN_008a3e00` / `FUN_008a3f70` / `FUN_008a4070` / `FUN_008a4190` map to liquid shader families
  - `FUN_008a1fa0` and `FUN_008a28f0` resolve liquid type through material/settings banks with water fallback
  - `FUN_007cefd0` and `FUN_007cf790` confirm `MapChunkLiquid.cpp` runtime object ownership
- particles
  - `FUN_00821100` is the merged emitter batch path (`ParticleBatch`)
  - `FUN_008214e0` is direct/model-linked particle submission path
  - `FUN_0081f330` initializes particle effect handles including `Particle_Unlit`
- lighting
  - `FUN_008bdfc0`..`FUN_008be1a0` expose `Light*.dbc` table seams
  - `FUN_0079e7c0` allocates `WLIGHT` and `WCACHELIGHT` map-light pools
  - debug/script light controls stay live via `AddLight` / `AddCharacterLight` / `AddPetLight` / `ResetLights`
- LIT status
  - no positive `.lit` / `.LIT` file-path or formatter seam found in this pass
  - `Unlit` references found were effect-mode labels, not standalone file-loader evidence

## 2026-04-02

### Restarted x64dbg session and live file-open sampling

- debugger was restarted and reattached to live Win32 WoW process
- targeted breakpoint at `0x004609b0` (`FUN_004609b0`, Storm open-wrapper path) hit successfully
- sampled argument path payloads included:
  - `sound\\emitters\\Emitter_Stormwind_BehindtheGate_03.wav`
  - `Shaders\\Pixel\\ps_3_0\\Desaturate.bls`
- this gives fresh runtime evidence that active open traffic in this scene is currently audio/shader-file driven and still does not show positive `.lit` activity in sampled hits

### Extension-gate corroboration during restarted pass

- `FUN_0081c390` remains the strict `Model2` extension gate in static/decompilation view
- compare logic in `FUN_0081c390` and `M2_NormalizeModelPathAndProbeSkins` still reflects:
  - `.m2` accepted path
  - `.mdl`/`.mdx` rewritten to `.m2`
  - no positive `.lit` extension branch recovered in this loader seam

### Current runtime blocker in restarted pass

- repeated `DebugRun` operations often re-paused in system DLL frames before reaching target M2 chain breakpoints
- this currently blocks efficient world-path M2 chain capture in the same pass despite breakpoints being armed
- follow-up still needed: run-until-user-code style stabilization before resuming full world-path chain harvest
