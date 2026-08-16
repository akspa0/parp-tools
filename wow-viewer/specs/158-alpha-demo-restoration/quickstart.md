# Quickstart: Alpha Demo Restoration

## US1 — already done, verify via Spec 159

```powershell
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- wtf sweep `
  --archive-root "H:/CLIENTS/Vanilla/0.x/0_5_3_3368/World of Warcraft" `
  --build "0.5.3.3368"
```

Expect (already measured, committed `f0dffdaa`): `WTF\Config.wtf [Loose]` and
`WTF\DefaultBindings.wtf [Archive]`, 245/245 lines recognized, 0 unrecognized. This is US1's acceptance
scenario 1-2 already satisfied — no new command needed for reading.

## US2 — worldport/teleport execution (once Phase 2 lands)

```powershell
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- wtf run `
  --archive-root "H:/CLIENTS/Vanilla/0.x/0_5_3_3368/World of Warcraft" `
  --build "0.5.3.3368" `
  --file <hand-written test command file>
```

Confirm a worldport line's `status` is `Applied` and the reported position matches the file; confirm a
teleport line never triggers a map load.

## US3 — Alt+P (once Phase 3 lands)

Manual: launch the viewer, press Alt+P, confirm the performance overlay appears exactly as the existing
"Show Perf" toolbar button already shows it. Press plain "P" separately, confirm its existing behavior
(`ui.focus_pm4`) is unchanged.

## US4 — camera follow (once Phase 4 lands)

Manual: load a scene with an animating character model, select it, attach the camera to it, confirm
visual tracking through an animation cycle, detach, confirm free-fly resumes from the last position.

## US5 — torch light (once Phase 5a+5b land)

Manual: equip a torch on a character in a dim real scene (a night/interior area is the closest available
match to the referenced 2001 screenshot), confirm visible terrain/object brightening that moves with the
character. Before this: run the existing Terrain Alpha Risk Area regression check with the feature
present but zero lights active, to confirm no baseline regression.

## US6 — blocked

No quickstart. Nothing to run until Spec 159 (or the user) supplies a real source file. Check Spec 159's
own status first: `wtf probe` is the live mechanism once a candidate name exists.
