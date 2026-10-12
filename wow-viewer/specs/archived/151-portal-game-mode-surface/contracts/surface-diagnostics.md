# Surface and Diagnostic Profile Contract

`SimpleInteractive` is a presentation/refresh policy, not a replacement application.

## SimpleInteractive

- Default on the simple surface and opt-in from the existing advanced shell.
- Shows concise scene status and controls for loading, camera/game mode, audio, and region overlay.
- Does not create raw payload tables, per-frame forensic panels, or unbounded log text.
- Retains essential errors, warnings, and bounded performance counters.

## AdvancedExplorer / Forensic

- Preserves the existing data-explorer panels and explicit diagnostic routes.
- May enable verbose logs and raw inspection after user action.

Switching profiles must be reversible while the scene is loaded. A profile change must not remove
the underlying decoded data or invalidate the current camera/world state.
