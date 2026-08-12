# Research: WoWViewer UI Overhaul

## Decision: Extend Spec 080 instead of replacing the shell wholesale

Spec 080 already owns the viewer UI consolidation and explicitly requires the left navigator to remain until replacement routes are proven. Spec 145 therefore starts with a safe shell slice: keybinding context, shortcut help, vertical main-page selection, bounded navigator content, and log layout. Named-frame migration and sidebar retirement remain later phases.

## Decision: Keep bottom bars unchanged in the first slice

The user identified the two bottom bars as working. The new work should preserve their ownership of persistent scene toggles and status information while improving the sidebars around them.

## Decision: Capture shortcuts are page-scoped

The current camera-path handler is called from the global keyboard loop whenever keyboard authoring is enabled. This is the collision described by the user. The handler will require the active Capture context, while global navigation and bottom-bar shortcuts retain their existing scope.

## Decision: Use a vertical main-page rail, not a second scrolling tab strip

The right workbench currently uses `FittingPolicyScroll` for Model/World/Tools. A fixed-width vertical selector avoids the main-page overflow and makes the page hierarchy visually obvious. Nested utility tabs remain a later consolidation target and will be audited separately.

## Decision: Wrap log content in the available width

The current log body prints one unwrapped string per entry inside a child with horizontal scrolling. The first fix is to disable horizontal scrolling and use wrapped text; filtering and history ownership are not changed.

## Decision: Version 0.5.2 is the current release line

The app title/About and viewer README already identify v0.5.2, while the two viewer project files and shared identity still report 0.5.0 or 0.5.0-dev. The metadata and READMEs will be aligned to v0.5.2. Support statements will distinguish implemented/provisional routes from pending real-client proof.

## Branch limitation

The prescribed feature script was invoked once with number 145 and short name `wow-ui-overhaul`. It failed before creating the branch or spec because Git could not create `I:/parp/parp-tools/.git/index.lock` under the managed workspace permissions. Manual Speckit artifacts are used on the current branch; no second branch-creation attempt will be made in this slice.
