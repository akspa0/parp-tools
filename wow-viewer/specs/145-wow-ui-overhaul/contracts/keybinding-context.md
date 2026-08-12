# Keybinding Context Contract

This is a viewer-shell contract, not a network API.

## Inputs

- Current main page selection.
- Current nested page selection.
- Whether ImGui is capturing keyboard text/input.
- Whether a feature's opt-in keyboard mode is enabled.

## Resolution

1. Global bindings are eligible when the viewer is not in a text-input capture state.
2. Page bindings are eligible only when their `ViewerKeyContext` matches the active page.
3. Capture authoring bindings additionally require Capture keyboard authoring mode.
4. A page binding must not mutate state when another page is active.

## Output

The resolver exposes the active bindings to the help surface and supplies the keyboard loop with the eligible action scope. The contract does not prescribe user-editable bindings in this phase.
