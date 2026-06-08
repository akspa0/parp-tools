# Plan: 049 Viewer UI Consolidation

1. Add `SceneInspector` to `ShellPanelId` enum
2. Add shell panel definition (right lane, 420px default)
3. Add `DrawSceneInspectorPanelContent()` with tab bar dispatching to existing methods
4. Wire into `DrawShellPanelContent` switch
5. Add View menu item toggle
6. Done — no existing panels removed, no functionality lost
