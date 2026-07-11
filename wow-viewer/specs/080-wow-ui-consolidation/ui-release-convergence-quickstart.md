# UI Release Convergence Quickstart

1. Build the viewer:

   `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`

2. Start with the surface inventory. Do not migrate a panel until it has a
   `UiSurfaceRecord` and a declared tabbed and legacy route.

3. Fix route-integrity defects first. The initial known repro is Settings:
   open File -> Settings with tabbed UI enabled and verify a Settings window
   appears; repeat in legacy mode.

4. For each surface, capture manual evidence with its required model/world
   prerequisite. Record a route result for both modes.

5. A release is blocked if any visible surface is `misrouted`, `missing`, or
   `placeholder`, or if a disabled surface lacks an explanatory reason.

6. Before optimizing, capture the same staged dense-map camera state with the
   minimap closed/open and each high-pressure overlay off/on. Select the next
   performance change only from the largest measured cost.
