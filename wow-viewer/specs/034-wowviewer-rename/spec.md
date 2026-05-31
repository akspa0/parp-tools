# WoWViewer Rename

## Goal
Mark the existing WowViewer.App as defunct and elevate the newly migrated MdxViewer to become the primary WoWViewer, bringing the application version to v0.5.0 across the GUI.

## Success Criteria
- The old viewer shell (WowViewer.App) is marked defunct, preserving its code for reference but removing it from active build targets.
- The MdxViewer project and all internal namespaces/string references are renamed to WoWViewer.
- The GUI title bar reads WoWViewer v0.5.0.
- The solution builds flawlessly without reference errors.
