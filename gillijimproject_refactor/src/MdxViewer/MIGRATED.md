# MdxViewer — Migrated

**This directory is a legacy stub.** MdxViewer has been migrated to:

```
wow-viewer/src/viewer/WoWViewer/
```

The MDX format types (`MdxFile`, `MdlModel`, `MdlBone`, `MdxHeaders`, etc.) have been
ported into the shared I/O library:

```
wow-viewer/src/core/WowViewer.Core.IO/Mdx/
```

The original `MdxLTool.Formats.Mdx` namespace has been replaced with
`WowViewer.Core.IO.Mdx` across all WoWViewer source files.

See `wow-viewer/specs/033-mdxviewer-migration/` for the full migration plan.
