using System;
using System.Collections.Generic;

namespace PM4NextExporter.Model
{
    internal enum ExportFormat { Obj, Gltf, Glb }
    internal enum AssemblyStrategy { ParentIndex, MsurIndexCount, SurfaceKey, SurfaceKeyAA, CompositeHierarchy, ContainerHierarchy8Bit, CompositeBytePair, Parent16, MslkParent, MslkInstance, MslkInstanceCk24 }
    internal enum GroupKey { Parent16, Parent16Container, Parent16Object, Surface, Flags, Type, SortKey, Tile }

    internal sealed class Options
    {
        // Positional
        public string? InputPath { get; set; }
        public List<string> AdditionalInputs { get; } = new();

        // Output
        public string? OutDirBaseName { get; set; }
        public bool LegacyObjParity { get; set; }
        public ExportFormat Format { get; set; } = ExportFormat.Obj;

        // Assembly & grouping
        public AssemblyStrategy AssemblyStrategy { get; set; } = AssemblyStrategy.CompositeHierarchy;
        public List<GroupKey> GroupKeys { get; } = new();
        public bool Parent16Swap { get; set; }
        // Optional: further split composite-hierarchy groups by dominant MSLK type
        public bool CkSplitByType { get; set; }
        // Optional: export MSCN vertices as separate per-tile OBJ layers
        public bool ExportMscnObj { get; set; }
        // Optional: append pm4 tile coords (xx/yy) to object names
        public bool NameObjectsWithTile { get; set; }
        // Optional: export one OBJ per PM4 tile into a subfolder
        public bool ExportTiles { get; set; }
        // Optional: project output to a local coordinate origin at export time
        public bool ProjectLocal { get; set; }

        // Assembly tuning (MSLK parent)
        // Minimum triangle count required to emit an object when using --assembly mslk-parent
        public int MslkParentMinTriangles { get; set; } = 300;
        // Allow cross-tile fallback scan when MSLK tile coords are invalid (may over-match); default off
        public bool MslkParentAllowFallbackScan { get; set; }

        // Cross-tile / audit
        public bool IncludeAdjacent { get; set; }
        public bool AuditOnly { get; set; }
        public bool NoRemap { get; set; }

        // Diagnostics
        public bool CsvDiagnostics { get; set; }
        public string? CsvOut { get; set; }
        public List<Tuple<string,string>> Correlates { get; } = new();

        // Batch
        public bool Batch { get; set; }
    }
}
