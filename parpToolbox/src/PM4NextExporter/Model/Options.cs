using System;
using System.Collections.Generic;

namespace PM4NextExporter.Model
{
    internal enum ExportFormat { Obj, Gltf, Glb }
    internal enum AssemblyStrategy { ParentIndex, MsurIndexCount, SurfaceKey, CompositeHierarchy, ContainerHierarchy8Bit, CompositeBytePair, Parent16 }
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
        public AssemblyStrategy AssemblyStrategy { get; set; } = AssemblyStrategy.ParentIndex;
        public List<GroupKey> GroupKeys { get; } = new();
        public bool Parent16Swap { get; set; }

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
