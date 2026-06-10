using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using CsvHelper;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Services.Export;
using WoWToolbox.Core.v2.Models.PM4.Mslk;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using MsvtVertex = WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsvtVertex;
using MSLKEntry = WoWToolbox.Core.v2.Foundation.PM4.Chunks.MSLKEntry;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class MslkExporter : IMslkExporter
    {
        private readonly ICoordinateService _coordinateService;
        private readonly IJsonExporter _jsonExporter;
        private readonly ICsvExporter _csvExporter;

        public MslkExporter(ICoordinateService coordinateService, IJsonExporter jsonExporter, ICsvExporter csvExporter)
        {
            _coordinateService = coordinateService;
            _jsonExporter = jsonExporter;
            _csvExporter = csvExporter;
        }

        public void Export(PM4File pm4File, string inputFilePath, MslkExportPaths paths, ISet<int> uniqueBuildingIds)
        {
            // Validate paths (prevent nullable warnings)
            if (paths.GeometryObjPath is null || paths.NodesObjPath is null || paths.HierarchyJsonPath is null ||
                paths.DoodadCsvPath is null || paths.SkippedLogPath is null)
            {
                throw new ArgumentNullException(nameof(paths), "All output paths must be provided.");
            }

            if (pm4File.MSLK == null || !pm4File.MSLK.Entries.Any())
            {
                File.WriteAllText(paths.SkippedLogPath, $"# No MSLK data in {Path.GetFileName(inputFilePath)}\n");
                return;
            }

            using var geometryWriter = new StreamWriter(paths.GeometryObjPath);
            using var nodesWriter = new StreamWriter(paths.NodesObjPath);
            using var skippedWriter = new StreamWriter(paths.SkippedLogPath);

            WriteHeaders(geometryWriter, nodesWriter, skippedWriter, inputFilePath);

            var mslkHierarchy = new Dictionary<uint, MslkGroupDto>();
            var doodadData = new List<DoodadDataRecord>();

            var rawMspvVertices = pm4File.MSPV?.Vertices?.ToList() ?? new List<Warcraft.NET.Files.Structures.C3Vector>();
            var transformedMspvVertices = rawMspvVertices.Select(v => _coordinateService.FromMspvVertex(v)).ToList();
            WriteVertices(geometryWriter, transformedMspvVertices, $"MSLK_Geometry_{Path.GetFileNameWithoutExtension(inputFilePath)}");

            var msvtVertices = pm4File.MSVT?.Vertices?.ToList() ?? new List<WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsvtVertex>();
            var msvtVertexCount = msvtVertices.Count;

            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                var entry = pm4File.MSLK.Entries[i];
                var groupId = entry.Unknown_0x04;
                if (!mslkHierarchy.ContainsKey(groupId))
                    mslkHierarchy[groupId] = new MslkGroupDto();

                if (entry.IsGeometryNode)
                    ProcessGeometryNode(entry, i, pm4File, geometryWriter, skippedWriter, mslkHierarchy[groupId]);
                else
                    ProcessAnchorNode(entry, i, pm4File, nodesWriter, skippedWriter, msvtVertices, mslkHierarchy[groupId]);
                
                ProcessDoodadData(entry, pm4File, doodadData, uniqueBuildingIds);
            }

            _jsonExporter.Export(mslkHierarchy, paths.HierarchyJsonPath);
            _csvExporter.Export(doodadData, paths.DoodadCsvPath);
        }

        private void WriteHeaders(StreamWriter geometryWriter, StreamWriter nodesWriter, StreamWriter skippedWriter, string inputFilePath)
        {
            var fileName = Path.GetFileName(inputFilePath);
            geometryWriter.WriteLine($"# PM4 MSLK Geometry (Generated: {DateTime.Now}) - File: {fileName}");
            geometryWriter.WriteLine("# Vertices (v) Transform: X, Y, Z (Standard)");
            nodesWriter.WriteLine($"# PM4 MSLK Node Anchor Points (Generated: {DateTime.Now}) - File: {fileName}");
            nodesWriter.WriteLine("# Vertices (v) Transform: Y, X, Z");
            skippedWriter.WriteLine($"# PM4 Skipped/Invalid MSLK Entries Log (Generated: {DateTime.Now}) - File: {fileName}");
        }

        private void WriteVertices(StreamWriter writer, List<Vector3> vertices, string objectName)
        {
            writer.WriteLine($"o {objectName}");
            foreach (var vertex in vertices)
            {
                writer.WriteLine(FormattableString.Invariant($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"));
            }
        }

        private void ProcessGeometryNode(MSLKEntry entry, int index, PM4File pm4File, StreamWriter writer, StreamWriter skippedWriter, MslkGroupDto groupDto)
        {
            groupDto.Geometry.Add(new MslkGeometryEntryDto { EntryIndex = index, MspiFirstIndex = entry.MspiFirstIndex, MspiIndexCount = entry.MspiIndexCount, Unk00 = entry.Unknown_0x00, Unk01 = entry.Unknown_0x01, Unk02 = entry.Unknown_0x02, Unk0C = entry.Unknown_0x0C, Unk10 = entry.Unknown_0x10, Unk12 = entry.Unknown_0x12 });
            var mspiIndices = pm4File.MSPI?.Indices?.ToList() ?? new List<uint>();
            if (entry.MspiFirstIndex < mspiIndices.Count && (entry.MspiFirstIndex + entry.MspiIndexCount) <= mspiIndices.Count)
            {
                for (int j = 0; j < entry.MspiIndexCount - 1; j++)
                {
                    var idx1 = (int)mspiIndices[entry.MspiFirstIndex + j] + 1; // OBJ is 1-based
                    var idx2 = (int)mspiIndices[entry.MspiFirstIndex + j + 1] + 1;
                    writer.WriteLine($"l {idx1} {idx2}");
                }
            }
            else
            {
                skippedWriter.WriteLine($"Invalid MSPI index range for MSLK geometry entry {index}.");
            }
        }

        private void ProcessAnchorNode(MSLKEntry entry, int index, PM4File pm4File, StreamWriter writer, StreamWriter skippedWriter, List<MsvtVertex> msvtVertices, MslkGroupDto groupDto)
        {
            groupDto.Nodes.Add(new MslkNodeEntryDto { EntryIndex = index, Unk00 = entry.Unknown_0x00, Unk01 = entry.Unknown_0x01, Unk02 = entry.Unknown_0x02, Unk0C = entry.Unknown_0x0C, Unk10 = entry.Unknown_0x10, Unk12 = entry.Unknown_0x12 });
            var msviIndices = pm4File.MSVI?.Indices ?? new List<uint>();
            if (entry.Unk10 < msviIndices.Count)
            {
                var msvtIndex = msviIndices[(int)entry.Unk10];
                if (msvtIndex < msvtVertices.Count)
                {
                    var vertex = msvtVertices[(int)msvtIndex];
                    var transformedVertex = _coordinateService.FromMsvtVertexSimple(vertex);
                    writer.WriteLine(FormattableString.Invariant($"v {transformedVertex.X:F6} {transformedVertex.Y:F6} {transformedVertex.Z:F6} # Node from MSLK[{index}] -> MSVI[{entry.Unk10}] -> MSVT[{msvtIndex}]"));
                }
                else
                {
                    skippedWriter.WriteLine($"Invalid MSVT index {msvtIndex} for MSLK anchor node {index}.");
                }
            }
            else
            {
                skippedWriter.WriteLine($"Invalid MSVI index {entry.Unk10} for MSLK anchor node {index}.");
            }
        }

        private void ProcessDoodadData(MSLKEntry entry, PM4File pm4File, List<DoodadDataRecord> doodadData, ISet<int> uniqueBuildingIds)
        {
            if (pm4File.MDBH != null && entry.Unk12 < pm4File.MDBH.Entries.Count)
            {
                var mdbhEntry = pm4File.MDBH.Entries[entry.Unk12];
                if (pm4File.MDOS != null && mdbhEntry.MdosIndex < pm4File.MDOS.Entries.Count)
                {
                    var mdosEntry = pm4File.MDOS.Entries[(int)mdbhEntry.MdosIndex];
                    uniqueBuildingIds.Add((int)mdosEntry.NameId); 
                    doodadData.Add(new DoodadDataRecord
                    {
                        BuildingId = mdosEntry.NameId,
                        DoodadSet = mdosEntry.DoodadSet,
                        NameId = mdosEntry.NameId,
                        Flags = (ushort)mdosEntry.Flags,
                        Scale = mdosEntry.Scale
                    });
                }
            }
        }
    }
}
