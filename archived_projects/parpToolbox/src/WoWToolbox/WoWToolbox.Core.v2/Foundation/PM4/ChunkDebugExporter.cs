using System.Globalization;
using System.Text;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using WoWToolbox.Core.v2.Foundation.Transforms;

namespace WoWToolbox.Core.v2.Foundation.PM4;

public static class ChunkDebugExporter
{
    public static async Task ExportChunksAsync(PM4File pm4, string baseOutputPath, CancellationToken cancellationToken = default)
    {
        var outputDir = Path.GetDirectoryName(baseOutputPath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(baseOutputPath);
        
        // Ensure output directory exists
        Directory.CreateDirectory(outputDir);

        // Export MSPV vertices and faces if present
        if (pm4.MSPV != null && pm4.MSVI != null)
        {
            await ExportMeshChunkAsync(Path.Combine(outputDir, $"{baseName}_MSPV.obj"), 
                pm4.MSPV.Vertices, 
                pm4.MSVI.Indices, 
                v => CoordinateTransforms.FromMspvVertex(v));
        }

        // Export MSVT vertices if present
        if (pm4.MSVT != null)
        {
            await ExportVertexChunkAsync(Path.Combine(outputDir, $"{baseName}_MSVT.obj"),
                pm4.MSVT.Vertices,
                v => CoordinateTransforms.FromMsvtVertexSimple(v));
        }

        // Export MSCN collision vertices if present
        if (pm4.MSCN != null)
        {
            await ExportVertexChunkAsync(Path.Combine(outputDir, $"{baseName}_MSCN.obj"),
                pm4.MSCN.ExteriorVertices,
                v => CoordinateTransforms.FromMscnVertex(v));
        }

        // Export MPRL reference points if present
        if (pm4.MPRL != null && pm4.MPRL.Entries.Count > 0)
        {
            await ExportVertexChunkAsync(Path.Combine(outputDir, $"{baseName}_MPRL.obj"),
                pm4.MPRL.Entries,
                e => CoordinateTransforms.FromMprlEntry(e));
        }

        // Export MSLK link points if present
        if (pm4.MSLK != null && pm4.MSPI != null && pm4.MSPI.Indices.Count > 0)
        {
            await ExportMslkChunkAsync(
                Path.Combine(outputDir, $"{baseName}_MSLK.obj"),
                pm4.MSLK,
                pm4.MSPI,
                pm4.MSPV?.Vertices.Count ?? 0);
        }
    }

    private static async Task ExportMeshChunkAsync<TVertex>(
        string path,
        IReadOnlyList<TVertex> vertices,
        IReadOnlyList<uint> indices,
        Func<TVertex, Vector3> transform)
    {
        var sb = new StringBuilder();
        
        // Write header
        sb.AppendLine("# Exported chunk: " + Path.GetFileNameWithoutExtension(path));
        sb.AppendLine("# Vertices: " + vertices.Count);
        sb.AppendLine("# Faces: " + (indices.Count / 3));
        sb.AppendLine();

        // Write vertices
        foreach (var vertex in vertices)
        {
            var coords = transform(vertex);
            sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
        }

        // Write faces (assuming triangles)
        for (int i = 0; i < indices.Count; i += 3)
        {
            if (i + 2 >= indices.Count) break;
            sb.AppendLine($"f {indices[i] + 1} {indices[i + 1] + 1} {indices[i + 2] + 1}");
        }

        await File.WriteAllTextAsync(path, sb.ToString());
    }

    private static async Task ExportVertexChunkAsync<TVertex>(
        string path,
        IReadOnlyList<TVertex> vertices,
        Func<TVertex, Vector3> transform)
    {
        var sb = new StringBuilder();
        
        // Write header
        sb.AppendLine("# Exported chunk: " + Path.GetFileNameWithoutExtension(path));
        sb.AppendLine("# Points: " + vertices.Count);
        sb.AppendLine();

        // Write vertices as points
        foreach (var vertex in vertices)
        {
            var coords = transform(vertex);
            sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
        }

        // Write point indices
        for (int i = 0; i < vertices.Count; i++)
        {
            sb.AppendLine($"p {i + 1}");
        }

        await File.WriteAllTextAsync(path, sb.ToString());
    }

    private static async Task ExportMslkChunkAsync(
        string path,
        MSLK mslk,
        MSPIChunk mspi,
        int vertexCount)
    {
        var sb = new StringBuilder();
        
        // Write header
        sb.AppendLine("# Exported chunk: " + Path.GetFileNameWithoutExtension(path));
        sb.AppendLine("# MSLK Entries: " + mslk.Entries.Count);
        sb.AppendLine("# MSPI Indices: " + mspi.Indices.Count);
        sb.AppendLine();

        // Write each MSLK entry's points
        for (int entryIdx = 0; entryIdx < mslk.Entries.Count; entryIdx++)
        {
            var entry = mslk.Entries[entryIdx];
            if (entry.MspiIndexCount <= 0 || entry.MspiFirstIndex < 0) continue;

            int start = entry.MspiFirstIndex;
            int end = Math.Min(start + entry.MspiIndexCount, mspi.Indices.Count);
            
            // Write group header
            sb.AppendLine($"g MSLK_{entryIdx:D4}");
            
            // Write points in this entry
            for (int i = start; i < end; i++)
            {
                uint idx = mspi.Indices[i];
                if (idx < vertexCount)
                {
                    sb.AppendLine($"p {idx + 1}");
                }
            }
            
            sb.AppendLine();
        }

        await File.WriteAllTextAsync(path, sb.ToString());
    }
}
