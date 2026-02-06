using System.Globalization;

namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDL text format writer.
/// Port of ModelDataToText (0x0078b590) and CallTextWriteHandlers.
/// Outputs human-readable Warcraft III MDL format.
/// </summary>
public class MdlWriter
{
    private int _indentLevel = 0;

    public void Write(MdxFile mdx, StreamWriter sw)
    {
        // Force invariant culture for consistent float formatting
        var culture = CultureInfo.InvariantCulture;

        WriteHeader(sw, mdx, culture);
        WriteModel(sw, mdx, culture);
        WriteSequences(sw, mdx, culture);
        WriteGlobalSequences(sw, mdx, culture);
        WriteTextures(sw, mdx, culture);
        WriteMaterials(sw, mdx, culture);
        WriteGeosets(sw, mdx, culture);
        WriteBones(sw, mdx, culture);
        WritePivotPoints(sw, mdx, culture);
    }

    void WriteHeader(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        sw.WriteLine($"// MDL exported by MDX-L_Tool");
        sw.WriteLine($"// Source version: {mdx.Version}");
        sw.WriteLine();
    }

    void WriteModel(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        sw.WriteLine($"Model \"{mdx.Model.Name}\" {{");
        _indentLevel++;

        if (mdx.Sequences.Count > 0)
            WriteIndented(sw, $"NumGeosets {mdx.Geosets.Count},");

        if (mdx.Bones.Count > 0)
            WriteIndented(sw, $"NumBones {mdx.Bones.Count},");

        if (mdx.Attachments.Count > 0)
            WriteIndented(sw, $"NumAttachments {mdx.Attachments.Count},");

        WriteIndented(sw, $"BlendTime {mdx.Model.BlendTime},");
        WriteBounds(sw, "MinimumExtent", mdx.Model.Bounds.Extent.Min, fmt);
        WriteBounds(sw, "MaximumExtent", mdx.Model.Bounds.Extent.Max, fmt);
        WriteIndented(sw, $"BoundsRadius {mdx.Model.Bounds.Radius.ToString(fmt)},");

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteSequences(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.Sequences.Count == 0) return;

        sw.WriteLine($"Sequences {mdx.Sequences.Count} {{");
        _indentLevel++;

        foreach (var seq in mdx.Sequences)
        {
            WriteIndented(sw, $"Anim \"{seq.Name}\" {{");
            _indentLevel++;

            WriteIndented(sw, $"Interval {{ {seq.Time.Start}, {seq.Time.End} }},");

            if (seq.MoveSpeed != 0)
                WriteIndented(sw, $"MoveSpeed {seq.MoveSpeed.ToString(fmt)},");

            if ((seq.Flags & 1) != 0)
                WriteIndented(sw, "NonLooping,");

            WriteBounds(sw, "MinimumExtent", seq.Bounds.Extent.Min, fmt);
            WriteBounds(sw, "MaximumExtent", seq.Bounds.Extent.Max, fmt);
            WriteIndented(sw, $"BoundsRadius {seq.Bounds.Radius.ToString(fmt)},");

            if (seq.Frequency != 1.0f)
                WriteIndented(sw, $"Rarity {seq.Frequency.ToString(fmt)},");

            _indentLevel--;
            WriteIndented(sw, "}");
        }

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteGlobalSequences(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.GlobalSequences.Count == 0) return;

        sw.WriteLine($"GlobalSequences {mdx.GlobalSequences.Count} {{");
        _indentLevel++;

        foreach (var gs in mdx.GlobalSequences)
            WriteIndented(sw, $"Duration {gs},");

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteTextures(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.Textures.Count == 0) return;

        sw.WriteLine($"Textures {mdx.Textures.Count} {{");
        _indentLevel++;

        foreach (var tex in mdx.Textures)
        {
            WriteIndented(sw, "Bitmap {");
            _indentLevel++;

            WriteIndented(sw, $"Image \"{tex.Path}\",");

            if (tex.ReplaceableId != 0)
                WriteIndented(sw, $"ReplaceableId {tex.ReplaceableId},");

            if ((tex.Flags & 1) != 0)
                WriteIndented(sw, "WrapWidth,");
            if ((tex.Flags & 2) != 0)
                WriteIndented(sw, "WrapHeight,");

            _indentLevel--;
            WriteIndented(sw, "}");
        }

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteMaterials(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.Materials.Count == 0) return;

        sw.WriteLine($"Materials {mdx.Materials.Count} {{");
        _indentLevel++;

        foreach (var mat in mdx.Materials)
        {
            WriteIndented(sw, "Material {");
            _indentLevel++;

            if (mat.PriorityPlane != 0)
                WriteIndented(sw, $"PriorityPlane {mat.PriorityPlane},");

            foreach (var layer in mat.Layers)
            {
                WriteIndented(sw, "Layer {");
                _indentLevel++;

                WriteIndented(sw, $"FilterMode {GetBlendModeName(layer.BlendMode)},");

                if (layer.TextureId >= 0)
                    WriteIndented(sw, $"static TextureID {layer.TextureId},");

                if (layer.StaticAlpha < 1.0f)
                    WriteIndented(sw, $"static Alpha {layer.StaticAlpha.ToString(fmt)},");

                if (layer.Flags.HasFlag(MdlGeoFlags.Unshaded))
                    WriteIndented(sw, "Unshaded,");
                if (layer.Flags.HasFlag(MdlGeoFlags.TwoSided))
                    WriteIndented(sw, "TwoSided,");
                if (layer.Flags.HasFlag(MdlGeoFlags.Unfogged))
                    WriteIndented(sw, "Unfogged,");
                if (layer.Flags.HasFlag(MdlGeoFlags.NoDepthTest))
                    WriteIndented(sw, "NoDepthTest,");
                if (layer.Flags.HasFlag(MdlGeoFlags.NoDepthSet))
                    WriteIndented(sw, "NoDepthSet,");

                _indentLevel--;
                WriteIndented(sw, "}");
            }

            _indentLevel--;
            WriteIndented(sw, "}");
        }

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteGeosets(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.Geosets.Count == 0) return;

        foreach (var geo in mdx.Geosets)
        {
            sw.WriteLine("Geoset {");
            _indentLevel++;

            // Vertices
            WriteIndented(sw, $"Vertices {geo.Vertices.Count} {{");
            _indentLevel++;
            foreach (var v in geo.Vertices)
                WriteIndented(sw, $"{{ {v.X.ToString(fmt)}, {v.Y.ToString(fmt)}, {v.Z.ToString(fmt)} }},");
            _indentLevel--;
            WriteIndented(sw, "}");

            // Normals
            WriteIndented(sw, $"Normals {geo.Normals.Count} {{");
            _indentLevel++;
            foreach (var n in geo.Normals)
                WriteIndented(sw, $"{{ {n.X.ToString(fmt)}, {n.Y.ToString(fmt)}, {n.Z.ToString(fmt)} }},");
            _indentLevel--;
            WriteIndented(sw, "}");

            // Texture coordinates
            WriteIndented(sw, $"TVertices {geo.TexCoords.Count} {{");
            _indentLevel++;
            foreach (var uv in geo.TexCoords)
                WriteIndented(sw, $"{{ {uv.U.ToString(fmt)}, {uv.V.ToString(fmt)} }},");
            _indentLevel--;
            WriteIndented(sw, "}");

            // Vertex groups
            WriteIndented(sw, $"VertexGroup {{");
            _indentLevel++;
            for (int i = 0; i < geo.VertexGroups.Count; i += 16)
            {
                var line = string.Join(", ", geo.VertexGroups.Skip(i).Take(16));
                WriteIndented(sw, $"{line},");
            }
            _indentLevel--;
            WriteIndented(sw, "}");

            // Faces (triangles)
            WriteIndented(sw, $"Faces 1 {geo.Indices.Count} {{");
            _indentLevel++;
            WriteIndented(sw, "Triangles {");
            _indentLevel++;
            WriteIndented(sw, $"{{ {string.Join(", ", geo.Indices)} }},");
            _indentLevel--;
            WriteIndented(sw, "}");
            _indentLevel--;
            WriteIndented(sw, "}");

            // Groups
            WriteIndented(sw, $"Groups {geo.MatrixGroups.Count} {geo.MatrixIndices.Count} {{");
            _indentLevel++;
            int matIdx = 0;
            foreach (var count in geo.MatrixGroups)
            {
                var mats = geo.MatrixIndices.Skip(matIdx).Take((int)count);
                WriteIndented(sw, $"Matrices {{ {string.Join(", ", mats)} }},");
                matIdx += (int)count;
            }
            _indentLevel--;
            WriteIndented(sw, "}");

            // Bounds
            WriteBounds(sw, "MinimumExtent", geo.Bounds.Extent.Min, fmt);
            WriteBounds(sw, "MaximumExtent", geo.Bounds.Extent.Max, fmt);
            WriteIndented(sw, $"BoundsRadius {geo.Bounds.Radius.ToString(fmt)},");

            WriteIndented(sw, $"MaterialID {geo.MaterialId},");
            WriteIndented(sw, $"SelectionGroup {geo.SelectionGroup},");

            if ((geo.Flags & 4) != 0)
                WriteIndented(sw, "Unselectable,");

            _indentLevel--;
            sw.WriteLine("}");
            sw.WriteLine();
        }
    }

    void WriteBones(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.Bones.Count == 0) return;

        foreach (var bone in mdx.Bones)
        {
            sw.WriteLine($"Bone \"{bone.Name}\" {{");
            _indentLevel++;

            WriteIndented(sw, $"ObjectId {bone.ObjectId},");

            if (bone.ParentId >= 0)
                WriteIndented(sw, $"Parent {bone.ParentId},");

            if (bone.GeosetId >= 0)
                WriteIndented(sw, $"GeosetId {bone.GeosetId},");
            else
                WriteIndented(sw, "GeosetId Multiple,");

            if (bone.GeosetAnimId >= 0)
                WriteIndented(sw, $"GeosetAnimId {bone.GeosetAnimId},");
            else
                WriteIndented(sw, "GeosetAnimId None,");

            _indentLevel--;
            sw.WriteLine("}");
        }
        sw.WriteLine();
    }

    void WritePivotPoints(StreamWriter sw, MdxFile mdx, IFormatProvider fmt)
    {
        if (mdx.PivotPoints.Count == 0) return;

        sw.WriteLine($"PivotPoints {mdx.PivotPoints.Count} {{");
        _indentLevel++;

        foreach (var p in mdx.PivotPoints)
            WriteIndented(sw, $"{{ {p.X.ToString(fmt)}, {p.Y.ToString(fmt)}, {p.Z.ToString(fmt)} }},");

        _indentLevel--;
        sw.WriteLine("}");
        sw.WriteLine();
    }

    void WriteIndented(StreamWriter sw, string text)
    {
        sw.Write(new string('\t', _indentLevel));
        sw.WriteLine(text);
    }

    void WriteBounds(StreamWriter sw, string name, C3Vector v, IFormatProvider fmt)
    {
        WriteIndented(sw, $"{name} {{ {v.X.ToString(fmt)}, {v.Y.ToString(fmt)}, {v.Z.ToString(fmt)} }},");
    }

    static string GetBlendModeName(MdlTexOp mode) => mode switch
    {
        MdlTexOp.Load => "None",
        MdlTexOp.Transparent => "Transparent",
        MdlTexOp.Blend => "Blend",
        MdlTexOp.Add => "Additive",
        MdlTexOp.AddAlpha => "AddAlpha",
        MdlTexOp.Modulate => "Modulate",
        MdlTexOp.Modulate2X => "Modulate2x",
        _ => "None"
    };
}
