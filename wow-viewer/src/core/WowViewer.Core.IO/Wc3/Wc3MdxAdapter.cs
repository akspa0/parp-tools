using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.IO.Wc3;

/// <summary>WC3-only framing adapter above the frozen shared MDX reader.</summary>
public static class Wc3MdxAdapter
{
    public static MdxFile Read(byte[] bytes)
    {
        if (bytes.Length < 16 || !bytes.AsSpan(0, 4).SequenceEqual("MDLX"u8))
            throw new InvalidDataException("Expected an MDLX model.");
        using MemoryStream output = new();
        using BinaryWriter writer = new(output, Encoding.ASCII, leaveOpen: true);
        writer.Write("MDLX"u8);
        uint version = 0;
        for (int pos = 4; pos < bytes.Length;)
        {
            if (bytes.Length - pos < 8) throw new InvalidDataException("Truncated WC3 MDX chunk header.");
            string tag = Encoding.ASCII.GetString(bytes, pos, 4);
            int size = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(pos + 4)));
            pos += 8;
            if (size > bytes.Length - pos) throw new InvalidDataException($"WC3 {tag} chunk exceeds file bounds.");
            ReadOnlySpan<byte> data = bytes.AsSpan(pos, size);
            if (tag == "VERS" && size >= 4) version = BinaryPrimitives.ReadUInt32LittleEndian(data);
            byte[]? adapted = null;
            if (version == 800 && tag == "MTLS") adapted = AdaptMaterials(data);
            if (version == 800 && tag == "LITE") adapted = AdaptLite(data);
            if (version == 800 && tag == "BONE") adapted = AdaptBones(data);
            if (version == 800 && tag == "HELP") adapted = AdaptHelpers(data);
            if (version == 800 && tag == "PRE2") adapted = AdaptPre2(data);
            if (version == 800 && tag == "RIBB") adapted = AdaptRibb(data);
            if (version == 800 && tag == "MODL" && size == 372)
            {
                // WC3 has animFile[260], then radius/min/max/blend; shared model
                // metadata expects name[80], min/max/blend. Geometry is unchanged.
                adapted = new byte[108];
                data[..80].CopyTo(adapted);
                data.Slice(344, 24).CopyTo(adapted.AsSpan(80));
                data.Slice(368, 4).CopyTo(adapted.AsSpan(104));
            }
            writer.Write(Encoding.ASCII.GetBytes(tag));
            writer.Write(adapted?.Length ?? size);
            if (adapted != null) writer.Write(adapted); else writer.Write(data);
            pos += size;
        }
        output.Position = 0;
        return MdxFile.Load(output);
    }

    private static byte[]? AdaptLite(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return [0, 0, 0, 0];
        uint firstDword = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (firstDword < 96 && data.Length >= 4) return null;

        uint count = 0;
        int pos = 0;
        while (pos < data.Length)
        {
            if (data.Length - pos < 4) break;
            uint lightSize = BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]);
            if (lightSize < 4 || lightSize > data.Length - pos) break;
            count++;
            pos += (int)lightSize;
        }

        byte[] result = new byte[4 + data.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(result, count);
        data.CopyTo(result.AsSpan(4));
        return result;
    }

    private static byte[]? AdaptBones(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return [0, 0, 0, 0];
        uint firstDword = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (firstDword < 96 && data.Length >= 4) return null;

        uint count = 0;
        int pos = 0;
        while (pos < data.Length)
        {
            if (data.Length - pos < 4) break;
            uint nodeSize = BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]);
            int boneSize = (int)nodeSize + 8;
            if (nodeSize < 4 || boneSize > data.Length - pos) break;
            count++;
            pos += boneSize;
        }

        byte[] result = new byte[4 + data.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(result, count);
        data.CopyTo(result.AsSpan(4));
        return result;
    }

    private static byte[]? AdaptHelpers(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return [0, 0, 0, 0];
        uint firstDword = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (firstDword < 96 && data.Length >= 4) return null;

        uint count = 0;
        int pos = 0;
        while (pos < data.Length)
        {
            if (data.Length - pos < 4) break;
            uint nodeSize = BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]);
            if (nodeSize < 4 || nodeSize > data.Length - pos) break;
            count++;
            pos += (int)nodeSize;
        }

        byte[] result = new byte[4 + data.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(result, count);
        data.CopyTo(result.AsSpan(4));
        return result;
    }

    private static byte[]? AdaptPre2(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return [0, 0, 0, 0];
        uint firstDword = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (firstDword < 96 && data.Length >= 4) return null;

        uint count = 0;
        int pos = 0;
        while (pos < data.Length)
        {
            if (data.Length - pos < 4) break;
            uint emitterSize = BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]);
            if (emitterSize < 4 || emitterSize > data.Length - pos) break;
            count++;
            pos += (int)emitterSize;
        }

        byte[] result = new byte[4 + data.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(result, count);
        data.CopyTo(result.AsSpan(4));
        return result;
    }

    private static byte[]? AdaptRibb(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return [0, 0, 0, 0];
        uint firstDword = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (firstDword < 96 && data.Length >= 4) return null;

        uint count = 0;
        int pos = 0;
        while (pos < data.Length)
        {
            if (data.Length - pos < 4) break;
            uint emitterSize = BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]);
            if (emitterSize < 4 || emitterSize > data.Length - pos) break;
            count++;
            pos += (int)emitterSize;
        }

        byte[] result = new byte[4 + data.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(result, count);
        data.CopyTo(result.AsSpan(4));
        return result;
    }

    private static byte[]? AdaptMaterials(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return null;
        // Already-counted shared framing is not this WC3 material variant.
        if (data.Length < 20 || !data.Slice(12, 4).SequenceEqual("LAYS"u8)) return null;
        List<byte[]> materials = [];
        for (int pos = 0; pos < data.Length;)
        {
            if (data.Length - pos < 20) throw new InvalidDataException("Truncated WC3 material record.");
            int size = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(data[pos..]));
            if (size < 20 || size > data.Length - pos || !data.Slice(pos + 12, 4).SequenceEqual("LAYS"u8))
                throw new InvalidDataException("Invalid WC3 material/LAYS record bounds.");
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(data[(pos + 16)..]);
            int layer = pos + 20, end = pos + size;
            for (uint i = 0; i < count; i++)
            {
                if (end - layer < 28) throw new InvalidDataException("Truncated WC3 material layer.");
                int length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(data[layer..]));
                if (length < 28 || length > end - layer) throw new InvalidDataException("Invalid WC3 layer bounds.");
                layer += length;
            }
            if (layer != end) throw new InvalidDataException("WC3 material has an unrecognized trailing layout.");
            byte[] material = new byte[size - 8];
            BinaryPrimitives.WriteInt32LittleEndian(material, material.Length);
            data.Slice(pos + 4, 4).CopyTo(material.AsSpan(4)); // Priority plane.
            BinaryPrimitives.WriteUInt32LittleEndian(material.AsSpan(8), count);
            data.Slice(pos + 20, size - 20).CopyTo(material.AsSpan(12));
            materials.Add(material);
            pos = end;
        }
        using MemoryStream result = new();
        using BinaryWriter writer = new(result);
        writer.Write(materials.Count); writer.Write(0);
        foreach (byte[] material in materials) writer.Write(material);
        return result.ToArray();
    }
}
