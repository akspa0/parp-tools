using System;
using System.Collections.Generic;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.PM4.Chunks;
using System.Numerics;

namespace ParpToolbox.Services.PM4
{
    /// <inheritdoc/>
    public sealed class Pm4Adapter : IPm4Loader
{
    public Pm4Scene Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("PM4 path must be provided", nameof(path));
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        // Simple chunk scan variables
        MspvChunk? mspv = null;
        MsvtChunk? msvt = null;
        MspiChunk? mspi = null;

        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            string sig = new string(br.ReadChars(4));
            uint size = br.ReadUInt32();
            long payloadStart = br.BaseStream.Position;
            byte[] data = br.ReadBytes((int)size);

            switch (sig)
            {
                case MspvChunk.Signature:
                    mspv ??= new MspvChunk();
                    mspv.LoadBinaryData(data);
                    break;
                case MsvtChunk.Signature:
                    msvt ??= new MsvtChunk();
                    msvt.LoadBinaryData(data);
                    break;
                case MspiChunk.Signature:
                    mspi ??= new MspiChunk();
                    mspi.LoadBinaryData(data);
                    break;
                default:
                    // skip unknown chunks for now
                    break;
            }

            // Ensure we really consumed expected bytes; seek to next aligned chunk.
            br.BaseStream.Position = payloadStart + size;
        }

        if (mspi == null || (msvt == null && mspv == null))
            throw new InvalidDataException("PM4 missing required chunks (MSPI + MSPV/MSVT)");

        IReadOnlyList<Vector3> verts = msvt?.Vertices.Count > 0 ? msvt.Vertices : mspv!.Vertices;
        var scene = new Pm4Scene
        {
            Vertices = verts,
            Triangles = mspi.Triangles
        };
        return scene;
    }
}
}
