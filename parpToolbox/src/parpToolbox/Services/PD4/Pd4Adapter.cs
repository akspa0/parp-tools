namespace ParpToolbox.Services.PD4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4; // Re-use Pm4Scene model for now (geometry is identical)
using ParpToolbox.Services.PM4;

/// <summary>
/// Minimal PD4 loader that reuses the shared P4 chunk definitions. PD4 has the same core geometry
/// chunks as PM4, so for a first pass we simply parse the same ones. Format-specific chunks can be
/// added later.
/// </summary>
public sealed class Pd4Adapter : IPm4Loader // temporary reuse of interface & scene model
{
    public Pm4Scene Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("PD4 path must be provided", nameof(path));

        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        MspvChunk? mspv = null;
        MsvtChunk? msvt = null;
        MspiChunk? mspi = null;

        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            string sig = FourCc.Read(br);
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
                    int vertCount = msvt?.Vertices.Count ?? mspv?.Vertices.Count ?? 0;
                    mspi.LoadBinaryData(data, vertCount);
                    break;
            }

            br.BaseStream.Position = payloadStart + size;
        }

        if (mspi == null || (msvt == null && mspv == null))
            throw new InvalidDataException("PD4 missing required chunks (MSPI + MSPV/MSVT)");

        IReadOnlyList<Vector3> verts = msvt?.Vertices.Count > 0 ? msvt.Vertices : mspv!.Vertices;
        var scene = new Pm4Scene
        {
            Vertices = verts,
            Triangles = mspi.Triangles
        };
        return scene;
    }
}
