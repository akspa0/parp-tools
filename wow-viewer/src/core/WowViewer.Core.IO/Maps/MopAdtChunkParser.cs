using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Parsers for late-Cataclysm (4.3.4) and Mists of Pandaria (5.0.1–5.1.0) ADT chunks.
/// </summary>
/// <remarks>
/// <para>
/// Every chunk here is one the 5.0.1.15464 client actually dispatches on. Two dispatchers
/// were read to establish that: <c>FUN_00bb6f10</c> (<c>MapAdtFileData.cpp</c>), which retains
/// only <c>MCNK</c>, <c>MTEX</c>, <c>MTXF</c> and <c>MTXP</c>; and <c>FUN_00bb0b50</c>
/// (<c>MapArea.cpp</c>), the root-ADT dispatcher, which adds <c>MHDR</c>, <c>MAMP</c>,
/// <c>MDDF</c>, <c>MODF</c>, <c>MMDX</c>, <c>MMID</c>, <c>MWMO</c>, <c>MWID</c>, <c>MFBO</c>,
/// <c>MH2O</c> and the blend-mesh trio <c>MBMH</c>/<c>MBMI</c>/<c>MBMV</c>.
/// </para>
/// <para>
/// <b>MDID, MHID and MCXH are not in either dispatcher and are not parsed here.</b> Earlier
/// revisions of this file carried parsers for all three plus a tile/chunk model that no code
/// ever produced. MDID/MHID are a later-expansion FileDataID scheme outside this project's
/// 0.5.3–5.1 range, and MCXH does not correspond to anything the client reads. The
/// height-blend parameters that MCXH was invented to carry are in <c>MTXP</c>.
/// </para>
/// <para>
/// Record sizes below are the client's own divisors, not conventions: MDDF <c>size / 0x24</c>
/// and MODF <c>size &gt;&gt; 6</c> as read at <c>FUN_00bb0b50</c>.
/// </para>
/// </remarks>
public static class MopAdtChunkParser
{
    /// <summary>MDDF record size as divided by the client (<c>size / 0x24</c>).</summary>
    public const int MddfRecordSize = 36;

    /// <summary>MODF record size as divided by the client (<c>size &gt;&gt; 6</c>).</summary>
    public const int ModfRecordSize = 64;

    /// <summary>
    /// Parses MTEX, a block of null-terminated texture paths.
    /// </summary>
    public static List<string> ParseMtexChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                    result.Add(Encoding.ASCII.GetString(data.Slice(start, i - start)));

                start = i + 1;
            }
        }
        return result;
    }

    /// <summary>
    /// Parses MTXP, the per-texture parameter block the client stores alongside MTEX and MTXF
    /// (file-data object <c>+0x430</c>, wired into the map area at <c>+0x98</c> by
    /// <c>FUN_00bb0b50</c>).
    /// </summary>
    /// <remarks>
    /// The record layout is deliberately <b>not</b> asserted. The client's consumer of
    /// <c>+0x98</c> has not been isolated, so the stride is unknown, and guessing one is how
    /// MCXH came to exist. <see cref="AdtTextureParameters.StrideBytes"/> derives it from real
    /// data instead — MTXP is parallel to MTEX, so payload length divided by texture count is
    /// a measurement, and a non-zero result across a corpus is what would settle the layout.
    /// </remarks>
    public static AdtTextureParameters ParseMtxpChunk(ReadOnlySpan<byte> data, int textureCount)
        => new(data.ToArray(), textureCount);

    /// <summary>
    /// Reads MODF (WMO placements).
    /// </summary>
    public static List<MopObjectPlacement> ParseModfChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<MopObjectPlacement>();
        int count = data.Length / ModfRecordSize;

        for (int i = 0; i < count; i++)
        {
            var span = data.Slice(i * ModfRecordSize, ModfRecordSize);
            result.Add(new MopObjectPlacement
            {
                IsWmo = true,
                NameId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(0, 4)),
                UniqueId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(4, 4)),
                Position = new Vector3(
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(8, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(12, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(16, 4))),
                Rotation = new Vector3(
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(20, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(24, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(28, 4))),
                Scale = 1.0f,
                Flags = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(56, 4)),
            });
        }
        return result;
    }

    /// <summary>
    /// Reads MDDF (M2 doodad placements).
    /// </summary>
    public static List<MopObjectPlacement> ParseMddfChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<MopObjectPlacement>();
        int count = data.Length / MddfRecordSize;

        for (int i = 0; i < count; i++)
        {
            var span = data.Slice(i * MddfRecordSize, MddfRecordSize);
            ushort scaleInt = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(32, 2));
            float scale = scaleInt / 1024.0f;
            if (scale <= 0.0001f)
                scale = 1.0f;

            result.Add(new MopObjectPlacement
            {
                IsWmo = false,
                NameId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(0, 4)),
                UniqueId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(4, 4)),
                Position = new Vector3(
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(8, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(12, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(16, 4))),
                Rotation = new Vector3(
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(20, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(24, 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(span.Slice(28, 4))),
                Scale = scale,
                Flags = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(34, 2)),
            });
        }
        return result;
    }
}

/// <summary>
/// The raw MTXP payload plus the texture count it runs parallel to, so the record stride can
/// be measured rather than assumed.
/// </summary>
public sealed record AdtTextureParameters(byte[] Payload, int TextureCount)
{
    /// <summary>
    /// Bytes per texture, or 0 when the payload does not divide evenly by the texture count —
    /// which would mean MTXP is not parallel to MTEX and the assumption needs revisiting.
    /// </summary>
    public int StrideBytes => TextureCount > 0 && Payload.Length % TextureCount == 0
        ? Payload.Length / TextureCount
        : 0;
}

/// <summary>
/// A doodad or WMO placement read from MDDF/MODF.
/// </summary>
public sealed class MopObjectPlacement
{
    public bool IsWmo { get; set; }
    public uint NameId { get; set; }
    public uint UniqueId { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; } = 1.0f;
    public uint Flags { get; set; }
}
