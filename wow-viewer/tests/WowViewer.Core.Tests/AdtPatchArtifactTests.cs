using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using ICSharpCode.SharpZipLib.BZip2;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtPatchArtifactTests
{
    [Fact]
    public void IsPatchArtifact_RejectsReconstructedAdt()
    {
        byte[] data = BuildAdt("REVM", 64);

        Assert.False(AdtPatchArtifactDecoder.IsPatchArtifact(data));
    }

    [Fact]
    public void IsPatchArtifact_AcceptsPtchMagic()
    {
        byte[] data = Encoding.ASCII.GetBytes("PTCH\0\0\0\0");

        Assert.True(AdtPatchArtifactDecoder.IsPatchArtifact(data));
    }

    [Fact]
    public void TryReconstruct_AppliesBsdiff_AndMatchesBaseByMd5()
    {
        byte[] baseData = BuildAdt("REVM", 64);
        byte[] expectedNew = (byte[])baseData.Clone();
        expectedNew[10] = (byte)(expectedNew[10] + 0x5A);
        expectedNew[11] = (byte)(expectedNew[11] + 0x01);
        expectedNew[12] = 0xFF;
        expectedNew[13] = 0x00;

        byte[] artifact = BuildPtchArtifact(baseData, expectedNew);

        byte[] wrongBase = BuildAdt("REVM", 64);
        wrongBase[0] ^= 0xFF;

        bool ok = AdtPatchArtifactDecoder.TryReconstruct(
            artifact,
            [wrongBase, baseData],
            out byte[]? reconstructed,
            out string failureReason);

        Assert.True(ok, failureReason);
        Assert.NotNull(reconstructed);
        Assert.Equal(expectedNew, reconstructed);
    }

    [Fact]
    public void TryReconstruct_NoMatchingBase_FailsWithMd5Reason()
    {
        byte[] baseData = BuildAdt("REVM", 64);
        byte[] expectedNew = (byte[])baseData.Clone();
        expectedNew[10] ^= 0xFF;

        byte[] artifact = BuildPtchArtifact(baseData, expectedNew);
        byte[] unrelated = BuildAdt("REVM", 32);

        bool ok = AdtPatchArtifactDecoder.TryReconstruct(
            artifact,
            [unrelated],
            out byte[]? reconstructed,
            out string failureReason);

        Assert.False(ok);
        Assert.Null(reconstructed);
        Assert.Contains("base MD5", failureReason);
    }

    [Fact]
    public void TryReconstruct_SkipsPatchArtifactsInCandidates()
    {
        byte[] baseData = BuildAdt("REVM", 64);
        byte[] expectedNew = (byte[])baseData.Clone();
        expectedNew[10] ^= 0xFF;

        byte[] artifact = BuildPtchArtifact(baseData, expectedNew);
        byte[] otherArtifact = BuildPtchArtifact(BuildAdt("REVM", 32), BuildAdt("REVM", 32));

        bool ok = AdtPatchArtifactDecoder.TryReconstruct(
            artifact,
            [otherArtifact, baseData],
            out byte[]? reconstructed,
            out string failureReason);

        Assert.True(ok, failureReason);
        Assert.Equal(expectedNew, reconstructed);
    }

    [Fact]
    public void TryApplyBsdiff_HandlesInsertAndSeek()
    {
        // old = "ABCDEF"
        // control: (add=2, insert=2, seek=-1) then (add=3, insert=0, seek=0)
        // new = "AB" + "XY" + old[1..4) adjusted = "ABXYDEF"
        byte[] oldData = Encoding.ASCII.GetBytes("ABCDEF");
        byte[] expectedNew = Encoding.ASCII.GetBytes("ABXYDEF");

        byte[] control = new byte[48];
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(0, 8), 2);   // add
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(8, 8), 2);   // insert
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(16, 8), -1); // seek
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(24, 8), 3);  // add
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(32, 8), 0);  // insert
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(40, 8), 0);  // seek

        // add run 1: new[0..2) = diff[0..2) + old[0..2) → want "AB" → diff bytes 0,0
        // insert run: literal "XY"
        // add run 2: new[4..7) = diff[4..7) + old[1..4) → want "DEF" → diff bytes 2,2,2
        byte[] diff = [(byte)0, (byte)0, (byte)'X', (byte)'Y', (byte)2, (byte)2, (byte)2];

        byte[] bsdiff = BuildBsdiff(control, diff, extra: []);

        bool ok = AdtPatchArtifactDecoder.TryApplyBsdiff(oldData, bsdiff, out byte[]? result, out string error);

        Assert.True(ok, error);
        Assert.Equal(expectedNew, result);
    }

    [Fact]
    public void TryApplyBsdiff_RejectsBadMagic()
    {
        bool ok = AdtPatchArtifactDecoder.TryApplyBsdiff([1, 2, 3], new byte[64], out byte[]? result, out string error);

        Assert.False(ok);
        Assert.Null(result);
        Assert.Contains("BSDIFF40", error);
    }

    private static byte[] BuildAdt(string fourCc, int length)
    {
        byte[] data = new byte[length];
        Encoding.ASCII.GetBytes(fourCc).CopyTo(data, 0);
        for (int i = 4; i < length; i++)
            data[i] = (byte)(i * 7);
        return data;
    }

    /// <summary>
    /// Build a PTCH artifact whose BSDIFF delta transforms baseData into newExpected using a
    /// single add run covering the whole file (diff bytes = new - old).
    /// </summary>
    private static byte[] BuildPtchArtifact(byte[] baseData, byte[] newExpected)
    {
        Assert.Equal(baseData.Length, newExpected.Length);

        byte[] control = new byte[24];
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(0, 8), newExpected.Length); // add
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(8, 8), 0);                   // insert
        BinaryPrimitives.WriteInt64BigEndian(control.AsSpan(16, 8), 0);                  // seek

        byte[] diff = new byte[newExpected.Length];
        for (int i = 0; i < newExpected.Length; i++)
            diff[i] = (byte)(newExpected[i] - baseData[i]);

        byte[] bsdiff = BuildBsdiff(control, diff, extra: []);

        using MemoryStream ms = new();
        ms.Write("PTCH"u8);
        WriteInt32(ms, 0); // header size field (parser scans for markers, value not load-bearing)
        ms.Write("MD5_"u8);
        ms.Write(MD5.HashData(baseData));
        ms.Write("BSD0"u8);
        WriteInt32(ms, bsdiff.Length);
        ms.Write(bsdiff);
        return ms.ToArray();
    }

    private static byte[] BuildBsdiff(byte[] control, byte[] diff, byte[] extra)
    {
        byte[] controlBz = Bzip2(control);
        byte[] diffBz = Bzip2(diff);
        byte[] extraBz = extra.Length > 0 ? Bzip2(extra) : [];

        using MemoryStream ms = new();
        ms.Write("BSDIFF40"u8);
        WriteInt64BigEndianAt(ms, controlBz.Length);
        WriteInt64BigEndianAt(ms, diffBz.Length);
        WriteInt64BigEndianAt(ms, diff.Length + extra.Length);
        ms.Write(controlBz);
        ms.Write(diffBz);
        ms.Write(extraBz);
        return ms.ToArray();
    }

    private static void WriteInt64BigEndianAt(MemoryStream ms, long value)
    {
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteInt64BigEndian(buffer, value);
        ms.Write(buffer);
    }

    private static byte[] Bzip2(byte[] input)
    {
        using MemoryStream inputMs = new(input, writable: false);
        using MemoryStream outputMs = new();
        using (BZip2OutputStream bzip = new(outputMs, 1))
        {
            bzip.Write(input);
        }

        return outputMs.ToArray();
    }

    private static void WriteInt32(MemoryStream ms, int value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(buffer, value);
        ms.Write(buffer);
    }
}
