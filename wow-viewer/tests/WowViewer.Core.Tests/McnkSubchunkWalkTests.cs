using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Lk;

namespace WowViewer.Core.Tests;

/// <summary>
/// The MCNK sub-chunk walk must consume the payload exactly — the invariant the client
/// asserts at MapChunk.cpp:0x461 ("dataSize == 0"). Two eras satisfy it under different
/// advance rules, and picking the wrong one silently drops every sub-chunk after the
/// mismatch (MCLY and MCAL sit after MCNR, so the tile renders untextured).
/// </summary>
public sealed class McnkSubchunkWalkTests
{
    private static readonly Mcnk.ParseOptions Headerless = new() { SkipHeader = true };

    [Fact]
    public void Scan_ModernMcnrDeclaringItsRealSize_StillFindsLayersAfterIt()
    {
        // 5.0.1 shape: every declared size is exact, so the walk must advance by the
        // declared size alone (MapChunk.cpp FUN_00ba3050). A walker that always consumes
        // 0x1C0 bytes for MCNR jumps clean over MCLY and MCAL and reports no layers.
        byte[] payload =
        [
            .. Subchunk("MCNR", new byte[16]),
            .. Subchunk("MCLY", Mcly(layerCount: 2)),
            .. Subchunk("MCAL", new byte[32]),
        ];

        Mcnk mcnk = new(payload, Headerless);

        Assert.Equal(Mcnk.SubchunkAdvanceRule.DeclaredSize, mcnk.AdvanceRuleUsed);
        Assert.True(mcnk.SubchunkWalkConsumedExactly);
        Assert.Equal(2, mcnk.MclyRecordCount);
        Assert.NotNull(mcnk.MclyRawData);
        Assert.NotNull(mcnk.McalRawData);
        Assert.Equal(32, mcnk.McalRawData!.Length);
    }

    [Fact]
    public void Scan_LegacyMcnrWithTrailingPadding_StillParses()
    {
        // 3.3.5 shape: MCNR declares 435 bytes but occupies 448. Only the legacy
        // over-consuming rule lands the walk on the next real token, so this era must keep
        // selecting it.
        byte[] mcnrPayload = new byte[435];
        byte[] payload =
        [
            .. Subchunk("MCNR", mcnrPayload),
            .. new byte[13], // the 13 bytes the declared size excludes
            .. Subchunk("MCLY", Mcly(layerCount: 3)),
        ];

        Mcnk mcnk = new(payload, Headerless);

        Assert.Equal(Mcnk.SubchunkAdvanceRule.LegacyOverConsume, mcnk.AdvanceRuleUsed);
        Assert.True(mcnk.SubchunkWalkConsumedExactly);
        Assert.Equal(3, mcnk.MclyRecordCount);
    }

    [Fact]
    public void Scan_NativeOnlySubchunks_AreCapturedWithClientDerivedCounts()
    {
        // Tokens the 5.0.1 dispatcher handles that this parser previously ignored.
        byte[] payload =
        [
            .. Subchunk("MCRF", new byte[8]),
            .. Subchunk("MCLV", new byte[16]),
            .. Subchunk("MCMT", UInt32Payload(0x2Au)),
            .. Subchunk("MCBB", new byte[0x14 * 3]),
            .. Subchunk("MCDD", new byte[8]),
            .. Subchunk("MCLY", Mcly(layerCount: 1)),
        ];

        Mcnk mcnk = new(payload, Headerless);

        Assert.True(mcnk.SubchunkWalkConsumedExactly);
        Assert.Equal(8, mcnk.McrfData?.Length);
        Assert.Equal(16, mcnk.MclvData?.Length);
        Assert.Equal(0x2Au, mcnk.McmtValue);
        Assert.Equal(3, mcnk.McbbRecordCount); // client: size / 0x14
        Assert.Equal(8, mcnk.McddData?.Length);
        Assert.Empty(mcnk.UnknownSubchunks);
    }

    [Fact]
    public void Scan_UnmodelledToken_IsRecordedAndDoesNotStopTheWalk()
    {
        // Native falls through unknown tokens and still advances by the declared size, so an
        // unmodelled chunk must not cost us the ones behind it — but we want to know it exists.
        byte[] payload =
        [
            .. Subchunk("MCVT", new byte[4]),
            .. Subchunk("MCZZ", new byte[12]),
            .. Subchunk("MCLY", Mcly(layerCount: 1)),
        ];

        Mcnk mcnk = new(payload, Headerless);

        Assert.True(mcnk.SubchunkWalkConsumedExactly);
        Assert.Equal(1, mcnk.MclyRecordCount);
        Assert.Contains("MCZZ", mcnk.UnknownSubchunks);
    }

    private static byte[] Subchunk(string token, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Encoding.ASCII.GetBytes(token).CopyTo(bytes, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        payload.CopyTo(bytes, 8);
        return bytes;
    }

    private static byte[] Mcly(int layerCount) => new byte[layerCount * 16];

    private static byte[] UInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }
}
