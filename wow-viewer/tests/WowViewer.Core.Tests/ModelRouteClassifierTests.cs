using System.Buffers.Binary;
using WowViewer.Core.IO.AssetReferences;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;

namespace WowViewer.Core.Tests;

/// <summary>
/// Pins ModelRouteClassifier against Spec 154's actual measured findings: MDLX and MD20 0x108 read,
/// MD20 0x102-0x107 is refused outright by the dispatcher itself, and an era Spec 154 never measured
/// either way stays Unknown rather than being guessed into Readable or Blocked.
/// </summary>
public sealed class ModelRouteClassifierTests
{
    [Fact]
    public void Mdlx_ClassifiesAsReadable()
    {
        ModelRouteClassification result = ModelRouteClassifier.Classify(MdlxHeader(), "test.mdx");

        Assert.Equal(ModelRouteStatus.Readable, result.Status);
    }

    [Fact]
    public void Md20V108_ClassifiesAsReadable()
    {
        ModelRouteClassification result = ModelRouteClassifier.Classify(Md20Header(0x108), "test.m2");

        Assert.Equal(ModelRouteStatus.Readable, result.Status);
    }

    [Theory]
    [InlineData(0x102u)]
    [InlineData(0x107u)]
    public void Md20TbcEraRange_ClassifiesAsBlocked(uint version)
    {
        // 0x107 is the version actually measured at 3.0.1.8303 (Spec 154). 0x102 pins the range's
        // lower boundary the dispatcher itself refuses via NotSupportedException.
        ModelRouteClassification result = ModelRouteClassifier.Classify(Md20Header(version), "test.m2");

        Assert.Equal(ModelRouteStatus.Blocked, result.Status);
        Assert.Contains("0x102-0x107", result.RouteLabel);
    }

    [Fact]
    public void Md20V109_ClassifiesAsBlocked()
    {
        // Measured broken at 4.0.0.11927 (Spec 154) and beyond this project's declared 4.0.0 scope
        // ceiling either way.
        ModelRouteClassification result = ModelRouteClassifier.Classify(Md20Header(0x109), "test.m2");

        Assert.Equal(ModelRouteStatus.Blocked, result.Status);
    }

    [Fact]
    public void BlockedRoute_CarriesANonEmptyReason()
    {
        ModelRouteClassification result = ModelRouteClassifier.Classify(Md20Header(0x107), "test.m2");

        Assert.False(string.IsNullOrWhiteSpace(result.Reason));
    }

    [Fact]
    public void TooSmallToClassify_IsUnknownNotBlocked()
    {
        // A truncated/garbage file is a per-asset problem, not evidence the format route is broken.
        ModelRouteClassification result = ModelRouteClassifier.Classify([0x01, 0x02], "test.m2");

        Assert.Equal(ModelRouteStatus.Unknown, result.Status);
    }

    private static byte[] Md20Header(uint version)
    {
        byte[] bytes = new byte[M2Era1121Constants.DispatchHeaderSizeBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, M2Era1121Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), version);
        return bytes;
    }

    private static byte[] MdlxHeader()
    {
        byte[] bytes = new byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, MdxMagic.Mdlx);
        return bytes;
    }
}
