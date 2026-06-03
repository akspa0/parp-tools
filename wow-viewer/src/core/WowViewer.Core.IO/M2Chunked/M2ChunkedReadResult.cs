using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.M2Chunked;

public sealed class M2ChunkedReadResult(
    M2ModelDocument model,
    MdxSummary summary,
    MdxGeometryFile geometry,
    MdxToM2ConversionResult conversion,
    IReadOnlyList<M2ChunkedChunkHeader> chunks)
{
    public M2ModelDocument Model { get; } = model ?? throw new ArgumentNullException(nameof(model));

    public MdxSummary Summary { get; } = summary ?? throw new ArgumentNullException(nameof(summary));

    public MdxGeometryFile Geometry { get; } = geometry ?? throw new ArgumentNullException(nameof(geometry));

    public MdxToM2ConversionResult Conversion { get; } = conversion ?? throw new ArgumentNullException(nameof(conversion));

    public IReadOnlyList<M2ChunkedChunkHeader> Chunks { get; } = chunks ?? throw new ArgumentNullException(nameof(chunks));

    public int VertexCount => Geometry.Geosets.Sum(static geoset => geoset.VertexCount);

    public int TriangleCount => Geometry.Geosets.Sum(static geoset => geoset.TriangleCount);
}
