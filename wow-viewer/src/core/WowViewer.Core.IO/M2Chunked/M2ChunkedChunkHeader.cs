namespace WowViewer.Core.IO.M2Chunked;

public readonly record struct M2ChunkedChunkHeader(string FourCC, uint Size, long Offset, bool IsTruncated = false);
