using System.Buffers.Binary;
using System.Security.Cryptography;
using ICSharpCode.SharpZipLib.BZip2;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// A parsed <c>PTCH</c> patch artifact header as embedded at the start of patched ADT/WDT files
/// in MPQ-era clients (Cataclysm through MoP). The artifact carries the MD5 of the base file it
/// patches plus a BSDIFF40 delta; the client applies the delta to the base copy from a
/// lower-priority archive before handing the reconstructed bytes to the map loaders.
/// Native evidence: 5.0.1 <c>MapArea</c> registers file-data objects whose bytes are already
/// reconstructed by the resource layer (FUN_00BB0850 → FUN_00BB7D10 → load callback
/// FUN_00BB70F0 in MapAdtFileData.cpp stores the final file data and size).
/// </summary>
public readonly record struct AdtPatchArtifact(byte[] ExpectedBaseMd5, byte[] BsdiffPatch);

/// <summary>
/// Decoder for PTCH/BSDIFF patch artifacts. Pure byte-level logic with no I/O so it can be
/// unit-tested and reused by any data source layer.
/// </summary>
public static class AdtPatchArtifactDecoder
{
    public static bool IsPatchArtifact(ReadOnlySpan<byte> data)
        => data.Length >= 4 && data[0] == (byte)'P' && data[1] == (byte)'T' && data[2] == (byte)'C' && data[3] == (byte)'H';

    /// <summary>
    /// Parse a PTCH artifact: locate the <c>MD5_</c> marker (16-byte base MD5 follows), then the
    /// <c>BSD0</c> marker (int32 patch size + BSDIFF payload follows). Marker scanning keeps the
    /// parser tolerant of minor header-layout variations between client builds.
    /// </summary>
    public static bool TryParse(ReadOnlySpan<byte> data, out AdtPatchArtifact artifact, out string error)
    {
        artifact = default;
        error = string.Empty;

        if (!IsPatchArtifact(data))
        {
            error = "data does not start with the PTCH magic";
            return false;
        }

        int md5Marker = IndexOf(data, "MD5_"u8, 0, Math.Min(data.Length, 64));
        if (md5Marker < 0)
        {
            error = "MD5_ marker not found in PTCH header";
            return false;
        }

        int md5Start = md5Marker + 4;
        if (md5Start + 16 > data.Length)
        {
            error = "PTCH header truncated inside the base MD5";
            return false;
        }

        byte[] baseMd5 = data.Slice(md5Start, 16).ToArray();

        int bsdMarker = IndexOf(data, "BSD0"u8, md5Start + 16, Math.Min(data.Length, md5Start + 16 + 64));
        if (bsdMarker < 0)
        {
            error = "BSD0 marker not found in PTCH header";
            return false;
        }

        int sizePosition = bsdMarker + 4;
        if (sizePosition + 4 > data.Length)
        {
            error = "PTCH header truncated inside the patch size";
            return false;
        }

        int patchSize = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(sizePosition, 4));
        if (patchSize <= 0 || sizePosition + 4 + patchSize > data.Length)
        {
            error = $"invalid BSDIFF patch size {patchSize} (file length {data.Length})";
            return false;
        }

        byte[] patch = data.Slice(sizePosition + 4, patchSize).ToArray();
        artifact = new AdtPatchArtifact(baseMd5, patch);
        return true;
    }

    /// <summary>
    /// Reconstruct the final file bytes from a PTCH artifact: select the base candidate whose MD5
    /// matches the artifact's embedded base MD5, apply the BSDIFF delta, and require the result to
    /// begin with the on-disk reversed MVER fourcc (<c>REVM</c>) used by all MPQ-era chunked map
    /// files (ADT/WDT).
    /// </summary>
    /// <param name="artifactBytes">Raw PTCH artifact bytes.</param>
    /// <param name="baseCandidates">Raw copies of the same virtual file from all backing sources.</param>
    /// <param name="reconstructed">Reconstructed file bytes on success.</param>
    /// <param name="failureReason">Human-readable failure reason on failure.</param>
    public static bool TryReconstruct(
        byte[] artifactBytes,
        IEnumerable<byte[]?> baseCandidates,
        out byte[]? reconstructed,
        out string failureReason)
    {
        reconstructed = null;

        if (!TryParse(artifactBytes, out AdtPatchArtifact artifact, out string parseError))
        {
            failureReason = parseError;
            return false;
        }

        foreach (byte[]? candidate in baseCandidates)
        {
            if (candidate is not { Length: > 0 })
                continue;

            // Other patch artifacts in the chain are not base material; a full multi-step chain
            // resolver would apply them in priority order, which is out of scope here.
            if (IsPatchArtifact(candidate))
                continue;

            if (!MD5.HashData(candidate).AsSpan().SequenceEqual(artifact.ExpectedBaseMd5))
                continue;

            if (!TryApplyBsdiff(candidate, artifact.BsdiffPatch, out byte[]? patched, out string applyError))
            {
                failureReason = $"base MD5 matched but BSDIFF apply failed: {applyError}";
                return false;
            }

            if (patched is not { Length: >= 4 } ||
                !patched.AsSpan(0, 4).SequenceEqual("REVM"u8))
            {
                failureReason = $"reconstructed payload does not start with REVM (length {(patched?.Length ?? 0)})";
                return false;
            }

            reconstructed = patched;
            failureReason = string.Empty;
            return true;
        }

        failureReason = "no base copy matched the artifact's embedded base MD5";
        return false;
    }

    /// <summary>
    /// Apply a BSDIFF40 binary delta (control/diff/extra blocks, each BZip2-compressed) to
    /// <paramref name="oldData"/>. Implements the standard bsdiff algorithm.
    /// </summary>
    public static bool TryApplyBsdiff(byte[] oldData, byte[] bsdiff, out byte[]? result, out string error)
    {
        result = null;
        error = string.Empty;

        if (bsdiff.Length < 32 || !bsdiff.AsSpan(0, 8).SequenceEqual("BSDIFF40"u8))
        {
            error = "payload is not BSDIFF40";
            return false;
        }

        long controlLength = BinaryPrimitives.ReadInt64BigEndian(bsdiff.AsSpan(8, 8));
        long diffLength = BinaryPrimitives.ReadInt64BigEndian(bsdiff.AsSpan(16, 8));
        long newSize = BinaryPrimitives.ReadInt64BigEndian(bsdiff.AsSpan(24, 8));

        if (controlLength < 0 || diffLength < 0 || newSize < 0 ||
            32L + controlLength + diffLength > bsdiff.Length)
        {
            error = $"invalid BSDIFF header (control={controlLength}, diff={diffLength}, newSize={newSize}, payload={bsdiff.Length})";
            return false;
        }

        if (!TryBzip2Decompress(bsdiff, 32, controlLength, out byte[]? control, out error) || control is null)
            return false;

        if (!TryBzip2Decompress(bsdiff, (int)(32 + controlLength), diffLength, out byte[]? diff, out error) || diff is null)
            return false;

        int extraStart = (int)(32 + controlLength + diffLength);
        int extraLength = bsdiff.Length - extraStart;
        byte[] extra = [];
        if (extraLength > 0 && (!TryBzip2Decompress(bsdiff, extraStart, extraLength, out extra, out error) || extra is null))
            return false;

        byte[] newData = new byte[newSize];
        long newPos = 0;
        long oldPos = 0;
        long diffPos = 0;
        long extraPos = 0;
        int controlPos = 0;

        while (newPos < newSize)
        {
            if (controlPos + 24 > control.Length)
            {
                error = "BSDIFF control stream exhausted";
                return false;
            }

            long add = BinaryPrimitives.ReadInt64BigEndian(control.AsSpan(controlPos, 8));
            long insert = BinaryPrimitives.ReadInt64BigEndian(control.AsSpan(controlPos + 8, 8));
            long seek = BinaryPrimitives.ReadInt64BigEndian(control.AsSpan(controlPos + 16, 8));
            controlPos += 24;

            // Diff run: bytes come from the DIFF block and are summed with the aligned old byte.
            // bsdiff allows the old cursor to walk outside the base file; those positions
            // contribute zero rather than aborting the apply.
            if (add < 0 || newPos + add > newSize)
            {
                error = $"BSDIFF invalid add run ({add}) at new offset {newPos}";
                return false;
            }

            if (diffPos + add > diff.Length)
            {
                error = $"BSDIFF diff block exhausted at new offset {newPos} (need {add}, have {diff.Length - diffPos})";
                return false;
            }

            for (long i = 0; i < add; i++)
            {
                byte delta = diff[diffPos + i];
                long oldIndex = oldPos + i;
                newData[newPos + i] = oldIndex >= 0 && oldIndex < oldData.Length
                    ? (byte)(delta + oldData[oldIndex])
                    : delta;
            }

            newPos += add;
            oldPos += add;
            diffPos += add;

            // Insert run: literal bytes come from the EXTRA block, not the diff block.
            if (insert < 0 || newPos + insert > newSize)
            {
                error = $"BSDIFF invalid insert run ({insert}) at new offset {newPos}";
                return false;
            }

            if (extraPos + insert > extra.Length)
            {
                error = $"BSDIFF extra block exhausted at new offset {newPos} (need {insert}, have {extra.Length - extraPos})";
                return false;
            }

            extra.AsSpan((int)extraPos, (int)insert).CopyTo(newData.AsSpan((int)newPos, (int)insert));

            newPos += insert;
            extraPos += insert;

            oldPos += seek;
        }

        result = newData;
        return true;
    }

    private static bool TryBzip2Decompress(byte[] source, int offset, long length, out byte[]? output, out string error)
    {
        output = null;
        error = string.Empty;

        if (length == 0)
        {
            output = [];
            return true;
        }

        try
        {
            using MemoryStream input = new(source, offset, (int)length, writable: false);
            using BZip2InputStream bzip = new(input);
            using MemoryStream outputMs = new();
            bzip.CopyTo(outputMs);
            output = outputMs.ToArray();
            return true;
        }
        catch (Exception ex)
        {
            error = $"BZip2 decompression failed: {ex.Message}";
            return false;
        }
    }

    private static int IndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, int start, int end)
    {
        if (start < 0 || end > haystack.Length || start >= end)
            return -1;

        ReadOnlySpan<byte> window = haystack[start..end];
        int relative = window.IndexOf(needle);
        return relative < 0 ? -1 : start + relative;
    }
}
