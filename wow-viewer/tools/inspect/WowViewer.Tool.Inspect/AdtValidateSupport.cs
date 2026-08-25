using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

/// <summary>
/// Validates that an ADT is structurally well-formed, by walking it exactly the way a client does.
/// </summary>
/// <remarks>
/// Written because a malformed file this project produced was caught by an ad-hoc script instead of by
/// the tooling. <c>LkAdtWriter</c> padded odd payloads without counting the pad in the declared size,
/// so a sequential walk landed a byte short and read the next tag as garbage - and nothing in the repo
/// would have said so. A check that only exists as a one-off is a check that will not run next time.
///
/// <para>The walk is deliberately strict: <c>offset + 8 + size</c>, no padding assumed, no recovery
/// attempted. That is the contract a reader relies on, so anything needing a lenient reader is broken
/// even if some readers tolerate it. Measured against real ADTs from the corpus, which have no
/// odd-sized chunks at all.</para>
///
/// <para>Reports the chunk inventory as well, since "parses" and "contains what an ADT must contain"
/// are different questions and a blank file can pass the first.</para>
/// </remarks>
internal static class AdtValidateSupport
{
    private static readonly string[] RequiredChunks = ["MVER", "MHDR", "MCIN", "MTEX", "MMDX", "MMID", "MWMO", "MWID", "MDDF", "MODF"];

    public static AdtValidateReport Validate(string path)
    {
        var files = new List<AdtFileResult>();

        IEnumerable<string> targets = Directory.Exists(path)
            ? Directory.EnumerateFiles(path, "*.adt", SearchOption.TopDirectoryOnly).OrderBy(static f => f)
            : [path];

        foreach (string file in targets)
            files.Add(ValidateOne(file));

        return new AdtValidateReport(
            path,
            files.Count,
            files.Count(static f => f.Ok),
            files.Count(static f => !f.Ok),
            files);
    }

    private static AdtFileResult ValidateOne(string path)
    {
        string name = Path.GetFileName(path);

        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new AdtFileResult(name, false, 0, 0, 0, 0, $"unreadable: {ex.Message}", []);
        }

        var seen = new List<string>();
        int offset = 0, chunks = 0, oddSized = 0, mcnk = 0;

        while (offset + 8 <= data.Length)
        {
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 4, 4));

            // Tags are stored reversed on disk.
            string tag = string.Create(4, magic, static (span, m) =>
            {
                span[0] = (char)(byte)(m >> 24);
                span[1] = (char)(byte)(m >> 16);
                span[2] = (char)(byte)(m >> 8);
                span[3] = (char)(byte)m;
            });

            foreach (char ch in tag)
            {
                if (!char.IsAsciiLetterUpper(ch) && !char.IsAsciiDigit(ch) && ch != ' ')
                {
                    return new AdtFileResult(
                        name, false, chunks, oddSized, mcnk, offset,
                        $"garbage tag at offset {offset} after {chunks} chunks - the previous chunk's size does not lead to a tag boundary",
                        seen);
                }
            }

            if (size > int.MaxValue || offset + 8L + size > data.Length)
            {
                return new AdtFileResult(
                    name, false, chunks, oddSized, mcnk, offset,
                    $"chunk '{tag}' at {offset} declares {size} bytes, which runs past the end of the file",
                    seen);
            }

            if ((size & 1) != 0)
                oddSized++;

            if (tag == "MCNK")
                mcnk++;
            else if (!seen.Contains(tag))
                seen.Add(tag);

            chunks++;
            offset += 8 + (int)size;
        }

        if (offset != data.Length)
        {
            return new AdtFileResult(
                name, false, chunks, oddSized, mcnk, offset,
                $"trailing bytes: consumed {offset} of {data.Length}",
                seen);
        }

        // A split tile spreads its chunks across root/_obj0/_tex0, so demanding the monolithic set
        // from every file marks perfectly good files as broken. Only a file that presents itself as
        // monolithic - it has the header chunks - is held to the full inventory.
        bool monolithic = seen.Contains("MHDR") && seen.Contains("MCIN");
        if (monolithic)
        {
            List<string> missing = [.. RequiredChunks.Where(r => !seen.Contains(r))];
            if (missing.Count > 0)
            {
                return new AdtFileResult(
                    name, false, chunks, oddSized, mcnk, offset,
                    $"monolithic but missing required chunks: {string.Join(", ", missing)}",
                    seen);
            }
        }

        if (mcnk is not 0 and not 256)
        {
            return new AdtFileResult(
                name, false, chunks, oddSized, mcnk, offset,
                $"has {mcnk} MCNK chunks where a tile has 256 (or none, for a placement-only file)",
                seen);
        }

        // An empty file is a real state in this corpus - split tiles keep a zero-byte root - and is
        // not a malformed file. Reported as such rather than counted against the writer.
        if (chunks == 0)
            return new AdtFileResult(name, true, 0, 0, 0, offset, "empty file (0 bytes)", seen);

        // Placements are read through the real reader rather than re-parsed here, but only for files
        // that carry them. A _tex0 has no placement chunks by design, so asking for them and calling
        // the refusal a failure would condemn a correct file.
        if (!seen.Contains("MODF") && !seen.Contains("MWMO"))
            return new AdtFileResult(name, true, chunks, oddSized, mcnk, offset, "ok, carries no placement chunks", seen);

        int placements;
        try
        {
            placements = AdtPlacementReader.Read(path).WorldModelPlacements.Count;
        }
        catch (Exception ex)
        {
            return new AdtFileResult(
                name, false, chunks, oddSized, mcnk, offset,
                $"walks cleanly but the placement reader threw: {ex.Message}",
                seen);
        }

        return new AdtFileResult(name, true, chunks, oddSized, mcnk, offset, $"ok, {placements} world-model placements", seen);
    }
}

internal sealed record AdtFileResult(
    string File,
    bool Ok,
    int Chunks,
    int OddSizedChunks,
    int McnkCount,
    int BytesConsumed,
    string Detail,
    IReadOnlyList<string> TopLevelChunks);

internal sealed record AdtValidateReport(
    string Input,
    int Files,
    int Passed,
    int Failed,
    IReadOnlyList<AdtFileResult> Results);
