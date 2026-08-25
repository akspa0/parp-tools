using System.Globalization;

namespace WowViewer.Core.Editor.Eras;

/// <summary>
/// A comparable client build identity, used to scope editor plugins and era-scoped handlers.
/// </summary>
/// <remarks>
/// WoW build strings span several shapes ("0.5.3.3368", "3.3.5.12340", "10.2.7"). The numeric
/// leading segments are compared numerically segment-by-segment; any trailing non-numeric text is
/// retained as the original form and used only as a deterministic tie-breaker. This keeps version
/// ranges (a plugin targets 1.x, another targets 3.3.5+) comparable without inventing a new build
/// catalog — the editor reuses the caller's existing build identity string.
/// </remarks>
public readonly struct EditorBuildVersion : IComparable<EditorBuildVersion>, IEquatable<EditorBuildVersion>
{
    private const int MaxSegments = 12;

    private readonly int[] _segments;
    private readonly string _original;

    private EditorBuildVersion(int[] segments, string original)
    {
        _segments = segments;
        _original = original;
    }

    public string Original => _original;

    public static EditorBuildVersion Parse(string value)
    {
        if (!TryParse(value, out EditorBuildVersion version))
            throw new FormatException($"'{value}' is not a supported build version identity.");

        return version;
    }

    public static bool TryParse(string? value, out EditorBuildVersion version)
    {
        version = default;
        if (string.IsNullOrWhiteSpace(value))
            return false;

        string trimmed = value.Trim();
        int[] segments = new int[MaxSegments];
        int segmentCount = 0;
        int cursor = 0;

        while (cursor < trimmed.Length && segmentCount < MaxSegments)
        {
            if (!char.IsAsciiDigit(trimmed[cursor]))
                break;

            int start = cursor;
            while (cursor < trimmed.Length && char.IsAsciiDigit(trimmed[cursor]))
                cursor++;

            // Overflow-guarded numeric parse; extremely long digit runs are compared as text.
            if (!int.TryParse(trimmed.AsSpan(start, cursor - start), NumberStyles.None, CultureInfo.InvariantCulture, out int parsed))
                return false;

            segments[segmentCount++] = parsed;

            if (cursor < trimmed.Length && trimmed[cursor] == '.')
            {
                cursor++;
                continue;
            }

            break;
        }

        if (segmentCount == 0)
            return false;

        int[] exact = new int[segmentCount];
        Array.Copy(segments, exact, segmentCount);
        version = new EditorBuildVersion(exact, trimmed);
        return true;
    }

    public int CompareTo(EditorBuildVersion other)
    {
        int shared = Math.Min(_segments.Length, other._segments.Length);
        for (int i = 0; i < shared; i++)
        {
            int comparison = _segments[i].CompareTo(other._segments[i]);
            if (comparison != 0)
                return comparison;
        }

        if (_segments.Length != other._segments.Length)
            return _segments.Length.CompareTo(other._segments.Length);

        return string.CompareOrdinal(_original, other._original);
    }

    public bool Equals(EditorBuildVersion other)
        => CompareTo(other) == 0;

    public override bool Equals(object? obj)
        => obj is EditorBuildVersion other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hash = default;
        foreach (int segment in _segments)
            hash.Add(segment);

        return hash.ToHashCode();
    }

    public override string ToString() => _original;

    public static bool operator <(EditorBuildVersion left, EditorBuildVersion right) => left.CompareTo(right) < 0;
    public static bool operator >(EditorBuildVersion left, EditorBuildVersion right) => left.CompareTo(right) > 0;
    public static bool operator <=(EditorBuildVersion left, EditorBuildVersion right) => left.CompareTo(right) <= 0;
    public static bool operator >=(EditorBuildVersion left, EditorBuildVersion right) => left.CompareTo(right) >= 0;
    public static bool operator ==(EditorBuildVersion left, EditorBuildVersion right) => left.Equals(right);
    public static bool operator !=(EditorBuildVersion left, EditorBuildVersion right) => !left.Equals(right);
}