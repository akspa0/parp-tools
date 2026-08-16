using System.Text.RegularExpressions;

namespace WowViewer.Core.IO.Wtf;

/// <summary>
/// Classifies one WTF line against the two statement shapes confirmed from real client data
/// (<c>SET name "value"</c> and <c>bind KEY ACTION</c>, both read directly from 0.5.3.3368's real
/// WTF\DefaultBindings.wtf and Config.wtf). Deliberately does not attempt to recognize anything wider —
/// the tool's job is telling recognized from not, not guessing at every possible statement grammar
/// before any has actually been seen in real unrecognized-line output.
/// </summary>
public static class WtfLineClassifier
{
    // Value is quoted in most files (Config.wtf) but bare in at least one real one (realmlist.wtf:
    // "set realmlist beta.us.logon.worldofwarcraft.com", no quotes, lowercase keyword).
    private static readonly Regex SetPattern =
        new(@"^SET\s+(\S+)\s+(?:""(.*)""|(\S+))\s*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex BindPattern =
        new(@"^bind\s+(\S+)\s+(\S+)\s*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>Classifies one non-blank line. Blank-line filtering is the caller's responsibility —
    /// a blank line is not a statement of any kind, recognized or not.</summary>
    public static WtfLine Classify(string rawLine)
    {
        ArgumentNullException.ThrowIfNull(rawLine);

        Match setMatch = SetPattern.Match(rawLine);
        if (setMatch.Success)
        {
            string value = setMatch.Groups[2].Success ? setMatch.Groups[2].Value : setMatch.Groups[3].Value;
            return new WtfLine(rawLine, WtfLineKind.Set, setMatch.Groups[1].Value, value);
        }

        Match bindMatch = BindPattern.Match(rawLine);
        if (bindMatch.Success)
            return new WtfLine(rawLine, WtfLineKind.Bind, bindMatch.Groups[1].Value, bindMatch.Groups[2].Value);

        return new WtfLine(rawLine, WtfLineKind.Unrecognized);
    }

    /// <summary>Splits raw file text into lines and classifies every non-blank one.</summary>
    public static IReadOnlyList<WtfLine> ClassifyFile(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        List<WtfLine> lines = [];
        foreach (string rawLine in text.Split(['\r', '\n']))
        {
            if (string.IsNullOrWhiteSpace(rawLine))
                continue;

            lines.Add(Classify(rawLine));
        }

        return lines;
    }
}
