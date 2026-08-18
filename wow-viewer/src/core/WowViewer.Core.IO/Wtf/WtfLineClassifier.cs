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

    // keyword followed by 3 (teleport-shaped) or 4 (worldport-shaped) numeric arguments.
    private static readonly Regex PortCommandPattern =
        new(@"^(\S+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?\s*$",
            RegexOptions.Compiled);

    // WoW's internal world coordinate system: a 64x64 ADT tile grid at 533.3333 yards/tile gives a
    // max extent of 32 * 533.3333 ~= 17066.67 from origin in each direction. Widened generously since
    // this is a plausibility check for a discovery tool, not a hard format boundary.
    private const double MaxPlausibleCoordinate = 20000.0;
    private const double MaxPlausibleMapId = 1000.0;

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

        Match portMatch = PortCommandPattern.Match(rawLine);
        if (portMatch.Success)
            return ClassifyPortCommand(rawLine, portMatch);

        return new WtfLine(rawLine, WtfLineKind.Unrecognized);
    }

    private static WtfLine ClassifyPortCommand(string rawLine, Match match)
    {
        string keyword = match.Groups[1].Value;
        List<double> args = [];
        for (int groupIndex = 2; groupIndex <= 5; groupIndex++)
        {
            if (match.Groups[groupIndex].Success)
                args.Add(double.Parse(match.Groups[groupIndex].Value, System.Globalization.CultureInfo.InvariantCulture));
        }

        // 4 args: worldport-shaped only if the first looks like a map ID (non-negative, small,
        // effectively an integer) rather than just a fourth coordinate component.
        bool hasMapIdArg = args.Count == 4
            && args[0] >= 0
            && args[0] <= MaxPlausibleMapId
            && args[0] == Math.Floor(args[0]);

        IReadOnlyList<double> positionArgs = hasMapIdArg ? args.Skip(1).ToList() : args;
        bool coordinatesPlausible = positionArgs.All(static v => Math.Abs(v) <= MaxPlausibleCoordinate);

        return new WtfLine(rawLine, WtfLineKind.PortCommandCandidate, keyword, null, args, hasMapIdArg, coordinatesPlausible);
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
