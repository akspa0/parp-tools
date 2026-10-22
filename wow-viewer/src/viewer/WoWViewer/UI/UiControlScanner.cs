using System.Text.RegularExpressions;

namespace WoWViewer.UI;

/// <summary>One visible interactive control found in a manager-owned surface.</summary>
/// <param name="Label">The visible label, without the ImGui identifier suffix.</param>
/// <param name="File">Repository-relative source path.</param>
/// <param name="Line">1-based line number.</param>
/// <param name="Kind">The ImGui or shared-widget call that draws it.</param>
public sealed record UiControlSource(string Label, string File, int Line, string Kind);

/// <summary>
/// Finds the visible controls drawn by the surfaces the UI manager owns.
///
/// The scanner is the enforcement half of the documentation contract: every control
/// it finds must have a registry entry, otherwise the build check fails.
/// </summary>
public static partial class UiControlScanner
{
    /// <summary>Surfaces owned by the UI manager (Spec 280 ui-manager.md §2).</summary>
    public static IReadOnlyList<string> CoveredRoots { get; } =
    [
        "Workbench/Services/UiManager",
        "Workbench/Services/MainMenu",
        "Workbench/Services/Chrome",
        "Workbench/Services/Navigator",
        "Workbench/Services/WorkbenchPanels",
    ];

    // The label is the first argument. It may be a plain literal or a conditional
    // whose branches are literals (for example `collapsed ? "Map" : "Collapse"`).
    [GeneratedRegex(@"ImGui\.(Button|SmallButton|Checkbox|RadioButton|MenuItem|Selectable|TreeNode|CollapsingHeader|Combo|SliderFloat|SliderInt)\(([^;]*)\)")]
    private static partial Regex ImGuiCallRegex();

    [GeneratedRegex(@"DrawActionBarIcon\(([^;]*)\)")]
    private static partial Regex ActionBarIconRegex();

    [GeneratedRegex(@"SharedUiWidgets\.(ActionButton|DrawActionGroup|SectionHeader|DrawSectionHeader)\(([^;]*)\)")]
    private static partial Regex SharedWidgetRegex();

    [GeneratedRegex(@"(?<!\$)""((?:[^""\\]|\\.)*)""")]
    private static partial Regex StringLiteralRegex();

    /// <summary>Scans the manager-owned surfaces under <paramref name="viewerRoot"/>.</summary>
    public static IReadOnlyList<UiControlSource> Scan(string viewerRoot)
    {
        List<UiControlSource> controls = [];
        foreach (string relativeRoot in CoveredRoots)
        {
            string directory = Path.Combine(viewerRoot, relativeRoot.Replace('/', Path.DirectorySeparatorChar));
            if (!Directory.Exists(directory)) continue;

            foreach (string file in Directory.EnumerateFiles(directory, "*.cs").OrderBy(path => path, StringComparer.Ordinal))
            {
                string[] lines = File.ReadAllLines(file);
                for (int index = 0; index < lines.Length; index++)
                {
                    string line = lines[index];
                    foreach (Match match in ImGuiCallRegex().Matches(line))
                        AddControls(controls, match, file, viewerRoot, index + 1);
                    foreach (Match match in SharedWidgetRegex().Matches(line))
                        AddControls(controls, match, file, viewerRoot, index + 1);
                    foreach (Match match in ActionBarIconRegex().Matches(line))
                        AddActionBarIconControl(controls, match, file, viewerRoot, index + 1);
                }
            }
        }

        return controls
            .GroupBy(control => control.Label, StringComparer.Ordinal)
            .Select(group => group.OrderBy(control => control.File, StringComparer.Ordinal)
                .ThenBy(control => control.Line)
                .First())
            .OrderBy(control => control.Label, StringComparer.Ordinal)
            .ToList();
    }

    private static void AddControls(
        List<UiControlSource> controls,
        Match match,
        string file,
        string viewerRoot,
        int line)
    {
        string relative = Path.GetRelativePath(viewerRoot, file).Replace('\\', '/');
        foreach (string label in ExtractFirstArgumentLabels(match))
        {
            controls.Add(new UiControlSource(label, relative, line, match.Groups[1].Value));
        }
    }

    private static void AddActionBarIconControl(
        List<UiControlSource> controls,
        Match match,
        string file,
        string viewerRoot,
        int line)
    {
        string relative = Path.GetRelativePath(viewerRoot, file).Replace('\\', '/');
        foreach (string label in ExtractFirstArgumentLabels(match.Groups[1].Value))
            controls.Add(new UiControlSource(label, relative, line, "ActionBarIcon"));
    }

    /// <summary>Returns the visible labels of the call's first argument.</summary>
    private static IEnumerable<string> ExtractFirstArgumentLabels(Match match) =>
        ExtractFirstArgumentLabels(match.Groups[2].Value);

    private static IEnumerable<string> ExtractFirstArgumentLabels(string arguments)
    {
        int depth = 0;
        int end = arguments.Length;
        for (int index = 0; index < arguments.Length; index++)
        {
            char c = arguments[index];
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ',' && depth == 0) { end = index; break; }
        }

        string firstArgument = arguments[..Math.Min(end, arguments.Length)];
        List<string> labels = [];
        foreach (Match literal in StringLiteralRegex().Matches(firstArgument))
        {
            string label = Regex.Replace(literal.Groups[1].Value, "##.*$", string.Empty).Trim();
            if (string.IsNullOrWhiteSpace(label)) continue;
            labels.Add(label);
        }

        return labels;
    }
}
