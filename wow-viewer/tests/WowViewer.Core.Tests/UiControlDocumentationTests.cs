using System.Text.Json;
using WoWViewer.UI;

namespace WowViewer.Core.Tests;

/// <summary>
/// The UI documentation contract (Spec 280, ui-manager.md §7).
///
/// Two rules are enforced here:
/// 1. Every visible control on a manager-owned surface has a registry entry.
/// 2. The committed <c>docs/WoWViewer/UI-REFERENCE.md</c> matches the registry exactly.
/// </summary>
public sealed class UiControlDocumentationTests
{
    [Fact]
    public void Registry_Parses_And_Has_Unique_Ids()
    {
        IReadOnlyList<UiControlDoc> controls = LoadRegistry();

        Assert.NotEmpty(controls);
        Assert.Equal(controls.Count, controls.Select(control => control.Id).Distinct(StringComparer.Ordinal).Count());
        Assert.All(controls, control =>
        {
            Assert.False(string.IsNullOrWhiteSpace(control.Label));
            Assert.False(string.IsNullOrWhiteSpace(control.Region));
            Assert.False(string.IsNullOrWhiteSpace(control.Help));
        });
    }

    [Fact]
    public void Every_Visible_Manager_Control_Has_A_Registry_Entry()
    {
        IReadOnlyList<UiControlDoc> controls = LoadRegistry();
        HashSet<string> documented = controls
            .Where(control => control.Visible)
            .Select(control => control.Label)
            .ToHashSet(StringComparer.Ordinal);

        IReadOnlyList<UiControlSource> found = UiControlScanner.Scan(ViewerRoot);
        Assert.NotEmpty(found);

        List<string> undocumented = found
            .Where(control => !documented.Contains(control.Label))
            .Select(control => $"{control.Label} ({control.File}:{control.Line})")
            .ToList();

        Assert.True(undocumented.Count == 0,
            "Visible controls without a UI control registry entry: " + string.Join(", ", undocumented));
    }

    [Fact]
    public void Committed_Ui_Reference_Documents_Every_Visible_Control()
    {
        IReadOnlyList<UiControlDoc> controls = LoadRegistry();
        string document = File.ReadAllText(ReferenceDocumentPath).Replace("\r\n", "\n");

        List<string> missing = controls
            .Where(control => control.Visible)
            .Where(control => !document.Contains($"| {control.Label} |", StringComparison.Ordinal)
                || !document.Contains(control.Help, StringComparison.Ordinal))
            .Select(control => control.Label)
            .ToList();

        Assert.True(missing.Count == 0,
            "UI reference is missing entries for: " + string.Join(", ", missing));
    }

    [Fact]
    public void IconActions_AreDocumentedByFullName_AndExactSourceLine()
    {
        var found = UiControlScanner.Scan(ViewerRoot).Where(control => control.Kind == "ActionBarIcon").ToList();
        string[] expected = ["Workspace", "File Browser", "Maps", "Assets", "Taxi", "Capture", "Diagnostics", "Inspector", "World", "Model", "PM4"];
        Assert.Equal(expected.OrderBy(label => label, StringComparer.Ordinal), found.Select(control => control.Label));
        foreach (var control in found)
        {
            string[] lines = File.ReadAllLines(Path.Combine(ViewerRoot, control.File));
            Assert.Contains($"DrawActionBarIcon(\"{control.Label}\"", lines[control.Line - 1]);
        }
    }

    [Fact]
    public void Registry_DoesNotExposeRejectedSpellbookOrRailAbbreviations()
    {
        string[] rejected = ["Spellbook", "Ws", "Op", "As", "Tx", "Cp", "In", "Wr", "Md", "P4"];
        Assert.DoesNotContain(LoadRegistry(), control => control.Visible && rejected.Contains(control.Label));
    }

    private static IReadOnlyList<UiControlDoc> LoadRegistry()
    {
        string json = File.ReadAllText(RegistryPath);
        using JsonDocument document = JsonDocument.Parse(json);
        Assert.True(document.RootElement.TryGetProperty("controls", out _), "Registry has no 'controls' array.");
        return UiReferenceDocument.ParseRegistry(json);
    }

    private static string WowViewerRoot
    {
        get
        {
            string? current = AppContext.BaseDirectory;
            while (!string.IsNullOrWhiteSpace(current))
            {
                if (File.Exists(Path.Combine(current, "WowViewer.slnx")))
                    return current;

                current = Directory.GetParent(current)?.FullName;
            }

            throw new DirectoryNotFoundException("Could not locate the wow-viewer workspace root from the test context.");
        }
    }

    private static string ViewerRoot => Path.Combine(WowViewerRoot, "src", "viewer", "WoWViewer");

    private static string RegistryPath => Path.Combine(WowViewerRoot, "eng", "ui", "ui-controls.json");

    private static string ReferenceDocumentPath => Path.Combine(WowViewerRoot, "docs", "WoWViewer", "UI-REFERENCE.md");
}
