using System.Text.Json.Serialization;

namespace WowViewer.Core.Runtime.World.Inspection;

/// <summary>
/// UI-agnostic content payload for the unified object Inspector (Spec 223).
///
/// The Inspector renders whatever is selected as a list of typed sections. The payload is plain
/// data — no ImGui types, no rendering — so the same builder output can feed the right-sidebar
/// tool today and the Spec 212 3D object HUD later without reimplementation.
/// </summary>
public sealed record InspectorContent
{
    /// <summary>Object type key the payload describes: "ADT", "MDX", "M2", "WMO", "PM4", "WL", or "" when nothing is selected.</summary>
    public string ObjectType { get; init; } = string.Empty;

    /// <summary>Human headline, e.g. "WMO 'shadowfang_keep.wmo' (placement 12)".</summary>
    public string Headline { get; init; } = string.Empty;

    /// <summary>Ordered sections; renderers show them in order.</summary>
    public IReadOnlyList<InspectorSection> Sections { get; init; } = [];

    /// <summary>True when the payload describes a real selection.</summary>
    [JsonIgnore]
    public bool HasContent => ObjectType.Length > 0;
}

/// <summary>One titled block inside an Inspector payload.</summary>
public sealed record InspectorSection
{
    public string Title { get; init; } = string.Empty;

    /// <summary>Simple label/value rows.</summary>
    public IReadOnlyList<InspectorRow> Rows { get; init; } = [];

    /// <summary>Nested key/value tables (e.g. per-MCNK facts, per-doodad-set entries).</summary>
    public IReadOnlyList<InspectorTable> Tables { get; init; } = [];

    /// <summary>Named actions the renderer offers (e.g. "Switch doodad set").</summary>
    public IReadOnlyList<InspectorAction> Actions { get; init; } = [];

    /// <summary>Optional pre-formatted note text (legacy summaries during migration).</summary>
    public string Note { get; init; } = string.Empty;
}

/// <summary>A label/value row. <see cref="IsImportant"/> rows render emphasized.</summary>
public sealed record InspectorRow(string Label, string Value, bool IsImportant = false);

/// <summary>A titled nested table of rows.</summary>
public sealed record InspectorTable(string Title, IReadOnlyList<InspectorRow> Rows);

/// <summary>
/// A named action with a stable id and an optional integer argument (e.g. the target doodad-set
/// index). The renderer maps ids to viewer operations; the payload itself stays UI-agnostic.
/// </summary>
public sealed record InspectorAction(string Id, string Label, int Argument = 0);

/// <summary>Mutable builder used by the per-type section builders.</summary>
public sealed class InspectorContentBuilder
{
    private readonly List<InspectorSectionBuilder> _sectionBuilders = [];

    public string ObjectType { get; set; } = string.Empty;
    public string Headline { get; set; } = string.Empty;

    public InspectorSectionBuilder AddSection(string title)
    {
        var builder = new InspectorSectionBuilder(title);
        _sectionBuilders.Add(builder);
        return builder;
    }

    public InspectorContent Build() => new()
    {
        ObjectType = ObjectType,
        Headline = Headline,
        Sections = _sectionBuilders.Select(b => b.ToSection()).ToList(),
    };
}

/// <summary>Fluent builder for one section; rows/tables/actions accumulate then freeze on ToSection.</summary>
public sealed class InspectorSectionBuilder(string title)
{
    private readonly List<InspectorRow> _rows = [];
    private readonly List<InspectorTable> _tables = [];
    private readonly List<InspectorAction> _actions = [];

    private string _note = string.Empty;

    public InspectorSectionBuilder Note(string note)
    {
        _note = note;
        return this;
    }

    public InspectorSectionBuilder Row(string label, string value, bool isImportant = false)
    {
        _rows.Add(new InspectorRow(label, value, isImportant));
        return this;
    }

    public InspectorSectionBuilder Table(string title, params InspectorRow[] rows)
    {
        _tables.Add(new InspectorTable(title, rows));
        return this;
    }

    public InspectorSectionBuilder Table(string title, IEnumerable<InspectorRow> rows)
    {
        _tables.Add(new InspectorTable(title, rows.ToList()));
        return this;
    }

    public InspectorSectionBuilder Action(string id, string label, int argument = 0)
    {
        _actions.Add(new InspectorAction(id, label, argument));
        return this;
    }

    public InspectorSection ToSection() => new()
    {
        Title = title,
        Rows = _rows,
        Tables = _tables,
        Actions = _actions,
        Note = _note,
    };
}