using ImGuiNET;
using WowViewer.Core.Runtime.World.Inspection;
using WoWViewer.UI;

namespace WoWViewer.Workbench;

/// <summary>
/// Renders an <see cref="InspectorContent"/> payload (Spec 223 T101).
///
/// The host is deliberately dumb: it draws sections/rows/tables/actions and reports action ids
/// through a dispatch callback. All selection-specific content lives in the payload builders, so
/// the same payload can later drive the Spec 212 3D object HUD.
/// </summary>
public static class InspectorContentHost
{
    public delegate void InspectorActionHandler(InspectorAction action);

    public static void Draw(InspectorContent content, InspectorActionHandler? dispatch)
    {
        if (!content.HasContent)
            return;

        if (!string.IsNullOrWhiteSpace(content.Headline))
        {
            ImGui.TextWrapped(content.Headline);
            ImGui.Separator();
        }

        foreach (InspectorSection section in content.Sections)
            DrawSection(section, dispatch);
    }

    private static void DrawSection(InspectorSection section, InspectorActionHandler? dispatch)
    {
        if (SharedUiWidgets.SectionHeader(section.Title, section.Note, defaultOpen: true, id: $"Inspector_{section.Title}"))
        {
            foreach (InspectorRow row in section.Rows)
                SharedUiWidgets.StatusReadout(row.Label, row.Value, row.IsImportant);

            foreach (InspectorTable table in section.Tables)
            {
                if (!ImGui.TreeNodeEx(table.Title, ImGuiTreeNodeFlags.SpanFullWidth))
                    continue;

                foreach (InspectorRow row in table.Rows)
                    SharedUiWidgets.StatusReadout(row.Label, row.Value);

                ImGui.TreePop();
            }

            for (int actionIndex = 0; actionIndex < section.Actions.Count; actionIndex++)
            {
                InspectorAction action = section.Actions[actionIndex];
                string actionLabel = $"{action.Label}##InspectorAction_{action.Id}_{actionIndex}";
                if (SharedUiWidgets.ActionButton(actionLabel, small: true) && dispatch != null)
                    dispatch(action);
            }
        }
    }
}
