using ImGuiNET;
using WowViewer.Core.Runtime.World.Inspection;

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
        if (ImGui.CollapsingHeader(section.Title, ImGuiTreeNodeFlags.DefaultOpen))
        {
            if (!string.IsNullOrWhiteSpace(section.Note))
                ImGui.TextWrapped(section.Note);

            foreach (InspectorRow row in section.Rows)
            {
                if (row.IsImportant)
                    ImGui.TextColored(new System.Numerics.Vector4(1f, 0.85f, 0.4f, 1f), $"{row.Label}: {row.Value}");
                else
                    ImGui.TextDisabled($"{row.Label}: {row.Value}");
            }

            foreach (InspectorTable table in section.Tables)
            {
                if (!ImGui.TreeNodeEx(table.Title, ImGuiTreeNodeFlags.SpanFullWidth))
                    continue;

                foreach (InspectorRow row in table.Rows)
                    ImGui.TextDisabled($"{row.Label}: {row.Value}");

                ImGui.TreePop();
            }

            foreach (InspectorAction action in section.Actions)
            {
                if (ImGui.SmallButton(action.Label) && dispatch != null)
                    dispatch(action);
            }
        }
    }
}
