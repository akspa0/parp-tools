using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.UI;

namespace WoWViewer;

/// <summary>
/// File Browser add-ons that sit under the one main browser list: opening a Warcraft III folder as the active
/// data source, PNG export of textures (mip 0 = full resolution) and unpacking original bytes. Holds only UI
/// state + one background job; all data access goes through the active <see cref="IDataSource"/>.
/// </summary>
internal sealed class FileBrowserExtrasPanel : IDisposable
{
    private string _wc3Root = "";
    private string _wc3Listfile = "";
    private string _output = "";
    private bool _allMips = true;
    private bool _overwrite;
    private Task<string>? _job;
    private CancellationTokenSource _cancel = new();

    /// <summary>Draws the "Open WC3 data" section. <paramref name="open"/> is invoked with (folder, optional listfile).</summary>
    internal void DrawOpenWc3(Action<string, string?> open, string? currentSourceSummary)
    {
        if (!SharedUiWidgets.SectionHeader("Open WC3 data", defaultOpen: false, id: "FileBrowserWc3"))
            return;
        ImGui.InputText("Folder##wc3open", ref _wc3Root, 2048);
        if (ImGui.SmallButton("Browse folder##wc3open"))
            Pick("Select WC3 install or extracted data folder", true, _wc3Root, null, p => _wc3Root = p);
        ImGui.InputText("Listfile (optional)##wc3open", ref _wc3Listfile, 2048);
        if (ImGui.SmallButton("Choose listfile##wc3open"))
            Pick("Select local WC3 listfile", false, _wc3Listfile, ".txt", p => _wc3Listfile = p);
        ImGui.SameLine();
        if (ImGui.SmallButton("Clear listfile##wc3open")) _wc3Listfile = "";
        ImGui.BeginDisabled(string.IsNullOrWhiteSpace(_wc3Root));
        if (ImGui.Button("Open WC3 data##wc3open"))
            open(_wc3Root.Trim(), string.IsNullOrWhiteSpace(_wc3Listfile) ? null : _wc3Listfile.Trim());
        ImGui.EndDisabled();
        ImGui.TextDisabled("Loose extracted data needs no listfile. MPQs use the existing reader.");
        if (!string.IsNullOrEmpty(currentSourceSummary))
        {
            ImGui.PushTextWrapPos(0f);
            ImGui.TextDisabled(currentSourceSummary);
            ImGui.PopTextWrapPos();
        }
    }

    /// <summary>Draws PNG export + unpack controls for the current selection / listed files.</summary>
    internal void DrawExport(IDataSource? dataSource, string? selected, IReadOnlyList<string> listed, Action<string> setStatus)
    {
        PollJob(setStatus);
        if (dataSource == null) return;
        bool canExportSelected = selected != null && DataFileViewerService.IsImagePath(selected);
        List<string> listedTextures = listed.Where(DataFileViewerService.IsImagePath).ToList();
        if (!canExportSelected && listedTextures.Count == 0 && dataSource is not Wc3DataSource) return;

        if (!SharedUiWidgets.SectionHeader("Export / unpack", defaultOpen: true, id: "FileBrowserExport")) return;
        ImGui.InputText("Output folder##export", ref _output, 2048);
        if (ImGui.SmallButton("Choose output##export"))
            Pick("Select output folder", true, _output, null, p => _output = p);
        ImGui.Checkbox("All mip levels (BLP0 .b00-.bNN)##export", ref _allMips);
        ImGui.SameLine();
        ImGui.Checkbox("Overwrite##export", ref _overwrite);

        bool busy = _job != null;
        bool hasOutput = !string.IsNullOrWhiteSpace(_output);
        ImGui.BeginDisabled(busy || !hasOutput || !canExportSelected);
        if (ImGui.Button("Export selected as PNG##export")) StartExport(dataSource, [selected!], setStatus);
        ImGui.EndDisabled();
        ImGui.BeginDisabled(busy || !hasOutput || listedTextures.Count == 0);
        ImGui.SameLine();
        if (ImGui.Button($"Export all {listedTextures.Count:N0} listed textures##export")) StartExport(dataSource, listedTextures, setStatus);
        ImGui.EndDisabled();

        if (dataSource is Wc3DataSource wc3)
        {
            ImGui.BeginDisabled(busy || !hasOutput || selected == null);
            if (ImGui.SmallButton("Unpack selected (original bytes)##export"))
            {
                string file = selected!, output = _output.Trim();
                RunJob(() => "Unpacked: " + wc3.ExtractFile(file, output, _cancel.Token), setStatus, "Unpacking original bytes...");
            }
            ImGui.EndDisabled();
        }
        if (busy && ImGui.SmallButton("Cancel##export")) _cancel.Cancel();
    }

    private void StartExport(IDataSource ds, IReadOnlyCollection<string> files, Action<string> setStatus)
    {
        string output = _output.Trim();
        bool allMips = _allMips, overwrite = _overwrite;
        string[] snapshot = files.ToArray();
        RunJob(() =>
        {
            TextureExportResult r = TextureExportService.Export(ds, snapshot, output, allMips, overwrite, null, _cancel.Token);
            string text = $"PNG export: {r.Written} written, {r.Skipped} skipped (exists), {r.Failed} failed -> {output}";
            if (r.FirstError != null) text += $" | first error: {r.FirstError}";
            return text;
        }, setStatus, $"Exporting {snapshot.Length:N0} texture(s) to PNG...");
    }

    private void RunJob(Func<string> work, Action<string> setStatus, string startMessage)
    {
        _cancel.Dispose();
        _cancel = new CancellationTokenSource();
        setStatus(startMessage);
        _job = Task.Run(work);
    }

    private void PollJob(Action<string> setStatus)
    {
        if (_job is not { IsCompleted: true } job) return;
        _job = null;
        try { setStatus(job.GetAwaiter().GetResult()); }
        catch (OperationCanceledException) { setStatus("Export cancelled; partial output may remain."); }
        catch (Exception ex) { setStatus("Export failed: " + ex.GetBaseException().Message); }
    }

    private static void Pick(string title, bool folder, string initial, string? extension, Action<string> selected)
        => ImGuiPathPicker.Instance.Open(title, folder,
            string.IsNullOrWhiteSpace(initial) ? Directory.GetCurrentDirectory() : initial, extension, selected);

    public void Dispose()
    {
        _cancel.Cancel();
        _cancel.Dispose();
    }
}
