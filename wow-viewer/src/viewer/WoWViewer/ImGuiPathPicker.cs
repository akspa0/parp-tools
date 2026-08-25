using System.Numerics;
using ImGuiNET;

namespace WoWViewer;

/// <summary>
/// In-app ImGui file/folder picker built exclusively on BCL filesystem APIs, so browsing behaves
/// identically on every platform — no WinForms or native dialogs anywhere in cross-platform builds.
/// One modal at a time; call <see cref="Open"/> from any UI surface and <see cref="Draw"/> once per
/// frame while that surface is visible.
/// </summary>
internal sealed class ImGuiPathPicker
{
    public static ImGuiPathPicker Instance { get; } = new();

    private bool _openRequested;
    private bool _popupVisible;
    private string _title = "Select path";
    private bool _pickFolder;
    private string _filterExtension = string.Empty;
    private Action<string>? _onPicked;
    private string _currentDirectory = Directory.GetCurrentDirectory();
    private string _fileName = string.Empty;
    private string _error = string.Empty;

    private ImGuiPathPicker()
    {
    }

    /// <summary>Opens the picker modal. <paramref name="filterExtension"/> like ".pm4" (file mode only).</summary>
    public void Open(string title, bool pickFolder, string? initialPath, string? filterExtension, Action<string> onPicked)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        ArgumentNullException.ThrowIfNull(onPicked);

        _title = title;
        _pickFolder = pickFolder;
        _filterExtension = string.IsNullOrWhiteSpace(filterExtension) ? string.Empty : filterExtension.TrimStart('.');
        _onPicked = onPicked;
        _error = string.Empty;

        if (!string.IsNullOrWhiteSpace(initialPath) && Directory.Exists(initialPath))
        {
            _currentDirectory = initialPath;
            _fileName = string.Empty;
        }
        else if (!string.IsNullOrWhiteSpace(initialPath) && File.Exists(initialPath))
        {
            _currentDirectory = Path.GetDirectoryName(initialPath) ?? Directory.GetCurrentDirectory();
            _fileName = Path.GetFileName(initialPath);
        }
        else
        {
            _currentDirectory = Directory.GetCurrentDirectory();
            _fileName = string.Empty;
        }

        _openRequested = true;
    }

    public void Draw()
    {
        if (_openRequested)
        {
            _openRequested = false;
            _popupVisible = true;
            ImGui.OpenPopup(_title);
        }

        if (!_popupVisible)
            return;

        ImGui.SetNextWindowSize(new Vector2(680, 480), ImGuiCond.Appearing);
        if (!ImGui.BeginPopupModal(_title, ref _popupVisible, ImGuiWindowFlags.NoSavedSettings))
            return;

        if (ImGui.Button("Up"))
        {
            DirectoryInfo? parent = Directory.GetParent(_currentDirectory);
            if (parent is not null)
                _currentDirectory = parent.FullName;
        }

        ImGui.SameLine();
        ImGui.TextWrapped(_currentDirectory);

        if (!string.IsNullOrEmpty(_error))
        {
            ImGui.TextColored(new Vector4(1f, 0.4f, 0.4f, 1f), _error);
        }

        ImGui.Separator();

        ImGui.BeginChild("##PathPickerEntries", new Vector2(0, -GetFooterHeight()), border: false);

        try
        {
            foreach (string directory in Directory.EnumerateDirectories(_currentDirectory).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
            {
                if (ImGui.Selectable($"[dir]  {Path.GetFileName(directory)}"))
                {
                    _currentDirectory = directory;
                    _error = string.Empty;
                }
            }

            if (!_pickFolder)
            {
                foreach (string file in Directory.EnumerateFiles(_currentDirectory).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
                {
                    if (_filterExtension.Length > 0 && !Path.GetExtension(file).Equals("." + _filterExtension, StringComparison.OrdinalIgnoreCase))
                        continue;

                    bool isSelected = string.Equals(Path.GetFileName(file), _fileName, StringComparison.OrdinalIgnoreCase);
                    if (ImGui.Selectable($"        {Path.GetFileName(file)}", isSelected))
                        _fileName = Path.GetFileName(file);
                }
            }
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException or IOException or DirectoryNotFoundException)
        {
            _error = $"Cannot read '{_currentDirectory}': {ex.Message}";
        }

        ImGui.EndChild();

        if (!_pickFolder)
        {
            ImGui.SetNextItemWidth(-1f);
            ImGui.InputText("##PathPickerFileName", ref _fileName, 512);
        }

        float buttonsWidth = 160f;
        ImGui.SameLine(ImGui.GetWindowWidth() - buttonsWidth - ImGui.GetStyle().FramePadding.X);
        if (ImGui.Button("Cancel", new Vector2(buttonsWidth * 0.48f, 0)))
        {
            _popupVisible = false;
            ImGui.CloseCurrentPopup();
        }

        ImGui.SameLine();
        string confirmLabel = _pickFolder ? "Use folder" : "Open";
        if (ImGui.Button(confirmLabel, new Vector2(buttonsWidth * 0.52f, 0)))
        {
            string? picked = ResolveSelection();
            if (picked is not null)
            {
                _popupVisible = false;
                ImGui.CloseCurrentPopup();
                Action<string>? callback = _onPicked;
                _onPicked = null;
                callback?.Invoke(picked);
            }
        }

        ImGui.EndPopup();
    }

    private float GetFooterHeight()
        => _pickFolder ? ImGui.GetFrameHeightWithSpacing() : ImGui.GetFrameHeightWithSpacing() * 2f;

    private string? ResolveSelection()
    {
        if (_pickFolder)
        {
            if (!Directory.Exists(_currentDirectory))
            {
                _error = "Folder does not exist.";
                return null;
            }

            return _currentDirectory;
        }

        if (string.IsNullOrWhiteSpace(_fileName))
        {
            _error = "Enter a file name.";
            return null;
        }

        string candidate = Path.Combine(_currentDirectory, _fileName);
        if (!File.Exists(candidate))
        {
            _error = "File does not exist.";
            return null;
        }

        if (_filterExtension.Length > 0 && !Path.GetExtension(candidate).Equals("." + _filterExtension, StringComparison.OrdinalIgnoreCase))
        {
            _error = $"Expected a .{_filterExtension} file.";
            return null;
        }

        return candidate;
    }
}
