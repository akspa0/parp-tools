using System.Numerics;
using ImGuiNET;

namespace WoWViewer;

public enum ImGuiPathPickerMode
{
    OpenFolder,
    OpenFile,
    SaveFile
}

/// <summary>
/// In-app ImGui file/folder picker built exclusively on BCL filesystem APIs, so browsing behaves
/// identically on every platform — no WinForms or native dialogs anywhere in cross-platform builds.
/// Supports drive shortcuts, editable path bar, multi-extension filters, search filter, new folder creation,
/// and both Open and Save modes.
/// </summary>
internal sealed class ImGuiPathPicker
{
    public static ImGuiPathPicker Instance { get; } = new();

    private bool _openRequested;
    private bool _popupVisible;
    private string _title = "Select path";
    private ImGuiPathPickerMode _mode = ImGuiPathPickerMode.OpenFolder;
    private string[] _filterExtensions = [];
    private Action<string>? _onPicked;
    private string _currentDirectory = Directory.GetCurrentDirectory();
    private string _pathInputBuffer = string.Empty;
    private string _fileName = string.Empty;
    private string _searchFilter = string.Empty;
    private string _error = string.Empty;
    private bool _showNewFolderInput = false;
    private string _newFolderBuffer = string.Empty;

    private ImGuiPathPicker()
    {
    }

    /// <summary>Opens the picker in Folder or File Open mode.</summary>
    public void Open(string title, bool pickFolder, string? initialPath, string? filterExtension, Action<string> onPicked)
    {
        Open(title, pickFolder ? ImGuiPathPickerMode.OpenFolder : ImGuiPathPickerMode.OpenFile, initialPath, filterExtension, onPicked);
    }

    /// <summary>Opens the picker in a specified mode (OpenFolder, OpenFile, SaveFile).</summary>
    public void Open(string title, ImGuiPathPickerMode mode, string? initialPath, string? filterExtension, Action<string> onPicked, string? defaultFileName = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        ArgumentNullException.ThrowIfNull(onPicked);

        _title = title;
        _mode = mode;
        _onPicked = onPicked;
        _error = string.Empty;
        _searchFilter = string.Empty;
        _showNewFolderInput = false;
        _newFolderBuffer = string.Empty;

        // Parse filter extensions (e.g. ".pm4;.pd4" or ".wdt|.mpq" or "*.json")
        if (string.IsNullOrWhiteSpace(filterExtension) || filterExtension.Contains("*.*"))
        {
            _filterExtensions = [];
        }
        else
        {
            _filterExtensions = filterExtension
                .Split(['|', ';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(ext => ext.TrimStart('*').TrimStart('.'))
                .Where(ext => !string.IsNullOrEmpty(ext))
                .ToArray();
        }

        if (!string.IsNullOrWhiteSpace(initialPath) && Directory.Exists(initialPath))
        {
            _currentDirectory = Path.GetFullPath(initialPath);
            _fileName = defaultFileName ?? string.Empty;
        }
        else if (!string.IsNullOrWhiteSpace(initialPath) && File.Exists(initialPath))
        {
            _currentDirectory = Path.GetDirectoryName(Path.GetFullPath(initialPath)) ?? Directory.GetCurrentDirectory();
            _fileName = Path.GetFileName(initialPath);
        }
        else
        {
            _currentDirectory = Directory.GetCurrentDirectory();
            _fileName = defaultFileName ?? string.Empty;
        }

        _pathInputBuffer = _currentDirectory;
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

        ImGui.SetNextWindowSize(new Vector2(740, 520), ImGuiCond.Appearing);
        if (!ImGui.BeginPopupModal(_title, ref _popupVisible, ImGuiWindowFlags.NoSavedSettings))
            return;

        // 1. Drive & Shortcut Bar
        DrawDriveShortcuts();

        // 2. Navigation bar: Up button, Path Input, and Go
        if (ImGui.Button("Up (..)") || ImGui.IsKeyPressed(ImGuiKey.Backspace) && !ImGui.IsAnyItemActive())
        {
            DirectoryInfo? parent = Directory.GetParent(_currentDirectory);
            if (parent is not null && Directory.Exists(parent.FullName))
            {
                _currentDirectory = parent.FullName;
                _pathInputBuffer = _currentDirectory;
                _error = string.Empty;
            }
        }

        ImGui.SameLine();
        ImGui.SetNextItemWidth(-80f);
        if (ImGui.InputText("##PathBar", ref _pathInputBuffer, 512, ImGuiInputTextFlags.EnterReturnsTrue))
        {
            NavigateToInputPath();
        }

        ImGui.SameLine();
        if (ImGui.Button("Go", new Vector2(70f, 0)))
        {
            NavigateToInputPath();
        }

        // Search in directory & New Folder button
        ImGui.SetNextItemWidth(240f);
        ImGui.InputTextWithHint("##SearchFilter", "Filter current directory...", ref _searchFilter, 64);

        if (_mode == ImGuiPathPickerMode.OpenFolder || _mode == ImGuiPathPickerMode.SaveFile)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("+ New Folder"))
            {
                _showNewFolderInput = !_showNewFolderInput;
                _newFolderBuffer = "NewFolder";
            }
        }

        if (_showNewFolderInput)
        {
            ImGui.SameLine();
            ImGui.SetNextItemWidth(160f);
            ImGui.InputText("##NewFolderName", ref _newFolderBuffer, 64);
            ImGui.SameLine();
            if (ImGui.SmallButton("Create"))
            {
                try
                {
                    string targetNew = Path.Combine(_currentDirectory, _newFolderBuffer.Trim());
                    if (!Directory.Exists(targetNew))
                    {
                        Directory.CreateDirectory(targetNew);
                        _currentDirectory = targetNew;
                        _pathInputBuffer = targetNew;
                    }
                    _showNewFolderInput = false;
                }
                catch (Exception ex)
                {
                    _error = $"Cannot create folder: {ex.Message}";
                }
            }
        }

        if (!string.IsNullOrEmpty(_error))
        {
            ImGui.TextColored(new Vector4(1f, 0.4f, 0.4f, 1f), _error);
        }

        ImGui.Separator();

        // 3. Entries List
        ImGui.BeginChild("##PathPickerEntries", new Vector2(0, -GetFooterHeight()), border: true);

        try
        {
            if (Directory.Exists(_currentDirectory))
            {
                // Directories
                foreach (string directory in Directory.EnumerateDirectories(_currentDirectory).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
                {
                    string dirName = Path.GetFileName(directory);
                    if (!string.IsNullOrWhiteSpace(_searchFilter) && !dirName.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase))
                        continue;

                    if (ImGui.Selectable($"[dir]  {dirName}"))
                    {
                        _currentDirectory = directory;
                        _pathInputBuffer = directory;
                        _error = string.Empty;
                    }
                }

                // Files (when not picking folders)
                if (_mode != ImGuiPathPickerMode.OpenFolder)
                {
                    foreach (string file in Directory.EnumerateFiles(_currentDirectory).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
                    {
                        string fileName = Path.GetFileName(file);
                        if (!string.IsNullOrWhiteSpace(_searchFilter) && !fileName.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase))
                            continue;

                        if (_filterExtensions.Length > 0)
                        {
                            string fileExt = Path.GetExtension(file).TrimStart('.');
                            if (!_filterExtensions.Any(ext => ext.Equals(fileExt, StringComparison.OrdinalIgnoreCase)))
                                continue;
                        }

                        bool isSelected = string.Equals(fileName, _fileName, StringComparison.OrdinalIgnoreCase);
                        if (ImGui.Selectable($"       {fileName}", isSelected))
                            _fileName = fileName;
                    }
                }
            }
            else
            {
                ImGui.TextDisabled("Directory does not exist.");
            }
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException or IOException or DirectoryNotFoundException)
        {
            _error = $"Cannot read '{_currentDirectory}': {ex.Message}";
        }

        ImGui.EndChild();
        ImGui.Spacing();

        // 4. Footer File Name Input & Action Buttons
        if (_mode != ImGuiPathPickerMode.OpenFolder)
        {
            ImGui.Text("File name:");
            ImGui.SameLine();
            ImGui.SetNextItemWidth(MathF.Max(200f, ImGui.GetContentRegionAvail().X - 230f));
            ImGui.InputText("##PathPickerFileName", ref _fileName, 512);
            ImGui.Spacing();
        }

        float buttonsWidth = 220f;
        float availWidth = ImGui.GetContentRegionAvail().X;
        if (availWidth > buttonsWidth)
        {
            ImGui.SetCursorPosX(ImGui.GetCursorPosX() + availWidth - buttonsWidth);
        }

        if (ImGui.Button("Cancel", new Vector2(105f, 26f)))
        {
            _popupVisible = false;
            ImGui.CloseCurrentPopup();
        }

        ImGui.SameLine();
        string confirmLabel = _mode switch
        {
            ImGuiPathPickerMode.OpenFolder => "Use Folder",
            ImGuiPathPickerMode.SaveFile => "Save",
            _ => "Open"
        };

        if (ImGui.Button(confirmLabel, new Vector2(105f, 26f)))
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

    private void DrawDriveShortcuts()
    {
        ImGui.TextDisabled("Drives & Shortcuts:");
        ImGui.SameLine();

        try
        {
            DriveInfo[] drives = DriveInfo.GetDrives();
            foreach (DriveInfo drive in drives)
            {
                if (drive.IsReady)
                {
                    string driveLabel = drive.Name.TrimEnd('\\');
                    if (ImGui.SmallButton(driveLabel))
                    {
                        _currentDirectory = drive.RootDirectory.FullName;
                        _pathInputBuffer = _currentDirectory;
                        _error = string.Empty;
                    }
                    ImGui.SameLine();
                }
            }
        }
        catch
        {
            // Ignore drive enumeration failures on restricted environments
        }

        // Common shortcuts if available
        if (Directory.Exists(@"H:\CLIENTS"))
        {
            if (ImGui.SmallButton("CLIENTS (H:)"))
            {
                _currentDirectory = @"H:\CLIENTS";
                _pathInputBuffer = _currentDirectory;
                _error = string.Empty;
            }
            ImGui.SameLine();
        }

        string appBase = AppContext.BaseDirectory;
        if (ImGui.SmallButton("App Directory"))
        {
            _currentDirectory = appBase;
            _pathInputBuffer = _currentDirectory;
            _error = string.Empty;
        }

        ImGui.NewLine();
    }

    private void NavigateToInputPath()
    {
        string target = _pathInputBuffer.Trim();
        if (Directory.Exists(target))
        {
            _currentDirectory = Path.GetFullPath(target);
            _pathInputBuffer = _currentDirectory;
            _error = string.Empty;
        }
        else if (File.Exists(target))
        {
            _currentDirectory = Path.GetDirectoryName(Path.GetFullPath(target)) ?? _currentDirectory;
            _fileName = Path.GetFileName(target);
            _pathInputBuffer = _currentDirectory;
            _error = string.Empty;
        }
        else
        {
            _error = $"Path '{target}' does not exist.";
        }
    }

    private float GetFooterHeight()
        => _mode == ImGuiPathPickerMode.OpenFolder ? 45f : 80f;

    private string? ResolveSelection()
    {
        if (_mode == ImGuiPathPickerMode.OpenFolder)
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

        string candidate = Path.Combine(_currentDirectory, _fileName.Trim());

        if (_mode == ImGuiPathPickerMode.OpenFile)
        {
            if (!File.Exists(candidate))
            {
                _error = "File does not exist.";
                return null;
            }

            if (_filterExtensions.Length > 0)
            {
                string fileExt = Path.GetExtension(candidate).TrimStart('.');
                if (!_filterExtensions.Any(ext => ext.Equals(fileExt, StringComparison.OrdinalIgnoreCase)))
                {
                    _error = $"Expected a file with extension: {string.Join(", ", _filterExtensions.Select(e => "." + e))}";
                    return null;
                }
            }
        }
        else if (_mode == ImGuiPathPickerMode.SaveFile)
        {
            // If saving and no extension was entered, append first filter extension
            if (string.IsNullOrEmpty(Path.GetExtension(candidate)) && _filterExtensions.Length > 0)
            {
                candidate += "." + _filterExtensions[0];
            }
        }

        return candidate;
    }
}
