using System.Numerics;
using ImGuiNET;

namespace WoWViewer;

/// <summary>One root for shell popup requests, close callbacks, escape and modal input ownership.</summary>
internal sealed class UiPopupRegistry
{
    private sealed record Popup(Func<bool> Draw, Action? Closed, Vector2 Size);
    private readonly Dictionary<string, Popup> _providers = new();
    private readonly Queue<string> _requests = new();
    private readonly List<string> _active = new();
    public bool IsOpen => _active.Count != 0 || _requests.Count != 0;

    public void Register(string id, Func<bool> draw, Action? closed = null, Vector2 size = default) =>
        _providers[id] = new(draw, closed, size == default ? new(600, 480) : size);
    public void Request(string id)
    {
        if (!_providers.ContainsKey(id)) throw new ArgumentOutOfRangeException(nameof(id), id, "Unregistered shell popup");
        if (!_active.Contains(id) && !_requests.Contains(id)) _requests.Enqueue(id);
    }

    public void CloseTopmost()
    {
        if (_active.Count > 0) CloseFrom(_active.Count - 1);
        else _requests.Clear();
    }

    public void Draw()
    {
        // Popups need a current window for their ID stack. Without this host, OpenPopup and
        // BeginPopupModal run outside any window and the modal never appears.
        ImGui.SetNextWindowPos(Vector2.Zero);
        ImGui.SetNextWindowSize(ImGui.GetIO().DisplaySize);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        bool hostOpen = ImGui.Begin("##UiPopupHost",
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoDocking |
            ImGuiWindowFlags.NoBackground | ImGuiWindowFlags.NoBringToFrontOnFocus |
            ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);

        // Nest queued dialogs instead of replacing an active modal and losing its close callback.
        if (_active.Count == 0 && _requests.TryDequeue(out string? first))
        {
            _active.Add(first);
            ImGui.OpenPopup(first);
        }
        DrawLevel(0);

        if (hostOpen) ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawLevel(int level)
    {
        if (level >= _active.Count) return;
        string id = _active[level];
        Popup provider = _providers[id];
        var display = ImGui.GetIO().DisplaySize;
        ImGui.SetNextWindowSize(Vector2.Min(provider.Size, Vector2.Max(new(1, 1), display)), ImGuiCond.Appearing);
        ImGui.SetNextWindowPos(display / 2, ImGuiCond.Appearing, new(0.5f));
        bool open = true;
        if (ImGui.BeginPopupModal(id, ref open, ImGuiWindowFlags.NoSavedSettings))
        {
            bool close = provider.Draw();
            bool top = level == _active.Count - 1;
            if (top && !ImGui.GetIO().WantTextInput && ImGui.IsKeyPressed(ImGuiKey.Escape)) close = true;
            if (close || !open)
            {
                ImGui.CloseCurrentPopup();
                CloseFrom(level);
            }
            else
            {
                if (top && _requests.TryDequeue(out string? next))
                {
                    _active.Add(next);
                    ImGui.OpenPopup(next);
                }
                DrawLevel(level + 1);
            }
            ImGui.EndPopup();
        }
        else if (!open || !ImGui.IsPopupOpen(id)) CloseFrom(level);
    }

    private void CloseFrom(int level)
    {
        for (int i = _active.Count - 1; i >= level; i--)
        {
            string id = _active[i];
            _active.RemoveAt(i);
            _providers[id].Closed?.Invoke();
        }
    }
}
