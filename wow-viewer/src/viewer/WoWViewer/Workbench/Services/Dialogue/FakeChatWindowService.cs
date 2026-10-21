using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Population;

namespace WoWViewer.Workbench.Services.Dialogue;

/// <summary>
/// Renders an authentic World of Warcraft-style chat window overlay in the viewport.
/// Displays ambient NPC barks, shouts, whispers, and system notifications with authentic channel colors.
/// </summary>
public sealed class FakeChatWindowService
{
    public sealed record ChatEntry(
        DateTime Timestamp,
        string Sender,
        string Text,
        DialogueChatType Type,
        Vector4 Color);

    private readonly List<ChatEntry> _messages = new(200);
    private bool _scrollToBottom = true;

    public bool IsVisible { get; set; } = true;
    public bool AutoScroll { get; set; } = true;
    public int MaxHistory { get; set; } = 150;

    // Authentic WoW channel colors
    public static readonly Vector4 ColorSay = new(0.92f, 0.92f, 0.92f, 1.0f);        // White / Pale Gray
    public static readonly Vector4 ColorYell = new(1.0f, 0.25f, 0.25f, 1.0f);        // Red
    public static readonly Vector4 ColorEmote = new(1.0f, 0.50f, 0.0f, 1.0f);        // Orange
    public static readonly Vector4 ColorWhisper = new(1.0f, 0.50f, 0.90f, 1.0f);      // Pink
    public static readonly Vector4 ColorSystem = new(1.0f, 0.85f, 0.0f, 1.0f);       // Yellow
    public static readonly Vector4 ColorTimestamp = new(0.55f, 0.55f, 0.55f, 0.8f);   // Muted Gray

    public void Clear()
    {
        _messages.Clear();
    }

    public void PostNpcBark(string sender, string text, DialogueChatType type)
    {
        Vector4 color = type switch
        {
            DialogueChatType.Yell => ColorYell,
            DialogueChatType.Emote or DialogueChatType.BossEmote => ColorEmote,
            DialogueChatType.Whisper => ColorWhisper,
            DialogueChatType.System => ColorSystem,
            _ => ColorSay
        };

        if (_messages.Count >= MaxHistory)
            _messages.RemoveAt(0);

        _messages.Add(new ChatEntry(DateTime.Now, sender, text, type, color));
        _scrollToBottom = true;
    }

    public void PostSystemMessage(string text)
    {
        if (_messages.Count >= MaxHistory)
            _messages.RemoveAt(0);

        _messages.Add(new ChatEntry(DateTime.Now, "System", text, DialogueChatType.System, ColorSystem));
        _scrollToBottom = true;
    }

    /// <summary>
    /// Renders the chat body into the manager-owned chat region. The manager owns the
    /// window, position, size, border and draw order.
    /// </summary>
    public void DrawContent(Action? drawPopulationControls = null)
    {
        if (!IsVisible)
            return;

        ImGui.TextColored(new Vector4(0.79f, 0.64f, 0.15f, 1f), "[General]");
        ImGui.SameLine();
        ImGui.TextDisabled("|");
        ImGui.SameLine();
        if (ImGui.SmallButton("Clear##ChatClear"))
            Clear();
        ImGui.SameLine();
        bool auto = AutoScroll;
        if (ImGui.Checkbox("Auto-Scroll##ChatAuto", ref auto))
            AutoScroll = auto;
        ImGui.SameLine();
        if (ImGui.SmallButton("Hide##ChatHide"))
            IsVisible = false;

        drawPopulationControls?.Invoke();

        ImGui.Separator();

        if (ImGui.BeginChild("##ChatLogScrollRegion", new Vector2(0f, 0f), false, ImGuiWindowFlags.AlwaysVerticalScrollbar))
        {
            if (_messages.Count == 0)
            {
                ImGui.TextDisabled("Ambient dialogue and NPC barks appear here when within range...");
            }
            else
            {
                ImGui.PushTextWrapPos(0.0f);
                for (int i = 0; i < _messages.Count; i++)
                {
                    var msg = _messages[i];
                    string timeStr = $"[{msg.Timestamp:HH:mm:ss}] ";

                    ImGui.TextColored(ColorTimestamp, timeStr);
                    ImGui.SameLine(0f, 0f);

                    switch (msg.Type)
                    {
                        case DialogueChatType.Yell:
                            ImGui.TextColored(msg.Color, $"[{msg.Sender}] yells: {msg.Text}");
                            break;
                        case DialogueChatType.Emote or DialogueChatType.BossEmote:
                            ImGui.TextColored(msg.Color, $"{msg.Sender} {msg.Text}");
                            break;
                        case DialogueChatType.Whisper:
                            ImGui.TextColored(msg.Color, $"[{msg.Sender}] whispers: {msg.Text}");
                            break;
                        case DialogueChatType.System:
                            ImGui.TextColored(msg.Color, $"{msg.Text}");
                            break;
                        default:
                            ImGui.TextColored(msg.Color, $"[{msg.Sender}] says: {msg.Text}");
                            break;
                    }
                }
                ImGui.PopTextWrapPos();
            }

            if (_scrollToBottom && AutoScroll)
            {
                ImGui.SetScrollHereY(1.0f);
                _scrollToBottom = false;
            }

            ImGui.EndChild();
        }
    }
}
