using System.Diagnostics;
using System.Globalization;
using System.Text;
using ImGuiNET;

namespace WoWViewer;

/// <summary>
/// TEMPORARY DIAGNOSTIC (UI flicker hunt, 2026-11-04 — remove once the root cause is fixed):
/// enabled only when PARP_UI_OSCILLOSCOPE=1. Samples per-frame shell state, layout rects and
/// ImGui draw-data sizes, appending one line per observed change (plus a periodic heartbeat) to
/// ui-oscilloscope.log next to the viewer settings, so a per-frame visual toggle can be
/// identified from a single operator run. Draw-data alternation (A,B,A,B vertex counts between
/// consecutive frames) is logged explicitly — that is the signature of UI content strobing.
/// </summary>
internal sealed class UiOscilloscope : IDisposable
{
    private const int MaxChangeLines = 4000;
    private const int HeartbeatFrames = 300;

    private readonly StreamWriter _writer;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly StringBuilder _diff = new();
    private readonly StringBuilder _line = new();
    private int _frame;
    private int _changes;
    private bool _firstSample = true;
    private string? _lastAlternationSignature;

    // Last sampled state.
    private bool _hideUiChrome, _showLeft, _showRight, _leftDock, _rightDock, _loading, _fullscreenMinimap, _chatVisible, _hasScene;
    private int _leftPanel, _viewerPanel, _topTab, _bottomTab;
    private float _sceneX, _sceneY, _sceneW, _sceneH;
    private float _dockLX, _dockLY, _dockLW, _dockLH, _dockRX, _dockRY, _dockRW, _dockRH;
    private int _vtx = -1, _idx = -1, _lists = -1;
    private int _vtxPrev, _idxPrev;
    private float _fps = -1f, _dtMs = -1f;

    public UiOscilloscope()
    {
        string path = Path.Combine(ViewerApp.SettingsDir, "ui-oscilloscope.log");
        Directory.CreateDirectory(ViewerApp.SettingsDir);
        _writer = new StreamWriter(path, append: false, Encoding.UTF8) { AutoFlush = true };
        _writer.WriteLine($"# ui-oscilloscope start; change lines capped at {MaxChangeLines}; heartbeat every {HeartbeatFrames} frames");
        Console.WriteLine($"[UiOscilloscope] logging to {path}");
    }

    public void SampleState(bool hideUiChrome, bool showLeft, bool showRight, bool leftDock, bool rightDock,
        int leftPanel, int viewerPanel, int topTab, int bottomTab,
        bool loading, bool fullscreenMinimap, bool chatVisible,
        bool hasScene, float sceneX, float sceneY, float sceneW, float sceneH,
        UiRect leftDockRect, UiRect rightDockRect, float fps, float dtMs)
    {
        _frame++;
        _diff.Clear();
        TrackB(ref _hideUiChrome, hideUiChrome, "chrome");
        TrackB(ref _showLeft, showLeft, "leftBar");
        TrackB(ref _showRight, showRight, "rightBar");
        TrackB(ref _leftDock, leftDock, "leftDock");
        TrackB(ref _rightDock, rightDock, "rightDock");
        TrackI(ref _leftPanel, leftPanel, "leftPanel");
        TrackI(ref _viewerPanel, viewerPanel, "viewerPanel");
        TrackI(ref _topTab, topTab, "dest");
        TrackI(ref _bottomTab, bottomTab, "page");
        TrackB(ref _loading, loading, "loading");
        TrackB(ref _fullscreenMinimap, fullscreenMinimap, "fsMinimap");
        TrackB(ref _chatVisible, chatVisible, "chat");
        TrackB(ref _hasScene, hasScene, "hasScene");
        TrackF(ref _sceneX, sceneX, "sceneX");
        TrackF(ref _sceneY, sceneY, "sceneY");
        TrackF(ref _sceneW, sceneW, "sceneW");
        TrackF(ref _sceneH, sceneH, "sceneH");
        TrackF(ref _dockLX, leftDockRect.X, "dockLX");
        TrackF(ref _dockLY, leftDockRect.Y, "dockLY");
        TrackF(ref _dockLW, leftDockRect.Width, "dockLW");
        TrackF(ref _dockLH, leftDockRect.Height, "dockLH");
        TrackF(ref _dockRX, rightDockRect.X, "dockRX");
        TrackF(ref _dockRY, rightDockRect.Y, "dockRY");
        TrackF(ref _dockRW, rightDockRect.Width, "dockRW");
        TrackF(ref _dockRH, rightDockRect.Height, "dockRH");
        _fps = fps;
        _dtMs = dtMs;

        if (_firstSample)
        {
            _firstSample = false;
            Write("BASELINE");
        }
        else if (_diff.Length > 0 && _changes < MaxChangeLines)
        {
            _changes++;
            Write("CHANGED " + _diff.ToString().TrimEnd());
        }
        else if (_diff.Length > 0 && _changes == MaxChangeLines)
        {
            _changes++;
            Write("CHANGE CAP HIT — further changes suppressed");
        }

        if (_frame % HeartbeatFrames == 0)
            Write("HB");
    }

    public void SampleDrawData(int vtx, int idx, int lists)
    {
        // Alternation detector: A,B,A,B across consecutive frames = UI content strobing.
        if (_vtxPrev2 == vtx && MathF.Abs(_vtxPrev - vtx) > 2f)
            LogAlternation("vtx", _vtxPrev, vtx);
        if (_idxPrev2 == idx && MathF.Abs(_idxPrev - idx) > 2f)
            LogAlternation("idx", _idxPrev, idx);
        _vtxPrev2 = _vtxPrev;
        _idxPrev2 = _idxPrev;
        _vtxPrev = _vtx;
        _idxPrev = _idx;
        _vtx = vtx;
        _idx = idx;
        _lists = lists;
    }

    private int _vtxPrev2 = -1, _idxPrev2 = -1;

    private void LogAlternation(string name, int a, int b)
    {
        string signature = $"{name} {a}<->{b}";
        if (signature == _lastAlternationSignature) return;
        _lastAlternationSignature = signature;
        if (_changes < MaxChangeLines)
        {
            _changes++;
            Write($"ALTERNATING {signature}  (per-frame UI content toggle — strobe source)");
        }
    }

    private void TrackB(ref bool field, bool value, string name)
    {
        if (field == value) return;
        _diff.Append(name).Append(':').Append(field ? "1" : "0").Append("->").Append(value ? "1" : "0").Append(' ');
        field = value;
    }

    private void TrackI(ref int field, int value, string name)
    {
        if (field == value) return;
        _diff.Append(name).Append(':').Append(field.ToString(CultureInfo.InvariantCulture))
            .Append("->").Append(value.ToString(CultureInfo.InvariantCulture)).Append(' ');
        field = value;
    }

    private void TrackF(ref float field, float value, string name)
    {
        if (MathF.Abs(field - value) <= 0.5f) return;
        _diff.Append(name).Append(':').Append(field.ToString("0", CultureInfo.InvariantCulture))
            .Append("->").Append(value.ToString("0", CultureInfo.InvariantCulture)).Append(' ');
        field = value;
    }

    private void Write(string kind)
    {
        _line.Clear();
        _line.Append("f=").Append(_frame)
            .Append(" t=").Append((_clock.ElapsedMilliseconds / 1000.0).ToString("0.0", CultureInfo.InvariantCulture)).Append("s")
            .Append(" dt=").Append(_dtMs.ToString("0.0", CultureInfo.InvariantCulture))
            .Append(" fps=").Append(_fps.ToString("0.0", CultureInfo.InvariantCulture))
            .Append(' ').Append(kind);
        if (kind != "HB" || _changes == 0)
        {
            _line.Append(" | chrome=").Append(_hideUiChrome ? 1 : 0)
                .Append(" leftBar=").Append(_showLeft ? 1 : 0)
                .Append(" rightBar=").Append(_showRight ? 1 : 0)
                .Append(" leftDock=").Append(_leftDock ? 1 : 0)
                .Append(" rightDock=").Append(_rightDock ? 1 : 0)
                .Append(" leftPanel=").Append(_leftPanel)
                .Append(" viewerPanel=").Append(_viewerPanel)
                .Append(" dest=").Append(_topTab)
                .Append(" page=").Append(_bottomTab)
                .Append(" loading=").Append(_loading ? 1 : 0)
                .Append(" fsMinimap=").Append(_fullscreenMinimap ? 1 : 0)
                .Append(" chat=").Append(_chatVisible ? 1 : 0)
                .Append(" hasScene=").Append(_hasScene ? 1 : 0)
                .Append(" scene=(").Append(_sceneX.ToString("0")).Append(',').Append(_sceneY.ToString("0"))
                .Append(',').Append(_sceneW.ToString("0")).Append(',').Append(_sceneH.ToString("0")).Append(')')
                .Append(" dockL=(").Append(_dockLX.ToString("0")).Append(',').Append(_dockLY.ToString("0"))
                .Append(',').Append(_dockLW.ToString("0")).Append(',').Append(_dockLH.ToString("0")).Append(')')
                .Append(" dockR=(").Append(_dockRX.ToString("0")).Append(',').Append(_dockRY.ToString("0"))
                .Append(',').Append(_dockRW.ToString("0")).Append(',').Append(_dockRH.ToString("0")).Append(')');
        }
        _line.Append(" vtx=").Append(_vtx).Append(" idx=").Append(_idx).Append(" lists=").Append(_lists);
        _writer.WriteLine(_line.ToString());
    }

    public void Dispose() => _writer.Dispose();
}
