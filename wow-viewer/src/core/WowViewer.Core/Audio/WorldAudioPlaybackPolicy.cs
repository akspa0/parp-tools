namespace WowViewer.Core.Audio;

/// <summary>
/// Explicit safety policy for world-level audio routes whose client hand-off is
/// not yet proven. Resident MCNK/MCSE emitters remain independently controllable.
/// </summary>
public static class WorldAudioPlaybackPolicy
{
    /// <summary>
    /// ZoneMusic resolution remains diagnostic-only until its client-era
    /// mapping and playback contract are proven.
    /// </summary>
    public const bool AutomaticZoneMusicPlaybackEnabled = false;
}
