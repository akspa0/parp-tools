using System.Runtime.CompilerServices;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

/// <summary>
/// Decoded key frames of one M2 track, per payload (model bytes or an external <c>.anim</c>) and track slot
/// (sequence index, or 0 for a global sequence). Decoding is deterministic, so a cached array is exactly what
/// decoding again would produce; the samplers used to decode every key frame of every track on every call,
/// which made per-frame animation cost proportional to the key count of the whole model. Entries live as long
/// as their track definition (that is, the model). A null entry records a track that has no readable data.
/// </summary>
/// <remarks>
/// <typeparamref name="TFrame"/> is the caller's own key frame type, so each caller (and each track value
/// type) has its own table: a table is never shared between two decoders.
/// </remarks>
internal static class M2TrackKeyFrameCache<TTrack, TFrame>
{
    private sealed class Entry
    {
        public readonly Dictionary<(byte[] Payload, int TrackIndex), TFrame[]?> Frames = new();
    }

    private static readonly ConditionalWeakTable<M2TrackDefinition<TTrack>, Entry> Entries = new();

    public static bool TryGet(M2TrackDefinition<TTrack> track, byte[] payload, int trackIndex, out TFrame[]? frames)
    {
        Entry entry = Entries.GetValue(track, static _ => new Entry());
        lock (entry)
            return entry.Frames.TryGetValue((payload, trackIndex), out frames);
    }

    public static void Store(M2TrackDefinition<TTrack> track, byte[] payload, int trackIndex, TFrame[]? frames)
    {
        Entry entry = Entries.GetValue(track, static _ => new Entry());
        lock (entry)
            entry.Frames[(payload, trackIndex)] = frames;
    }
}
