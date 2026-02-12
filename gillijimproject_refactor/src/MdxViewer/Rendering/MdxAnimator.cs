using System.Numerics;
using MdxLTool.Formats.Mdx;

namespace MdxViewer.Rendering;

/// <summary>
/// MDX skeletal animation engine.
/// Evaluates bone hierarchy transforms per-frame using keyframe interpolation.
/// Based on the wow-mdx-viewer reference implementation.
/// </summary>
public class MdxAnimator
{
    private readonly MdxFile _mdx;
    private readonly Matrix4x4[] _boneMatrices;
    private readonly Dictionary<int, MdlBone> _bonesByObjectId = new();
    private readonly Dictionary<int, List<int>> _childrenByParent = new();
    private readonly List<int> _rootBoneIds = new();

    private int _sequenceIndex;
    private float _currentFrame;
    private float[] _globalSeqFrames;

    /// <summary>Number of bones in the skeleton</summary>
    public int BoneCount => _mdx.Bones.Count;

    /// <summary>True if the model has any bones with animation tracks</summary>
    public bool HasAnimation { get; }

    /// <summary>Current bone matrices (indexed by bone list position, not ObjectId)</summary>
    public Matrix4x4[] BoneMatrices => _boneMatrices;

    /// <summary>Available animation sequences</summary>
    public IReadOnlyList<MdlSequence> Sequences => _mdx.Sequences;

    /// <summary>Current sequence index</summary>
    public int CurrentSequence => _sequenceIndex;

    /// <summary>Current frame within the sequence</summary>
    public float CurrentFrame => _currentFrame;

    public MdxAnimator(MdxFile mdx)
    {
        _mdx = mdx;
        _boneMatrices = new Matrix4x4[Math.Max(mdx.Bones.Count, 1)];

        // Index bones by ObjectId and build parent-child hierarchy
        foreach (var bone in mdx.Bones)
        {
            _bonesByObjectId[bone.ObjectId] = bone;
        }

        foreach (var bone in mdx.Bones)
        {
            if (bone.ParentId < 0 || !_bonesByObjectId.ContainsKey(bone.ParentId))
            {
                _rootBoneIds.Add(bone.ObjectId);
            }
            else
            {
                if (!_childrenByParent.TryGetValue(bone.ParentId, out var children))
                {
                    children = new List<int>();
                    _childrenByParent[bone.ParentId] = children;
                }
                children.Add(bone.ObjectId);
            }
        }

        // Initialize global sequence frames
        _globalSeqFrames = new float[mdx.GlobalSequences.Count];

        // Check if any bones have animation tracks
        HasAnimation = mdx.Bones.Any(b =>
            b.TranslationTrack?.Keys.Count > 0 ||
            b.RotationTrack?.Keys.Count > 0 ||
            b.ScalingTrack?.Keys.Count > 0);

        // Initialize to identity
        for (int i = 0; i < _boneMatrices.Length; i++)
            _boneMatrices[i] = Matrix4x4.Identity;

        // Set first sequence if available
        if (mdx.Sequences.Count > 0)
            SetSequence(0);
    }

    /// <summary>Set the active animation sequence</summary>
    public void SetSequence(int index)
    {
        if (index < 0 || index >= _mdx.Sequences.Count) return;
        _sequenceIndex = index;
        _currentFrame = _mdx.Sequences[index].Time.Start;
    }

    /// <summary>Advance animation by deltaMs milliseconds and recompute bone matrices</summary>
    public void Update(float deltaMs)
    {
        if (!HasAnimation || _mdx.Sequences.Count == 0) return;

        var seq = _mdx.Sequences[_sequenceIndex];
        _currentFrame += deltaMs;

        // Loop animation
        if (_currentFrame > seq.Time.End)
        {
            float duration = seq.Time.End - seq.Time.Start;
            if (duration > 0)
                _currentFrame = seq.Time.Start + ((_currentFrame - seq.Time.Start) % duration);
            else
                _currentFrame = seq.Time.Start;
        }

        // Update global sequences
        for (int i = 0; i < _globalSeqFrames.Length; i++)
        {
            _globalSeqFrames[i] += deltaMs;
            float gsDuration = _mdx.GlobalSequences[i];
            if (gsDuration > 0 && _globalSeqFrames[i] > gsDuration)
                _globalSeqFrames[i] %= gsDuration;
        }

        // Traverse bone hierarchy from roots
        foreach (int rootId in _rootBoneIds)
            UpdateBone(rootId, Matrix4x4.Identity);
    }

    private void UpdateBone(int objectId, Matrix4x4 parentMatrix)
    {
        if (!_bonesByObjectId.TryGetValue(objectId, out var bone)) return;

        var pivot = new Vector3(bone.Pivot.X, bone.Pivot.Y, bone.Pivot.Z);

        // Evaluate animation tracks
        var translation = EvalVec3Track(bone.TranslationTrack);
        var rotation = EvalQuatTrack(bone.RotationTrack);
        var scaling = EvalVec3Track(bone.ScalingTrack);

        bool hasT = translation.HasValue;
        bool hasR = rotation.HasValue;
        bool hasS = scaling.HasValue;

        Matrix4x4 localMatrix;

        if (!hasT && !hasR && !hasS)
        {
            localMatrix = Matrix4x4.Identity;
        }
        else
        {
            // Build local transform around pivot point (same as reference viewer)
            // M = T(-pivot) * S * R * T(pivot) * T(translation)
            var t = hasT ? translation.Value : Vector3.Zero;
            var r = hasR ? rotation.Value : Quaternion.Identity;
            var s = hasS ? scaling.Value : Vector3.One;

            localMatrix = Matrix4x4.CreateTranslation(-pivot)
                        * Matrix4x4.CreateScale(s)
                        * Matrix4x4.CreateFromQuaternion(r)
                        * Matrix4x4.CreateTranslation(pivot)
                        * Matrix4x4.CreateTranslation(t);
        }

        // Combine with parent
        var worldMatrix = localMatrix * parentMatrix;

        // Store in bone matrices array (indexed by bone list position)
        int boneIndex = _mdx.Bones.IndexOf(bone);
        if (boneIndex >= 0 && boneIndex < _boneMatrices.Length)
            _boneMatrices[boneIndex] = worldMatrix;

        // Recurse to children
        if (_childrenByParent.TryGetValue(objectId, out var children))
        {
            foreach (int childId in children)
                UpdateBone(childId, worldMatrix);
        }
    }

    private Vector3? EvalVec3Track(MdlAnimTrack<C3Vector>? track)
    {
        if (track == null || track.Keys.Count == 0) return null;

        float frame;
        int from, to;
        GetFrameRange(track.GlobalSeqId, out frame, out from, out to);

        var keys = track.Keys;

        // Find surrounding keyframes using binary search
        var (left, right) = FindKeyframes(keys, frame, from, to);
        if (left == null) return null;

        var lv = left.Value;
        if (left == right || left.Frame == right!.Frame)
            return new Vector3(lv.X, lv.Y, lv.Z);

        float t = (float)(frame - left.Frame) / (right.Frame - left.Frame);
        t = Math.Clamp(t, 0f, 1f);

        var rv = right.Value;

        switch (track.InterpolationType)
        {
            case MdlTrackType.NoInterp:
                return new Vector3(lv.X, lv.Y, lv.Z);

            case MdlTrackType.Hermite:
            {
                var lo = left.OutTan;
                var ri = right.InTan;
                return new Vector3(
                    Hermite(lv.X, lo.X, ri.X, rv.X, t),
                    Hermite(lv.Y, lo.Y, ri.Y, rv.Y, t),
                    Hermite(lv.Z, lo.Z, ri.Z, rv.Z, t));
            }

            case MdlTrackType.Bezier:
            {
                var lo = left.OutTan;
                var ri = right.InTan;
                return new Vector3(
                    Bezier(lv.X, lo.X, ri.X, rv.X, t),
                    Bezier(lv.Y, lo.Y, ri.Y, rv.Y, t),
                    Bezier(lv.Z, lo.Z, ri.Z, rv.Z, t));
            }

            default: // Linear
                return Vector3.Lerp(new Vector3(lv.X, lv.Y, lv.Z), new Vector3(rv.X, rv.Y, rv.Z), t);
        }
    }

    private Quaternion? EvalQuatTrack(MdlAnimTrack<C4Quaternion>? track)
    {
        if (track == null || track.Keys.Count == 0) return null;

        float frame;
        int from, to;
        GetFrameRange(track.GlobalSeqId, out frame, out from, out to);

        var keys = track.Keys;
        var (left, right) = FindKeyframes(keys, frame, from, to);
        if (left == null) return null;

        var lv = left.Value;
        if (left == right || left.Frame == right!.Frame)
            return new Quaternion(lv.X, lv.Y, lv.Z, lv.W);

        float t = (float)(frame - left.Frame) / (right.Frame - left.Frame);
        t = Math.Clamp(t, 0f, 1f);

        var rv = right.Value;
        var lq = new Quaternion(lv.X, lv.Y, lv.Z, lv.W);
        var rq = new Quaternion(rv.X, rv.Y, rv.Z, rv.W);

        // For hermite/bezier quaternion interpolation, use slerp (simplified)
        // Full sqlerp would need tangent quaternions, but slerp is sufficient for most cases
        return Quaternion.Slerp(lq, rq, t);
    }

    private void GetFrameRange(int globalSeqId, out float frame, out int from, out int to)
    {
        if (globalSeqId >= 0 && globalSeqId < _globalSeqFrames.Length)
        {
            frame = _globalSeqFrames[globalSeqId];
            from = 0;
            to = (int)_mdx.GlobalSequences[globalSeqId];
        }
        else
        {
            frame = _currentFrame;
            var seq = _mdx.Sequences[_sequenceIndex];
            from = seq.Time.Start;
            to = seq.Time.End;
        }
    }

    private static (MdlTrackKey<T>? left, MdlTrackKey<T>? right) FindKeyframes<T>(
        List<MdlTrackKey<T>> keys, float frame, int from, int to)
    {
        if (keys.Count == 0) return (null, null);
        if (keys[0].Frame > to) return (null, null);
        if (keys[^1].Frame < from) return (null, null);

        // Binary search for the first key > frame
        int first = 0, count = keys.Count;
        while (count > 0)
        {
            int step = count >> 1;
            if (keys[first + step].Frame <= frame)
            {
                first = first + step + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }

        if (first == keys.Count || keys[first].Frame > to)
        {
            if (first > 0 && keys[first - 1].Frame >= from)
                return (keys[first - 1], keys[first - 1]);
            return (null, null);
        }

        if (first == 0 || keys[first - 1].Frame < from)
        {
            if (keys[first].Frame <= to)
                return (keys[first], keys[first]);
            return (null, null);
        }

        return (keys[first - 1], keys[first]);
    }

    private static float Hermite(float a, float aOutTan, float bInTan, float b, float t)
    {
        float t2 = t * t;
        float f1 = t2 * (2 * t - 3) + 1;
        float f2 = t2 * (t - 2) + t;
        float f3 = t2 * (t - 1);
        float f4 = t2 * (3 - 2 * t);
        return a * f1 + aOutTan * f2 + bInTan * f3 + b * f4;
    }

    private static float Bezier(float a, float aOutTan, float bInTan, float b, float t)
    {
        float inv = 1 - t;
        float inv2 = inv * inv;
        float t2 = t * t;
        return a * (inv2 * inv) + aOutTan * (3 * t * inv2) + bInTan * (3 * t2 * inv) + b * (t2 * t);
    }
}
