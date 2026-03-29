namespace WowViewer.Core.Mdx;

public sealed class MdxTextureAnimation
{
    public MdxTextureAnimation(int index, MdxVector3NodeTrack? translationTrack, MdxQuaternionNodeTrack? rotationTrack, MdxVector3NodeTrack? scalingTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
    }

    public int Index { get; }

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public bool HasTranslationTrack => TranslationTrack is not null;

    public bool HasRotationTrack => RotationTrack is not null;

    public bool HasScalingTrack => ScalingTrack is not null;
}