using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Runtime.World.Visibility;

public readonly record struct WorldVisibleMdxEntry(
    WorldObjectInstance Instance,
    float CenterDistanceSq,
    float OpaqueFade,
    float TransparentFade,
    bool IsTaxiActor);