using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Runtime.World.Visibility;

public readonly record struct WorldVisibleWmoEntry(WorldObjectInstance Instance, float CenterDistanceSq);