using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

public sealed class ManualCurationProjectPolicyTests
{
    [Fact]
    public void TryAddAssociation_AddsExplicitLink_AndRejectsDuplicate()
    {
        ManualCurationProject project = CreateProject(out ManualObjectPlacement placement);
        Pm4HullIdentity hull = new(12, 20, 0x401234, 3, 42);

        ManualAssociationResult first = ManualCurationProjectPolicy.TryAddAssociation(
            project, placement.PlacementId, hull, DateTimeOffset.UnixEpoch);
        ManualAssociationResult duplicate = ManualCurationProjectPolicy.TryAddAssociation(
            project, placement.PlacementId, hull, DateTimeOffset.UnixEpoch);

        Assert.Equal(ManualAssociationResult.Added, first);
        Assert.Equal(ManualAssociationResult.AlreadyExists, duplicate);
        Assert.Single(project.Associations);
        Assert.Equal(placement.PlacementId, project.Associations[0].PlacementId);
        Assert.Equal(hull, project.Associations[0].Hull);
    }

    [Fact]
    public void DistinctPlacementsMayShareHull()
    {
        ManualCurationProject project = CreateProject(out ManualObjectPlacement first);
        ManualObjectPlacement second = CreatePlacement(Guid.NewGuid(), uniqueId: 18);
        Assert.True(ManualCurationProjectPolicy.TryAddPlacement(project, second));
        Pm4HullIdentity hull = new(8, 9, 0x401111, 1);

        Assert.Equal(ManualAssociationResult.Added,
            ManualCurationProjectPolicy.TryAddAssociation(project, first.PlacementId, hull));
        Assert.Equal(ManualAssociationResult.Added,
            ManualCurationProjectPolicy.TryAddAssociation(project, second.PlacementId, hull));
        Assert.Equal(2, project.Associations.Count);
    }

    [Fact]
    public void ReplaceAssociation_IsExplicitAndKeepsOriginalOnInvalidOrDuplicateTarget()
    {
        ManualCurationProject project = CreateProject(out ManualObjectPlacement placement);
        Pm4HullIdentity firstHull = new(1, 2, 3, 4);
        Pm4HullIdentity otherHull = new(5, 6, 7, 8);
        Assert.Equal(ManualAssociationResult.Added,
            ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, firstHull));
        Assert.Equal(ManualAssociationResult.Added,
            ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, otherHull));

        Assert.Equal(ManualAssociationResult.AlreadyExists,
            ManualCurationProjectPolicy.TryReplaceAssociation(project, placement.PlacementId, firstHull, otherHull));
        Assert.Equal(ManualAssociationResult.InvalidHull,
            ManualCurationProjectPolicy.TryReplaceAssociation(project, placement.PlacementId, firstHull, new(-1, 0, 0, 0)));
        Assert.Contains(project.Associations, association => association.Hull == firstHull);

        Pm4HullIdentity replacement = new(20, 21, 22, 23);
        Assert.Equal(ManualAssociationResult.Replaced,
            ManualCurationProjectPolicy.TryReplaceAssociation(project, placement.PlacementId, firstHull, replacement));
        Assert.DoesNotContain(project.Associations, association => association.Hull == firstHull);
        Assert.Contains(project.Associations, association => association.Hull == replacement);
        Assert.Contains(project.Associations, association => association.Hull == otherHull);
    }

    [Fact]
    public void RemoveAssociation_RemovesOnlyTheExplicitRelation()
    {
        ManualCurationProject project = CreateProject(out ManualObjectPlacement placement);
        Pm4HullIdentity firstHull = new(1, 2, 3, 4);
        Pm4HullIdentity secondHull = new(5, 6, 7, 8);
        ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, firstHull);
        ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, secondHull);

        Assert.Equal(ManualAssociationResult.Removed,
            ManualCurationProjectPolicy.TryRemoveAssociation(project, placement.PlacementId, firstHull));
        Assert.Equal(ManualAssociationResult.AssociationNotFound,
            ManualCurationProjectPolicy.TryRemoveAssociation(project, placement.PlacementId, firstHull));
        Assert.Single(project.Associations);
        Assert.Equal(secondHull, project.Associations[0].Hull);
    }

    [Fact]
    public void UpdatingPlacementPositionPreservesItsExplicitHullAssociation()
    {
        ManualCurationProject project = CreateProject(out ManualObjectPlacement placement);
        Pm4HullIdentity hull = new(11, 12, 0x401122, 7);
        ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, hull);
        ManualObjectPlacement moved = placement with { Position = new Vector3(100f, 200f, 30f), RotationDegrees = new Vector3(0f, 0f, 17.5f) };

        Assert.True(ManualCurationProjectPolicy.TryUpdatePlacement(project, moved));
        Assert.Equal(moved.Position, Assert.Single(project.Placements).Position);
        ManualPm4Association association = Assert.Single(project.Associations);
        Assert.Equal(placement.PlacementId, association.PlacementId);
        Assert.Equal(hull, association.Hull);
    }

    [Fact]
    public void AddAssociation_RejectsUnknownPlacementAndInvalidHull()
    {
        ManualCurationProject project = new() { MapName = "Azeroth" };
        Pm4HullIdentity invalidHull = new(64, 1, 0, 0);

        Assert.Equal(ManualAssociationResult.PlacementNotFound,
            ManualCurationProjectPolicy.TryAddAssociation(project, Guid.NewGuid(), new(0, 0, 0, 0)));

        ManualObjectPlacement placement = CreatePlacement(Guid.NewGuid(), uniqueId: 12);
        ManualCurationProjectPolicy.TryAddPlacement(project, placement);
        Assert.Equal(ManualAssociationResult.InvalidHull,
            ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, invalidHull));
    }

    private static ManualCurationProject CreateProject(out ManualObjectPlacement placement)
    {
        var project = new ManualCurationProject { MapName = "Azeroth" };
        placement = CreatePlacement(Guid.NewGuid(), uniqueId: 17);
        Assert.True(ManualCurationProjectPolicy.TryAddPlacement(project, placement));
        return project;
    }

    private static ManualObjectPlacement CreatePlacement(Guid id, int uniqueId)
        => new(
            id,
            "World\\Azeroth\\Building.m2",
            "M2",
            "Building",
            TileX: 12,
            TileY: 20,
            PlacementEntryIndex: 2,
            UniqueId: uniqueId,
            Position: new Vector3(1f, 2f, 3f),
            RotationDegrees: Vector3.Zero,
            Scale: 1f,
            CreatedFromAssetCatalog: false,
            AuthoredUtc: DateTimeOffset.UnixEpoch);
}
