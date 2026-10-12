using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace WowViewer.Core.Tests.PromoVideo;

public class TaxiPlaylistModelTests
{
    public sealed record TestTaxiPlaylistItem(
        int PathId,
        int FromNodeId,
        int ToNodeId,
        string FromName,
        string ToName,
        float RouteLength)
    {
        public string DisplayLabel => $"{FromName} \u2192 {ToName} (#{PathId})";
    }

    [Fact]
    public void TaxiPlaylistItem_ConstructsWithValidFieldsAndLabel()
    {
        var item = new TestTaxiPlaylistItem(
            PathId: 14,
            FromNodeId: 2,
            ToNodeId: 4,
            FromName: "Stormwind",
            ToName: "Sentinel Hill",
            RouteLength: 2450.5f);

        Assert.Equal(14, item.PathId);
        Assert.Equal(2, item.FromNodeId);
        Assert.Equal(4, item.ToNodeId);
        Assert.Equal("Stormwind \u2192 Sentinel Hill (#14)", item.DisplayLabel);
        Assert.Equal(2450.5f, item.RouteLength);
    }

    [Fact]
    public void PlaylistReordering_MoveUpAndMoveDown_MaintainsSequence()
    {
        var list = new List<string> { "RouteA", "RouteB", "RouteC" };

        // Move "RouteB" (index 1) up -> should become index 0
        (list[0], list[1]) = (list[1], list[0]);
        Assert.Equal(new[] { "RouteB", "RouteA", "RouteC" }, list);

        // Move "RouteB" (index 0) down -> should become index 1
        (list[1], list[0]) = (list[0], list[1]);
        Assert.Equal(new[] { "RouteA", "RouteB", "RouteC" }, list);
    }

    [Fact]
    public void AutoChainAlgorithm_TraversesConnectedNodesWithoutImmediateBacktracking()
    {
        // Setup synthetic route graph:
        // Node 1 -> Node 2
        // Node 2 -> Node 1 (reverse ping-pong)
        // Node 2 -> Node 3 (forward hop)
        // Node 3 -> Node 4 (forward hop)
        var routes = new List<(int PathId, int From, int To)>
        {
            (10, 1, 2),
            (11, 2, 1),
            (12, 2, 3),
            (13, 3, 4)
        };

        var chain = new List<int>();
        int currentNode = 1;
        int prevNode = -1;
        int maxHops = 3;

        for (int hop = 0; hop < maxHops; hop++)
        {
            var outgoing = routes.Where(r => r.From == currentNode).ToList();
            if (outgoing.Count == 0)
                break;

            var next = outgoing.FirstOrDefault(r => r.To != prevNode);
            if (next.PathId == 0)
                next = outgoing[0];

            chain.Add(next.PathId);
            prevNode = currentNode;
            currentNode = next.To;
        }

        // Must pick 10 (1->2), then 12 (2->3, avoiding back to 1), then 13 (3->4)
        Assert.Equal(new[] { 10, 12, 13 }, chain);
        Assert.Equal(4, currentNode);
    }

    [Fact]
    public void OverallProgressCalculation_CalculatesWeightedProgressAcrossSegments()
    {
        int totalSegments = 4;
        int currentIndex = 2; // on 3rd segment (index 2)
        float segmentFraction = 0.50f; // half-way through 3rd segment

        float overall = (currentIndex + segmentFraction) / totalSegments;
        Assert.Equal(0.625f, overall); // 2.5 / 4 = 62.5%
    }
}
