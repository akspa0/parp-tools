using System.Numerics;
using WoWViewer.Rendering;

namespace WowViewer.Core.Tests;

/// <summary>
/// Epic 249 R-10d: the spatially indexed light query must return exactly what a full linear scan
/// returns — the same lights, nearest first, ties in insertion order — for every query shape.
/// </summary>
public sealed class SceneLightManagerTests
{
    [Fact]
    public void QueryAffecting_MatchesLinearScan_ForRandomScenes()
    {
        var random = new Random(24923);
        for (int scene = 0; scene < 40; scene++)
        {
            List<SceneLight> lights = RandomLights(random, count: random.Next(0, 400));
            var manager = new SceneLightManager();
            manager.AddRange(lights);

            for (int query = 0; query < 200; query++)
            {
                (Vector3 min, Vector3 max) = RandomBounds(random);
                SceneLight[] expected = ReferenceQuery(manager.Lights, min, max);

                Span<SceneLight> actual = new SceneLight[SceneLightManager.MaxShaderLights];
                int count = manager.QueryAffecting(min, max, actual);

                Assert.Equal(expected, actual[..count].ToArray());
                Assert.Equal(expected.Length > 0, manager.AnyAffecting(min, max));
            }
        }
    }

    [Fact]
    public void QueryAffecting_IncludesHugeRadiusLights_AndHugeQueries()
    {
        var manager = new SceneLightManager();
        manager.Add(Light(new Vector3(0, 0, 0), radius: 50_000f));          // spans far more than the grid
        manager.Add(Light(new Vector3(10_000, 10_000, 0), radius: 10f));
        manager.Add(Light(new Vector3(-20_000, 5_000, 30), radius: 100f));

        Span<SceneLight> destination = new SceneLight[SceneLightManager.MaxShaderLights];

        int nearFar = manager.QueryAffecting(new Vector3(9_995, 9_995, -5), new Vector3(10_005, 10_005, 5), destination);
        Assert.Equal(2, nearFar);
        Assert.Equal(10f, destination[0].AttenuationEnd); // containing light first (distance 0), then insertion order
        Assert.Equal(50_000f, destination[1].AttenuationEnd);

        int whole = manager.QueryAffecting(new Vector3(-40_000, -40_000, -1_000), new Vector3(40_000, 40_000, 1_000), destination);
        Assert.Equal(3, whole);
    }

    [Fact]
    public void QueryAffecting_RespectsZ_EvenThoughTheGridIsTwoDimensional()
    {
        var manager = new SceneLightManager();
        manager.Add(Light(new Vector3(0, 0, 500), radius: 20f));

        Span<SceneLight> destination = new SceneLight[SceneLightManager.MaxShaderLights];
        Assert.Equal(0, manager.QueryAffecting(new Vector3(-5, -5, 0), new Vector3(5, 5, 10), destination));
        Assert.False(manager.AnyAffecting(new Vector3(-5, -5, 0), new Vector3(5, 5, 10)));
        Assert.Equal(1, manager.QueryAffecting(new Vector3(-5, -5, 470), new Vector3(5, 5, 490), destination));
    }

    [Fact]
    public void AddingLightsAfterAQuery_RebuildsTheIndex()
    {
        var manager = new SceneLightManager();
        manager.Add(Light(new Vector3(0, 0, 0), radius: 10f));
        Assert.False(manager.AnyAffecting(new Vector3(500, 500, 0), new Vector3(510, 510, 1)));

        manager.Add(Light(new Vector3(505, 505, 0), radius: 10f));
        Assert.True(manager.AnyAffecting(new Vector3(500, 500, 0), new Vector3(510, 510, 1)));
    }

    [Fact]
    public void TakeFrameCounters_ReportsWorkloadAndRestartsQueryCounters()
    {
        var manager = new SceneLightManager();
        manager.AddRange(
            [Light(new Vector3(0, 0, 0), 10f), Light(new Vector3(1_000, 0, 0), 10f), Light(new Vector3(2_000, 0, 0), 10f)],
            light => light.Position.X < 1_500);
        manager.AnyAffecting(new Vector3(-1, -1, -1), new Vector3(1, 1, 1));
        manager.RecordWmoPartition(batched: 7, litFallback: 3, selfLit: 2);

        var stats = manager.TakeFrameCounters();
        Assert.Equal(3, stats.LightsCollected);
        Assert.Equal(2, stats.LightsKept);
        Assert.Equal(1, stats.QueryCount);
        Assert.Equal(7, stats.WmoPlacementsBatched);
        Assert.Equal(3, stats.WmoPlacementsLitFallback);
        Assert.Equal(2, stats.WmoPlacementsSelfLit);

        Assert.Equal(0, manager.TakeFrameCounters().QueryCount);
    }

    private static SceneLight[] ReferenceQuery(IReadOnlyList<SceneLight> lights, Vector3 min, Vector3 max)
    {
        var hits = new List<(int Index, float DistanceSq)>();
        for (int i = 0; i < lights.Count; i++)
        {
            float d = DistanceSquared(lights[i].Position, min, max);
            float r = MathF.Max(lights[i].AttenuationEnd, 0f);
            if (d <= r * r)
                hits.Add((i, d));
        }

        return hits
            .OrderBy(hit => hit.DistanceSq)
            .ThenBy(hit => hit.Index)
            .Take(SceneLightManager.MaxShaderLights)
            .Select(hit => lights[hit.Index])
            .ToArray();
    }

    private static float DistanceSquared(Vector3 p, Vector3 min, Vector3 max)
    {
        static float Axis(float v, float a, float b)
        {
            float lo = MathF.Min(a, b), hi = MathF.Max(a, b);
            return v < lo ? lo - v : v > hi ? v - hi : 0f;
        }

        float dx = Axis(p.X, min.X, max.X), dy = Axis(p.Y, min.Y, max.Y), dz = Axis(p.Z, min.Z, max.Z);
        return dx * dx + dy * dy + dz * dz;
    }

    private static List<SceneLight> RandomLights(Random random, int count)
    {
        var lights = new List<SceneLight>(count);
        for (int i = 0; i < count; i++)
        {
            var position = new Vector3(
                (float)(random.NextDouble() * 4_000 - 2_000),
                (float)(random.NextDouble() * 4_000 - 2_000),
                (float)(random.NextDouble() * 400 - 200));

            // Mostly torch-sized, some large, a few enormous (the "large light" list), and duplicates
            // of a shared position so equal distances exercise the insertion-order tie break.
            float radius = random.Next(10) switch
            {
                0 => (float)(random.NextDouble() * 5_000 + 1_000),
                1 or 2 => (float)(random.NextDouble() * 400 + 100),
                _ => (float)(random.NextDouble() * 60 + 2),
            };

            if (i % 17 == 0)
                position = new Vector3(100, 100, 0);

            lights.Add(Light(position, radius));
        }

        return lights;
    }

    private static (Vector3 Min, Vector3 Max) RandomBounds(Random random)
    {
        var center = new Vector3(
            (float)(random.NextDouble() * 4_400 - 2_200),
            (float)(random.NextDouble() * 4_400 - 2_200),
            (float)(random.NextDouble() * 400 - 200));
        float size = random.Next(8) == 0 ? (float)(random.NextDouble() * 6_000) : (float)(random.NextDouble() * 150);
        var half = new Vector3(size, size * (float)random.NextDouble(), (float)(random.NextDouble() * 60)) * 0.5f;

        // Occasionally hand the query inverted bounds: the scan accepts min/max in either order.
        return random.Next(5) == 0 ? (center + half, center - half) : (center - half, center + half);
    }

    private static SceneLight Light(Vector3 position, float radius)
        => new(position, Vector3.One, 1f, 0f, radius, "test", "test");
}
