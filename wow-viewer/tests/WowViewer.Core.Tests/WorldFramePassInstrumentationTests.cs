using System.Reflection;

using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 153 FR-001. <c>PrepareObjectPhase</c> was declared as a pass and given no stage timer, so a
/// ~212 ms periodic stall lived inside it while the stage table showed nothing. These tests make
/// that shape impossible to reintroduce: a pass with no timer, or a stage no one records, fails here.
/// </summary>
public sealed class WorldFramePassInstrumentationTests
{
    private static IReadOnlyList<PropertyInfo> PassMembers =>
        typeof(WorldFramePasses)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(static property => property.PropertyType == typeof(Action))
            .ToArray();

    [Fact]
    public void EveryPass_HasAtLeastOneStageTimer()
    {
        List<string> passesWithoutTimers = new();

        foreach (PropertyInfo pass in PassMembers)
        {
            if (!WorldFramePassInstrumentation.StagesByPass.TryGetValue(pass.Name, out IReadOnlyList<WorldRenderStage>? stages)
                || stages.Count == 0)
            {
                passesWithoutTimers.Add(pass.Name);
            }
        }

        Assert.True(
            passesWithoutTimers.Count == 0,
            "Every WorldFramePasses member must record at least one stage, otherwise its cost lands in "
            + "the unaccounted pass gap and no consumer can see it. Untimed passes: "
            + string.Join(", ", passesWithoutTimers));
    }

    [Fact]
    public void InstrumentationTable_DescribesOnlyRealPasses()
    {
        HashSet<string> passNames = PassMembers.Select(static property => property.Name).ToHashSet(StringComparer.Ordinal);
        string[] unknown = WorldFramePassInstrumentation.StagesByPass.Keys
            .Where(name => !passNames.Contains(name))
            .ToArray();

        Assert.True(unknown.Length == 0, "Instrumentation table names passes that do not exist: " + string.Join(", ", unknown));
    }

    [Fact]
    public void EveryStage_IsOwnedByAPassOrDeclaredPrePass()
    {
        HashSet<WorldRenderStage> owned = WorldFramePassInstrumentation.StagesByPass.Values
            .SelectMany(static stages => stages)
            .Concat(WorldFramePassInstrumentation.PrePassStages)
            .ToHashSet();

        WorldRenderStage[] orphaned = Enum.GetValues<WorldRenderStage>()
            .Where(stage => !owned.Contains(stage))
            .ToArray();

        Assert.True(orphaned.Length == 0, "Stages recorded by nothing: " + string.Join(", ", orphaned));
    }

    [Fact]
    public void NoStage_IsClaimedByTwoPasses()
    {
        WorldRenderStage[] duplicated = WorldFramePassInstrumentation.StagesByPass.Values
            .SelectMany(static stages => stages)
            .Concat(WorldFramePassInstrumentation.PrePassStages)
            .GroupBy(static stage => stage)
            .Where(static group => group.Count() > 1)
            .Select(static group => group.Key)
            .ToArray();

        // Double-counting a stage would inflate the instrumented total and make unaccounted time
        // read lower than it is — the exact failure mode this instrumentation exists to prevent.
        Assert.True(duplicated.Length == 0, "Stages claimed more than once: " + string.Join(", ", duplicated));
    }

    [Fact]
    public void HistoryStageCount_MatchesTheStageEnum()
    {
        Assert.Equal(WorldRenderFrameHistory.StageCount, Enum.GetValues<WorldRenderStage>().Length);
    }

    [Fact]
    public void PrepareObjectPhase_IsTimed()
    {
        // The specific regression: this pass existed with no timer at all.
        Assert.Contains(
            WorldRenderStage.PrepareObjectPhase,
            WorldFramePassInstrumentation.StagesByPass[nameof(WorldFramePasses.PrepareObjectPhase)]);
    }
}
