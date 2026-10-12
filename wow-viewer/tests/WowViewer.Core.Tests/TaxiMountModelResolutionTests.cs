using WoWViewer.Terrain;
using WowViewer.Core.IO.Files;
using Xunit;

namespace WowViewer.Core.Tests;

/// <summary>
/// Taxi mount models come from <c>CreatureModelData</c>. Legacy layouts (0.5.3-5.x) name the model
/// file; from 6.0.0 the field is a FileDataID. Both eras must resolve, so neither replaces the
/// other — a miss here pushes every taxi actor onto the hardcoded default candidate list.
/// </summary>
/// <remarks>
/// <see cref="FileDataIdPaths.Resolver"/> is a process-wide static, so every test that installs one
/// shares this collection and runs sequentially.
/// </remarks>
[Collection("FileDataIdPaths")]
public sealed class TaxiMountModelResolutionTests
{
    private static Func<string, object?> Row(params (string Key, object Value)[] fields)
    {
        Dictionary<string, object?> values = new(StringComparer.Ordinal);
        foreach ((string key, object value) in fields)
            values[key] = value;
        return name => values.TryGetValue(name, out object? value) ? value : null;
    }

    [Fact]
    public void ModernCreatureModelDataRowResolvesMountModelFromFileDataId()
    {
        Func<uint, string?> previous = FileDataIdPaths.Resolver!;
        FileDataIdPaths.Resolver = id => id == 116912 ? "Creature\\Gryphon\\Gryphon.m2" : null;
        try
        {
            Assert.Equal(
                "Creature\\Gryphon\\Gryphon.m2",
                TaxiPathLoader.ReadCreatureModelName(Row(("FileDataID", 116912u))));
        }
        finally
        {
            FileDataIdPaths.Resolver = previous;
        }
    }

    [Fact]
    public void LegacyCreatureModelDataRowStillReadsItsModelName()
    {
        Assert.Equal(
            "Creature\\Gryphon\\Gryphon.mdx",
            TaxiPathLoader.ReadCreatureModelName(Row(("ModelName", "Creature\\Gryphon\\Gryphon.mdx"))));
    }

    [Fact]
    public void RowWithNeitherNameNorIdResolvesToNothing()
    {
        Assert.Null(TaxiPathLoader.ReadCreatureModelName(Row(("Flags", 1u))));
    }
}
