using System.Collections.Generic;
using System.Linq;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// Options controlling liquid conversion behavior.
/// </summary>
public sealed class LiquidsOptions
{
    /// <summary>Master switch for liquid conversion.</summary>
    public bool EnableLiquids { get; init; } = true;

    /// <summary>Liquid type precedence when multiple types overlap. Default: magma &gt; slime &gt; river &gt; ocean.</summary>
    public IReadOnlyList<MclqLiquidType> Precedence { get; init; }
        = new[] { MclqLiquidType.Magma, MclqLiquidType.Slime, MclqLiquidType.River, MclqLiquidType.Ocean };

    /// <summary>Special mode to treat lava as green variant.</summary>
    public bool GreenLava { get; init; } = false;

    /// <summary>Mapping MCLQ types ↔ LiquidType IDs. Defaults can be overridden (e.g., via CLI JSON).</summary>
    public LiquidTypeMapping Mapping { get; init; } = LiquidTypeMapping.CreateDefault();

    public LiquidsOptions WithMapping(LiquidTypeMapping mapping) => new()
    {
        EnableLiquids = EnableLiquids,
        Precedence = Precedence.ToArray(),
        GreenLava = GreenLava,
        Mapping = mapping
    };
}
