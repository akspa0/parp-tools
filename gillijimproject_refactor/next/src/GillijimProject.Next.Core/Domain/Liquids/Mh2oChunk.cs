using System.Collections.Generic;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MH2O chunk container (decoded model): a collection of instances plus optional attributes.
/// </summary>
public sealed class Mh2oChunk
{
    public List<Mh2oInstance> Instances { get; } = new();

    /// <summary>Optional 8x8 attribute masks for the chunk.</summary>
    public Mh2oAttributes? Attributes { get; init; }

    /// <summary>True when there are no instances.</summary>
    public bool IsEmpty => Instances.Count == 0;

    /// <summary>Adds an instance after validating it.</summary>
    public void Add(Mh2oInstance instance)
    {
        instance.Validate();
        Instances.Add(instance);
    }
}
