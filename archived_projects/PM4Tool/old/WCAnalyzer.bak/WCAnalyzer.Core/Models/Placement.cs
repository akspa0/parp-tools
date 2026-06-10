using System;
using System.Numerics;

namespace WCAnalyzer.Core.Models;

/// <summary>
/// Base class for model and WMO placements in an ADT file.
/// </summary>
public abstract class Placement
{
    /// <summary>
    /// Gets or sets the unique ID of the placement.
    /// </summary>
    public int UniqueId { get; set; }

    /// <summary>
    /// Gets or sets the name ID (index into the model/WMO reference list).
    /// </summary>
    public int NameId { get; set; }

    /// <summary>
    /// Gets or sets the name of the model/WMO.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the position of the placement.
    /// </summary>
    public Vector3 Position { get; set; }

    /// <summary>
    /// Gets or sets the rotation of the placement.
    /// </summary>
    public Vector3 Rotation { get; set; }

    /// <summary>
    /// Gets or sets the flags of the placement.
    /// </summary>
    /// <remarks>
    /// In Warcraft.NET, this is a ushort that represents various flags:
    /// For MDDFFlags: 0x40 = NameIdIsFiledataId (among other flags)
    /// For MODFFlags: 0x8 = NameIdIsFiledataId (among other flags)
    /// </remarks>
    public ushort Flags { get; set; }
    
    /// <summary>
    /// Gets or sets the FileDataID of the placement.
    /// </summary>
    /// <remarks>
    /// Only populated in newer ADT formats where NameId represents a FileDataID instead of an index.
    /// </remarks>
    public uint FileDataId { get; set; }
    
    /// <summary>
    /// Gets or sets whether this placement uses a FileDataID in the NameId field.
    /// </summary>
    public bool UsesFileDataId { get; set; }
}

/// <summary>
/// Represents a model (M2) placement in an ADT file.
/// </summary>
public class ModelPlacement : Placement
{
    /// <summary>
    /// Gets or sets the scale of the model.
    /// </summary>
    /// <remarks>
    /// In Warcraft.NET MDDFEntry, the raw data is a ushort ScalingFactor where 1024 = 1.0f.
    /// This property contains the converted float value after division by 1024.
    /// </remarks>
    public float Scale { get; set; }
}

/// <summary>
/// Represents a world model object (WMO) placement in an ADT file.
/// </summary>
public class WmoPlacement : Placement
{
    /// <summary>
    /// Gets or sets the doodad set index.
    /// </summary>
    public ushort DoodadSet { get; set; }

    /// <summary>
    /// Gets or sets the name set index.
    /// </summary>
    public ushort NameSet { get; set; }
    
    /// <summary>
    /// Gets or sets the scale of the WMO. Scale flag must be set to use this value.
    /// </summary>
    /// <remarks>
    /// In Warcraft.NET, when the MODFFlags.HasScale flag (0x4) is set, this value is used to scale the WMO.
    /// </remarks>
    public ushort Scale { get; set; }
}