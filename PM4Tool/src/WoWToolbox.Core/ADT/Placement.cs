using System;
using System.Numerics;
using Warcraft.NET.Files.Structures; // For BoundingBox

namespace WoWToolbox.Core.ADT // Adjusted namespace
{

    /// <summary>
    /// Base class for model and WMO placements in an ADT file.
    /// </summary>
    public abstract class Placement
    {
        /// <summary>
        /// Gets or sets the unique ID of the placement.
        /// </summary>
        public uint UniqueId { get; set; } // Changed to uint

        /// <summary>
        /// Gets or sets the name ID (index into the model/WMO reference list OR FileDataId).
        /// </summary>
        public uint NameId { get; set; }

        /// <summary>
        /// Gets or sets the name of the model/WMO (looked up via NameId).
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the position of the placement.
        /// </summary>
        public Vector3 Position { get; set; }

        /// <summary>
        /// Gets or sets the rotation of the placement (degrees).
        /// </summary>
        public Vector3 Rotation { get; set; } // Note: Warcraft.NET uses Rotator, may need conversion

        /// <summary>
        /// Gets or sets the raw flags from MDDF/MODF chunks.
        /// </summary>
        public ushort Flags { get; set; }

        /// <summary>
        /// Gets or sets the FileDataID if Flags indicate NameId is a FileDataID.
        /// </summary>
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
        /// Gets or sets the scale of the model (converted from ushort/1024.0f).
        /// </summary>
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
        /// Gets or sets the calculated scale of the WMO (float).
        /// </summary>
        public float Scale { get; set; } // Changed to float

        /// <summary>
        /// Gets or sets the bounding box of the WMO.
        /// </summary>
        public BoundingBox BoundingBox { get; set; }
    }
} 