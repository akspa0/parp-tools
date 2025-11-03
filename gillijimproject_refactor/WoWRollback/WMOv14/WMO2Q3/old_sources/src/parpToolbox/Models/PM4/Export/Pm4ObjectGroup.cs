using System.Collections.Generic;

namespace parpToolbox.Models.PM4.Export
{
    /// <summary>
    /// Represents a collection of surfaces that form a coherent object or a logical group within the scene.
    /// </summary>
    public class Pm4ObjectGroup
    {
        /// <summary>
        /// The identifier for this object group, derived from fields like MSUR.GroupKey or MSLK.ParentId.
        /// </summary>
        public uint GroupId { get; set; }

        /// <summary>
        /// The collection of surfaces that belong to this object group.
        /// </summary>
        public List<Pm4SurfaceData> Surfaces { get; set; } = new List<Pm4SurfaceData>();
    }
}
