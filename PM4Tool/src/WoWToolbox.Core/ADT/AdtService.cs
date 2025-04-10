using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.ADT; // Use correct namespace for Placement models
using Warcraft.NET.Files.ADT.Chunks; // Entries should be defined here directly
// using Warcraft.NET.Files.ADT.Chunks.Subchunks; // Removed - Subchunks likely not needed
using Warcraft.NET.Files.Structures; // Correct namespace for Rotator
using Warcraft.NET.Files.ADT.Flags; // Added for MODFFlags

namespace WoWToolbox.Core.ADT
{
    /// <summary>
    /// Service for processing and extracting data from ADT files.
    /// </summary>
    public class AdtService
    {
        /// <summary>
        /// Extracts WMO and M2 model placement information from an ADT file.
        /// </summary>
        /// <param name="adtFile">The loaded ADT file.</param>
        /// <returns>An enumerable collection of placements.</returns>
        public IEnumerable<Placement> ExtractPlacements(ADTFile adtFile)
        {
            var placements = new List<Placement>();

            // Extract M2 Model Placements from MDDF chunk
            if (adtFile.MDDF != null && adtFile.MDDF.MDDFEntries != null) // Use MDDFEntries property
            {
                placements.AddRange(adtFile.MDDF.MDDFEntries.Select(entry => new ModelPlacement // Use MDDFEntries
                {
                    UniqueId = entry.UniqueID, // Use capitalized UniqueID property
                    NameId = entry.NameId,     // uint
                    Position = entry.Position, // Vector3
                    Rotation = ConvertRotation(entry.Rotation), // Vector3 (Degrees)
                    Scale = entry.ScalingFactor / 1024.0f, // Use ScalingFactor property
                    Flags = (ushort)entry.Flags        // Cast Flags enum to ushort
                }));
            }

            // Extract WMO Placements from MODF chunk
            if (adtFile.MODF != null && adtFile.MODF.MODFEntries != null) // Use MODFEntries property
            {
                placements.AddRange(adtFile.MODF.MODFEntries.Select(entry => new WmoPlacement // Use MODFEntries
                {
                    UniqueId = (uint)entry.UniqueId,      // Cast int UniqueId to uint
                    NameId = entry.NameId,          // uint
                    Position = entry.Position,      // Vector3
                    Rotation = ConvertRotation(entry.Rotation), // Vector3 (Degrees)
                    BoundingBox = entry.BoundingBox, // Assign BoundingBox property
                    Flags = (ushort)entry.Flags,            // Cast Flags enum to ushort
                    DoodadSet = entry.DoodadSet,    // ushort
                    NameSet = entry.NameSet,        // ushort
                    // Calculate float scale based on flag
                    Scale = (entry.Flags & MODFFlags.HasScale) != 0 ? entry.Scale / 1024.0f : 1.0f 
                }));
            }

            return placements;
        }

        /// <summary>
        /// Converts Warcraft.NET Rotator (Pitch, Roll, Yaw) to Vector3 (Degrees).
        /// Assumes standard game coordinate system mapping: Pitch -> X, Roll -> Y, Yaw -> Z.
        /// Adjust if Placement model expects a different order.
        /// </summary>
        private static Vector3 ConvertRotation(Rotator rotator)
        {
            // Ensure degrees are within standard ranges if necessary, though Vector3 doesn't inherently enforce this.
            // Example: Adjust yaw to be 0-360 if needed. Rotator likely provides this already.
            return new Vector3(rotator.Pitch, rotator.Roll, rotator.Yaw);
        }
    }
} 