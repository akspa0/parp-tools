using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.ADT; // Use correct namespace for Placement models
using Warcraft.NET.Files.ADT.Chunks; // Entries should be defined here directly
// using Warcraft.NET.Files.ADT.Chunks.Subchunks; // Removed - Subchunks likely not needed
using Warcraft.NET.Files.Structures; // Correct namespace for Rotator
using Warcraft.NET.Files.ADT.Flags; // Added for MODFFlags
using System.IO; // Added for Path.GetFileName
using Warcraft.NET.Files.ADT.TerrainObject.Zero; // Added for _obj0 ADT class

namespace WoWToolbox.Core.ADT
{
    /// <summary>
    /// Service for processing and extracting data from ADT files.
    /// </summary>
    public class AdtService
    {
        // Define flags locally for clarity, mirroring Warcraft.NET definitions
        private const MDDFFlags MDDF_FILE_DATA_ID_FLAG = (MDDFFlags)0x40; 
        private const MODFFlags MODF_FILE_DATA_ID_FLAG = (MODFFlags)0x8;

        /// <summary>
        /// Extracts WMO and M2 model placement information from an ADT file.
        /// Assumes the necessary data (MDDF, MODF, MMDX, MWMO) is present in the _obj0 file.
        /// </summary>
        /// <param name="obj0Data">The loaded _obj0 ADT data (TerrainObjectZero).</param>
        /// <param name="listfileData">A dictionary mapping FileDataId to FilePath.</param>
        /// <returns>An enumerable collection of placements.</returns>
        public IEnumerable<Placement> ExtractPlacements(TerrainObjectZero obj0Data, Dictionary<uint, string> listfileData)
        {
            var placements = new List<Placement>();

            // Extract M2 Model Placements from MDDF chunk (in obj0Data)
            if (obj0Data.ModelPlacementInfo != null && obj0Data.ModelPlacementInfo.MDDFEntries != null) 
            {
                // Use obj0Data.Models (from TerrainObjectBase) for MMDX filename lookup
                placements.AddRange(obj0Data.ModelPlacementInfo.MDDFEntries.Select(entry => 
                {
                    string filePath = "";
                    uint fileDataId = 0;
                    bool usesFileDataId = (entry.Flags & MDDF_FILE_DATA_ID_FLAG) != 0;

                    if (usesFileDataId)
                    {
                        fileDataId = entry.NameId; // In this case, NameId IS the FileDataId
                        if (!listfileData.TryGetValue(fileDataId, out filePath) || string.IsNullOrWhiteSpace(filePath))
                        {
                            filePath = $"UNKNOWN_FID_{fileDataId}"; // Placeholder if not found
                            Console.WriteLine($"Warning: FileDataId {fileDataId} (from MDDF NameId) not found in listfile.");
                        }
                    }
                    else
                    {
                        // Fallback to MMDX Filenames lookup (in obj0Data)
                        if (obj0Data.Models?.Filenames != null && entry.NameId >= 0 && entry.NameId < obj0Data.Models.Filenames.Count)
                        {
                            string? foundPath = obj0Data.Models.Filenames[(int)entry.NameId];
                            filePath = foundPath ?? ""; 
                        }
                        else
                        {
                             Console.WriteLine($"Warning: Could not find MMDX path for MDDF NameId {entry.NameId}");
                             filePath = ""; 
                        }
                        
                        // Attempt reverse lookup in listfile if path was found
                        if (!string.IsNullOrWhiteSpace(filePath))
                        {
                            string normalizedFilePath = NormalizePath(filePath);
                            foreach (var kvp in listfileData)
                            {
                                if (NormalizePath(kvp.Value) == normalizedFilePath)
                                {
                                    fileDataId = kvp.Key;
                                    break; // Found match
                                }
                            }
                            if (fileDataId == 0)
                            {
                                Console.WriteLine($"Warning: Could not find FileDataId for path (from MMDX): {filePath}");
                            }
                        }
                    }

                    var placement = new ModelPlacement
                    {
                        UniqueId = entry.UniqueID,
                        NameId = entry.NameId, // Keep original NameId regardless of flag
                        UsesFileDataId = usesFileDataId,
                        FileDataId = fileDataId, 
                        FilePath = filePath,
                        Position = entry.Position, 
                        Rotation = ConvertRotation(entry.Rotation), 
                        Scale = entry.ScalingFactor / 1024.0f, 
                        Flags = (ushort)entry.Flags
                    };
                    placement.FlagNames = GetMddfFlagNames(placement.Flags);
                    #pragma warning disable CS0618 // Type or member is obsolete
                    placement.Name = string.IsNullOrWhiteSpace(filePath) ? "" : Path.GetFileName(filePath);
                    #pragma warning restore CS0618
                    return placement;
                }));
            }

            // Extract WMO Placements from MODF chunk (in obj0Data)
            if (obj0Data.WorldModelObjectPlacementInfo != null && obj0Data.WorldModelObjectPlacementInfo.MODFEntries != null) 
            {
                // Use obj0Data.WorldModelObjects (from TerrainObjectBase) for MWMO filename lookup
                placements.AddRange(obj0Data.WorldModelObjectPlacementInfo.MODFEntries.Select(entry => 
                {
                    string filePath = "";
                    uint fileDataId = 0;
                    bool usesFileDataId = (entry.Flags & MODF_FILE_DATA_ID_FLAG) != 0;

                    if (usesFileDataId)
                    {
                        fileDataId = entry.NameId; // NameId IS the FileDataId
                        if (!listfileData.TryGetValue(fileDataId, out filePath) || string.IsNullOrWhiteSpace(filePath))
                        {
                            filePath = $"UNKNOWN_FID_{fileDataId}"; // Placeholder
                            Console.WriteLine($"Warning: FileDataId {fileDataId} (from MODF NameId) not found in listfile.");
                        }
                    }
                    else
                    {
                        // Fallback to MWMO Filenames lookup (in obj0Data)
                        if (obj0Data.WorldModelObjects?.Filenames != null && entry.NameId >= 0 && entry.NameId < obj0Data.WorldModelObjects.Filenames.Count)
                        {
                            string? foundPath = obj0Data.WorldModelObjects.Filenames[(int)entry.NameId];
                            filePath = foundPath ?? ""; 
                        }
                        else
                        {
                            Console.WriteLine($"Warning: Could not find MWMO path for MODF NameId {entry.NameId}");
                            filePath = ""; 
                        }

                        // Attempt reverse lookup in listfile if path was found
                        if (!string.IsNullOrWhiteSpace(filePath))
                        {
                            string normalizedFilePath = NormalizePath(filePath);
                            foreach (var kvp in listfileData)
                            {
                                if (NormalizePath(kvp.Value) == normalizedFilePath)
                                {
                                    fileDataId = kvp.Key;
                                    break; // Found match
                                }
                            }
                            if (fileDataId == 0)
                            {
                                Console.WriteLine($"Warning: Could not find FileDataId for path (from MWMO): {filePath}");
                            }
                        }
                    }
                    
                    var placement = new WmoPlacement 
                    {
                        UniqueId = (uint)entry.UniqueId, 
                        NameId = entry.NameId, // Keep original NameId 
                        UsesFileDataId = usesFileDataId,
                        FileDataId = fileDataId,
                        FilePath = filePath, 
                        Position = entry.Position, 
                        Rotation = ConvertRotation(entry.Rotation), 
                        BoundingBox = entry.BoundingBox, 
                        Flags = (ushort)entry.Flags,
                        DoodadSet = entry.DoodadSet,
                        NameSet = entry.NameSet,
                        Scale = (entry.Flags & MODFFlags.HasScale) != 0 ? entry.Scale / 1024.0f : 1.0f 
                    };
                    placement.FlagNames = GetModfFlagNames(placement.Flags);
                    #pragma warning disable CS0618 // Type or member is obsolete
                    placement.Name = string.IsNullOrWhiteSpace(filePath) ? "" : Path.GetFileName(filePath);
                    #pragma warning restore CS0618
                    return placement;
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

        private static List<string> GetMddfFlagNames(ushort flagsValue)
        {
            var names = new List<string>();
            MDDFFlags flags = (MDDFFlags)flagsValue;

            if ((flags & MDDFFlags.Biodome) != 0) names.Add(nameof(MDDFFlags.Biodome));
            if ((flags & MDDFFlags.Shrubbery) != 0) names.Add(nameof(MDDFFlags.Shrubbery));
            if ((flags & MDDFFlags.Unk4) != 0) names.Add(nameof(MDDFFlags.Unk4));
            if ((flags & MDDFFlags.Unk8) != 0) names.Add(nameof(MDDFFlags.Unk8));
            if ((flags & MDDFFlags.Unk10) != 0) names.Add(nameof(MDDFFlags.Unk10));
            if ((flags & MDDFFlags.LiquidKnown) != 0) names.Add(nameof(MDDFFlags.LiquidKnown));
            if ((flags & MDDFFlags.NameIdIsFiledataId) != 0) names.Add(nameof(MDDFFlags.NameIdIsFiledataId));
            if ((flags & MDDFFlags.Unk100) != 0) names.Add(nameof(MDDFFlags.Unk100));
            if ((flags & MDDFFlags.AcceptProjTextures) != 0) names.Add(nameof(MDDFFlags.AcceptProjTextures));
            // Add checks for other known flags if needed, ignore Unknown ones for now

            return names;
        }

        private static List<string> GetModfFlagNames(ushort flagsValue)
        {
            var names = new List<string>();
            MODFFlags flags = (MODFFlags)flagsValue;

            if ((flags & MODFFlags.Destroyable) != 0) names.Add(nameof(MODFFlags.Destroyable));
            if ((flags & MODFFlags.UseLod) != 0) names.Add(nameof(MODFFlags.UseLod));
            if ((flags & MODFFlags.HasScale) != 0) names.Add(nameof(MODFFlags.HasScale));
            if ((flags & MODFFlags.NameIdIsFiledataId) != 0) names.Add(nameof(MODFFlags.NameIdIsFiledataId));
            if ((flags & MODFFlags.UseDoodadSetsFromMWDS) != 0) names.Add(nameof(MODFFlags.UseDoodadSetsFromMWDS));

            return names;
        }

        /// <summary>
        /// Normalizes a file path for case-insensitive and slash-insensitive comparison.
        /// </summary>
        private static string NormalizePath(string path)
        {
            return path.Replace('\\', '/').ToLowerInvariant();
        }
    }
} 