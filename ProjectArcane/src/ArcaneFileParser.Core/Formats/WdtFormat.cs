using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using ArcaneFileParser.Core.Chunks.Common;
using ArcaneFileParser.Core.Chunks.Wdt;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Common.Types;
using System.Text;
using System.Linq;

namespace ArcaneFileParser.Core.Formats;

/// <summary>
/// Handler for World of Warcraft WDT (World Definition Table) files.
/// </summary>
public class WdtFormat : FileFormatBase
{
    /// <summary>
    /// Gets the MVER chunk containing version information.
    /// </summary>
    public MverChunk? Mver => GetChunk<MverChunk>();

    /// <summary>
    /// Gets the MAIN chunk containing map tile information.
    /// </summary>
    public MainChunk? Main => GetChunk<MainChunk>();

    /// <summary>
    /// Gets the MPHD chunk containing map header information.
    /// </summary>
    public MapHeaderChunk? MapHeader => GetChunk<MapHeaderChunk>();

    /// <summary>
    /// Gets the MODF chunk containing model placement information.
    /// </summary>
    public ModelPlacementChunk? ModelPlacements => GetChunk<ModelPlacementChunk>();

    /// <summary>
    /// Gets the MWID chunk containing model file paths.
    /// </summary>
    public ModelNameChunk? ModelNames => GetChunk<ModelNameChunk>();

    /// <summary>
    /// Gets whether the listfile has been initialized for path validation.
    /// </summary>
    public bool HasListfile => ListfileManager.Instance._isInitialized;

    /// <summary>
    /// Creates a new instance of the WDT format handler.
    /// </summary>
    /// <param name="filePath">Path to the WDT file to parse.</param>
    public WdtFormat(string filePath) : base(filePath)
    {
    }

    /// <summary>
    /// Initializes the listfile for path validation.
    /// </summary>
    /// <param name="localCachePath">Optional path to cache the downloaded listfile.</param>
    /// <returns>A task representing the initialization operation.</returns>
    public async Task InitializeListfile(string? localCachePath = null)
    {
        await ListfileManager.Instance.Initialize(localCachePath);
    }

    /// <summary>
    /// Gets whether a specific tile exists at the given coordinates.
    /// </summary>
    /// <param name="x">The X coordinate (0-63).</param>
    /// <param name="y">The Y coordinate (0-63).</param>
    /// <returns>True if the tile exists, false otherwise.</returns>
    public bool HasTile(int x, int y)
    {
        return Main?.HasTile(x, y) ?? false;
    }

    /// <summary>
    /// Gets the tile entry at the specified coordinates.
    /// </summary>
    /// <param name="x">The X coordinate (0-63).</param>
    /// <param name="y">The Y coordinate (0-63).</param>
    /// <returns>The tile entry at the specified coordinates, or null if the format is invalid.</returns>
    public MainChunk.MapTileEntry? GetTile(int x, int y)
    {
        return Main?.GetTile(x, y);
    }

    /// <summary>
    /// Gets whether the map has a specific flag set.
    /// </summary>
    /// <param name="flag">The flag to check.</param>
    /// <returns>True if the flag is set, false otherwise.</returns>
    public bool HasMapFlag(MapHeaderChunk.MapFlags flag)
    {
        return MapHeader?.Flags.HasFlag(flag) ?? false;
    }

    /// <summary>
    /// Gets all model placements within a specific bounding box.
    /// </summary>
    /// <param name="bounds">The bounding box to check.</param>
    /// <returns>An enumerable of model placements within the bounds.</returns>
    public IEnumerable<ModelPlacementChunk.ModelPlacement> GetModelsInBounds(BoundingBox bounds)
    {
        if (ModelPlacements == null)
            yield break;

        foreach (var placement in ModelPlacements.Placements)
        {
            if (bounds.Intersects(placement.Bounds))
                yield return placement;
        }
    }

    /// <summary>
    /// Gets a model placement by its unique ID.
    /// </summary>
    /// <param name="uniqueId">The unique ID to search for.</param>
    /// <returns>The model placement with the specified ID, or null if not found.</returns>
    public ModelPlacementChunk.ModelPlacement? GetModelById(uint uniqueId)
    {
        return ModelPlacements?.GetPlacementById(uniqueId);
    }

    /// <summary>
    /// Gets the file path for a model placement.
    /// </summary>
    /// <param name="placement">The model placement to get the path for.</param>
    /// <returns>The model's file path, or null if not found.</returns>
    public string? GetModelPath(ModelPlacementChunk.ModelPlacement placement)
    {
        return ModelNames?.GetPath(placement.NameId);
    }

    /// <summary>
    /// Gets the FileDataID for a model placement.
    /// </summary>
    /// <param name="placement">The model placement to get the FileDataID for.</param>
    /// <returns>The FileDataID if found in the listfile, or null if not found.</returns>
    public uint? GetModelFileDataId(ModelPlacementChunk.ModelPlacement placement)
    {
        return ModelNames?.GetFileDataId(placement.NameId);
    }

    /// <summary>
    /// Gets all unique model paths used in this WDT.
    /// </summary>
    /// <returns>A dictionary mapping unique IDs to model paths.</returns>
    public Dictionary<uint, string> GetAllModelPaths()
    {
        var paths = new Dictionary<uint, string>();
        
        if (ModelPlacements == null || ModelNames == null)
            return paths;

        foreach (var placement in ModelPlacements.Placements)
        {
            var path = ModelNames.GetPath(placement.NameId);
            if (path != null)
            {
                paths[placement.UniqueId] = path;
            }
        }

        return paths;
    }

    /// <summary>
    /// Gets validation statistics for all model paths in this WDT.
    /// </summary>
    /// <returns>A tuple containing (total paths, valid paths, invalid paths).</returns>
    public (int Total, int Valid, int Invalid) GetValidationStats()
    {
        return ModelNames?.GetValidationStats() ?? (0, 0, 0);
    }

    /// <summary>
    /// Gets all invalid model paths in this WDT.
    /// </summary>
    /// <returns>An enumerable of invalid paths and their indices.</returns>
    public IEnumerable<(uint Index, string Path)> GetInvalidPaths()
    {
        if (ModelNames == null)
            yield break;

        foreach (var invalid in ModelNames.GetInvalidPaths())
        {
            yield return invalid;
        }
    }

    /// <summary>
    /// Gets a report of missing files in this WDT.
    /// </summary>
    /// <returns>A formatted report string.</returns>
    public string GetMissingFilesReport()
    {
        var report = new StringBuilder();
        report.AppendLine("Missing Files Report");
        report.AppendLine("-----------------");

        // Get missing files from the validator
        var missingFiles = AssetPathValidator.MissingFiles;
        var missingFileDataIds = AssetPathValidator.MissingFileDataIds;

        report.AppendLine($"Total Missing Files: {missingFiles.Count}");
        report.AppendLine($"Total Missing FileDataIDs: {missingFileDataIds.Count}");

        if (missingFiles.Any())
        {
            report.AppendLine("\nMissing Files:");
            foreach (var file in missingFiles.Take(10)) // Show first 10 missing files
            {
                report.AppendLine($"  {file}");
            }
            if (missingFiles.Count > 10)
                report.AppendLine($"  ... and {missingFiles.Count - 10} more");
        }

        if (missingFileDataIds.Any())
        {
            report.AppendLine("\nMissing FileDataIDs:");
            foreach (var id in missingFileDataIds.Take(10)) // Show first 10 missing FileDataIDs
            {
                report.AppendLine($"  {id}");
            }
            if (missingFileDataIds.Count > 10)
                report.AppendLine($"  ... and {missingFileDataIds.Count - 10} more");
        }

        return report.ToString();
    }

    /// <summary>
    /// Internal method to parse the WDT file format from a binary reader.
    /// </summary>
    protected override void ParseInternal(BinaryReader reader)
    {
        // First chunk should be MVER
        var mver = new MverChunk(reader);
        Chunks.Add(mver);

        if (!mver.IsValid)
        {
            IsValid = false;
            return;
        }

        Version = mver.Version;

        // Continue reading chunks until we reach the end of the file
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            try
            {
                var (signature, size) = reader.ReadChunkHeader();

                // Store the start position for the chunk
                long startPosition = reader.BaseStream.Position;

                switch (signature)
                {
                    case MverChunk.ExpectedSignature:
                        // We already read MVER, this is an error
                        IsValid = false;
                        return;

                    case MainChunk.ExpectedSignature:
                        var main = new MainChunk(reader);
                        Chunks.Add(main);
                        if (!main.IsValid)
                        {
                            IsValid = false;
                            return;
                        }
                        break;

                    case MapHeaderChunk.ExpectedSignature:
                        var mapHeader = new MapHeaderChunk(reader);
                        Chunks.Add(mapHeader);
                        if (!mapHeader.IsValid)
                        {
                            IsValid = false;
                            return;
                        }
                        break;

                    case ModelPlacementChunk.ExpectedSignature:
                        var modelPlacements = new ModelPlacementChunk(reader);
                        Chunks.Add(modelPlacements);
                        if (!modelPlacements.IsValid)
                        {
                            IsValid = false;
                            return;
                        }
                        break;

                    case ModelNameChunk.ExpectedSignature:
                        var modelNames = new ModelNameChunk(reader);
                        Chunks.Add(modelNames);
                        if (!modelNames.IsValid)
                        {
                            IsValid = false;
                            return;
                        }
                        break;

                    default:
                        // Skip unknown chunks
                        reader.BaseStream.Position = startPosition + size;
                        break;
                }
            }
            catch (EndOfStreamException)
            {
                // Reached end of file
                break;
            }
        }

        // A valid WDT must have both MVER and MAIN chunks
        IsValid = Mver != null && Main != null;
    }

    /// <summary>
    /// Gets a detailed report of the WDT file contents.
    /// </summary>
    /// <returns>A formatted report string.</returns>
    public override string GetReport()
    {
        var report = new StringBuilder(base.GetReport());

        if (ModelNames != null && ModelNames.IsValid)
        {
            report.AppendLine("\nModel Names Report:");
            report.AppendLine(ModelNames.GetDetailedReport());
        }

        report.AppendLine("\nMissing Files Report:");
        report.AppendLine(GetMissingFilesReport());

        return report.ToString();
    }
} 