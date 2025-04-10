using System;

namespace WCAnalyzer.Core.Models;

/// <summary>
/// Represents the type of file reference.
/// </summary>
public enum FileReferenceType
{
    /// <summary>
    /// A texture file reference.
    /// </summary>
    Texture,

    /// <summary>
    /// A model (M2) file reference.
    /// </summary>
    Model,

    /// <summary>
    /// A world model object (WMO) file reference.
    /// </summary>
    WorldModel,
    
    /// <summary>
    /// A world model object (WMO) file reference (alias).
    /// </summary>
    Wmo = WorldModel
}

/// <summary>
/// Represents a file reference found in an ADT file.
/// </summary>
public class FileReference
{
    /// <summary>
    /// Gets or sets the original path as found in the ADT file.
    /// </summary>
    public string OriginalPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the normalized path for consistent comparison.
    /// </summary>
    public string NormalizedPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of reference.
    /// </summary>
    public FileReferenceType Type { get; set; }

    /// <summary>
    /// Gets or sets whether the reference is valid (exists in the listfile).
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets whether the reference exists in the listfile.
    /// </summary>
    public bool ExistsInListfile { get; set; }

    /// <summary>
    /// Gets or sets the repaired path if the reference is invalid and a repair was attempted.
    /// </summary>
    public string? RepairedPath { get; set; }
    
    /// <summary>
    /// Gets or sets the FileDataID for this file reference.
    /// </summary>
    /// <remarks>
    /// Only populated for newer ADT formats that use FileDataIDs instead of filenames.
    /// </remarks>
    public uint FileDataId { get; set; }
    
    /// <summary>
    /// Gets or sets whether this reference uses a FileDataID instead of a filename.
    /// </summary>
    public bool UsesFileDataId { get; set; }
    
    /// <summary>
    /// Gets or sets whether the reference was matched by FileDataID in the listfile.
    /// </summary>
    public bool MatchedByFileDataId { get; set; }
    
    /// <summary>
    /// Gets or sets the alternative extension path (for handling m2/mdx compatibility).
    /// </summary>
    /// <remarks>
    /// For .m2 files, this would store the .mdx path, and vice versa.
    /// Used to handle the change from .mdx to .m2 that occurred in version 0.11.
    /// </remarks>
    public string? AlternativeExtensionPath { get; set; }
    
    /// <summary>
    /// Gets or sets whether the alternative extension path exists in the listfile.
    /// </summary>
    public bool AlternativeExtensionFound { get; set; }

    /// <summary>
    /// Returns a string representation of the file reference.
    /// </summary>
    /// <returns>A string representation of the file reference.</returns>
    public override string ToString()
    {
        string fileDataIdInfo = UsesFileDataId ? $", FileDataID: {FileDataId}" : "";
        return $"{Type}: {OriginalPath} ({(IsValid ? "Valid" : "Invalid")}, {(ExistsInListfile ? "In Listfile" : "Not In Listfile")}{fileDataIdInfo})";
    }
}