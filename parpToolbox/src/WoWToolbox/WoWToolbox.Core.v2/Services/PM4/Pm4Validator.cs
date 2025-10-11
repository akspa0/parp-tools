using System;
using System.Linq;
using WoWToolbox.Core.v2.Models.Validation;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class Pm4Validator : IPm4Validator
    {
        /// <summary>
        /// Validates that indices in various chunks (MSVI, MSUR) point to valid vertices in MSVT.
        /// </summary>
        /// <param name="pm4File">The PM4 file to validate.</param>
        /// <returns>True if all indices are valid, otherwise false.</returns>
        public ValidationResult ValidateChunkIndices(PM4File pm4File)
        {
            var result = new ValidationResult();
            if (pm4File == null) { result.AddError("PM4File object cannot be null."); return result; }

            var msvtCount = pm4File.MSVT?.Vertices?.Count ?? 0;
            if (msvtCount == 0)
            {
                var hasIndices = (pm4File.MSVI?.Indices?.Any() ?? false) || (pm4File.MSUR?.Surfaces?.Any() ?? false);
                if (hasIndices)
                {
                    result.AddError("MSVT vertex list is empty but index or surface data exists.");
                }
                return result;
            }

            // Validate MSVI indices
            if (pm4File.MSVI?.Indices != null)
            {
                foreach (var index in pm4File.MSVI.Indices)
                {
                    if (index >= msvtCount)
                    {
                        result.AddError($"MSVI index {index} is out of bounds for MSVT vertex count {msvtCount}");
                    }
                }
            }

            // Validate MSUR surface indices
            if (pm4File.MSUR?.Surfaces != null)
            {
                foreach (var surface in pm4File.MSUR.Surfaces)
                {
                    if (surface.FirstIndex + surface.IndexCount > (pm4File.MSVI?.Indices?.Count ?? 0))
                    {
                        result.AddError($"MSUR entry has an invalid index count ({surface.IndexCount}) or start index ({surface.FirstIndex}) which points outside the bounds of the MSVI chunk size ({pm4File.MSVI?.Indices?.Count ?? 0}).");
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Validates that the number of faces defined by indices is consistent.
        /// </summary>
        public ValidationResult ValidateFaceCounts(PM4File pm4File)
        {
            var result = new ValidationResult();
            if (pm4File?.MSVI?.Indices == null)
            {
                return result;
            }

            if (pm4File.MSVI.Indices.Count % 3 != 0)
            {
                result.AddError($"MSVI index count ({pm4File.MSVI.Indices.Count}) is not divisible by 3. Likely not all triangle-based.");
            }

            return result;
        }

        /// <summary>
        /// Validates the structural integrity of MSLK nodes and their references.
        /// </summary>
        public ValidationResult ValidateMslkStructure(PM4File pm4File)
        {
            var result = new ValidationResult();
            if (pm4File?.MSLK?.Entries == null)
            {
                return result; // No MSLK to validate.
            }

            var mspvCount = pm4File.MSPV?.Vertices?.Count ?? 0;
            var mscnCount = pm4File.MSCN?.ExteriorVertices?.Count ?? 0;

            foreach (var entry in pm4File.MSLK.Entries)
            {
                var referenceIndex = entry.ReferenceIndex;
                var objectType = entry.ObjectType;

                if (objectType == 1 && referenceIndex >= mspvCount)
                {
                    result.AddError($"MSLK entry has an invalid vertex index {referenceIndex} for MSPV chunk with {mspvCount} vertices.");
                }
                else if (objectType == 2 && referenceIndex >= mscnCount)
                {
                    result.AddError($"MSLK entry references an out-of-bounds MSCN index ({referenceIndex}).");
                }
            }

            return result;
        }

        public ValidationResult ValidateMsviIndices(PM4File pm4File)
        {
            var result = new ValidationResult();
            var msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;

            if (pm4File.MSVI == null || msvtVertexCount == 0)
            {
                return result; // Not an error, just nothing to validate.
            }

            for (int i = 0; i < pm4File.MSVI.Indices.Count; i++)
            {
                var msviIndex = pm4File.MSVI.Indices[i];
                if (msviIndex >= msvtVertexCount)
                {
                    result.AddError($"MSVI index {msviIndex} at position {i} is out of bounds for MSVT vertex count {msvtVertexCount}");
                }
            }

            return result;
        }

        public ValidationResult ValidateMspiIndices(PM4File pm4File)
        {
            var result = new ValidationResult();
            var mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;

            if (pm4File.MSPI == null || mspvVertexCount == 0)
            {
                return result; // Not an error, just nothing to validate.
            }

            for (int i = 0; i < pm4File.MSPI.Indices.Count; i++)
            {
                var mspiIndex = pm4File.MSPI.Indices[i];
                if (mspiIndex >= mspvVertexCount)
                {
                    // This is more of a warning, as the structure is not fully understood.
                    result.AddError($"MSPI index {mspiIndex} at position {i} is out of bounds for MSPV vertex count {mspvVertexCount}");
                }
            }

            return result;
        }
    }
}
