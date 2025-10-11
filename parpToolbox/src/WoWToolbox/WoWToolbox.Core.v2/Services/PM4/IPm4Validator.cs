using WoWToolbox.Core.v2.Models.Validation;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IPm4Validator
    {
        ValidationResult ValidateChunkIndices(PM4File pm4File);
        ValidationResult ValidateFaceCounts(PM4File pm4File);
        ValidationResult ValidateMslkStructure(PM4File pm4File);

        /// <summary>
        /// Validates that all MSVI indices are within the bounds of the MSVT vertex list.
        /// </summary>
        /// <param name="pm4File">The PM4 file to validate.</param>
        /// <returns>A ValidationResult indicating success or failure with error details.</returns>
        ValidationResult ValidateMsviIndices(PM4File pm4File);

        /// <summary>
        /// Validates that all MSPI indices are within the bounds of the MSPV vertex list.
        /// </summary>
        /// <param name="pm4File">The PM4 file to validate.</param>
        /// <returns>A ValidationResult indicating success or failure with error details.</returns>
        ValidationResult ValidateMspiIndices(PM4File pm4File);
        // Add other validation methods as needed
    }
}
