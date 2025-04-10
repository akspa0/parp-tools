using System;
using System.Collections.Generic;
using System.Linq;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Legacy.Interfaces;
using WoWToolbox.Validation.Chunkvault.Models;

namespace WoWToolbox.Validation.Chunkvault.Validators
{
    /// <summary>
    /// Validates chunks against their specifications
    /// </summary>
    public class ChunkValidator
    {
        private readonly Dictionary<string, ChunkSpecification> _specifications;

        /// <summary>
        /// Initializes a new instance of the ChunkValidator class
        /// </summary>
        /// <param name="specifications">Dictionary of chunk specifications keyed by chunk ID</param>
        public ChunkValidator(Dictionary<string, ChunkSpecification> specifications)
        {
            _specifications = specifications ?? throw new ArgumentNullException(nameof(specifications));
        }

        /// <summary>
        /// Validates a chunk against its specification
        /// </summary>
        /// <param name="chunk">The chunk to validate</param>
        /// <returns>A list of validation errors, empty if validation succeeds</returns>
        public IEnumerable<ValidationError> ValidateChunk(IIFFChunk chunk)
        {
            var errors = new List<ValidationError>();

            // Get the specification for this chunk
            string chunkSignature = chunk.GetSignature();
            if (!_specifications.TryGetValue(chunkSignature, out var spec))
            {
                errors.Add(new ValidationError
                {
                    ChunkId = chunkSignature,
                    ErrorType = ValidationErrorType.MissingSpecification,
                    Message = "No specification found for chunk"
                });
                return errors;
            }

            // Validate version if it's a legacy chunk
            if (chunk is ILegacyChunk legacyChunk)
            {
                if (!spec.SupportedVersions.Contains(legacyChunk.Version))
                {
                    errors.Add(new ValidationError
                    {
                        ChunkId = chunkSignature,
                        ErrorType = ValidationErrorType.UnsupportedVersion,
                        Message = $"Version {legacyChunk.Version} is not supported"
                    });
                }
            }

            /* // Commented out Size Validation Block - IIFFChunk interface does not provide size
            // Validate size constraints
            if (spec.Rules?.Size != null)
            {
                var size = chunk.GetSize(); // GetSize() does not exist on IIFFChunk
                
                if (spec.Rules.Size.MinSize.HasValue && size < spec.Rules.Size.MinSize.Value)
                {
                    errors.Add(new ValidationError
                    {
                        ChunkId = chunkSignature, // Use variable
                        ErrorType = ValidationErrorType.SizeConstraintViolation,
                        Message = $"Chunk size {size} is less than minimum {spec.Rules.Size.MinSize.Value}"
                    });
                }

                if (spec.Rules.Size.MaxSize.HasValue && size > spec.Rules.Size.MaxSize.Value)
                {
                    errors.Add(new ValidationError
                    {
                        ChunkId = chunkSignature, // Use variable
                        ErrorType = ValidationErrorType.SizeConstraintViolation,
                        Message = $"Chunk size {size} is greater than maximum {spec.Rules.Size.MaxSize.Value}"
                    });
                }

                if (spec.Rules.Size.MultipleOf.HasValue && size % spec.Rules.Size.MultipleOf.Value != 0)
                {
                    errors.Add(new ValidationError
                    {
                        ChunkId = chunkSignature, // Use variable
                        ErrorType = ValidationErrorType.SizeConstraintViolation,
                        Message = $"Chunk size {size} is not a multiple of {spec.Rules.Size.MultipleOf.Value}"
                    });
                }
            }
            */

            // TODO: Implement field validation
            // This will require reflection or a more sophisticated way to access chunk fields

            return errors;
        }
    }

    /// <summary>
    /// Represents a validation error
    /// </summary>
    public class ValidationError
    {
        /// <summary>
        /// The ID of the chunk that failed validation
        /// </summary>
        public string ChunkId { get; set; } = string.Empty;

        /// <summary>
        /// The type of validation error
        /// </summary>
        public ValidationErrorType ErrorType { get; set; }

        /// <summary>
        /// A description of the error
        /// </summary>
        public string Message { get; set; } = string.Empty;
    }

    /// <summary>
    /// Types of validation errors
    /// </summary>
    public enum ValidationErrorType
    {
        /// <summary>
        /// No specification found for the chunk
        /// </summary>
        MissingSpecification,

        /// <summary>
        /// The chunk version is not supported
        /// </summary>
        UnsupportedVersion,

        /// <summary>
        /// The chunk size violates constraints
        /// </summary>
        SizeConstraintViolation,

        /// <summary>
        /// A field value violates constraints
        /// </summary>
        FieldConstraintViolation,

        /// <summary>
        /// A relationship constraint is violated
        /// </summary>
        RelationshipConstraintViolation
    }
} 