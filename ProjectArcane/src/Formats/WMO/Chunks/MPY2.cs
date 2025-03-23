using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MPY2 chunk - Material info for triangles (v10+)
    /// Four bytes per triangle, containing flags and material ID
    /// Replaces MOPY chunk in v10+
    /// </summary>
    public class MPY2 : IChunk
    {
        /// <summary>
        /// Gets the list of polygon material info.
        /// </summary>
        public List<PolyMaterial2> PolyMaterials { get; } = new();

        public class PolyMaterial2
        {
            /// <summary>
            /// Gets or sets the flags for this polygon.
            /// </summary>
            public ushort Flags { get; set; }

            /// <summary>
            /// Gets or sets the material ID.
            /// </summary>
            public ushort MaterialId { get; set; }

            /// <summary>
            /// Gets whether this is a transition face (blends lighting from exterior to interior).
            /// </summary>
            public bool IsTransFace => (Flags & 0x01) != 0 && ((Flags & 0x04) != 0 || (Flags & 0x20) != 0);

            /// <summary>
            /// Gets whether this face has color.
            /// </summary>
            public bool IsColor => (Flags & 0x08) == 0;

            /// <summary>
            /// Gets whether this is a render face.
            /// </summary>
            public bool IsRenderFace => (Flags & 0x20) != 0 && (Flags & 0x04) == 0;

            /// <summary>
            /// Gets whether this face is collidable.
            /// </summary>
            public bool IsCollidable => (Flags & 0x08) != 0 || IsRenderFace;
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each poly material is 4 bytes:
            // - 2 bytes flags
            // - 2 bytes material ID
            var polyCount = (int)size / 4;

            // Clear existing data
            PolyMaterials.Clear();

            // Read poly materials
            for (int i = 0; i < polyCount; i++)
            {
                var poly = new PolyMaterial2
                {
                    Flags = reader.ReadUInt16(),
                    MaterialId = reader.ReadUInt16()
                };

                PolyMaterials.Add(poly);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var poly in PolyMaterials)
            {
                writer.Write(poly.Flags);
                writer.Write(poly.MaterialId);
            }
        }

        /// <summary>
        /// Gets a validation report for the poly materials.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MPY2 Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Polygons: {PolyMaterials.Count}");
            report.AppendLine();

            // Analyze polygons
            var uniqueMaterials = new HashSet<ushort>();
            var renderFaces = 0;
            var transFaces = 0;
            var colorFaces = 0;

            foreach (var poly in PolyMaterials)
            {
                uniqueMaterials.Add(poly.MaterialId);

                if (poly.IsRenderFace)
                    renderFaces++;
                if (poly.IsTransFace)
                    transFaces++;
                if (poly.IsColor)
                    colorFaces++;
            }

            report.AppendLine($"Unique Materials: {uniqueMaterials.Count}");
            report.AppendLine($"Render Faces: {renderFaces}");
            report.AppendLine($"Transition Faces: {transFaces}");
            report.AppendLine($"Color Faces: {colorFaces}");

            // Flag distribution
            var flagCounts = new Dictionary<ushort, int>();
            foreach (var poly in PolyMaterials)
            {
                if (!flagCounts.ContainsKey(poly.Flags))
                    flagCounts[poly.Flags] = 0;
                flagCounts[poly.Flags]++;
            }

            report.AppendLine();
            report.AppendLine("Flag Distribution:");
            foreach (var kvp in flagCounts.OrderBy(k => k.Key))
            {
                report.AppendLine($"  0x{kvp.Key:X4}: {kvp.Value} faces");
                // Add flag descriptions
                var flags = new List<string>();
                if ((kvp.Key & 0x0001) != 0) flags.Add("UNK_0x01");
                if ((kvp.Key & 0x0002) != 0) flags.Add("NOCAMCOLLIDE");
                if ((kvp.Key & 0x0004) != 0) flags.Add("DETAIL");
                if ((kvp.Key & 0x0008) != 0) flags.Add("COLLISION");
                if ((kvp.Key & 0x0010) != 0) flags.Add("HINT");
                if ((kvp.Key & 0x0020) != 0) flags.Add("RENDER");
                if ((kvp.Key & 0x0040) != 0) flags.Add("CULL_OBJECTS");
                if ((kvp.Key & 0x0080) != 0) flags.Add("COLLIDE_HIT");
                if (flags.Count > 0)
                    report.AppendLine($"    Flags: {string.Join(", ", flags)}");
            }

            return report.ToString();
        }
    }
} 