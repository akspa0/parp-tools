using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOPY chunk - Material info for triangles
    /// Two bytes per triangle, containing flags and material ID
    /// </summary>
    public class MOPY : IChunk
    {
        /// <summary>
        /// Gets the list of polygon material info.
        /// </summary>
        public List<PolyMaterial> PolyMaterials { get; } = new();

        public class PolyMaterial
        {
            /// <summary>
            /// Gets or sets the flags for this polygon.
            /// </summary>
            public byte Flags { get; set; }

            /// <summary>
            /// Gets or sets the material ID (index into MOMT, 0xFF for collision faces).
            /// </summary>
            public byte MaterialId { get; set; }

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

            // Each poly material is 2 bytes:
            // - 1 byte flags
            // - 1 byte material ID
            var polyCount = (int)size / 2;

            // Clear existing data
            PolyMaterials.Clear();

            // Read poly materials
            for (int i = 0; i < polyCount; i++)
            {
                var poly = new PolyMaterial
                {
                    Flags = reader.ReadByte(),
                    MaterialId = reader.ReadByte()
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
            report.AppendLine("MOPY Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Polygons: {PolyMaterials.Count}");
            report.AppendLine();

            // Analyze polygons
            var uniqueMaterials = new HashSet<byte>();
            var collisionFaces = 0;
            var renderFaces = 0;
            var transFaces = 0;
            var colorFaces = 0;

            foreach (var poly in PolyMaterials)
            {
                if (poly.MaterialId != 0xFF)
                    uniqueMaterials.Add(poly.MaterialId);
                else
                    collisionFaces++;

                if (poly.IsRenderFace)
                    renderFaces++;
                if (poly.IsTransFace)
                    transFaces++;
                if (poly.IsColor)
                    colorFaces++;
            }

            report.AppendLine($"Unique Materials: {uniqueMaterials.Count}");
            report.AppendLine($"Collision-only Faces: {collisionFaces}");
            report.AppendLine($"Render Faces: {renderFaces}");
            report.AppendLine($"Transition Faces: {transFaces}");
            report.AppendLine($"Color Faces: {colorFaces}");

            // Flag distribution
            var flagCounts = new Dictionary<byte, int>();
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
                report.AppendLine($"  0x{kvp.Key:X2}: {kvp.Value} faces");
                // Add flag descriptions
                var flags = new List<string>();
                if ((kvp.Key & 0x01) != 0) flags.Add("UNK_0x01");
                if ((kvp.Key & 0x02) != 0) flags.Add("NOCAMCOLLIDE");
                if ((kvp.Key & 0x04) != 0) flags.Add("DETAIL");
                if ((kvp.Key & 0x08) != 0) flags.Add("COLLISION");
                if ((kvp.Key & 0x10) != 0) flags.Add("HINT");
                if ((kvp.Key & 0x20) != 0) flags.Add("RENDER");
                if ((kvp.Key & 0x40) != 0) flags.Add("CULL_OBJECTS");
                if ((kvp.Key & 0x80) != 0) flags.Add("COLLIDE_HIT");
                if (flags.Count > 0)
                    report.AppendLine($"    Flags: {string.Join(", ", flags)}");
            }

            return report.ToString();
        }
    }
} 