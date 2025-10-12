using System;
using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Formats.WMO.Chunks;
using ArcaneFileParser.Core.Formats.WMO.Validation;

namespace ArcaneFileParser.Core.Formats.WMO.Converters
{
    /// <summary>
    /// Handles conversion of WMO files from v17 to v14 format
    /// </summary>
    public class WmoV17ToV14Converter
    {
        private readonly WMOFile sourceWmo;
        private readonly WMOFile targetWmo;
        private readonly List<string> conversionErrors;

        // Flag conversion mappings (v17 to v14)
        private static readonly Dictionary<uint, uint> MohdFlagMap = new Dictionary<uint, uint>
        {
            { 0x1, 0x1 },     // DO_NOT_ATTENUATE_VERTICES_BASED_ON_DISTANCE_TO_PORTAL
            { 0x2, 0x2 },     // USE_UNIFIED_RENDER_PATH
            { 0x4, 0x4 },     // USE_LIQUID_TYPE_DBC_ID
            { 0x8, 0x8 },     // DO_NOT_FIX_VERTEX_COLOR_ALPHA
            { 0x10, 0x0 },    // LOD -> removed in v14
            { 0x20, 0x0 }     // DEFAULT_MAX_LOD -> removed in v14
        };

        private static readonly Dictionary<uint, uint> MogpFlagMap = new Dictionary<uint, uint>
        {
            { 0x1, 0x1 },     // HAS_BSP
            { 0x4, 0x4 },     // HAS_VERTEX_COLORS
            { 0x8, 0x8 },     // EXTERIOR
            { 0x20, 0x20 },   // NO_COLLISION
            { 0x40, 0x40 },   // EXTERIOR_LIT
            { 0x80, 0x80 },   // UNREACHABLE
            { 0x100, 0x100 }, // SHOW_SKY
            { 0x200, 0x200 }, // HAS_LIGHTS
            { 0x800, 0x800 }, // HAS_DOODADS
            { 0x1000, 0x1000 }, // HAS_WATER
            { 0x2000, 0x2000 }, // INDOOR
            { 0x8000, 0x8000 }, // HAS_BSP2
            { 0x10000, 0x10000 } // ALWAYS_DRAW
        };

        public WmoV17ToV14Converter(WMOFile source)
        {
            if (source.Version != 17)
                throw new ArgumentException("Source WMO must be version 17");

            sourceWmo = source;
            targetWmo = new WMOFile();
            conversionErrors = new List<string>();
        }

        public WMOFile Convert()
        {
            try
            {
                // Validate source WMO collision integrity
                var sourceValidator = new WmoCollisionValidator(sourceWmo);
                if (!sourceValidator.Validate())
                {
                    foreach (var error in sourceValidator.GetValidationErrors())
                    {
                        conversionErrors.Add($"Source validation: {error}");
                    }
                    throw new InvalidDataException("Source WMO failed collision validation");
                }

                // Convert root chunks
                ConvertRootChunks();

                // Convert group chunks
                ConvertGroupChunks();

                // Validate target WMO collision integrity
                var targetValidator = new WmoCollisionValidator(targetWmo);
                if (!targetValidator.Validate())
                {
                    foreach (var error in targetValidator.GetValidationErrors())
                    {
                        conversionErrors.Add($"Target validation: {error}");
                    }
                    throw new InvalidDataException("Converted WMO failed collision validation");
                }

                return targetWmo;
            }
            catch (Exception ex)
            {
                conversionErrors.Add($"Conversion error: {ex.Message}");
                throw;
            }
        }

        public IReadOnlyList<string> GetConversionErrors() => conversionErrors;

        private void ConvertRootChunks()
        {
            // Set version to 14
            var mver = new MVER { Version = 14 };
            targetWmo.Chunks["MVER"] = mver;

            // Convert MOHD
            ConvertMOHD();

            // Copy unchanged chunks
            CopyChunk("MOTX");
            CopyChunk("MOMT");
            CopyChunk("MOGN");
            CopyChunk("MOGI");
            CopyChunk("MOSB");

            // Convert portal chunks
            ConvertPortalChunks();

            // Convert lighting
            ConvertLighting();

            // Convert doodads
            ConvertDoodadChunks();

            // Copy fog
            CopyChunk("MFOG");
        }

        private void ConvertMOHD()
        {
            var sourceMohd = sourceWmo.GetChunk<MOHD>("MOHD");
            var targetMohd = new MOHD
            {
                TextureCount = sourceMohd.TextureCount,
                GroupCount = sourceMohd.GroupCount,
                PortalCount = sourceMohd.PortalCount,
                LightCount = sourceMohd.LightCount,
                DoodadNameCount = sourceMohd.DoodadNameCount,
                DoodadDefinitionCount = sourceMohd.DoodadDefinitionCount,
                DoodadSetCount = sourceMohd.DoodadSetCount,
                AmbientColor = sourceMohd.AmbientColor,
                WmoId = sourceMohd.WmoId,
                BoundingBox = sourceMohd.BoundingBox.Clone(),
                IsV14 = true,
                InMemoryPadding = new byte[8] // v14 requires 8 bytes of padding
            };

            targetWmo.Chunks["MOHD"] = targetMohd;
        }

        private void ConvertPortalChunks()
        {
            // Convert MOPT (Portal definitions)
            var sourceMopt = sourceWmo.GetChunk<MOPT>("MOPT");
            if (sourceMopt != null)
            {
                var targetMopt = new MOPT();
                foreach (var portal in sourceMopt.Portals)
                {
                    targetMopt.AddPortal(new Portal
                    {
                        BasePosition = portal.BasePosition.Clone(),
                        Normal = portal.Normal.Clone(),
                        VertexCount = portal.VertexCount
                    });
                }
                targetWmo.Chunks["MOPT"] = targetMopt;
            }

            // Copy MOPV (Portal vertices)
            CopyChunk("MOPV");

            // Convert MOPR (Portal references)
            var sourceMopr = sourceWmo.GetChunk<MOPR>("MOPR");
            if (sourceMopr != null)
            {
                var targetMopr = new MOPR();
                foreach (var reference in sourceMopr.References)
                {
                    targetMopr.AddReference(new PortalReference
                    {
                        PortalIndex = reference.PortalIndex,
                        GroupIndex = reference.GroupIndex,
                        Side = reference.Side
                    });
                }
                targetWmo.Chunks["MOPR"] = targetMopr;
            }
        }

        private void ConvertLighting()
        {
            // Convert modern lighting to v14 lightmaps
            var molt = sourceWmo.GetChunk<MOLT>("MOLT");
            if (molt != null)
            {
                // Create MOLM chunk for v14 lightmap info
                var molm = new MOLM();
                var mold = new MOLD(); // For lightmap textures

                // Convert each light to a lightmap entry
                foreach (var light in molt.Lights)
                {
                    if (light.Type == LightType.Ambient)
                    {
                        var entry = new MOLM.LightmapEntry
                        {
                            X = 0,
                            Y = 0,
                            Width = 16,
                            Height = 16
                        };
                        molm.Lightmaps.Add(entry);

                        // Create a simple lightmap texture with the light's color
                        var texture = new MOLD.LightmapTexture
                        {
                            Texels = CreateSolidColorTexture(light.Color)
                        };
                        mold.Textures.Add(texture);
                    }
                }

                if (molm.Lightmaps.Count > 0)
                {
                    targetWmo.Chunks["MOLM"] = molm;
                    targetWmo.Chunks["MOLD"] = mold;
                }
            }
        }

        private byte[] CreateSolidColorTexture(uint color)
        {
            // Create a 16x16 DXT1 texture with a solid color
            // This is a simplified version - in practice you'd want proper DXT1 compression
            var texels = new byte[128]; // 16x16 pixels in DXT1 format = 128 bytes
            for (int i = 0; i < texels.Length; i += 8)
            {
                BitConverter.GetBytes(color).CopyTo(texels, i);
                BitConverter.GetBytes(color).CopyTo(texels, i + 4);
            }
            return texels;
        }

        private void ConvertDoodadChunks()
        {
            // Copy doodad chunks if they exist
            CopyChunk("MODS");
            CopyChunk("MODN");
            CopyChunk("MODD");
        }

        private void CopyChunk(string chunkId)
        {
            if (sourceWmo.Chunks.TryGetValue(chunkId, out var chunk))
            {
                targetWmo.Chunks[chunkId] = chunk;
            }
        }

        private void ConvertGroupChunks()
        {
            foreach (var sourceGroup in sourceWmo.Groups)
            {
                var targetGroup = new WMOGroup();

                // Set version to 14
                targetGroup.Chunks["MVER"] = new MVER { Version = 14 };

                // Convert MOGP
                ConvertGroupMOGP(sourceGroup, targetGroup);

                // Copy geometry chunks
                CopyGroupChunk(sourceGroup, targetGroup, "MOPY");
                CopyGroupChunk(sourceGroup, targetGroup, "MOVI");
                CopyGroupChunk(sourceGroup, targetGroup, "MOVT");
                CopyGroupChunk(sourceGroup, targetGroup, "MONR");
                CopyGroupChunk(sourceGroup, targetGroup, "MOTV");

                // Convert batches
                ConvertGroupBatches(sourceGroup, targetGroup);

                // Copy BSP tree
                CopyGroupChunk(sourceGroup, targetGroup, "MOBN");
                CopyGroupChunk(sourceGroup, targetGroup, "MOBR");

                targetWmo.Groups.Add(targetGroup);
            }
        }

        private void ConvertGroupMOGP(WMOGroup source, WMOGroup target)
        {
            var sourceMogp = source.GetChunk<MOGP>("MOGP");
            var targetMogp = new MOGP(14) // Create MOGP with v14 version
            {
                GroupNameOffset = sourceMogp.GroupNameOffset,
                DescriptiveGroupNameOffset = sourceMogp.DescriptiveGroupNameOffset,
                Flags = ConvertMogpFlags(sourceMogp.Flags),
                BoundingBox = sourceMogp.BoundingBox.Clone(),
                PortalStart = sourceMogp.PortalStart,
                PortalCount = sourceMogp.PortalCount,
                TransBatchCount = sourceMogp.TransBatchCount,
                IntBatchCount = sourceMogp.IntBatchCount,
                ExtBatchCount = sourceMogp.ExtBatchCount,
                BatchTypeD = sourceMogp.BatchTypeD,
                FogIndices = (byte[])sourceMogp.FogIndices.Clone(),
                GroupLiquid = sourceMogp.GroupLiquid,
                UniqueId = sourceMogp.UniqueId
            };

            target.Chunks["MOGP"] = targetMogp;
        }

        private uint ConvertMogpFlags(uint sourceFlags)
        {
            uint targetFlags = 0;
            foreach (var flagPair in MogpFlagMap)
            {
                if ((sourceFlags & flagPair.Key) != 0)
                {
                    targetFlags |= flagPair.Value;
                }
            }
            return targetFlags;
        }

        private void CopyGroupChunk(WMOGroup source, WMOGroup target, string chunkId)
        {
            if (source.Chunks.TryGetValue(chunkId, out var chunk))
            {
                target.Chunks[chunkId] = chunk;
            }
        }

        private void ConvertGroupBatches(WMOGroup source, WMOGroup target)
        {
            var sourceMogp = source.GetChunk<MOGP>("MOGP");
            var sourceMoba = source.GetChunk<MOBA>("MOBA");
            var targetMogp = target.GetChunk<MOGP>("MOGP");

            if (sourceMoba != null)
            {
                // Convert MOBA batches to v14 batch format
                foreach (var batch in sourceMoba.Batches)
                {
                    var v14Batch = new SMOGxBatch
                    {
                        StartVertex = batch.StartVertex,
                        Count = batch.VertexCount,
                        MinIndex = batch.MinIndex,
                        MaxIndex = batch.MaxIndex,
                        MaterialId = batch.MaterialId
                    };

                    if (batch.Flags == 0) // Interior batch
                        targetMogp.InteriorBatches.Add(v14Batch);
                    else // Exterior batch
                        targetMogp.ExteriorBatches.Add(v14Batch);
                }
            }
        }
    }
} 