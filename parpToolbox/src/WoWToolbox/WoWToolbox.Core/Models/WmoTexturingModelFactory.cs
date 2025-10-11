using System.Collections.Generic;
using WoWToolbox.Core.WMO;
using System.Numerics;

namespace WoWToolbox.Core.Models
{
    /// <summary>
    /// Factory for converting parsed WMO chunk data into the unified texturing model.
    /// </summary>
    public static class WmoTexturingModelFactory
    {
        /// <summary>
        /// Converts WmoMotxChunk (MOTX) chunk data (byte array) to a WmoTextureBlock.
        /// </summary>
        public static WmoTextureBlock FromMotx(byte[] motxData)
        {
            return new WmoTextureBlock
            {
                TexturePaths = ReadNullTerminatedStrings(motxData)
            };
        }

        /// <summary>
        /// Converts a list of MOMT structs to a WmoMaterialBlock.
        /// </summary>
        public static WmoMaterialBlock FromMomt(List<MOMT> momtList)
        {
            var block = new WmoMaterialBlock();
            foreach (var m in momtList)
            {
                block.Materials.Add(new WmoMaterial
                {
                    Flags = m.Flags,
                    Shader = m.Shader,
                    BlendMode = m.BlendMode,
                    Texture1Index = (int)m.Texture1,
                    Texture2Index = (int)m.Texture2,
                    Texture3Index = (int)m.Texture3,
                    Color1 = m.Color1,
                    Color1b = m.Color1b,
                    Color2 = m.Color2,
                    Color3 = m.Color3,
                    GroupType = m.GroupType,
                    Flags3 = m.Flags3,
                    RuntimeData = m.RuntimeData != null ? (uint[])m.RuntimeData.Clone() : new uint[4]
                });
            }
            return block;
        }

        /// <summary>
        /// Converts MOGN chunk data (byte array) to a WmoGroupNameBlock.
        /// </summary>
        public static WmoGroupNameBlock FromMogn(byte[] mognData)
        {
            return new WmoGroupNameBlock
            {
                GroupNames = ReadNullTerminatedStrings(mognData) // MOGN uses same string block logic as MOTX
            };
        }

        /// <summary>
        /// Converts a list of MOGI structs to a WmoGroupInfoBlock.
        /// </summary>
        public static WmoGroupInfoBlock FromMogi(List<MOGI> mogiList)
        {
            var block = new WmoGroupInfoBlock();
            foreach (var g in mogiList)
            {
                block.Groups.Add(new WmoGroupInfo
                {
                    Flags = g.Flags,
                    BoundingBoxMin = g.BoundingBoxMin,
                    BoundingBoxMax = g.BoundingBoxMax,
                    NameIndex = g.NameIndex
                });
            }
            return block;
        }

        private static List<string> ReadNullTerminatedStrings(byte[] data)
        {
            var result = new List<string>();
            int start = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0)
                {
                    if (i > start)
                        result.Add(System.Text.Encoding.UTF8.GetString(data, start, i - start));
                    start = i + 1;
                }
            }
            if (start < data.Length)
                result.Add(System.Text.Encoding.UTF8.GetString(data, start, data.Length - start));
            return result;
        }
    }
} 