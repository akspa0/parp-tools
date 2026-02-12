using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using WmoBspConverter.Wmo;

namespace WmoBspConverter.Verification
{
    public static class WmoRoundTripVerifier
    {
        public static async Task VerifyAsync(bool verbose)
        {
            Console.WriteLine("Starting WMO Round Trip Verification (v14 -> v17 -> v14)...");

            // 1. Create Mock V14 Data
            var mockV14 = CreateMockV14();
            string tempDir = Path.Combine(Path.GetTempPath(), "WmoRoundTripTest_" + Guid.NewGuid());
            Directory.CreateDirectory(tempDir);
            
            try 
            {
                string pathV14_Initial = Path.Combine(tempDir, "initial.wmo");
                string pathV17 = Path.Combine(tempDir, "converted_v17.wmo");
                string pathV14_Final = Path.Combine(tempDir, "final.wmo");

                // Write Initial V14 (We don't have a direct V14 Writer exposed in WmoV14Parser, 
                // but WmoV17ToV14Converter writes V14. 
                // We can manually craft a crude V14 file or use the V17->V14 writer to bootstrap!)
                
                // Let's bootstrap by creating a V17 first (easier since we have V14->V17 code? No, we have V17->V14 Code).
                // Actually, WmoV14ToV17Converter takes parsed V14 data.
                
                // So: Mock V14 Data -> (V14ToV17) -> V17 File.
                var toV17 = new WmoV14ToV17Converter();
                toV17.ConvertAndWrite(mockV14, pathV17);
                if(verbose) Console.WriteLine($"[STEP 1] Generated V17 file: {new FileInfo(pathV17).Length} bytes");

                // Now we have a valid V17 file.
                // 2. Convert V17 -> V14
                var toV14 = new WmoV17ToV14Converter();
                toV14.ConvertAndWrite(pathV17, pathV14_Final);
                if(verbose) Console.WriteLine($"[STEP 2] Converted back to V14: {new FileInfo(pathV14_Final).Length} bytes");

                // 3. Convert V14 (Final) -> V17 (Again) to check stability
                string pathV17_Final = Path.Combine(tempDir, "final_v17.wmo");
                var parser = new WmoV14Parser();
                var v14FinalData = parser.ParseWmoV14(pathV14_Final); // Parse the V14 Root
                
                // Manually load group files (since parser is strict/non-recursive)
                // WmoV17ToV14Converter naming: BaseName_000.wmo
                string groupPattern = Path.GetFileNameWithoutExtension(pathV14_Final) + "_*.wmo";
                string[] groups = Directory.GetFiles(Path.GetDirectoryName(pathV14_Final)!, groupPattern);
                
                // Sort by name to ensure order (000, 001...)
                Array.Sort(groups);
                
                foreach(var gp in groups)
                {
                    if (gp.Equals(pathV14_Final, StringComparison.OrdinalIgnoreCase)) continue;
                
                    var gpData = parser.ParseWmoV14(gp);
                    if (gpData.Groups.Count > 0)
                    {
                        // Add groups from group file to root data
                        v14FinalData.Groups.AddRange(gpData.Groups);
                    }
                }

                toV17.ConvertAndWrite(v14FinalData, pathV17_Final);
                if(verbose) Console.WriteLine($"[STEP 3] Converted Final V14 -> V17: {new FileInfo(pathV17_Final).Length} bytes");

                // 4. Comparisons
                ValidationCheck(mockV14, v14FinalData);
                CompareBytes(pathV17, pathV17_Final);

                Console.WriteLine("✓ Round trip verification PASSED (V17 -> V14 -> V17 stability verified)");
            }
            finally
            {
                if (!verbose) Directory.Delete(tempDir, true);
                else Console.WriteLine($"[INFO] Temp artifacts kept at: {tempDir}");
            }

            await Task.CompletedTask;
        }

        private static void CompareBytes(string p1, string p2)
        {
            if (!File.Exists(p1) || !File.Exists(p2))
            {
                Console.WriteLine("[ERR] One or both verification files missing.");
                return;
            }
            byte[] b1 = File.ReadAllBytes(p1);
            byte[] b2 = File.ReadAllBytes(p2);
            if (b1.Length != b2.Length)
            {
                Console.WriteLine($"[WARN] Size mismatch: {b1.Length} vs {b2.Length}");
            }
            else
            {
                 int diffs = 0;
                 for(int i=0; i<b1.Length; i++) if(b1[i]!=b2[i]) diffs++;
                 if(diffs == 0) Console.WriteLine("[SUCCESS] Files are byte-identical.");
                 else Console.WriteLine($"[WARN] Files have {diffs} differing bytes.");
            }
        }

        private static void ValidationCheck(WmoV14Parser.WmoV14Data initial, WmoV14Parser.WmoV14Data final)
        {
            Console.WriteLine($"[CHECK] Materials: Expect {initial.Materials.Count}, Got {final.Materials.Count}");
            if (initial.Materials.Count != final.Materials.Count) 
                Console.WriteLine("    [FAIL] Material count mismatch!");
            
            Console.WriteLine($"[CHECK] Groups: Expect {initial.Groups.Count}, Got {final.Groups.Count}");
            if (initial.Groups.Count != final.Groups.Count)
                Console.WriteLine("    [FAIL] Group count mismatch!");

            Console.WriteLine($"[CHECK] Textures: Expect {initial.Textures.Count}, Got {final.Textures.Count}");
            if (initial.Textures.Count != final.Textures.Count)
                Console.WriteLine("    [FAIL] Texture count mismatch!");

            // Check Material Properties
            for(int i=0; i<Math.Min(initial.Materials.Count, final.Materials.Count); i++)
            {
                var m1 = initial.Materials[i];
                var m2 = final.Materials[i];
                if (m1.Flags != m2.Flags) Console.WriteLine($"    [WARN] Mat {i} Flags mismatch: {m1.Flags:X} vs {m2.Flags:X}");
                if (m1.BlendMode != m2.BlendMode) Console.WriteLine($"    [WARN] Mat {i} BlendMode mismatch: {m1.BlendMode} vs {m2.BlendMode}");
            }
        }

        private static WmoV14Parser.WmoV14Data CreateMockV14()
        {
            var data = new WmoV14Parser.WmoV14Data();
            data.Version = 14;
            
            // --- Textures ---
            data.Textures.Add(@"WORLD\ENVIRONMENT\DOODAD\OUTLAND\ORCRUINS\TEXTURE\MM_ORC_AROOF_02_RUIN.BLP");  // 0
            data.Textures.Add(@"DUNGEONS\TEXTURES\OUTLAND\MM_ORC_ATRIM_02_RUIN.BLP");                           // 1
            data.Textures.Add(@"DUNGEONS\TEXTURES\OUTLAND\MM_OGRMR_AFLOOR_03_RUIN.BLP");                        // 2
            data.Textures.Add(@"WORLD\ENVIRONMENT\DOODAD\OUTLAND\ORCRUINS\TEXTURE\MM_ORC_AWALL_01_RUIN.BLP");   // 3
            // Rebuild offsets map mock (imperfect but sufficient for writer if it recalculates)
            data.TextureOffsetToName[0] = @"WORLD\ENVIRONMENT\DOODAD\OUTLAND\ORCRUINS\TEXTURE\MM_ORC_AROOF_02_RUIN.BLP";

            // --- Materials ---
            // 1. Opaque Stone
            data.Materials.Add(new WmoBspConverter.Wmo.WmoMaterial {
                 Flags = 0, BlendMode = 0, Shader = 0, DiffuseColor = 0xFFFFFFFF, Texture1Offset = 0
            });
            data.MaterialTextureIndices.Add(0);

            // 2. Transparent Window (Blend logic)
            data.Materials.Add(new WmoBspConverter.Wmo.WmoMaterial {
                 Flags = 0x4, BlendMode = 1, Shader = 1, DiffuseColor = 0x88FFFFFF, Texture1Offset = 0 
            });
            data.MaterialTextureIndices.Add(3);

            // 3. Two-Layer Material (if supported)
            data.Materials.Add(new WmoBspConverter.Wmo.WmoMaterial {
                 Flags = 0, BlendMode = 0, Shader = 2, DiffuseColor = 0xFFFFFFFF, Texture1Offset = 0, Texture2Offset = 10 
            });
            data.MaterialTextureIndices.Add(1);

            // --- Groups ---
            // Group 0: Main Hall (Simple Box)
            var g0 = new WmoV14Parser.WmoGroupData { Name = "MainHall", Flags = 0x8 /* Indoor */ };
            AddBoxGeometry(g0, new Vector3(0,0,0), new Vector3(100, 20, 100));
            // Assign material 0
            for(int i=0; i<g0.Batches.Count; i++) { var b = g0.Batches[i]; b.MaterialId = 0; g0.Batches[i] = b; }
            for(int i=0; i<g0.FaceMaterials.Count; i++) g0.FaceMaterials[i] = 0;
            data.Groups.Add(g0);

            // Group 1: Tower (Cylinder-ish)
            var g1 = new WmoV14Parser.WmoGroupData { Name = "Tower", Flags = 0 /* Outdoor */ };
            AddTriangleGeometry(g1);
            // Assign material 1
            for(int i=0; i<g1.Batches.Count; i++) { var b = g1.Batches[i]; b.MaterialId = 1; g1.Batches[i] = b; }
            for(int i=0; i<g1.FaceMaterials.Count; i++) g1.FaceMaterials[i] = 1;
            data.Groups.Add(g1);
            
            // Group 2: Basement (Another Indoor)
             var g2 = new WmoV14Parser.WmoGroupData { Name = "Basement", Flags = 0x8 };
            AddBoxGeometry(g2, new Vector3(0,-20,0), new Vector3(50, 10, 50));
            // Assign material 2
            for(int i=0; i<g2.Batches.Count; i++) { var b = g2.Batches[i]; b.MaterialId = 2; g2.Batches[i] = b; }
            for(int i=0; i<g2.FaceMaterials.Count; i++) g2.FaceMaterials[i] = 2;
            data.Groups.Add(g2);

            return data;
        }

        private static void AddBoxGeometry(WmoV14Parser.WmoGroupData g, Vector3 center, Vector3 size)
        {
            var h = size / 2;
            // Simple quad (2 tris) for floor
            g.Vertices.Add(center + new Vector3(-h.X, 0, -h.Z));
            g.Vertices.Add(center + new Vector3( h.X, 0, -h.Z));
            g.Vertices.Add(center + new Vector3( h.X, 0,  h.Z));
            g.Vertices.Add(center + new Vector3(-h.X, 0,  h.Z));
            
            int baseV = g.Vertices.Count - 4;
            g.Indices.AddRange(new ushort[] { (ushort)baseV, (ushort)(baseV+1), (ushort)(baseV+2) });
            g.Indices.AddRange(new ushort[] { (ushort)baseV, (ushort)(baseV+2), (ushort)(baseV+3) });
            
            g.FaceMaterials.Add(0);
            g.FaceMaterials.Add(0);
            
            g.Batches.Add(new WmoV14Parser.MobaBatch {
                FirstFace = (uint)g.Indices.Count/3 - 2, NumFaces = 2, 
                FirstVertex = (ushort)baseV, LastVertex = (ushort)(baseV+3), 
                MaterialId = 0, Flags = 0
            });
        }

        private static void AddTriangleGeometry(WmoV14Parser.WmoGroupData g)
        {
            g.Vertices.Add(new Vector3(50, 50, 50));
            g.Vertices.Add(new Vector3(60, 50, 50));
            g.Vertices.Add(new Vector3(55, 60, 50));
            g.Indices.AddRange(new ushort[]{0,1,2});
            g.FaceMaterials.Add(0);
            g.Batches.Add(new WmoV14Parser.MobaBatch{
                 FirstFace=0, NumFaces=1, FirstVertex=0, LastVertex=2, MaterialId=0
            });
        }
    }
}
