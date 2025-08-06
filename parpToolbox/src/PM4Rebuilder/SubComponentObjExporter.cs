using Microsoft.Data.Sqlite;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace PM4Rebuilder
{
    /// <summary>
    /// Extremely thin prototype that exports the vertices referenced by all Links rows
    /// that share a given ParentIndex (ObjectId) into a Wavefront OBJ.  Faces are
    /// not emitted yet; the goal is a quick visualisation of the point cloud / sub-component
    /// for verification.
    /// </summary>
    internal static class SubComponentObjExporter
    {
        public static int Export(string dbPath, uint objectId, string outPath)
        {
            using var conn = new SqliteConnection($"Data Source={dbPath};Mode=ReadOnly");
            conn.Open();

            // Step 1: fetch all Links rows for this object
            string sqlLinks = @"SELECT MspiFirstIndex, MspiIndexCount FROM Links WHERE ParentIndex = @objectId";
            using var cmdLinks = new SqliteCommand(sqlLinks, conn);
            cmdLinks.Parameters.AddWithValue("@objectId", objectId);

            using var reader = cmdLinks.ExecuteReader();
            if (!reader.HasRows)
            {
                Console.WriteLine($"[OBJ-EXPORT] No Links rows found for object {objectId}");
                return 0;
            }

            // Collect all vertex index ranges referenced by the MSPI spans
            var vertexIndices = new HashSet<int>();
            while (reader.Read())
            {
                int first = reader.GetInt32(0);
                int count = reader.GetInt32(1);
                for (int i = first; i < first + count; i++)
                {
                    vertexIndices.Add(i);
                }
            }

            if (vertexIndices.Count == 0)
            {
                Console.WriteLine($"[OBJ-EXPORT] No vertex indices gathered for object {objectId}");
                return 0;
            }

            // Step 2: fetch vertex positions
            string sqlVertices = @"SELECT GlobalIndex, X, Y, Z FROM Vertices WHERE GlobalIndex IN (" + string.Join(',', vertexIndices) + ") ORDER BY GlobalIndex";
            using var cmdVerts = new SqliteCommand(sqlVertices, conn);
            using var rVerts = cmdVerts.ExecuteReader();

            var verts = new List<(int Index, float X, float Y, float Z)>();
            while (rVerts.Read())
            {
                int idx = rVerts.GetInt32(0);
                float x = rVerts.GetFloat(1);
                float y = rVerts.GetFloat(2);
                float z = rVerts.GetFloat(3);
                verts.Add((idx, x, y, z));
            }

            if (verts.Count == 0)
            {
                Console.WriteLine($"[OBJ-EXPORT] No vertices retrieved for object {objectId}");
                return 0;
            }

            // Map original vertex indices to sequential OBJ indices
            var objIndexMap = verts.Select((v, i) => new { v.Index, ObjIdx = i + 1 })
                                   .ToDictionary(k => k.Index, v => v.ObjIdx);

            // Step 2b: fetch triangles covering the MSPI range to build faces
            var faces = new List<(int A,int B,int C)>();
            string sqlTri = @"SELECT GlobalIndex, VertexA, VertexB, VertexC FROM Triangles WHERE GlobalIndex BETWEEN @start AND @end";
            foreach(var linkRow in vertexIndices.Select(idx=>idx/3).Distinct())
            {
                int start = linkRow;
                int end = start; // single triangle
                using var cmdT = new SqliteCommand(sqlTri, conn);
                cmdT.Parameters.AddWithValue("@start", start);
                cmdT.Parameters.AddWithValue("@end", start);
                using var rt = cmdT.ExecuteReader();
                while(rt.Read())
                {
                    int a = rt.GetInt32(1);
                    int b = rt.GetInt32(2);
                    int c = rt.GetInt32(3);
                    if(objIndexMap.TryGetValue(a,out int va)&& objIndexMap.TryGetValue(b,out int vb)&& objIndexMap.TryGetValue(c,out int vc))
                        faces.Add((va,vb,vc));
                }
            }

            // Step 3: write OBJ file – vertices and faces
            var sb = new StringBuilder();
            sb.AppendLine($"# Sub-component OBJ for ObjectId {objectId}");
            foreach (var v in verts)
            {
                sb.Append("v ");
                sb.Append(v.X.ToString("F6", CultureInfo.InvariantCulture)).Append(' ');
                sb.Append(v.Y.ToString("F6", CultureInfo.InvariantCulture)).Append(' ');
                sb.Append(v.Z.ToString("F6", CultureInfo.InvariantCulture)).Append('\n');
            }
            sb.AppendLine("\n# No faces emitted – point cloud only");

            Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
            File.WriteAllText(outPath, sb.ToString());
            Console.WriteLine($"[OBJ-EXPORT] Wrote {verts.Count:N0} vertices to {outPath}");
            return verts.Count;
        }
    }
}
