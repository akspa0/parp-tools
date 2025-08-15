using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace PM4Rebuilder
{
    /// <summary>
    /// Simple helper that scans a directory for .obj files and removes duplicates based on SHA256 hash of file contents.
    /// Only the first occurrence of a unique mesh is kept. Optionally, duplicates can be renamed with a suffix instead
    /// of deletion by setting <paramref name="deleteDuplicates"/> to false.
    /// </summary>
    internal static class ObjDeduplicator
    {
        public static void Deduplicate(string rootDir, bool deleteDuplicates = true)
        {
            if (!Directory.Exists(rootDir)) return;

            var seenHashes = new Dictionary<string, string>(); // hash -> path kept
            int duplicateCount = 0;
            using var sha = SHA256.Create();

            foreach (var objPath in Directory.EnumerateFiles(rootDir, "*.obj", SearchOption.AllDirectories))
            {
                try
                {
                    byte[] data = File.ReadAllBytes(objPath);
                    byte[] hashBytes = sha.ComputeHash(data);
                    string hash = Convert.ToHexString(hashBytes);

                    if (!seenHashes.ContainsKey(hash))
                    {
                        seenHashes[hash] = objPath;
                    }
                    else
                    {
                        duplicateCount++;
                        string keptPath = seenHashes[hash];
                        Console.WriteLine($"[DEDUP] Duplicate OBJ detected → '{objPath}' duplicates '{keptPath}'.");

                        if (deleteDuplicates)
                        {
                            File.Delete(objPath);
                        }
                        else
                        {
                            string newName = Path.Combine(Path.GetDirectoryName(objPath)!, Path.GetFileNameWithoutExtension(objPath) + "_DUP.obj");
                            File.Move(objPath, newName, overwrite: true);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[DEDUP WARNING] Could not process '{objPath}': {ex.Message}");
                }
            }

            Console.WriteLine($"[DEDUP] Completed. Unique OBJs: {seenHashes.Count}, Duplicates removed: {duplicateCount}");
        }
    }
}
