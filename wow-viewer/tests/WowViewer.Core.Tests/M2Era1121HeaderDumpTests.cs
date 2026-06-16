using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;

namespace WowViewer.Core.Tests;

public sealed class M2Era1121HeaderDumpTests
{
    [Fact]
    public void Dump_1121_Bear_Header_For_Real_Data_Analysis()
    {
        if (!TryReadStagedVirtualFile(
                "1.X_Retail_Windows_enUS_1.12.1.5875",
                "creature\\bear\\bear.m2",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            string root = GetWowViewerRoot();
            string clientDataPath = Path.Combine(root, "..", "output", "tmp", "wowarchive-clients", "1.X_Retail_Windows_enUS_1.12.1.5875", "World of Warcraft", "Data");
            string listfilePath = Path.Combine(root, "libs", "wowdev", "wow-listfile", "listfile.txt");
            throw new InvalidOperationException($"Could not load staged virtual file. ClientDataPath: {clientDataPath} (Exists={Directory.Exists(clientDataPath)}), ListfilePath: {listfilePath} (Exists={File.Exists(listfilePath)})");
        }

        Assert.True(bytes.Length > 0);
        string signature = Encoding.ASCII.GetString(bytes, 0, 4);
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4));

        var sb = new StringBuilder();
        sb.AppendLine($"bear.mdx total size = {bytes.Length} bytes, signature = {signature}, version = 0x{version:X}");
        sb.AppendLine();

        // Dump every uint32 in the full 0x144-byte header
        sb.AppendLine("=== FULL HEADER (uint32 at every 4-byte offset) ===");
        int fullHeaderSize = Math.Min(0x144, bytes.Length);
        for (int offset = 0; offset + 4 <= fullHeaderSize; offset += 4)
        {
            uint val = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
            float fval = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4)));
            string floatStr = float.IsFinite(fval) && Math.Abs(fval) > 0.0001f && Math.Abs(fval) < 100000f ? $"  float={fval:F4}" : "";
            sb.AppendLine($"  [0x{offset:X3}] = 0x{val:X8} ({val,10}){floatStr}");
        }
        sb.AppendLine();

        // Dump the known count/offset pairs from native trace
        sb.AppendLine("=== NAMED HEADER TABLES (count/offset pairs) ===");
        (string name, int countOff, int offsetOff, int stride)[] tables = [
            ("name", 0x08, 0x0C, 1),
            ("global_seq", 0x14, 0x18, 4),
            ("anim", 0x1C, 0x20, 0x6C),
            ("anim_lookup", 0x24, 0x28, 2),
            ("tex_anim", 0x2C, 0x30, 2),
            ("bone", 0x34, 0x38, 4),
            ("view", 0x3C, 0x40, 0x2C),
            ("color", 0x44, 0x48, 0x1C),
            ("texture", 0x4C, 0x50, 0x1C),
            ("tex_weight", 0x54, 0x58, 8),
            ("tex_lookup", 0x5C, 0x60, 4),
            ("tex_unit_lookup", 0x64, 0x68, 2),
            ("tex_repl_lookup", 0x6C, 0x70, 2),
            ("tex_flag_lookup", 0x74, 0x78, 2),
            ("bounding_tri", 0x7C, 0x80, 0x0C),
            ("bounding_vert", 0x84, 0x88, 0x0C),
            ("render_flag", 0x8C, 0x90, 0x10),
            ("lod_table", 0x94, 0x98, 0x38),
            ("collision", 0x9C, 0xA0, 2),
            ("attach", 0xA4, 0xA8, 0x0C),
            ("light", 0xAC, 0xB0, 0x0C),
            ("camera", 0xB4, 0xB8, 0x2C),
            ("cam_perframe", 0xBC, 0xC0, 0xD4),
            ("ribbon", 0xC4, 0xC8, 0x7C),
            ("particle", 0xCC, 0xD0, 0xDC),
        ];
        foreach (var (name, cOff, oOff, stride) in tables)
        {
            if (cOff + 4 > bytes.Length || oOff + 4 > bytes.Length) continue;
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(cOff, 4));
            uint dataOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(oOff, 4));
            long dataEnd = (long)dataOffset + (long)count * stride;
            sb.AppendLine($"  {name,-20} [0x{cOff:X2}/0x{oOff:X2}] count={count,6}  offset=0x{dataOffset:X6}  stride=0x{stride:X2}  end=0x{dataEnd:X6}  inFile={dataEnd <= bytes.Length}");
        }
        sb.AppendLine();

        // Dump bounds region as floats
        sb.AppendLine("=== POTENTIAL BOUNDS REGIONS (as floats) ===");
        foreach (int startOff in new[] { 0xA0, 0xB4, 0xD4, 0xE4, 0xEC })
        {
            sb.Append($"  Starting at 0x{startOff:X2}: ");
            for (int i = 0; i < 7 && startOff + i * 4 + 4 <= bytes.Length; i++)
            {
                float f = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(startOff + i * 4, 4)));
                sb.Append($"[{f:F3}] ");
            }
            sb.AppendLine();
        }
        sb.AppendLine();

        // Also dump the geometry tables (what the current constants say)
        sb.AppendLine("=== GEOMETRY TABLES (current constants) ===");
        (string gname, int gcOff, int goOff, int gstride)[] geoTables = [
            ("vertex_idx", 0xEC, 0xF0, 4),
            ("position", 0xF4, 0xF8, 0x0C),
            ("normal", 0xFC, 0x100, 0x0C),
            ("uv", 0x104, 0x108, 8),
            ("triangle", 0x10C, 0x110, 2),
            ("batch", 0x114, 0x118, 0x1C),
            ("extra0", 0x11C, 0x120, 2),
            ("extra1", 0x124, 0x128, 0x1F8),
            ("extra2", 0x12C, 0x130, 2),
        ];
        foreach (var (gname, gcOff, goOff, gstride) in geoTables)
        {
            if (gcOff + 4 > bytes.Length || goOff + 4 > bytes.Length) continue;
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(gcOff, 4));
            uint dataOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(goOff, 4));
            long dataEnd = (long)dataOffset + (long)count * gstride;
            sb.AppendLine($"  {gname,-14} [0x{gcOff:X3}/0x{goOff:X3}] count={count,6}  offset=0x{dataOffset:X6}  stride=0x{gstride:X2}  end=0x{dataEnd:X6}  inFile={dataEnd <= bytes.Length}");
        }

        // Dump texture table
        sb.AppendLine("=== TEXTURE TABLE DUMP ===");
        if (0x4C + 4 <= bytes.Length && 0x50 + 4 <= bytes.Length)
        {
            uint texCount = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x4C, 4));
            uint texOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x50, 4));
            sb.AppendLine($"  Texture count: {texCount}, offset: 0x{texOffset:X6}");
            for (int i = 0; i < texCount; i++)
            {
                int start = checked((int)texOffset + i * 28);
                if (start + 28 <= bytes.Length)
                {
                    var hex = BitConverter.ToString(bytes, start, 28).Replace("-", " ");
                    sb.AppendLine($"    Record {i}: {hex}");
                }
            }
        }
        sb.AppendLine();

        // Dump render flag table
        sb.AppendLine("=== RENDER FLAG TABLE DUMP ===");
        if (0x8C + 4 <= bytes.Length && 0x90 + 4 <= bytes.Length)
        {
            uint rfCount = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x8C, 4));
            uint rfOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x90, 4));
            sb.AppendLine($"  RenderFlag count: {rfCount}, offset: 0x{rfOffset:X6}");
            for (int i = 0; i < rfCount; i++)
            {
                int start = checked((int)rfOffset + i * 16);
                if (start + 16 <= bytes.Length)
                {
                    ushort flags = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(start, 2));
                    ushort blendMode = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(start + 2, 2));
                    uint r1 = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(start + 4, 4));
                    uint r2 = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(start + 8, 4));
                    uint r3 = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(start + 12, 4));
                    sb.AppendLine($"    RenderFlag[{i}]: flags=0x{flags:X4}, blendMode={blendMode}, remainder: 0x{r1:X8} 0x{r2:X8} 0x{r3:X8}");
                }
            }
        }
        sb.AppendLine();

        // Try parsing textures with stride 16 vs 28
        sb.AppendLine("=== TEXTURE STRIDE ANALYSIS ===");
        if (0x4C + 4 <= bytes.Length && 0x50 + 4 <= bytes.Length)
        {
            uint texCount = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x4C, 4));
            uint texOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0x50, 4));
            sb.AppendLine($"Texture count: {texCount}, offset: 0x{texOffset:X6}");

            // Try stride 16
            sb.AppendLine("  --- Trying Stride 16 ---");
            for (int i = 0; i < texCount; i++)
            {
                int entry = checked((int)texOffset + i * 16);
                if (entry + 16 <= bytes.Length)
                {
                    uint type = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry, 4));
                    uint flags = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 4, 4));
                    uint nameLen = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 8, 4));
                    uint nameOfs = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 12, 4));
                    string name = "";
                    if (nameLen > 0 && nameOfs > 0 && nameOfs + nameLen <= bytes.Length)
                    {
                        name = Encoding.ASCII.GetString(bytes, (int)nameOfs, (int)nameLen).TrimEnd('\0');
                    }
                    sb.AppendLine($"    Tex[{i}]: type={type}, flags={flags}, nameLen={nameLen}, nameOfs=0x{nameOfs:X6}, name=\"{name}\"");
                }
            }

            // Try stride 28
            sb.AppendLine("  --- Trying Stride 28 ---");
            for (int i = 0; i < texCount; i++)
            {
                int entry = checked((int)texOffset + i * 28);
                if (entry + 28 <= bytes.Length)
                {
                    uint type = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry, 4));
                    uint flags = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 4, 4));
                    uint nameLen = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 8, 4));
                    uint nameOfs = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(entry + 12, 4));
                    string name = "";
                    if (nameLen > 0 && nameOfs > 0 && nameOfs + nameLen <= bytes.Length)
                    {
                        name = Encoding.ASCII.GetString(bytes, (int)nameOfs, (int)nameLen).TrimEnd('\0');
                    }
                    sb.AppendLine($"    Tex[{i}]: type={type}, flags={flags}, nameLen={nameLen}, nameOfs=0x{nameOfs:X6}, name=\"{name}\"");
                }
            }
        }
        sb.AppendLine();
        // Scan for all printable ASCII strings of length >= 4 in the bytes
        sb.AppendLine("=== ALL PRINTABLE ASCII STRINGS (>= 4 chars) ===");
        int currentLen = 0;
        int startIdx = -1;
        for (int i = 0; i < bytes.Length; i++)
        {
            byte b = bytes[i];
            if (b >= 0x20 && b <= 0x7E)
            {
                if (startIdx == -1) startIdx = i;
                currentLen++;
            }
            else
            {
                if (currentLen >= 4)
                {
                    string s = Encoding.ASCII.GetString(bytes, startIdx, currentLen);
                    sb.AppendLine($"  [0x{startIdx:X5}] (len={currentLen}): \"{s}\"");
                }
                startIdx = -1;
                currentLen = 0;
            }
        }
        if (currentLen >= 4)
        {
            string s = Encoding.ASCII.GetString(bytes, startIdx, currentLen);
            sb.AppendLine($"  [0x{startIdx:X5}] (len={currentLen}): \"{s}\"");
        }

        Console.WriteLine(sb.ToString());
    }

    [Fact]
    public void Dump_1121_Bonepile_Header_For_Real_Data_Analysis()
    {
        if (!TryReadStagedVirtualFile(
                "1.X_Retail_Windows_enUS_1.12.1.5875",
                "world\\azeroth\\theblastedlands\\passivedoodads\\bones\\blastedlandsbonepile02.mdx",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            throw new InvalidOperationException("Could not load bonepile virtual file.");
        }

        Assert.True(bytes.Length > 0);
        string signature = Encoding.ASCII.GetString(bytes, 0, 4);
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4));

        var sb = new StringBuilder();
        sb.AppendLine($"bonepile total size = {bytes.Length} bytes, signature = {signature}, version = 0x{version:X}");
        sb.AppendLine();

        sb.AppendLine("=== FULL HEADER (uint32 at every 4-byte offset) ===");
        int fullHeaderSize = Math.Min(0x144, bytes.Length);
        for (int offset = 0; offset + 4 <= fullHeaderSize; offset += 4)
        {
            uint val = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
            float fval = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset, 4)));
            string floatStr = float.IsFinite(fval) && Math.Abs(fval) > 0.0001f && Math.Abs(fval) < 100000f ? $"  float={fval:F4}" : "";
            sb.AppendLine($"  [0x{offset:X3}] = 0x{val:X8} ({val,10}){floatStr}");
        }
        sb.AppendLine();

        Console.WriteLine(sb.ToString());
    }

    private static bool TryReadStagedVirtualFile(
        string stagedBuildDirectory,
        string virtualPath,
        out byte[] bytes,
        out string sourcePath,
        out Func<string, byte[]?> companionReader)
    {
        string root = GetWowViewerRoot();
        string clientDataPath = Path.Combine(root, "..", "output", "tmp", "wowarchive-clients", stagedBuildDirectory, "World of Warcraft", "Data");
        string listfilePath = Path.Combine(root, "libs", "wowdev", "wow-listfile", "listfile.txt");
        sourcePath = virtualPath.Replace('/', '\\');

        if (!Directory.Exists(clientDataPath) || !File.Exists(listfilePath))
        {
            bytes = [];
            companionReader = static _ => null;
            return false;
        }

        try
        {
            bytes = ArchiveVirtualFileReader.ReadVirtualFile(sourcePath, [clientDataPath], listfilePath);
            companionReader = path =>
            {
                try
                {
                    return ArchiveVirtualFileReader.ReadVirtualFile(path.Replace('/', '\\'), [clientDataPath], listfilePath);
                }
                catch (FileNotFoundException)
                {
                    return null;
                }
            };

            return true;
        }
        catch (FileNotFoundException)
        {
            bytes = [];
            companionReader = static _ => null;
            return false;
        }
    }

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }
}
