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
                "creature\\bear\\bear.mdx",
                out byte[] bytes,
                out string sourcePath,
                out Func<string, byte[]?> companionReader))
        {
            return;
        }

        Assert.True(bytes.Length > 0);
        string signature = Encoding.ASCII.GetString(bytes, 0, 4);
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4));

        int headerEnd = version == 0x101 ? 0xE8 : 0xD4;
        int stride = 8;

        var sb = new StringBuilder();
        sb.AppendLine($"bear.mdx total size = {bytes.Length} bytes, signature = {signature}, version = 0x{version:X}, headerEnd = 0x{headerEnd:X}");

        for (int offset = 0x08; offset + 8 <= headerEnd; offset += stride)
        {
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
            uint dataOffset = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            if (count == 0 && dataOffset == 0)
            {
                continue;
            }

            long end = (long)dataOffset + (long)count * 1;
            sb.AppendLine($"  [0x{offset:X2}] count=0x{count:X8} ({count})  offset=0x{dataOffset:X8}  tentativeEnd=0x{end:X8}  (inFile={(end <= bytes.Length)})");
        }

        throw new InvalidOperationException(sb.ToString());
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
