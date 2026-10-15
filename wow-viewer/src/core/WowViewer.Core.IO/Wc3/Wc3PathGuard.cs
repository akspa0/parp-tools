using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace WowViewer.Core.IO.Wc3;

// Windows directory handles deny rename/delete while an extraction is in flight.
// OPEN_REPARSE_POINT also lets us inspect the opened object rather than its target.
// No source writes, new package, or shared-reader change is involved.
internal sealed class Wc3PathGuard : IDisposable
{
    private readonly Dictionary<string, SafeFileHandle> _handles = new(StringComparer.OrdinalIgnoreCase);

    internal static Wc3PathGuard Pin(string path)
    {
        Wc3PathGuard guard = new();
        try { guard.Add(path); return guard; }
        catch { guard.Dispose(); throw; }
    }

    internal void Add(string path)
    {
        Wc3PathSafety.CheckAncestors(path);
        if (!OperatingSystem.IsWindows()) return;
        Stack<string> chain = new();
        for (string? current = path; current is not null; current = Path.GetDirectoryName(current))
            if (!_handles.ContainsKey(current) && Wc3PathSafety.Attributes(current).HasValue)
                chain.Push(current);
        while (chain.TryPop(out string? current))
        {
            SafeFileHandle handle = CreateFile(current, 0, FileShare.Read | FileShare.Write,
                IntPtr.Zero, 3, 0x02200000, IntPtr.Zero); // backup semantics + open reparse point
            if (handle.IsInvalid)
            {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new IOException($"Cannot safely access '{current}': {new Win32Exception(error).Message}");
            }
            try
            {
                if (!GetFileInformationByHandle(handle, out HandleInformation information))
                    throw new IOException($"Cannot inspect '{current}': {new Win32Exception(Marshal.GetLastWin32Error()).Message}");
                if ((information.Attributes & (uint)FileAttributes.ReparsePoint) != 0)
                    throw new IOException($"Refusing reparse-point/symbolic-link path '{current}'.");
                _handles.Add(current, handle);
            }
            catch { handle.Dispose(); throw; }
        }
        Wc3PathSafety.CheckAncestors(path);
    }

    internal static string CanonicalPath(string path)
    {
        Wc3PathSafety.CheckAncestors(path);
        if (!OperatingSystem.IsWindows()) return path;
        using Wc3PathGuard guard = Pin(path);
        Stack<string> suffix = new();
        string current = path;
        while (!guard._handles.ContainsKey(current))
        {
            suffix.Push(Path.GetFileName(current));
            current = Path.GetDirectoryName(current) ?? throw new IOException("Cannot resolve the physical output root.");
        }
        StringBuilder buffer = new(512);
        uint count = GetFinalPathNameByHandle(guard._handles[current], buffer, (uint)buffer.Capacity, 0);
        if (count >= buffer.Capacity)
        {
            buffer = new StringBuilder(checked((int)count + 1));
            count = GetFinalPathNameByHandle(guard._handles[current], buffer, (uint)buffer.Capacity, 0);
        }
        if (count == 0 || count >= buffer.Capacity)
            throw new IOException($"Cannot canonicalize '{current}': {new Win32Exception(Marshal.GetLastWin32Error()).Message}");
        string physical = buffer.ToString();
        if (physical.StartsWith("\\\\?\\UNC\\", StringComparison.OrdinalIgnoreCase)) physical = "\\\\" + physical[8..];
        else if (physical.StartsWith("\\\\?\\", StringComparison.Ordinal)) physical = physical[4..];
        while (suffix.TryPop(out string? part)) physical = Path.Combine(physical, part);
        return Wc3PathSafety.FullPath(physical);
    }

    public void Dispose()
    {
        foreach (SafeFileHandle handle in _handles.Values.Reverse()) handle.Dispose();
        _handles.Clear();
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct HandleInformation
    {
        internal uint Attributes;
        internal uint CreationLow, CreationHigh, AccessLow, AccessHigh, WriteLow, WriteHigh;
        internal uint VolumeSerial, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, EntryPoint = "CreateFileW")]
    private static extern SafeFileHandle CreateFile(string path, uint access, FileShare share,
        IntPtr security, uint disposition, uint flags, IntPtr template);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(SafeFileHandle handle, out HandleInformation information);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, EntryPoint = "GetFinalPathNameByHandleW")]
    private static extern uint GetFinalPathNameByHandle(SafeFileHandle handle, StringBuilder path, uint size, uint flags);
}
