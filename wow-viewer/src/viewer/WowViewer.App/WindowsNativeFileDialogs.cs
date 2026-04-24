using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace WowViewer.App;

internal static class WindowsNativeFileDialogs
{
    private const int HResultCancelled = unchecked((int)0x800704C7);

    public static string? PickFolder(string title, string? initialDir = null)
    {
        if (!OperatingSystem.IsWindows())
            return null;

        return PickFolderWindows(title, initialDir);
    }

    public static string? PickFile(string title, string? initialDir, IReadOnlyList<FileDialogFilter> fileTypes)
    {
        ArgumentNullException.ThrowIfNull(fileTypes);

        if (!OperatingSystem.IsWindows())
            return null;

        return PickFileWindows(title, initialDir, fileTypes);
    }

    [SupportedOSPlatform("windows")]
    private static string? PickFolderWindows(string title, string? initialDir)
    {
        return ShowOnStaWindows(() => ShowOpenDialog(title, initialDir, pickFolders: true, fileTypes: null));
    }

    [SupportedOSPlatform("windows")]
    private static string? PickFileWindows(string title, string? initialDir, IReadOnlyList<FileDialogFilter> fileTypes)
    {
        return ShowOnStaWindows(() => ShowOpenDialog(title, initialDir, pickFolders: false, fileTypes));
    }

    [SupportedOSPlatform("windows")]
    private static string? ShowOnStaWindows(Func<string?> action)
    {
        string? result = null;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                result = action();
            }
            catch (COMException exception) when (exception.HResult == HResultCancelled)
            {
                result = null;
            }
            catch (Exception exception)
            {
                failure = exception;
            }
        });

        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();

        if (failure is not null)
            return null;

        return result;
    }

    [SupportedOSPlatform("windows")]
    private static string? ShowOpenDialog(string title, string? initialDir, bool pickFolders, IReadOnlyList<FileDialogFilter>? fileTypes)
    {
        IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialogCom();
        IShellItem? initialFolder = null;
        IShellItem? resultItem = null;

        try
        {
            dialog.GetOptions(out FileOpenDialogOptions options);
            options |= FileOpenDialogOptions.ForceFileSystem | FileOpenDialogOptions.PathMustExist;
            options |= pickFolders
                ? FileOpenDialogOptions.PickFolders
                : FileOpenDialogOptions.FileMustExist;
            dialog.SetOptions(options);
            dialog.SetTitle(title);

            if (!string.IsNullOrWhiteSpace(initialDir) && Directory.Exists(initialDir))
            {
                Guid shellItemGuid = typeof(IShellItem).GUID;
                SHCreateItemFromParsingName(initialDir, IntPtr.Zero, ref shellItemGuid, out initialFolder);
                dialog.SetFolder(initialFolder);
            }

            if (!pickFolders && fileTypes is { Count: > 0 })
            {
                ComDlgFilterSpec[] filters = fileTypes
                    .Select(static filter => new ComDlgFilterSpec(filter.Name, filter.Pattern))
                    .ToArray();
                dialog.SetFileTypes((uint)filters.Length, filters);
                dialog.SetFileTypeIndex(1);
            }

            int showResult = dialog.Show(IntPtr.Zero);
            if (showResult == HResultCancelled)
                return null;

            Marshal.ThrowExceptionForHR(showResult);
            dialog.GetResult(out resultItem);
            return GetFileSystemPath(resultItem);
        }
        finally
        {
            if (resultItem is not null)
                Marshal.ReleaseComObject(resultItem);
            if (initialFolder is not null)
                Marshal.ReleaseComObject(initialFolder);
            Marshal.ReleaseComObject(dialog);
        }
    }

    private static string? GetFileSystemPath(IShellItem shellItem)
    {
        shellItem.GetDisplayName(ShellItemDisplayName.FileSystemPath, out IntPtr displayNamePointer);
        if (displayNamePointer == IntPtr.Zero)
            return null;

        try
        {
            return Marshal.PtrToStringUni(displayNamePointer);
        }
        finally
        {
            Marshal.FreeCoTaskMem(displayNamePointer);
        }
    }

    internal readonly record struct FileDialogFilter(string Name, string Pattern);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private readonly struct ComDlgFilterSpec(string name, string pattern)
    {
        [MarshalAs(UnmanagedType.LPWStr)]
        public readonly string Name = name;

        [MarshalAs(UnmanagedType.LPWStr)]
        public readonly string Pattern = pattern;
    }

    [Flags]
    private enum FileOpenDialogOptions : uint
    {
        PickFolders = 0x00000020,
        ForceFileSystem = 0x00000040,
        PathMustExist = 0x00000800,
        FileMustExist = 0x00001000,
    }

    private enum ShellItemDisplayName : uint
    {
        FileSystemPath = 0x80058000,
    }

    [ComImport]
    [Guid("42F85136-DB7E-439C-85F1-E4075D135FC8")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileDialog
    {
        [PreserveSig]
        int Show(IntPtr parent);

        void SetFileTypes(uint fileTypeCount, [MarshalAs(UnmanagedType.LPArray)] ComDlgFilterSpec[] filterSpec);
        void SetFileTypeIndex(uint fileTypeIndex);
        void GetFileTypeIndex(out uint fileTypeIndex);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(FileOpenDialogOptions options);
        void GetOptions(out FileOpenDialogOptions options);
        void SetDefaultFolder(IShellItem shellItem);
        void SetFolder(IShellItem shellItem);
        void GetFolder(out IShellItem shellItem);
        void GetCurrentSelection(out IShellItem shellItem);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem shellItem);
        void AddPlace(IShellItem shellItem, uint alignment);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string defaultExtension);
        void Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
    }

    [ComImport]
    [Guid("D57C7288-D4AD-4768-BE02-9D969532D960")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileOpenDialog : IFileDialog
    {
        [PreserveSig]
        new int Show(IntPtr parent);

        new void SetFileTypes(uint fileTypeCount, [MarshalAs(UnmanagedType.LPArray)] ComDlgFilterSpec[] filterSpec);
        new void SetFileTypeIndex(uint fileTypeIndex);
        new void GetFileTypeIndex(out uint fileTypeIndex);
        new void Advise(IntPtr events, out uint cookie);
        new void Unadvise(uint cookie);
        new void SetOptions(FileOpenDialogOptions options);
        new void GetOptions(out FileOpenDialogOptions options);
        new void SetDefaultFolder(IShellItem shellItem);
        new void SetFolder(IShellItem shellItem);
        new void GetFolder(out IShellItem shellItem);
        new void GetCurrentSelection(out IShellItem shellItem);
        new void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        new void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        new void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        new void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        new void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        new void GetResult(out IShellItem shellItem);
        new void AddPlace(IShellItem shellItem, uint alignment);
        new void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string defaultExtension);
        new void Close(int hr);
        new void SetClientGuid(ref Guid guid);
        new void ClearClientData();
        new void SetFilter(IntPtr filter);

        void GetResults(out IntPtr items);
        void GetSelectedItems(out IntPtr items);
    }

    [ComImport]
    [Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem
    {
        void BindToHandler(IntPtr bindContext, ref Guid bindingHandler, ref Guid riid, out IntPtr result);
        void GetParent(out IShellItem shellItem);
        void GetDisplayName(ShellItemDisplayName displayName, out IntPtr name);
        void GetAttributes(uint attributes, out uint itemAttributes);
        void Compare(IShellItem shellItem, uint hint, out int order);
    }

    [ComImport]
    [Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    private class FileOpenDialogCom
    {
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    private static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string path,
        IntPtr bindContext,
        ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] out IShellItem shellItem);
}