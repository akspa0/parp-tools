using System.Runtime.InteropServices;

namespace WowViewer.Core.Audio;

/// <summary>
/// Checks for an OpenAL native library before Silk.NET's AudioContext is
/// touched. Audio is optional; a missing native backend must not terminate the
/// viewer during startup or finalization.
/// </summary>
public static class OpenAlNativeLibraryProbe
{
    private static readonly object Sync = new();
    private static IntPtr _loadedHandle;
    private static string? _loadedLibraryName;

    public static bool TryFind(out string? libraryName)
    {
        lock (Sync)
        {
            if (_loadedHandle != IntPtr.Zero)
            {
                libraryName = _loadedLibraryName;
                return true;
            }
        }

        return TryFindAndRetain(GetPlatformCandidates(), out libraryName);
    }

    public static bool TryFind(IEnumerable<string> candidates, out string? libraryName)
    {
        ArgumentNullException.ThrowIfNull(candidates);

        foreach (string candidate in candidates.Where(static value => !string.IsNullOrWhiteSpace(value)))
        {
            try
            {
                if (!NativeLibrary.TryLoad(candidate, out IntPtr handle))
                    continue;

                NativeLibrary.Free(handle);
                libraryName = candidate;
                return true;
            }
            catch (Exception)
            {
                // A missing or incompatible optional audio library is not a
                // viewer startup failure. Try the next platform spelling.
            }
        }

        libraryName = null;
        return false;
    }

    private static bool TryFindAndRetain(IEnumerable<string> candidates, out string? libraryName)
    {
        ArgumentNullException.ThrowIfNull(candidates);

        foreach (string candidate in candidates.Where(static value => !string.IsNullOrWhiteSpace(value)))
        {
            try
            {
                if (!NativeLibrary.TryLoad(candidate, out IntPtr handle))
                    continue;

                lock (Sync)
                {
                    if (_loadedHandle == IntPtr.Zero)
                    {
                        // Silk.NET's OpenAL loader resolves the canonical
                        // library name separately. Keep the app-local native
                        // module resident so that name-based resolution can
                        // reuse the already-loaded OpenAL Soft module.
                        _loadedHandle = handle;
                        _loadedLibraryName = candidate;
                        libraryName = candidate;
                        return true;
                    }
                }

                NativeLibrary.Free(handle);
                libraryName = _loadedLibraryName;
                return true;
            }
            catch (Exception)
            {
                // A missing or incompatible optional audio library is not a
                // viewer startup failure. Try the next platform spelling.
            }
        }

        libraryName = null;
        return false;
    }

    private static IEnumerable<string> GetPlatformCandidates()
    {
        if (OperatingSystem.IsWindows())
        {
            yield return Path.Combine(AppContext.BaseDirectory, "openal32.dll");
            yield return Path.Combine(AppContext.BaseDirectory, "soft_oal.dll");
            string rid = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X86 => "win-x86",
                Architecture.Arm64 => "win-arm64",
                _ => "win-x64"
            };
            yield return Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "soft_oal.dll");
            yield return Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "openal32.dll");
            yield return "openal32.dll";
            yield return "soft_oal.dll";
            yield return "openal32";
            yield return "soft_oal";
            yield break;
        }

        if (OperatingSystem.IsMacOS())
        {
            yield return "/System/Library/Frameworks/OpenAL.framework/OpenAL";
            string rid = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "osx-arm64",
                _ => "osx-x64"
            };
            yield return Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "libopenal.dylib");
            yield return "OpenAL";
            yield return "openal";
            yield break;
        }

        string linuxRid = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.Arm => "linux-arm",
            Architecture.Arm64 => "linux-arm64",
            _ => "linux-x64"
        };
        yield return Path.Combine(AppContext.BaseDirectory, "runtimes", linuxRid, "native", "libopenal.so");
        yield return "libopenal.so.1";
        yield return "libopenal.so";
        yield return "openal";
    }
}
