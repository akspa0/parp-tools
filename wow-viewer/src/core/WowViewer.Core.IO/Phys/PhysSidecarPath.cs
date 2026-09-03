namespace WowViewer.Core.IO.Phys;

/// <summary>
/// Resolves the physics sidecar path for a model.
/// </summary>
/// <remarks>
/// MEASURED from the 5.0.1 client (<c>FUN_005a29a0</c>, anchored by <c>Physics.cpp</c>): the sidecar
/// path is the model path with its extension replaced by <c>.phys</c>. Association is by filename
/// alone — there is no id, index or lookup table.
/// <para>
/// The client writes the extension as two immediates (<c>0x7968702e</c> then <c>0x73</c>) rather than
/// as a string literal, which is why no <c>.phys</c> string exists in the binary to search for.
/// </para>
/// </remarks>
public static class PhysSidecarPath
{
    public const string Extension = ".phys";

    /// <summary>
    /// MEASURED: <c>Physics.cpp:47</c> asserts <c>(fileName-ext)+6 &lt;= 260</c>, so the client
    /// composes the sidecar name in a 260-byte buffer.
    /// </summary>
    public const int ClientPathBufferLength = 260;

    /// <summary>
    /// Produces the sidecar path for <paramref name="modelPath"/> by replacing its extension.
    /// </summary>
    /// <returns>
    /// <see langword="true"/> and the resolved path, or <see langword="false"/> when
    /// <paramref name="modelPath"/> is blank or the composed path would exceed the client's buffer.
    /// </returns>
    public static bool TryResolve(string? modelPath, out string sidecarPath)
    {
        sidecarPath = string.Empty;
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        string trimmed = modelPath.Trim();
        int lastSeparator = trimmed.LastIndexOfAny(['/', '\\']);
        int lastDot = trimmed.LastIndexOf('.');

        // A dot belonging to a directory name is not an extension.
        string withoutExtension = lastDot > lastSeparator && lastDot >= 0
            ? trimmed[..lastDot]
            : trimmed;

        if (withoutExtension.Length == 0)
            return false;

        string candidate = withoutExtension + Extension;

        // Mirror the client's own bound rather than inventing one.
        if (candidate.Length + 1 > ClientPathBufferLength)
            return false;

        sidecarPath = candidate;
        return true;
    }
}
