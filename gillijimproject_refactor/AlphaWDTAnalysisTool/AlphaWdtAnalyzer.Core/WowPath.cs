using System;
using System.Text;

namespace AlphaWdtAnalyzer.Core;

public static class WowPath
{
    public static string Normalize(string? path)
    {
        if (string.IsNullOrWhiteSpace(path)) return string.Empty;
        // Preserve casing; just trim
        var p = path.Trim();
        // Replace backslashes with forward slashes
        p = p.Replace('\\', '/');
        // Trim leading separators
        p = p.TrimStart('/');
        // Collapse duplicate '/'
        var sb = new StringBuilder(p.Length);
        bool lastWasSlash = false;
        foreach (var ch in p)
        {
            if (ch == '/')
            {
                if (!lastWasSlash) sb.Append(ch);
                lastWasSlash = true;
            }
            else
            {
                sb.Append(ch);
                lastWasSlash = false;
            }
        }
        return sb.ToString();
    }
}
