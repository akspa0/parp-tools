using WoWToolbox.Core.v2.Infrastructure;
using System.IO;

namespace WoWToolbox.Tests
{
    internal static class OutputLocator
    {
        public static string Central(params string[] segments)
        {
            return ProjectOutput.GetPath(Path.Combine(segments));
        }
    }
}
