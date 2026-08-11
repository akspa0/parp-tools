namespace WowViewer.Tools.ValidationCapture;

internal static class Program
{
    public static int Main(string[] args)
    {
        string? outputPath = GetOption(args, "--output", "-o");
        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            AppDomain.CurrentDomain.UnhandledException += (_, eventArgs) =>
            {
                if (eventArgs.ExceptionObject is Exception exception)
                    ProductionWorldSceneProfiler.PublishUnhandledFailure(outputPath, exception);
            };
        }

        return ValidationCaptureCommand.Execute(args);
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return null;
    }
}
