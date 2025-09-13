using System;

namespace AlphaWdtAnalyzer.Core.Export;

public enum LogLevel
{
    Quiet = 0,
    Normal = 1,
    Verbose = 2
}

public static class Log
{
    private static LogLevel _level = LogLevel.Normal;

    public static void SetLevel(LogLevel level) => _level = level;

    public static void Info(string message)
    {
        if (_level >= LogLevel.Normal) Console.WriteLine(message);
    }

    public static void Verbose(string message)
    {
        if (_level >= LogLevel.Verbose) Console.WriteLine(message);
    }

    public static void Error(string message)
    {
        Console.Error.WriteLine(message);
    }
}
