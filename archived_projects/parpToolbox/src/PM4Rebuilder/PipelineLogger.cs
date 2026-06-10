using System;
using System.IO;
using System.Text;

namespace PM4Rebuilder
{
    /// <summary>
    /// Lightweight logger that mirrors all Console output to <see cref="pipeline.log"/> with timestamps.
    /// </summary>
    internal static class PipelineLogger
    {
        private static TextWriter? _originalOut;
        private static TextWriter? _originalErr;
        private static StreamWriter? _fileWriter;

        /// <summary>
        /// Sets up a tee logger that mirrors <see cref="Console.Out"/> and <see cref="Console.Error"/> to a log file.
        /// </summary>
        /// <param name="outputDirectory">Directory where <c>pipeline.log</c> will be written.</param>
        /// <param name="enableFileLogging">If false, console redirection is skipped.</param>
        public static void Initialize(string outputDirectory, bool enableFileLogging = true)
        {
            if (!enableFileLogging)
                return;

            Directory.CreateDirectory(outputDirectory);
            var logPath = Path.Combine(outputDirectory, "pipeline.log");
            _fileWriter = new StreamWriter(logPath, append: true, Encoding.UTF8) { AutoFlush = true };

            _originalOut = Console.Out;
            _originalErr = Console.Error;

            var teeOut = new TeeTextWriter(_originalOut, _fileWriter);
            var teeErr = new TeeTextWriter(_originalErr, _fileWriter);

            Console.SetOut(teeOut);
            Console.SetError(teeErr);
        }

        /// <summary>
        /// Helper: write line with timestamp through the tee so callers don’t need Console.WriteLine directly.
        /// </summary>
        public static void Log(string? message) => Console.WriteLine(message);

        private sealed class TeeTextWriter : TextWriter
        {
            private readonly TextWriter _first;
            private readonly TextWriter _second;

            public TeeTextWriter(TextWriter first, TextWriter second)
            {
                _first = first;
                _second = second;
            }

            public override Encoding Encoding => _first.Encoding;

            public override void Write(char value)
            {
                _first.Write(value);
                _second.Write(value);
            }

            public override void Write(string? value)
            {
                _first.Write(value);
                _second.Write(value);
            }

            public override void WriteLine(string? value)
            {
                var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
                var lineWithTs = $"[{timestamp}] {value}";
                _first.WriteLine(lineWithTs);
                _second.WriteLine(lineWithTs);
            }

            public override void Flush()
            {
                _first.Flush();
                _second.Flush();
            }
        }
    }
}
