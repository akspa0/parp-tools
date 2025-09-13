using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace AlphaWdtAnalyzer.Core.Export;

public static class CsvReportBus
{
    private static BlockingCollection<CsvLineEvent>? _queue;
    private static Task? _writerTask;
    private static readonly object _sync = new();
    private static bool _running;

    private sealed class WriterState
    {
        public StreamWriter Writer { get; }
        public string Header { get; }
        public HashSet<string> Seen { get; } = new(StringComparer.OrdinalIgnoreCase);
        public WriterState(StreamWriter writer, string header)
        {
            Writer = writer;
            Header = header;
        }
    }

    private static readonly Dictionary<string, WriterState> _writers = new(StringComparer.OrdinalIgnoreCase);

    public static bool IsRunning => _running;

    public static void Start()
    {
        if (_running) return;
        _queue = new BlockingCollection<CsvLineEvent>(boundedCapacity: 8192);
        _writerTask = Task.Run(ConsumeLoop);
        _running = true;
    }

    public static void Enqueue(CsvLineEvent ev)
    {
        if (!_running || _queue is null) return;
        _queue.Add(ev);
    }

    public static void CompleteAndDrain()
    {
        if (!_running || _queue is null) return;
        _queue.CompleteAdding();
        try { _writerTask?.Wait(); } catch { /* ignore */ }
        CloseWriters();
        _running = false;
        _queue = null;
        _writerTask = null;
    }

    private static void ConsumeLoop()
    {
        try
        {
            foreach (var ev in _queue!.GetConsumingEnumerable())
            {
                try { WriteLine(ev); } catch { /* ignore individual errors */ }
            }
        }
        catch { /* ignore */ }
    }

    private static void WriteLine(CsvLineEvent ev)
    {
        lock (_sync)
        {
            if (!_writers.TryGetValue(ev.TargetPath, out var ws))
            {
                var dir = Path.GetDirectoryName(ev.TargetPath);
                if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
                var sw = new StreamWriter(ev.TargetPath, append: false);
                ws = new WriterState(sw, ev.Header);
                // write header once upon first creation
                sw.WriteLine(ev.Header);
                _writers[ev.TargetPath] = ws;
            }

            if (!string.IsNullOrEmpty(ev.DedupKey))
            {
                if (ws.Seen.Contains(ev.DedupKey!)) return;
                ws.Seen.Add(ev.DedupKey!);
            }

            ws.Writer.WriteLine(ev.Line);
        }
    }

    private static void CloseWriters()
    {
        lock (_sync)
        {
            foreach (var ws in _writers.Values)
            {
                try { ws.Writer.Flush(); ws.Writer.Dispose(); } catch { }
            }
            _writers.Clear();
        }
    }
}
