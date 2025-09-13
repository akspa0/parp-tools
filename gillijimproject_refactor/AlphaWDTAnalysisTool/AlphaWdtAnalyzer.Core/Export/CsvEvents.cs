using System;

namespace AlphaWdtAnalyzer.Core.Export;

public sealed record CsvLineEvent(
    string TargetPath,
    string Header,
    string Line,
    string? DedupKey
);
