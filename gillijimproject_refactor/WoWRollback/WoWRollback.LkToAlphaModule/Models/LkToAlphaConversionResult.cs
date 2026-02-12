namespace WoWRollback.LkToAlphaModule;

public sealed record LkToAlphaConversionResult(
    string AlphaOutputDirectory,
    int TilesProcessed,
    bool Success,
    string? ErrorMessage = null);
