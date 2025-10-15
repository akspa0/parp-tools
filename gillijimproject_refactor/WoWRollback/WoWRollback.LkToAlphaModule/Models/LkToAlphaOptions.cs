namespace WoWRollback.LkToAlphaModule;

public sealed record LkToAlphaOptions
{
    public bool SkipLiquids { get; init; } = true;
    public bool SkipWmos { get; init; }
    public string? TextureMappingPath { get; init; }
    public bool Validate { get; init; }
    public bool Verbose { get; init; }
}
