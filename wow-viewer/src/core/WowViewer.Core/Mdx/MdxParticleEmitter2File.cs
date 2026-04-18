namespace WowViewer.Core.Mdx;

public sealed class MdxParticleEmitter2File
{
    public MdxParticleEmitter2File(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxParticleEmitter2> particleEmitters)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(particleEmitters);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = string.IsNullOrWhiteSpace(modelName) ? null : modelName;
        ParticleEmitters = particleEmitters;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxParticleEmitter2> ParticleEmitters { get; }

    public int ParticleEmitterCount => ParticleEmitters.Count;
}