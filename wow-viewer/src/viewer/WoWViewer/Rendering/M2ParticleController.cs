using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

/// <summary>
/// Controls particle emitters and manages rendering for an M2 model instance.
/// </summary>
public sealed class M2ParticleController : IDisposable
{
    private static ParticleRenderer? _particleRenderer;
    private static readonly object _rendererLock = new();

    private readonly GL _gl;
    private readonly M2StaticRenderModel _runtimeModel;
    private readonly Func<ushort, uint> _textureResolver;
    private readonly List<M2ParticleEmitter> _emitters = new();
    private bool _disposed;

    public IReadOnlyList<M2ParticleEmitter> Emitters => _emitters;

    public bool HasActiveParticles
    {
        get
        {
            if (_emitters.Count == 0)
                return false;

            for (int i = 0; i < _emitters.Count; i++)
            {
                var emitter = _emitters[i];
                if (emitter.IsActive && (emitter.Particles.Count > 0 || emitter.Definition.EmissionRateTrack.TimestampArray.Count > 0))
                    return true;
            }

            return false;
        }
    }

    public M2ParticleController(
        GL gl,
        M2StaticRenderModel runtimeModel,
        Func<ushort, uint> textureResolver)
    {
        ArgumentNullException.ThrowIfNull(gl);
        ArgumentNullException.ThrowIfNull(runtimeModel);
        ArgumentNullException.ThrowIfNull(textureResolver);

        _gl = gl;
        _runtimeModel = runtimeModel;
        _textureResolver = textureResolver;

        lock (_rendererLock)
        {
            _particleRenderer ??= new ParticleRenderer(gl);
        }

        byte[] rawBytes = runtimeModel.Model.RawBytes;
        foreach (M2ParticleDefinition def in runtimeModel.Model.Particles)
        {
            var emitter = new M2ParticleEmitter(def, rawBytes, runtimeModel.Model);
            uint texId = textureResolver(def.TextureIndex);
            emitter.TextureId = texId;
            _emitters.Add(emitter);
        }
    }

    public void Update(float deltaSeconds, Matrix4x4[]? boneMatrices, int sequenceIndex, int timeMs)
    {
        if (_disposed || _emitters.Count == 0)
            return;

        byte[] rawBytes = _runtimeModel.Model.RawBytes;
        for (int i = 0; i < _emitters.Count; i++)
        {
            _emitters[i].Update(deltaSeconds, boneMatrices, sequenceIndex, timeMs, rawBytes, _runtimeModel.Model);
        }
    }

    public void Render(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 cameraPos,
        Vector3 fogColor,
        float fogStart,
        float fogEnd)
    {
        if (_disposed || _emitters.Count == 0 || _particleRenderer == null)
            return;

        // Lazy-resolve textures if any were pending when the controller was created
        for (int i = 0; i < _emitters.Count; i++)
        {
            var emitter = _emitters[i];
            if (emitter.TextureId == 0)
                emitter.TextureId = _textureResolver(emitter.Definition.TextureIndex);
        }

        _particleRenderer.RenderM2(
            _emitters,
            view,
            proj,
            cameraPos,
            fogColor,
            fogStart,
            fogEnd,
            modelMatrix);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _emitters.Clear();
    }
}
