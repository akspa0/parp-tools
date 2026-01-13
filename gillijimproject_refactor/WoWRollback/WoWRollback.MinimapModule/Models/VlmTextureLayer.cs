using System.Text.Json.Serialization;

namespace WoWRollback.MinimapModule.Models
{
    public record VlmTextureLayer(
        [property: JsonPropertyName("texture_id")] uint TextureId,
        [property: JsonPropertyName("flags")] uint Flags,
        [property: JsonPropertyName("offset")] uint Offset,
        [property: JsonPropertyName("effect_id")] uint EffectId
    );
}
