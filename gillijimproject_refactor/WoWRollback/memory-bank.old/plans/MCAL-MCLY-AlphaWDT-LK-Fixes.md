**Planning document structure**

I’m thinking about how to structure this markdown document. It’ll include sections, lists, and bold lead-ins for clarity. I won’t need citations since there aren’t any, and I'll avoid LaTeX for math as it doesn’t seem necessary.

I’ll organize things with clear headings, use horizontal dividers, and provide technical guidance with steps for implementation, an outline for RLE decoding, and testing checklists. I also need to remember to address CASC and unlimited layers but will avoid mentioning tools. Let's move forward.
# Lk adt to alphawdt terrain conversion: mcly/mcal fix guide

This is a focused, drop-in guide to remind future-you how to correctly handle MCLY/MCAL when converting Wrath (3.3.5) ADTs down to Alpha/Vanilla (0.5.3–1.12) WDT/ADT. It assumes you already have geometry (WDT/MCNK) working, and that the current bug is “only base texture shows” due to writing LK-style MCAL into alpha ADTs.

---

## Key differences at a glance

### Mcly and mcal structure

- **Alpha/Vanilla MCLY:**
  - **Layers per chunk:** Up to 4.
  - **Flags:** Indicate presence of alpha and whether the alpha is compressed.
  - **Texture reference:** Legacy texture index/path mapping; ensure listfile resolution.

- **Alpha/Vanilla MCAL:**
  - **Header:** No explicit size field before payload.
  - **Alpha maps:** 64×64 logical resolution, but often stored at 4-bit precision (2048 bytes uncompressed) or RLE-compressed.
  - **Ordering:** Alpha payloads are written sequentially for layers that have alpha; layers without alpha have no MCAL payload.

- **Wrath (3.3.5) MCLY:**
  - **Layers per chunk:** Up to 4.
  - **Flags:** Has alpha; compression rare.
  - **Texture reference:** LK indices; you already remap.

- **Wrath (3.3.5) MCAL:**
  - **Header:** Commonly an 8-byte chunk header preceding payload (e.g., magic + size).
  - **Alpha maps:** Typically 8-bit, raw 4096 bytes per layer (64×64).
  - **Ordering:** Each layer’s alpha block is read using the size field; generally uncompressed.

> Sources: Your current exporter was reading MCAL like LK (skip 8 bytes, copy “size” bytes), which is wrong for alpha. Alpha MCAL should be emitted as raw alpha data matching alpha’s flags and encoding—no LK-style size field.

---

## Conversion strategy overview

- **Goal:** Down-convert LK texture layers (1–4 per chunk) into Alpha-compatible layers with correct alpha payload encoding.
- **Constraints:** Alpha hard-caps at 4 layers; Alpha MCAL uses 4-bit uncompressed or RLE-compressed maps; Alpha expects no size header.

---

## Implementation steps

### 1. Inspect lk layers and produce an alpha layer plan

- **Collect LK layers:**
  - Extract layer order, texture IDs/paths, and per-layer alpha maps (64×64, 8-bit).
- **Decide which layers to keep:**
  - **Rule of thumb:** Keep up to 4 most visually dominant layers based on coverage (sum of alpha values).
- **Normalize order:**
  - **Base layer:** The one with highest coverage becomes layer 0 (no alpha).
  - **Blend layers:** Remaining up to 3 layers flagged as “has alpha.”

### 2. Convert 8-bit alpha maps to alpha’s expected format

- **Uncompressed path (simplest, recommended to start):**
  - **Quantize:** Convert 8-bit 0–255 alpha to 4-bit 0–15.
    - Example: value4 = (value8 + 8) / 16 (rounded to nearest).
  - **Pack two 4-bit values per byte:** High nibble = first pixel, low nibble = second pixel.
  - **Output size:** 2048 bytes per layer (64×64 pixels × 4 bits).
  - **MCAL write:** Append 2048 raw bytes per “has alpha” layer, in the same order as MCLY entries.

- **Compressed path (RLE):**
  - **Use MCLY flag to mark compressed.**
  - **Encode:** Run-length encode alpha values across the 64×64 stream (row-major), using alpha’s RLE scheme.
  - **MCAL write:** Append compressed byte stream with no size header; client decodes on load.
  - **Note:** Start with uncompressed; add RLE after uncompressed is verified.

### 3. Write mcly entries correctly

- **For each layer in order:**
  - **Texture index/path:** Map LK texture reference to alpha’s expected index via your listfile resolver.
  - **Flags:**
    - **Layer 0 (base):** hasAlpha = false; compressed = false.
    - **Layers 1–3:** hasAlpha = true; compressed = false (initially; set true when you add RLE).
  - **Other bits:** Keep consistent with alpha spec for “no animation,” “no shadow,” etc., if present.

### 4. Emit mcal payload correctly

- **No size header:** Do not write LK-style `[size]` or skip 8 bytes.
- **Sequential payload:** Concatenate per-layer alpha payloads in the same order as MCLY, but only for layers where hasAlpha = true.
- **Consistency check:** The number of alpha payloads must equal the count of layers flagged hasAlpha.

### 5. Validate geometry and texture references

- **MCNK alignment:** Ensure MCNK references (MCLY/MCAL offsets) point to your newly written payload.
- **Listfile:** Confirm texture paths exist and resolve to alpha-era IDs; fallback to vanilla equivalents if missing.
- **Layer count:** Enforce cap at 4; drop extras with least coverage to avoid client confusion.

---

## Byte-level sanity checks

### Alpha uncompressed alpha map (per layer)

- **Pixels:** 64×64 = 4096 pixels.
- **Precision:** 4-bit per pixel.
- **Packing:** 2 pixels per byte.
- **Total:** 2048 bytes continuous, row-major scanline order (typical).

### Wrath raw alpha map (per layer)

- **Pixels:** 64×64 = 4096 pixels.
- **Precision:** 8-bit per pixel.
- **Total:** 4096 bytes continuous; commonly preceded by chunk header with size.

> If you currently read/write 4096 bytes and/or prepend size fields for alpha, you’re in LK mode. Switch to 2048 packed bytes (or compressed stream) with no size prefix for alpha.

---

## RLE encoding outline (alpha compressed path)

- **Input:** 4-bit alpha stream (4096 values).
- **Process:**
  - **Run detection:** Scan left-to-right, top-to-bottom; group consecutive identical 4-bit values.
  - **Emit tokens:** For each run, write [length][value] pairs; ensure token format matches alpha client expectations (consult your alpha spec doc).
  - **Edge cases:** Clamp run length to token max; split runs as needed.
- **Output:** Concatenated tokens; no size header added to MCAL.

> Start uncompressed first. Once verified, implement RLE and flip the MCLY compressed flag for layers you encode.

---

## Testing checklist

- **Single-chunk test:** Use a chunk with 2–3 layers and clear blends. Verify all layers render.
- **Alpha-flag match:** Count hasAlpha layers in MCLY equals the number of alpha payloads in MCAL.
- **Payload size:** For uncompressed, MCAL length equals 2048 × hasAlphaLayers.
- **Visual parity:** Compare LK source and alpha output for major texture boundaries; expect slightly blockier transitions due to 4-bit quantization.
- **Noggit/alpha client smoke test:** Ensure no crashes; verify base + blended textures appear.

---

## Casc-era note: layers can exceed four

- **Modern versions (Cataclysm → Dragonflight → 12.0 beta, CASC-based):**
  - **Layers per chunk:** Can exceed 4; some pipelines treat it as unbounded in practice.
  - **Blending:** More advanced and shader-driven; per-chunk layer count is no longer the bottleneck.
- **Down-conversion rule:**
  - **Select top 4 by coverage:** Compute per-layer dominance (sum of alpha) and keep the most visually impactful.
  - **Bake complex overlaps:** If necessary, pre-bake minor layers into the dominant layers’ alphas (composite) to avoid abrupt loss.
  - **Document loss:** Note that rich multi-layer detail will be collapsed—this is expected when targeting alpha constraints.

---

## Practical code reminders

- **Do not read/write LK MCAL headers** when targeting alpha. Emit raw payload only.
- **Quantize alpha:** 8-bit to 4-bit before packing or RLE.
- **Pack nibbles:** Two 4-bit pixels per byte; maintain row-major order.
- **Align offsets:** Ensure MCNK’s offset/size fields reflect the new MCAL layout.
- **Respect flags:** MCLY hasAlpha and compressed must mirror the payload you wrote.

---

## Roadmap

- **Phase 1:** Implement uncompressed 4-bit MCAL, fix MCLY flags, validate in Noggit and alpha client.
- **Phase 2:** Add RLE compressed MCAL support; toggle flags; test multiple chunks with varied textures.
- **Phase 3:** Add smart layer selection for CASC-era maps; introduce coverage-based pruning and optional pre-baking.

---

## Quick tl;dr for future-you

- You’re writing LK-style MCAL into alpha. Stop adding size headers and 4096-byte 8-bit maps.
- Alpha wants 4-bit 64×64 maps (2048 bytes) or RLE-compressed streams, ordered by MCLY hasAlpha layers.
- Keep max 4 layers; base has no alpha; others have alpha payloads.
- For modern CASC maps with >4 layers, pick top 4 by coverage and quantize/blend down.
