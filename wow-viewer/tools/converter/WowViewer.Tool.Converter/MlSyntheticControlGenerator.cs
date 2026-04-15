using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

internal static class MlSyntheticControlGenerator
{
	private const int MinimapSize = 256;
	private const int HeightmapSize = 257;
	private const int ChunkGridSize = 16;
	private const int ChunkAlphaSize = 64;
	private const int TileAlphaSize = ChunkGridSize * ChunkAlphaSize;
	private const int LayerCount = 3;
	private static readonly string DefaultHybridSourceRoot = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "datasets", "original_development", "development"));

	public static void Run(string[] args)
	{
		string datasetRoot = Path.GetFullPath(
			GetOption(args, "--dataset-root", "-d")
			?? GetOption(args, "--output-dir", "-o")
			?? Path.Combine(Environment.CurrentDirectory, "output", "build-validation", "synthetic-controls"));
		string mapName = GetOption(args, "--map-name", "-m") ?? "synthetic_controls";
		string hybridSourceRoot = Path.GetFullPath(GetOption(args, "--hybrid-source-root", "-r") ?? DefaultHybridSourceRoot);
		string? hybridSourceTile = GetOption(args, "--hybrid-source-tile", "-t");

		Directory.CreateDirectory(datasetRoot);
		string datasetDirectory = Path.Combine(datasetRoot, "dataset");
		string imagesDirectory = Path.Combine(datasetRoot, "images");
		string stitchedDirectory = Path.Combine(datasetRoot, "stitched");
		Directory.CreateDirectory(datasetDirectory);
		Directory.CreateDirectory(imagesDirectory);
		Directory.CreateDirectory(stitchedDirectory);

		List<SyntheticControlSpec> controls =
		[
			CreateWhitePlateSpec(),
			CreateDiagonalRampSpec(),
			CreateRingMoundSpec(),
			CreateTerraceStepsSpec()
		];

		HybridSourceData? hybridSource = TryLoadHybridSource(hybridSourceRoot, hybridSourceTile);
		if (hybridSource is not null)
		{
			controls.Add(CreateHybridBrushPlateSpec(hybridSource));
		}

		List<SyntheticControlManifestEntry> manifestEntries = new(controls.Count);
		List<SyntheticMetadataRow> metadataRows = new(controls.Count);

		Console.WriteLine("WowViewer.Tool.Converter ml-generate-controls report");
		Console.WriteLine($"DatasetRoot: {datasetRoot}");
		Console.WriteLine($"MapName: {mapName}");
		Console.WriteLine($"HybridSourceRoot: {hybridSourceRoot}");
		Console.WriteLine($"HybridSourceTile: {hybridSource?.TileName ?? "none"}");
		Console.WriteLine($"ControlCount: {controls.Count}");

		for (int index = 0; index < controls.Count; index++)
		{
			SyntheticControlSpec spec = controls[index];
			int tileX = 0;
			int tileY = index;
			string tileName = string.Create(CultureInfo.InvariantCulture, $"{mapName}_{tileX}_{tileY}");

			SyntheticControlTile tile = BuildTile(tileName, mapName, spec);

			string minimapPath = Path.Combine(imagesDirectory, tileName + ".png");
			using (Image<Rgba32> minimap = RenderMinimap(spec))
			{
				minimap.SaveAsPng(minimapPath);
			}

			string heightmapLocalPath = Path.Combine(imagesDirectory, tileName + "_heightmap.png");
			string heightmapGlobalPath = Path.Combine(imagesDirectory, tileName + "_heightmap_global.png");
			using (Image<L16> localHeightmap = RenderHeightmap(spec))
			{
				localHeightmap.SaveAsPng(heightmapLocalPath);
				localHeightmap.Clone().SaveAsPng(heightmapGlobalPath);
			}

			List<string> alphaMaskRelativePaths = [];
			for (int layerIndex = 0; layerIndex < LayerCount; layerIndex++)
			{
				string alphaMaskPath = Path.Combine(stitchedDirectory, string.Create(CultureInfo.InvariantCulture, $"{tileName}_alpha_l{layerIndex + 1}.png"));
				using Image<L8> alphaMask = RenderAlphaLayer(spec, layerIndex);
				{
					alphaMask.SaveAsPng(alphaMaskPath);
				}
				alphaMaskRelativePaths.Add(Relativize(datasetRoot, alphaMaskPath));
			}

			string shadowPath = Path.Combine(stitchedDirectory, tileName + "_shadow.png");
			using (Image<L8> shadow = RenderShadow(spec))
			{
				shadow.SaveAsPng(shadowPath);
			}

			string alphaAtlasPath = Path.Combine(stitchedDirectory, tileName + "_alpha_atlas.png");
			using (Image<Rgba32> alphaAtlas = RenderAlphaAtlas(spec))
			{
				alphaAtlas.SaveAsPng(alphaAtlasPath);
			}

			SyntheticDatasetSample sample = new(
				ImagePath: Relativize(datasetRoot, minimapPath),
				SyntheticControl: new SyntheticControlMetadata(
					Name: spec.Name,
					ExpectedInterestClass: spec.ExpectedInterestClass,
					Description: spec.Description,
					ExpectedBrushGroups: spec.ExpectedBrushGroups,
					ExpectedLayerStackDepth: spec.ExpectedLayerStackDepth),
				TerrainData: new SyntheticTerrainData(
					AdtTile: tileName,
					HeightmapPath: Relativize(datasetRoot, heightmapLocalPath),
					HeightmapLocalPath: Relativize(datasetRoot, heightmapLocalPath),
					HeightmapGlobalPath: Relativize(datasetRoot, heightmapGlobalPath),
					AlphaMasks: alphaMaskRelativePaths.ToArray(),
					AlphaAtlasPath: Relativize(datasetRoot, alphaAtlasPath),
					ShadowMaps: [Relativize(datasetRoot, shadowPath)],
					Textures: tile.TexturePaths,
					ChunkLayers: tile.ChunkLayers.ToArray(),
					Holes: new int[ChunkGridSize * ChunkGridSize],
					Objects: [],
					Liquids: [],
					HeightMin: 0f,
					HeightMax: 1f,
					HeightGlobalMin: 0f,
					HeightGlobalMax: 1f,
					IsInterleaved: false));

			string tileJsonPath = Path.Combine(datasetDirectory, tileName + ".json");
			File.WriteAllText(tileJsonPath, JsonSerializer.Serialize(sample, CreateJsonOptions()));

			manifestEntries.Add(new SyntheticControlManifestEntry(
				TileName: tileName,
				MapName: mapName,
				ControlName: spec.Name,
				ExpectedInterestClass: spec.ExpectedInterestClass,
				Description: spec.Description,
				ExpectedBrushGroups: spec.ExpectedBrushGroups,
				ExpectedLayerStackDepth: spec.ExpectedLayerStackDepth,
				ImagePath: Relativize(datasetRoot, minimapPath),
				HeightmapGlobalPath: Relativize(datasetRoot, heightmapGlobalPath),
				AlphaAtlasPath: Relativize(datasetRoot, alphaAtlasPath),
				TileJsonPath: Relativize(datasetRoot, tileJsonPath)));

			metadataRows.Add(new SyntheticMetadataRow(
				FileName: Relativize(datasetRoot, minimapPath),
				TileName: tileName,
				MapName: mapName,
				TileJson: Relativize(datasetRoot, tileJsonPath),
				HeightmapGlobal: Relativize(datasetRoot, heightmapGlobalPath),
				AlphaAtlas: Relativize(datasetRoot, alphaAtlasPath),
				ExpectedInterestClass: spec.ExpectedInterestClass,
				SyntheticControl: spec.Name));
		}

		SyntheticControlManifest manifest = new(
			SchemaVersion: "wowviewer-ml-synthetic-controls.v1",
			GeneratedUtc: DateTime.UtcNow,
			DatasetRoot: datasetRoot,
			MapName: mapName,
			ControlCount: manifestEntries.Count,
			Controls: manifestEntries);

		string manifestPath = Path.Combine(datasetRoot, "synthetic_control_manifest.json");
		File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, CreateJsonOptions()));

		string metadataPath = Path.Combine(datasetRoot, "metadata.jsonl");
		File.WriteAllLines(metadataPath, metadataRows.Select(static row => JsonSerializer.Serialize(row)));

		string datasetInfoPath = Path.Combine(datasetRoot, "dataset_info.json");
		File.WriteAllText(datasetInfoPath, JsonSerializer.Serialize(new
		{
			schema_version = "wowviewer-ml-synthetic-controls-dataset-info.v1",
			map_name = mapName,
			control_count = manifestEntries.Count,
			includes_white_plate = manifestEntries.Any(static entry => string.Equals(entry.ControlName, "white_plate", StringComparison.Ordinal))
		}, CreateJsonOptions()));

		Console.WriteLine($"Wrote {manifestPath}");
		Console.WriteLine($"Wrote {metadataPath}");
		Console.WriteLine($"Wrote {datasetInfoPath}");
	}

	private static SyntheticControlTile BuildTile(string tileName, string mapName, SyntheticControlSpec spec)
	{
		List<SyntheticChunkLayers> chunkLayers = new(ChunkGridSize * ChunkGridSize);
		HashSet<string> texturePaths = new(StringComparer.OrdinalIgnoreCase)
		{
			spec.BaseTexturePath
		};

		for (int chunkY = 0; chunkY < ChunkGridSize; chunkY++)
		{
			for (int chunkX = 0; chunkX < ChunkGridSize; chunkX++)
			{
				int chunkIndex = (chunkY * ChunkGridSize) + chunkX;
				List<SyntheticTextureLayer> layers =
				[
					new SyntheticTextureLayer(
						TextureId: 0,
						TexturePath: spec.BaseTexturePath,
						Flags: 0,
						AlphaOffset: 0,
						EffectId: 0,
						AlphaBitsBase64: null)
				];

				for (int layerIndex = 0; layerIndex < LayerCount; layerIndex++)
				{
					byte[] alphaChunk = BuildChunkAlphaBytes(spec, layerIndex, chunkX, chunkY);
					if (!alphaChunk.Any(static value => value > 0))
						continue;

					string overlayTexturePath = spec.OverlayTexturePaths[layerIndex];
					texturePaths.Add(overlayTexturePath);
					layers.Add(new SyntheticTextureLayer(
						TextureId: (uint)(layerIndex + 1),
						TexturePath: overlayTexturePath,
						Flags: 0,
						AlphaOffset: 0,
						EffectId: (uint)(layerIndex + 1),
						AlphaBitsBase64: Convert.ToBase64String(alphaChunk)));
				}

				chunkLayers.Add(new SyntheticChunkLayers(ChunkIndex: chunkIndex, Layers: layers.ToArray()));
			}
		}

		return new SyntheticControlTile(tileName, mapName, texturePaths.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase).ToList(), chunkLayers);
	}

	private static Image<Rgba32> RenderMinimap(SyntheticControlSpec spec)
	{
		Image<Rgba32> image = new(MinimapSize, MinimapSize);
		for (int y = 0; y < MinimapSize; y++)
		{
			float v = y / (float)(MinimapSize - 1);
			for (int x = 0; x < MinimapSize; x++)
			{
				float u = x / (float)(MinimapSize - 1);
				float height = spec.HeightFunction(u, v);
				byte alpha1 = spec.AlphaFunctions[0](u, v);
				byte alpha2 = spec.AlphaFunctions[1](u, v);
				byte alpha3 = spec.AlphaFunctions[2](u, v);
				image[x, y] = spec.ColorFunction(u, v, height, alpha1, alpha2, alpha3);
			}
		}

		return image;
	}

	private static Image<L16> RenderHeightmap(SyntheticControlSpec spec)
	{
		Image<L16> image = new(HeightmapSize, HeightmapSize);
		for (int y = 0; y < HeightmapSize; y++)
		{
			float v = y / (float)(HeightmapSize - 1);
			for (int x = 0; x < HeightmapSize; x++)
			{
				float u = x / (float)(HeightmapSize - 1);
				ushort value = (ushort)Math.Clamp(MathF.Round(spec.HeightFunction(u, v) * 65535f), 0f, 65535f);
				image[x, y] = new L16(value);
			}
		}

		return image;
	}

	private static Image<L8> RenderAlphaLayer(SyntheticControlSpec spec, int layerIndex)
	{
		Image<L8> image = new(TileAlphaSize, TileAlphaSize);
		for (int y = 0; y < TileAlphaSize; y++)
		{
			float v = y / (float)(TileAlphaSize - 1);
			for (int x = 0; x < TileAlphaSize; x++)
			{
				float u = x / (float)(TileAlphaSize - 1);
				image[x, y] = new L8(spec.AlphaFunctions[layerIndex](u, v));
			}
		}

		return image;
	}

	private static Image<L8> RenderShadow(SyntheticControlSpec spec)
	{
		Image<L8> image = new(TileAlphaSize, TileAlphaSize);
		for (int y = 0; y < TileAlphaSize; y++)
		{
			float v = y / (float)(TileAlphaSize - 1);
			for (int x = 0; x < TileAlphaSize; x++)
			{
				float u = x / (float)(TileAlphaSize - 1);
				image[x, y] = new L8(spec.ShadowFunction(u, v));
			}
		}

		return image;
	}

	private static Image<Rgba32> RenderAlphaAtlas(SyntheticControlSpec spec)
	{
		Image<Rgba32> image = new(TileAlphaSize, TileAlphaSize);
		for (int y = 0; y < TileAlphaSize; y++)
		{
			float v = y / (float)(TileAlphaSize - 1);
			for (int x = 0; x < TileAlphaSize; x++)
			{
				float u = x / (float)(TileAlphaSize - 1);
				image[x, y] = new Rgba32(
					spec.AlphaFunctions[0](u, v),
					spec.AlphaFunctions[1](u, v),
					spec.AlphaFunctions[2](u, v),
					spec.ShadowFunction(u, v));
			}
		}

		return image;
	}

	private static byte[] BuildChunkAlphaBytes(SyntheticControlSpec spec, int layerIndex, int chunkX, int chunkY)
	{
		byte[] bytes = new byte[ChunkAlphaSize * ChunkAlphaSize];
		int baseX = chunkX * ChunkAlphaSize;
		int baseY = chunkY * ChunkAlphaSize;
		for (int y = 0; y < ChunkAlphaSize; y++)
		{
			float v = (baseY + y) / (float)(TileAlphaSize - 1);
			for (int x = 0; x < ChunkAlphaSize; x++)
			{
				float u = (baseX + x) / (float)(TileAlphaSize - 1);
				bytes[(y * ChunkAlphaSize) + x] = spec.AlphaFunctions[layerIndex](u, v);
			}
		}

		return bytes;
	}

	private static string Relativize(string root, string path)
	{
		return Path.GetRelativePath(root, path).Replace('\\', '/');
	}

	private static JsonSerializerOptions CreateJsonOptions()
	{
		JsonSerializerOptions options = new()
		{
			WriteIndented = true,
			DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
		};
		options.Converters.Add(new JsonStringEnumConverter());
		return options;
	}

	private static string? GetOption(string[] args, string longName, string shortName)
	{
		for (int index = 0; index < args.Length - 1; index++)
		{
			if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
				|| string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
			{
				return args[index + 1];
			}
		}

		return null;
	}

	private static HybridSourceData? TryLoadHybridSource(string datasetRoot, string? preferredTileName)
	{
		if (!Directory.Exists(datasetRoot))
			return null;

		string manifestPath = Path.Combine(datasetRoot, "brush_imprints", "brush_imprint_manifest.json");
		if (!File.Exists(manifestPath))
			return null;

		using JsonDocument manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
		if (!manifestDocument.RootElement.TryGetProperty("tiles", out JsonElement tilesElement) || tilesElement.ValueKind != JsonValueKind.Array)
			return null;

		JsonElement? selectedTile = null;
		foreach (JsonElement tile in tilesElement.EnumerateArray())
		{
			string tileName = tile.GetProperty("tile_name").GetString() ?? string.Empty;
			if (!string.IsNullOrWhiteSpace(preferredTileName))
			{
				if (string.Equals(tileName, preferredTileName, StringComparison.OrdinalIgnoreCase))
				{
					selectedTile = tile;
					break;
				}
				continue;
			}

			string? brushMaskRelative = tile.TryGetProperty("brush_mask_path", out JsonElement brushMaskPathElement)
				? brushMaskPathElement.GetString()
				: null;
			if (string.IsNullOrWhiteSpace(brushMaskRelative))
				continue;

			if (selectedTile is null)
			{
				selectedTile = tile;
				continue;
			}

			int currentGroups = tile.TryGetProperty("groups_written", out JsonElement currentGroupsElement) ? currentGroupsElement.GetInt32() : 0;
			int currentPatches = tile.TryGetProperty("patch_candidates", out JsonElement currentPatchesElement) ? currentPatchesElement.GetInt32() : 0;
			int bestGroups = selectedTile.Value.TryGetProperty("groups_written", out JsonElement bestGroupsElement) ? bestGroupsElement.GetInt32() : 0;
			int bestPatches = selectedTile.Value.TryGetProperty("patch_candidates", out JsonElement bestPatchesElement) ? bestPatchesElement.GetInt32() : 0;

			if (currentGroups > bestGroups || (currentGroups == bestGroups && currentPatches > bestPatches))
				selectedTile = tile;
		}

		if (selectedTile is null)
			return null;

		string sourceTileName = selectedTile.Value.GetProperty("tile_name").GetString() ?? string.Empty;
		if (string.IsNullOrWhiteSpace(sourceTileName))
			return null;

		string jsonPath = Path.Combine(datasetRoot, "dataset", sourceTileName + ".json");
		if (!File.Exists(jsonPath))
			return null;

		using JsonDocument datasetDocument = JsonDocument.Parse(File.ReadAllText(jsonPath));
		JsonElement rootElement = datasetDocument.RootElement;
		JsonElement terrainElement = rootElement.GetProperty("terrain_data");

		string? imageRelativePath = rootElement.TryGetProperty("image", out JsonElement imageElement)
			? imageElement.GetString()
			: null;
		string? heightmapRelativePath = terrainElement.TryGetProperty("heightmap_global", out JsonElement heightmapElement)
			? heightmapElement.GetString()
			: null;
		string? brushMaskRelativePath = selectedTile.Value.TryGetProperty("brush_mask_path", out JsonElement brushMaskElement)
			? brushMaskElement.GetString()
			: null;
		string? fractalRelativePath = selectedTile.Value.TryGetProperty("fractal_detail_path", out JsonElement fractalElement)
			? fractalElement.GetString()
			: null;

		if (string.IsNullOrWhiteSpace(imageRelativePath) || string.IsNullOrWhiteSpace(heightmapRelativePath) || string.IsNullOrWhiteSpace(brushMaskRelativePath))
			return null;

		string minimapPath = Path.Combine(datasetRoot, imageRelativePath.Replace('/', Path.DirectorySeparatorChar));
		string heightmapPath = Path.Combine(datasetRoot, heightmapRelativePath.Replace('/', Path.DirectorySeparatorChar));
		string brushMaskPath = Path.Combine(datasetRoot, brushMaskRelativePath.Replace('/', Path.DirectorySeparatorChar));
		string? fractalPath = string.IsNullOrWhiteSpace(fractalRelativePath)
			? null
			: Path.Combine(datasetRoot, fractalRelativePath.Replace('/', Path.DirectorySeparatorChar));

		if (!File.Exists(minimapPath) || !File.Exists(heightmapPath) || !File.Exists(brushMaskPath))
			return null;

		List<string> textures = [];
		if (terrainElement.TryGetProperty("textures", out JsonElement texturesElement) && texturesElement.ValueKind == JsonValueKind.Array)
		{
			foreach (JsonElement texture in texturesElement.EnumerateArray())
			{
				string? texturePath = texture.GetString();
				if (!string.IsNullOrWhiteSpace(texturePath))
					textures.Add(texturePath);
			}
		}

		using Image<Rgba32> minimapImage = Image.Load<Rgba32>(minimapPath);
		using Image<L16> heightmapImage = Image.Load<L16>(heightmapPath);
		using Image<L8> brushMaskImage = Image.Load<L8>(brushMaskPath);
		using Image<L8>? fractalImage = !string.IsNullOrWhiteSpace(fractalPath) && File.Exists(fractalPath)
			? Image.Load<L8>(fractalPath)
			: null;

		Rgba32[] minimapPixels = new Rgba32[minimapImage.Width * minimapImage.Height];
		minimapImage.CopyPixelDataTo(minimapPixels);
		L16[] heightPixels = new L16[heightmapImage.Width * heightmapImage.Height];
		heightmapImage.CopyPixelDataTo(heightPixels);
		L8[] brushPixels = new L8[brushMaskImage.Width * brushMaskImage.Height];
		brushMaskImage.CopyPixelDataTo(brushPixels);
		L8[] fractalPixels = new L8[brushMaskImage.Width * brushMaskImage.Height];
		if (fractalImage is not null)
		{
			fractalImage.CopyPixelDataTo(fractalPixels);
		}

		return new HybridSourceData(
			SourceDatasetRoot: datasetRoot,
			TileName: sourceTileName,
			MapName: selectedTile.Value.TryGetProperty("map_name", out JsonElement mapNameElement) ? mapNameElement.GetString() ?? string.Empty : string.Empty,
			TexturePaths: textures,
			MinimapPixels: minimapPixels,
			MinimapWidth: minimapImage.Width,
			MinimapHeight: minimapImage.Height,
			HeightPixels: heightPixels.Select(static pixel => pixel.PackedValue / 65535f).ToArray(),
			HeightWidth: heightmapImage.Width,
			HeightHeight: heightmapImage.Height,
			BrushPixels: brushPixels.Select(static pixel => pixel.PackedValue).ToArray(),
			BrushWidth: brushMaskImage.Width,
			BrushHeight: brushMaskImage.Height,
			FractalPixels: fractalPixels.Select(static pixel => pixel.PackedValue).ToArray(),
			FractalWidth: brushMaskImage.Width,
			FractalHeight: brushMaskImage.Height);
	}

	private static SyntheticControlSpec CreateHybridBrushPlateSpec(HybridSourceData source)
	{
		string[] texturePaths = ResolveHybridTexturePaths(source.TexturePaths);
		return new SyntheticControlSpec(
			Name: "hybrid_brush_plate",
			Description: string.Create(CultureInfo.InvariantCulture, $"Hybrid control seeded from {source.TileName}: real minimap, height, and brush signal blended with a synthetic terrace carrier."),
			ExpectedInterestClass: "control-hybrid",
			ExpectedBrushGroups: 2,
			ExpectedLayerStackDepth: 2,
			BaseTexturePath: texturePaths[0],
			OverlayTexturePaths: [texturePaths[1], texturePaths[2], texturePaths[3]],
			HeightFunction: (u, v) =>
			{
				float sourceHeight = SampleFloat(source.HeightPixels, source.HeightWidth, source.HeightHeight, u, v);
				float terrace = MathF.Floor((u * 5f) + (v * 2f)) / 6f;
				return Math.Clamp((sourceHeight * 0.82f) + (terrace * 0.18f), 0f, 1f);
			},
			ColorFunction: (u, v, height, alpha1, alpha2, alpha3) =>
			{
				Rgba32 sourceColor = SampleColor(source.MinimapPixels, source.MinimapWidth, source.MinimapHeight, u, v);
				float accent = ((alpha1 * 0.35f) + (alpha2 * 0.45f) + (alpha3 * 0.20f)) / 255f;
				return new Rgba32(
					(byte)Math.Clamp((sourceColor.R * 0.78f) + (height * 70f) + (accent * 35f), 0f, 255f),
					(byte)Math.Clamp((sourceColor.G * 0.82f) + (height * 55f) + (accent * 18f), 0f, 255f),
					(byte)Math.Clamp((sourceColor.B * 0.70f) + (height * 28f) + (accent * 42f), 0f, 255f),
					255);
			},
			AlphaFunctions:
			[
				(u, v) =>
				{
					byte brush = SampleByte(source.BrushPixels, source.BrushWidth, source.BrushHeight, u, v);
					return brush > 0 ? brush : (MathF.Abs(u - v) < 0.10f ? (byte)150 : (byte)0);
				},
				(u, v) =>
				{
					byte fractal = SampleByte(source.FractalPixels, source.FractalWidth, source.FractalHeight, u, v);
					byte ring = EvaluateRing(u, v, 0.18f, 0.34f, 140);
					return (byte)Math.Clamp(fractal + ring, 0, 255);
				},
				(u, v) =>
				{
					byte brush = SampleByte(source.BrushPixels, source.BrushWidth, source.BrushHeight, u, v);
					float centered = 1f - MathF.Min(1f, MathF.Abs(u - 0.5f) * 2.2f);
					return brush > 96 ? (byte)Math.Clamp(centered * 180f, 0f, 255f) : (byte)0;
				}
			],
			ShadowFunction: (u, v) =>
			{
				byte fractal = SampleByte(source.FractalPixels, source.FractalWidth, source.FractalHeight, u, v);
				return fractal > 0 ? (byte)Math.Clamp(fractal * 0.5f, 0f, 255f) : (u + v > 1.2f ? (byte)70 : (byte)0);
			});
	}

	private static string[] ResolveHybridTexturePaths(List<string> sourceTextures)
	{
		string[] fallback =
		[
			"synthetic/textures/hybrid_base.blp",
			"synthetic/textures/hybrid_overlay_1.blp",
			"synthetic/textures/hybrid_overlay_2.blp",
			"synthetic/textures/hybrid_overlay_3.blp"
		];

		string[] resolved = fallback.ToArray();
		for (int index = 0; index < Math.Min(sourceTextures.Count, resolved.Length); index++)
		{
			if (!string.IsNullOrWhiteSpace(sourceTextures[index]))
				resolved[index] = sourceTextures[index];
		}

		return resolved;
	}

	private static int SampleIndex(int width, int height, float u, float v)
	{
		int x = Math.Clamp((int)MathF.Round(u * (width - 1)), 0, width - 1);
		int y = Math.Clamp((int)MathF.Round(v * (height - 1)), 0, height - 1);
		return (y * width) + x;
	}

	private static float SampleFloat(float[] values, int width, int height, float u, float v)
	{
		return values[SampleIndex(width, height, u, v)];
	}

	private static byte SampleByte(byte[] values, int width, int height, float u, float v)
	{
		return values[SampleIndex(width, height, u, v)];
	}

	private static Rgba32 SampleColor(Rgba32[] values, int width, int height, float u, float v)
	{
		return values[SampleIndex(width, height, u, v)];
	}

	private static SyntheticControlSpec CreateWhitePlateSpec()
	{
		return new SyntheticControlSpec(
			Name: "white_plate",
			Description: "Completely flat non-interesting control tile with a single z=0-like plate and no overlay structure.",
			ExpectedInterestClass: "non-interesting",
			ExpectedBrushGroups: 0,
			ExpectedLayerStackDepth: 0,
			BaseTexturePath: "synthetic/textures/white_plate_base.blp",
			OverlayTexturePaths:
			[
				"synthetic/textures/white_plate_overlay_1.blp",
				"synthetic/textures/white_plate_overlay_2.blp",
				"synthetic/textures/white_plate_overlay_3.blp"
			],
			HeightFunction: static (_, _) => 0f,
			ColorFunction: static (_, _, _, _, _, _) => new Rgba32(255, 255, 255, 255),
			AlphaFunctions:
			[
				static (_, _) => (byte)0,
				static (_, _) => (byte)0,
				static (_, _) => (byte)0
			],
			ShadowFunction: static (_, _) => (byte)0);
	}

	private static SyntheticControlSpec CreateDiagonalRampSpec()
	{
		return new SyntheticControlSpec(
			Name: "diagonal_ramp",
			Description: "A smooth diagonal rise with a single dominant blend band for baseline slope supervision.",
			ExpectedInterestClass: "control-interesting",
			ExpectedBrushGroups: 1,
			ExpectedLayerStackDepth: 1,
			BaseTexturePath: "synthetic/textures/ramp_base.blp",
			OverlayTexturePaths:
			[
				"synthetic/textures/ramp_overlay_1.blp",
				"synthetic/textures/ramp_overlay_2.blp",
				"synthetic/textures/ramp_overlay_3.blp"
			],
			HeightFunction: static (u, v) => Math.Clamp((u * 0.75f) + (v * 0.25f), 0f, 1f),
			ColorFunction: static (_, _, height, alpha1, _, _) => new Rgba32(
				(byte)Math.Clamp(80f + (height * 140f) + (alpha1 * 0.2f), 0f, 255f),
				(byte)Math.Clamp(110f + (height * 100f), 0f, 255f),
				(byte)Math.Clamp(90f + (height * 70f), 0f, 255f),
				255),
			AlphaFunctions:
			[
				static (u, v) => MathF.Abs(u - v) < 0.12f ? (byte)220 : (byte)0,
				static (_, _) => (byte)0,
				static (_, _) => (byte)0
			],
			ShadowFunction: static (u, v) => u + v > 1.15f ? (byte)80 : (byte)0);
	}

	private static SyntheticControlSpec CreateRingMoundSpec()
	{
		return new SyntheticControlSpec(
			Name: "ring_mound",
			Description: "Radial mound with stacked concentric blend rings to exercise atlas and layer-depth validation.",
			ExpectedInterestClass: "control-interesting",
			ExpectedBrushGroups: 2,
			ExpectedLayerStackDepth: 3,
			BaseTexturePath: "synthetic/textures/ring_base.blp",
			OverlayTexturePaths:
			[
				"synthetic/textures/ring_overlay_1.blp",
				"synthetic/textures/ring_overlay_2.blp",
				"synthetic/textures/ring_overlay_3.blp"
			],
			HeightFunction: static (u, v) =>
			{
				float dx = u - 0.5f;
				float dy = v - 0.5f;
				float radius = MathF.Sqrt((dx * dx) + (dy * dy));
				return Math.Clamp(1f - (radius * 1.8f), 0f, 1f);
			},
			ColorFunction: static (_, _, height, alpha1, alpha2, alpha3) => new Rgba32(
				(byte)Math.Clamp(50f + (height * 110f) + (alpha2 * 0.35f), 0f, 255f),
				(byte)Math.Clamp(70f + (height * 120f) + (alpha1 * 0.25f), 0f, 255f),
				(byte)Math.Clamp(40f + (height * 80f) + (alpha3 * 0.4f), 0f, 255f),
				255),
			AlphaFunctions:
			[
				static (u, v) => EvaluateRing(u, v, 0.18f, 0.30f, 235),
				static (u, v) => EvaluateRing(u, v, 0.28f, 0.40f, 210),
				static (u, v) => EvaluateRing(u, v, 0.08f, 0.18f, 180)
			],
			ShadowFunction: static (u, v) =>
			{
				float dx = u - 0.62f;
				float dy = v - 0.42f;
				return (dx * dx) + (dy * dy) < 0.035f ? (byte)150 : (byte)0;
			});
	}

	private static SyntheticControlSpec CreateTerraceStepsSpec()
	{
		return new SyntheticControlSpec(
			Name: "terrace_steps",
			Description: "Stepped terrace control with checker transitions for fault finding in quantized terrain and mask decoding.",
			ExpectedInterestClass: "control-interesting",
			ExpectedBrushGroups: 3,
			ExpectedLayerStackDepth: 2,
			BaseTexturePath: "synthetic/textures/terrace_base.blp",
			OverlayTexturePaths:
			[
				"synthetic/textures/terrace_overlay_1.blp",
				"synthetic/textures/terrace_overlay_2.blp",
				"synthetic/textures/terrace_overlay_3.blp"
			],
			HeightFunction: static (u, v) =>
			{
				float steps = MathF.Floor((u * 6f) + (v * 2f)) / 7f;
				return Math.Clamp(steps, 0f, 1f);
			},
			ColorFunction: static (_, _, height, alpha1, alpha2, _) => new Rgba32(
				(byte)Math.Clamp(90f + (height * 100f) + (alpha2 * 0.3f), 0f, 255f),
				(byte)Math.Clamp(60f + (height * 80f) + (alpha1 * 0.25f), 0f, 255f),
				(byte)Math.Clamp(50f + (height * 60f), 0f, 255f),
				255),
			AlphaFunctions:
			[
				static (u, v) => ((int)MathF.Floor(u * 8f) + (int)MathF.Floor(v * 8f)) % 2 == 0 ? (byte)200 : (byte)0,
				static (u, _) => MathF.Abs(u - 0.5f) < 0.16f ? (byte)180 : (byte)0,
				static (_, _) => (byte)0
			],
			ShadowFunction: static (_, v) => v > 0.72f ? (byte)90 : (byte)0);
	}

	private static byte EvaluateRing(float u, float v, float minRadius, float maxRadius, byte value)
	{
		float dx = u - 0.5f;
		float dy = v - 0.5f;
		float radius = MathF.Sqrt((dx * dx) + (dy * dy));
		return radius >= minRadius && radius <= maxRadius ? value : (byte)0;
	}
}

internal sealed record SyntheticDatasetSample(
	[property: JsonPropertyName("image")] string ImagePath,
	[property: JsonPropertyName("synthetic_control")] SyntheticControlMetadata SyntheticControl,
	[property: JsonPropertyName("terrain_data")] SyntheticTerrainData TerrainData);

internal sealed record SyntheticControlMetadata(
	[property: JsonPropertyName("name")] string Name,
	[property: JsonPropertyName("expected_interest_class")] string ExpectedInterestClass,
	[property: JsonPropertyName("description")] string Description,
	[property: JsonPropertyName("expected_brush_groups")] int ExpectedBrushGroups,
	[property: JsonPropertyName("expected_layer_stack_depth")] int ExpectedLayerStackDepth);

internal sealed record SyntheticTerrainData(
	[property: JsonPropertyName("adt_tile")] string AdtTile,
	[property: JsonPropertyName("heightmap")] string HeightmapPath,
	[property: JsonPropertyName("heightmap_local")] string HeightmapLocalPath,
	[property: JsonPropertyName("heightmap_global")] string HeightmapGlobalPath,
	[property: JsonPropertyName("alpha_masks")] string[] AlphaMasks,
	[property: JsonPropertyName("alpha_atlas")] string AlphaAtlasPath,
	[property: JsonPropertyName("shadow_maps")] string[] ShadowMaps,
	[property: JsonPropertyName("textures")] List<string> Textures,
	[property: JsonPropertyName("chunk_layers")] SyntheticChunkLayers[] ChunkLayers,
	[property: JsonPropertyName("holes")] int[] Holes,
	[property: JsonPropertyName("objects")] object[] Objects,
	[property: JsonPropertyName("liquids")] object[] Liquids,
	[property: JsonPropertyName("height_min")] float HeightMin,
	[property: JsonPropertyName("height_max")] float HeightMax,
	[property: JsonPropertyName("height_global_min")] float HeightGlobalMin,
	[property: JsonPropertyName("height_global_max")] float HeightGlobalMax,
	[property: JsonPropertyName("is_interleaved")] bool IsInterleaved);

internal sealed record SyntheticChunkLayers(
	[property: JsonPropertyName("idx")] int ChunkIndex,
	[property: JsonPropertyName("layers")] SyntheticTextureLayer[] Layers);

internal sealed record SyntheticTextureLayer(
	[property: JsonPropertyName("tex_id")] uint TextureId,
	[property: JsonPropertyName("texture_path")] string TexturePath,
	[property: JsonPropertyName("flags")] uint Flags,
	[property: JsonPropertyName("alpha_off")] uint AlphaOffset,
	[property: JsonPropertyName("effect_id")] uint EffectId,
	[property: JsonPropertyName("alpha_bits")] string? AlphaBitsBase64);

internal sealed record SyntheticControlManifest(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("control_count")] int ControlCount,
	[property: JsonPropertyName("controls")] List<SyntheticControlManifestEntry> Controls);

internal sealed record SyntheticControlManifestEntry(
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("control_name")] string ControlName,
	[property: JsonPropertyName("expected_interest_class")] string ExpectedInterestClass,
	[property: JsonPropertyName("description")] string Description,
	[property: JsonPropertyName("expected_brush_groups")] int ExpectedBrushGroups,
	[property: JsonPropertyName("expected_layer_stack_depth")] int ExpectedLayerStackDepth,
	[property: JsonPropertyName("image_path")] string ImagePath,
	[property: JsonPropertyName("heightmap_global_path")] string HeightmapGlobalPath,
	[property: JsonPropertyName("alpha_atlas_path")] string AlphaAtlasPath,
	[property: JsonPropertyName("tile_json_path")] string TileJsonPath);

internal sealed record SyntheticMetadataRow(
	[property: JsonPropertyName("file_name")] string FileName,
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("tile_json")] string TileJson,
	[property: JsonPropertyName("heightmap_global")] string HeightmapGlobal,
	[property: JsonPropertyName("alpha_atlas")] string AlphaAtlas,
	[property: JsonPropertyName("expected_interest_class")] string ExpectedInterestClass,
	[property: JsonPropertyName("synthetic_control")] string SyntheticControl);

internal sealed record SyntheticControlTile(
	string TileName,
	string MapName,
	List<string> TexturePaths,
	List<SyntheticChunkLayers> ChunkLayers);

internal sealed record HybridSourceData(
	string SourceDatasetRoot,
	string TileName,
	string MapName,
	List<string> TexturePaths,
	Rgba32[] MinimapPixels,
	int MinimapWidth,
	int MinimapHeight,
	float[] HeightPixels,
	int HeightWidth,
	int HeightHeight,
	byte[] BrushPixels,
	int BrushWidth,
	int BrushHeight,
	byte[] FractalPixels,
	int FractalWidth,
	int FractalHeight);

internal sealed record SyntheticControlSpec(
	string Name,
	string Description,
	string ExpectedInterestClass,
	int ExpectedBrushGroups,
	int ExpectedLayerStackDepth,
	string BaseTexturePath,
	string[] OverlayTexturePaths,
	Func<float, float, float> HeightFunction,
	Func<float, float, float, byte, byte, byte, Rgba32> ColorFunction,
	Func<float, float, byte>[] AlphaFunctions,
	Func<float, float, byte> ShadowFunction);