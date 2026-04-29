using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Tool.Converter;

public static class V10TilesetTaxonomyCommand
{
	private static readonly Dictionary<string, string> ZoneAbbreviations = new(StringComparer.OrdinalIgnoreCase)
	{
		["BT"] = "BoreanTundra",
		["DB"] = "Dragonblight",
		["DH"] = "Deepholm",
		["GH"] = "GrizzlyHills",
		["GN"] = "Gilneas",
		["HF"] = "HowlingFjord",
		["HFJORDS"] = "HowlingFjord",
		["HFjords"] = "HowlingFjord",
		["IC"] = "Icecrown",
		["IG"] = "Icecrown",
		["IceC"] = "Icecrown",
		["LI"] = "LostIsles",
		["LW"] = "LakeWintergrasp",
		["Org"] = "Orgrimmar",
		["Orgrim"] = "Orgrimmar",
		["SB"] = "SholazarBasin",
		["SH"] = "SholazarBasin",
		["SP"] = "StormPeaks",
		["SW"] = "StormwindCity",
		["SWC"] = "StormwindCity",
		["TH"] = "TwilightHighlands",
		["UL"] = "Uldum",
		["VJ"] = "Vashjir",
		["WG"] = "LakeWintergrasp",
		["ZD"] = "ZulDrak",
	};

	private static readonly string[] LayerRoleKeywords =
	[
		"Base", "Highlight", "Shadow", "Light", "Dark", "Darker", "Lighter",
		"Alpha", "Crack", "Cracked", "Smooth", "Rough",
	];

	private static readonly string[] TypeKeywords =
	[
		"Grass", "Dirt", "Rock", "Sand", "Soil", "Stone", "Mud", "Snow", "Ice",
		"Gravel", "Clay", "Cobble", "CobbleStone", "Brick", "Wood", "Leaf", "Leaves",
		"Flower", "Moss", "Slime", "Lava", "Ash", "Bone", "Fur", "Skin", "Scale",
		"Metal", "Crystal", "Water", "Coral", "Reef", "Shore", "Beach",
		"Roots", "Ferns", "Thorns", "Weeds", "Vines", "Bush", "Brush",
		"Crop", "Rubble", "Pebbles", "Shale", "Muck", "Mulch", "PineNeedles",
		"Road", "Path", "Tile", "Floor", "Paver", "Flagstones",
		"Corrupt", "Creep", "Blight", "Decay", "Dead", "Dying",
		"Termite", "Mosaic", "Barnacle", "Mold", "Fungus", "Membrane",
		"Caustics", "PackIce", "Lumpysnow", "PackedSnow",
	];

	private static readonly string[] ModifierKeywords =
	[
		"Darker", "Lighter", "Dark", "Light", "Wet", "Dry", "Cracked", "Smooth",
		"Rough", "Cold", "Warm", "Frozen", "Melted", "Burnt", "Fresh", "Old",
		"New", "Solid", "Soft", "Hard", "Green", "Brown", "Red", "Blue", "Purple",
		"Orange", "Pink", "Grey", "Gray", "Black", "White", "Yellow",
		"Dead", "Dying", "Corrupt", "Purple", "Barnacle",
	];

	private static readonly string[] VariantPatterns =
	[
		"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
		"A", "B", "C", "D", "E", "F", "G", "H",
	];

	public static void Run(string[] args)
	{
		string? input = GetOption(args, "--input", "-i");
		string? outputDir = GetOption(args, "--output-dir", "-o");

		if (string.IsNullOrWhiteSpace(input))
		{
			Console.Error.WriteLine("Error: --input <merged_tileset_index.json> is required.");
			Environment.ExitCode = 1;
			return;
		}

		if (!File.Exists(input))
		{
			Console.Error.WriteLine($"Error: input file not found: {input}");
			Environment.ExitCode = 1;
			return;
		}

		string outputRoot = string.IsNullOrWhiteSpace(outputDir)
			? Path.Combine(Environment.CurrentDirectory, "output", "ml-training", "v10_tileset_taxonomy")
			: Path.GetFullPath(outputDir);

		Directory.CreateDirectory(outputRoot);

		TilesetDatabase? db = JsonSerializer.Deserialize<TilesetDatabase>(File.ReadAllText(input), CreateJsonOptions());
		if (db is null || db.Entries.Count == 0)
		{
			Console.Error.WriteLine("Error: no entries found in input database.");
			Environment.ExitCode = 1;
			return;
		}

		Console.WriteLine($"Loaded {db.Entries.Count} tileset entries");

		Dictionary<string, TextureTaxonomyEntry> taxonomy = BuildTaxonomy(db.Entries);
		Dictionary<string, TextureFamily> families = IdentifyTextureFamilies(taxonomy);
		Dictionary<string, CrossEraTextureGroup> crossEraGroups = FindCrossEraMatches(taxonomy);
		List<LayerStack> layerStacks = IdentifyLayerStacks(families);
		List<TextureCategory> categories = CategorizeTextures(taxonomy);

		string taxonomyPath = Path.Combine(outputRoot, "texture_taxonomy.json");
		File.WriteAllText(taxonomyPath, JsonSerializer.Serialize(new TextureTaxonomyReport(
			SchemaVersion: "v10-tileset-taxonomy.v1",
			GeneratedAtUtc: DateTimeOffset.UtcNow,
			TotalTextures: taxonomy.Count,
			TextureFamilies: families.Values.OrderBy(f => f.Textures.Count).ToList(),
			LayerStacks: layerStacks,
			CrossEraGroups: crossEraGroups.Values.OrderBy(g => g.TextureName).ToList(),
			Categories: categories.OrderBy(c => c.Textures.Count).ToList(),
			NamingConventions: BuildNamingConventions(taxonomy)), CreateJsonOptions()));

		Console.WriteLine("WowViewer.Tool.Converter analyze-tileset-taxonomy report");
		Console.WriteLine($"Input: {input}");
		Console.WriteLine($"Total unique textures: {taxonomy.Count}");
		Console.WriteLine($"Texture families: {families.Count}");
		Console.WriteLine($"Layer stacks: {layerStacks.Count}");
		Console.WriteLine($"Cross-era matches: {crossEraGroups.Count}");
		Console.WriteLine($"Texture categories: {categories.Count}");
		Console.WriteLine($"Wrote {taxonomyPath}");

		Console.WriteLine("");
		Console.WriteLine("=== Top Texture Families (by texture count) ===");
		foreach (TextureFamily family in families.Values.OrderByDescending(f => f.Textures.Count).Take(20))
		{
			Console.WriteLine($"  {family.FamilyId}: {family.Textures.Count} textures, eras=[{string.Join(", ", family.Eras.Take(5))}]");
		}

		Console.WriteLine("");
		Console.WriteLine("=== Layer Stacks (complete texture sets per zone+type) ===");
		foreach (LayerStack stack in layerStacks.OrderByDescending(s => s.TotalTextures).Take(20))
		{
			Console.WriteLine($"  {stack.Zone}/{stack.TextureType}: {stack.TotalTextures} textures, layers=[{string.Join(", ", stack.Layers)}]");
		}
	}

	private static Dictionary<string, TextureTaxonomyEntry> BuildTaxonomy(List<TilesetEntry> entries)
	{
		Dictionary<string, TextureTaxonomyEntry> taxonomy = new(StringComparer.OrdinalIgnoreCase);

		foreach (TilesetEntry entry in entries)
		{
			string key = $"{entry.EraTag}|{entry.RelativePath}";
			if (taxonomy.ContainsKey(key))
				continue;

			TextureClassification classification = ClassifyTexture(entry);

			taxonomy[key] = new TextureTaxonomyEntry(
				RelativePath: entry.RelativePath,
				EraTag: entry.EraTag,
				DesignKit: entry.DesignKit,
				ZoneName: entry.ZoneName,
				FileName: entry.FileName,
				Classification: classification,
				Width: entry.Width,
				Height: entry.Height,
				HasAlpha: entry.HasAlpha,
				IsSpecularVariant: entry.IsSpecularVariant);
		}

		return taxonomy;
	}

	private static TextureClassification ClassifyTexture(TilesetEntry entry)
	{
		string name = entry.FileName;
		string nameLower = entry.FileNameLower;

		string cleanName = StripCopyOfPrefix(name);
		string namingStyle = DetermineNamingStyle(cleanName);
		string rawPrefix = ExtractRawPrefix(cleanName, namingStyle);
		string zonePrefix = ExpandZoneAbbreviation(rawPrefix);
		string coreName = ExtractCoreName(cleanName, namingStyle, rawPrefix);

		string textureType = TypeKeywords.FirstOrDefault(kw => name.IndexOf(kw, StringComparison.OrdinalIgnoreCase) >= 0) ?? "unknown";
		string layerRole = DetermineLayerRole(name);
		string variantSuffix = ExtractVariantSuffix(name);
		string modifier = ModifierKeywords.FirstOrDefault(kw => name.IndexOf(kw, StringComparison.OrdinalIgnoreCase) >= 0) ?? "";

		bool isSpecular = nameLower.EndsWith("_s") || nameLower.EndsWith("_s.blp");
		bool isHeight = nameLower.EndsWith("_h") || nameLower.EndsWith("_h.blp");
		bool isBase = layerRole == "base";
		bool isOverlay = layerRole is "highlight" or "shadow" or "light" or "dark";

		string coreKey = BuildCoreKey(zonePrefix, textureType, coreName);

		return new TextureClassification(
			TextureType: textureType,
			LayerRole: layerRole,
			NamingStyle: namingStyle,
			ZonePrefix: zonePrefix,
			CoreName: coreName,
			CoreKey: coreKey,
			Modifier: modifier,
			VariantSuffix: variantSuffix,
			IsBase: isBase,
			IsOverlay: isOverlay,
			IsSpecular: isSpecular,
			IsHeight: isHeight);
	}

	private static string DetermineNamingStyle(string name)
	{
		string cleanName = StripCopyOfPrefix(name);

		if (cleanName.StartsWith("BT_") || cleanName.StartsWith("DB_") || cleanName.StartsWith("GH_") ||
			cleanName.StartsWith("DH_") || cleanName.StartsWith("ZD_") || cleanName.StartsWith("ITK_") ||
			cleanName.StartsWith("SR_") || cleanName.StartsWith("SM_") || cleanName.StartsWith("HM_") ||
			cleanName.StartsWith("NG_") || cleanName.StartsWith("SA_") || cleanName.StartsWith("UDM_") ||
			cleanName.StartsWith("DUR_") || cleanName.StartsWith("ELW_") || cleanName.StartsWith("ESW_") ||
			cleanName.StartsWith("VS_") || cleanName.StartsWith("OG_") || cleanName.StartsWith("CAN_") ||
			cleanName.StartsWith("CAV_") || cleanName.StartsWith("DG_") || cleanName.StartsWith("ED_") ||
			cleanName.StartsWith("GSL_") || cleanName.StartsWith("HGL_") || cleanName.StartsWith("TI_") ||
			cleanName.StartsWith("AR_") || cleanName.StartsWith("NR_") || cleanName.StartsWith("RIV_") ||
			cleanName.StartsWith("WAR_") || cleanName.StartsWith("NAJ_") || cleanName.StartsWith("SWA_") ||
			cleanName.StartsWith("UND_") || cleanName.StartsWith("JF_") || cleanName.StartsWith("7SR_") ||
			cleanName.StartsWith("8SWA_") || cleanName.StartsWith("8RIV_") || cleanName.StartsWith("8UND_") ||
			cleanName.StartsWith("8WAR_") || cleanName.StartsWith("8NAJ_") ||
			cleanName.StartsWith("IC_") || cleanName.StartsWith("ND_") || cleanName.StartsWith("SP_") ||
			cleanName.StartsWith("SB_") || cleanName.StartsWith("HFjords_") || cleanName.StartsWith("HF_") ||
			cleanName.StartsWith("ZM_"))
			return "abbreviated";

		if (cleanName.Any(char.IsDigit) && cleanName.IndexOf('_') > 0 && cleanName[0] >= '0' && cleanName[0] <= '9')
			return "numeric";

		if (cleanName.Contains('_') && char.IsUpper(cleanName[0]))
		{
			int firstUnderscore = cleanName.IndexOf('_');
			if (firstUnderscore > 2 && firstUnderscore < 20)
				return "underscore_zone";
		}

		return "camelcase";
	}

	private static string StripCopyOfPrefix(string name)
	{
		if (name.StartsWith("Copy of ", StringComparison.OrdinalIgnoreCase))
			return name.Substring(8);
		if (name.StartsWith("Copyof", StringComparison.OrdinalIgnoreCase))
			return name.Substring(6);
		return name;
	}

	private static string ExtractZonePrefix(string name, string namingStyle)
	{
		string raw = namingStyle switch
		{
			"abbreviated" => name.Split('_')[0],
			"underscore_zone" => name.Split('_')[0],
			"camelcase" => ExtractCamelCasePrefix(name),
			"numeric" => name.Split('_')[0],
			_ => ""
		};

		if (ZoneAbbreviations.TryGetValue(raw, out string? expanded))
			return expanded;

		return raw;
	}

	private static string ExtractCamelCasePrefix(string name)
	{
		int pos = 1;
		while (pos < name.Length && char.IsUpper(name[pos]))
			pos++;

		while (pos < name.Length && char.IsLower(name[pos]))
			pos++;

		return name[..pos];
	}

	private static string ExtractRawPrefix(string name, string namingStyle)
	{
		return namingStyle switch
		{
			"abbreviated" => name.Split('_')[0],
			"underscore_zone" => name.Split('_')[0],
			"camelcase" => ExtractCamelCasePrefix(name),
			"numeric" => name.Split('_')[0],
			_ => ""
		};
	}

	private static string ExpandZoneAbbreviation(string raw)
	{
		if (ZoneAbbreviations.TryGetValue(raw, out string? expanded))
			return expanded;
		return raw;
	}

	private static string ExtractCoreName(string name, string namingStyle, string rawPrefix)
	{
		string core = namingStyle switch
		{
			"abbreviated" => name[(rawPrefix.Length + 1)..],
			"underscore_zone" => name[(rawPrefix.Length + 1)..],
			"numeric" => name[(rawPrefix.Length + 1)..],
			_ => name[rawPrefix.Length..]
		};

		foreach (string kw in LayerRoleKeywords.OrderByDescending(k => k.Length))
		{
			if (core.EndsWith(kw, StringComparison.OrdinalIgnoreCase))
			{
				core = core[..^kw.Length];
				break;
			}
		}

		foreach (string kw in ModifierKeywords.OrderByDescending(k => k.Length))
		{
			if (core.EndsWith(kw, StringComparison.OrdinalIgnoreCase))
			{
				core = core[..^kw.Length];
				break;
			}
		}

		foreach (string variant in VariantPatterns.OrderByDescending(v => v.Length))
		{
			if (core.EndsWith(variant, StringComparison.OrdinalIgnoreCase))
			{
				core = core[..^variant.Length];
				break;
			}
		}

		if (core.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
			core = core[..^2];
		if (core.EndsWith("_h", StringComparison.OrdinalIgnoreCase))
			core = core[..^2];

		return core.TrimEnd('_');
	}

	private static string DetermineLayerRole(string name)
	{
		string nameLower = name.ToLowerInvariant();

		if (nameLower.Contains("highlight") || nameLower.Contains("highlig"))
			return "highlight";
		if (nameLower.Contains("shadow") || nameLower.Contains("shad"))
			return "shadow";
		if (nameLower.Contains("base"))
			return "base";
		if (nameLower.Contains("darker") || nameLower.Contains("dark"))
			return "dark";
		if (nameLower.Contains("lighter") || nameLower.Contains("light"))
			return "light";
		if (nameLower.Contains("alpha"))
			return "alpha-blend";
		if (nameLower.Contains("crack"))
			return "crack-detail";
		if (nameLower.Contains("smooth"))
			return "smooth";
		if (nameLower.Contains("rough"))
			return "rough";
		if (nameLower.Contains("road") || nameLower.Contains("path") || nameLower.Contains("cobble") ||
			nameLower.Contains("brick") || nameLower.Contains("tile") || nameLower.Contains("floor") ||
			nameLower.Contains("paver") || nameLower.Contains("flagstones"))
			return "hardsurface";
		if (nameLower.Contains("moss") || nameLower.Contains("root") || nameLower.Contains("leaf") ||
			nameLower.Contains("fern") || nameLower.Contains("thorn") || nameLower.Contains("weed") ||
			nameLower.Contains("vine") || nameLower.Contains("bush") || nameLower.Contains("brush") ||
			nameLower.Contains("flower") || nameLower.Contains("crop") || nameLower.Contains("pine"))
			return "organic-detail";
		if (nameLower.Contains("shore") || nameLower.Contains("beach") || nameLower.Contains("coast"))
			return "shoreline";
		if (nameLower.Contains("dead") || nameLower.Contains("corrupt") || nameLower.Contains("creep") ||
			nameLower.Contains("blight") || nameLower.Contains("decay") || nameLower.Contains("dying"))
			return "corrupted";
		if (nameLower.Contains("lava") || nameLower.Contains("fire") || nameLower.Contains("ash") ||
			nameLower.Contains("burn") || nameLower.Contains("chard") || nameLower.Contains("magma"))
			return "volcanic";
		if (nameLower.Contains("snow") || nameLower.Contains("ice") || nameLower.Contains("frost") ||
			nameLower.Contains("frozen"))
			return "snow-ice";

		return "unknown";
	}

	private static string ExtractVariantSuffix(string name)
	{
		string core = name;
		if (core.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
			core = core[..^2];
		if (core.EndsWith("_h", StringComparison.OrdinalIgnoreCase))
			core = core[..^2];

		foreach (string variant in VariantPatterns.OrderByDescending(v => v.Length))
		{
			if (core.EndsWith(variant, StringComparison.OrdinalIgnoreCase))
				return variant;
		}
		return "";
	}

	private static string BuildCoreKey(string zonePrefix, string textureType, string coreName)
	{
		return $"{zonePrefix.ToLowerInvariant()}_{textureType.ToLowerInvariant()}_{coreName.ToLowerInvariant()}";
	}

	private static Dictionary<string, TextureFamily> IdentifyTextureFamilies(Dictionary<string, TextureTaxonomyEntry> taxonomy)
	{
		Dictionary<string, TextureFamily> families = new(StringComparer.OrdinalIgnoreCase);

		foreach ((_, TextureTaxonomyEntry entry) in taxonomy)
		{
			string familyKey = entry.Classification.CoreKey;
			if (string.IsNullOrWhiteSpace(familyKey) || entry.Classification.TextureType == "unknown")
				continue;

			if (!families.TryGetValue(familyKey, out TextureFamily? family))
			{
			family = new TextureFamily(
				familyKey,
				entry.Classification.ZonePrefix,
				entry.Classification.TextureType,
				entry.Classification.CoreName,
				[],
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				new HashSet<string>(StringComparer.OrdinalIgnoreCase));
				families[familyKey] = family;
			}

			family.Textures.Add(entry.RelativePath);
			family.LayerRoles.Add(entry.Classification.LayerRole);
			if (!string.IsNullOrWhiteSpace(entry.Classification.VariantSuffix))
				family.Variants.Add(entry.Classification.VariantSuffix);
			family.Eras.Add(entry.EraTag);
		}

		return families;
	}

	private static List<LayerStack> IdentifyLayerStacks(Dictionary<string, TextureFamily> families)
	{
		Dictionary<string, LayerStack> stacks = new(StringComparer.OrdinalIgnoreCase);

		foreach ((_, TextureFamily family) in families)
		{
			if (family.Textures.Count < 2)
				continue;

			string stackKey = $"{family.ZonePrefix}_{family.TextureType}";
			if (!stacks.TryGetValue(stackKey, out LayerStack? stack))
			{
		stack = new LayerStack(
			family.ZonePrefix,
			family.TextureType,
			new HashSet<string>(StringComparer.OrdinalIgnoreCase),
			0,
			[]);
		stacks[stackKey] = stack;
	}

	foreach (string role in family.LayerRoles)
		stack.Layers.Add(role);
	stack.TotalTextures += family.Textures.Count;
	stack.Families.Add(family.FamilyId);
		}

		return stacks.Values.Where(s => s.Layers.Count >= 2).ToList();
	}

	private static Dictionary<string, CrossEraTextureGroup> FindCrossEraMatches(Dictionary<string, TextureTaxonomyEntry> taxonomy)
	{
		Dictionary<string, CrossEraTextureGroup> groups = new(StringComparer.OrdinalIgnoreCase);

		foreach ((_, TextureTaxonomyEntry entry) in taxonomy)
		{
			string normalizedName = NormalizeTextureName(entry.FileName);
			if (string.IsNullOrWhiteSpace(normalizedName))
				continue;

			if (!groups.TryGetValue(normalizedName, out CrossEraTextureGroup? group))
			{
			group = new CrossEraTextureGroup(
				normalizedName,
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				0);
				groups[normalizedName] = group;
			}

			group.Eras.Add(entry.EraTag);
			group.DesignKits.Add(entry.DesignKit);
			group.ZoneNames.Add(entry.ZoneName);
			group.EraCount = group.Eras.Count;
		}

		return groups.Where(g => g.Value.EraCount >= 2).ToDictionary(g => g.Key, g => g.Value, StringComparer.OrdinalIgnoreCase);
	}

	private static string NormalizeTextureName(string fileName)
	{
		string name = fileName;

		if (name.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
			name = name[..^2];
		if (name.EndsWith("_h", StringComparison.OrdinalIgnoreCase))
			name = name[..^2];
		if (name.EndsWith("_S", StringComparison.Ordinal))
			name = name[..^2];
		if (name.EndsWith("_H", StringComparison.Ordinal))
			name = name[..^2];

		foreach (string kw in LayerRoleKeywords.OrderByDescending(k => k.Length))
		{
			if (name.EndsWith(kw, StringComparison.OrdinalIgnoreCase))
			{
				name = name[..^kw.Length];
				break;
			}
		}

		foreach (string kw in ModifierKeywords.OrderByDescending(k => k.Length))
		{
			if (name.EndsWith(kw, StringComparison.OrdinalIgnoreCase))
			{
				name = name[..^kw.Length];
				break;
			}
		}

		foreach (string variant in VariantPatterns.OrderByDescending(v => v.Length))
		{
			if (name.EndsWith(variant, StringComparison.OrdinalIgnoreCase))
			{
				name = name[..^variant.Length];
				break;
			}
		}

		return name.TrimEnd('_');
	}

	private static List<TextureCategory> CategorizeTextures(Dictionary<string, TextureTaxonomyEntry> taxonomy)
	{
		Dictionary<string, TextureCategory> categories = new(StringComparer.OrdinalIgnoreCase);

		foreach ((_, TextureTaxonomyEntry entry) in taxonomy)
		{
			string categoryKey = $"{entry.Classification.TextureType}_{entry.Classification.LayerRole}";
			if (!categories.TryGetValue(categoryKey, out TextureCategory? category))
			{
			category = new TextureCategory(
				categoryKey,
				entry.Classification.TextureType,
				entry.Classification.LayerRole,
				[],
				new HashSet<string>(StringComparer.OrdinalIgnoreCase),
				new HashSet<string>(StringComparer.OrdinalIgnoreCase));
				categories[categoryKey] = category;
			}

			category.Textures.Add(entry.RelativePath);
			category.DesignKits.Add(entry.DesignKit);
			category.Eras.Add(entry.EraTag);
		}

		return categories.Values.ToList();
	}

	private static Dictionary<string, object> BuildNamingConventions(Dictionary<string, TextureTaxonomyEntry> taxonomy)
	{
		var styleCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		var roleCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		var typeCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

		foreach ((_, TextureTaxonomyEntry entry) in taxonomy)
		{
			string style = entry.Classification.NamingStyle;
			if (!styleCounts.TryGetValue(style, out int sc)) sc = 0;
			styleCounts[style] = sc + 1;

			string role = entry.Classification.LayerRole;
			if (!roleCounts.TryGetValue(role, out int rc)) rc = 0;
			roleCounts[role] = rc + 1;

			string type = entry.Classification.TextureType;
			if (!typeCounts.TryGetValue(type, out int tc)) tc = 0;
			typeCounts[type] = tc + 1;
		}

		return new Dictionary<string, object>
		{
			["naming_styles"] = styleCounts.OrderByDescending(p => p.Value).ToDictionary(p => p.Key, p => p.Value),
			["layer_role_distribution"] = roleCounts.OrderByDescending(p => p.Value).ToDictionary(p => p.Key, p => p.Value),
			["texture_type_distribution"] = typeCounts.OrderByDescending(p => p.Value).ToDictionary(p => p.Key, p => p.Value),
		};
	}

	private static string? GetOption(string[] args, string longName, string shortName)
	{
		for (int i = 0; i < args.Length - 1; i++)
		{
			if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase)
				|| string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
				return args[i + 1];
		}
		return null;
	}

	private static JsonSerializerOptions CreateJsonOptions()
	{
		JsonSerializerOptions options = new() { WriteIndented = true };
		options.Converters.Add(new JsonStringEnumConverter());
		return options;
	}
}

public sealed record TextureClassification(
	[property: JsonPropertyName("texture_type")] string TextureType,
	[property: JsonPropertyName("layer_role")] string LayerRole,
	[property: JsonPropertyName("naming_style")] string NamingStyle,
	[property: JsonPropertyName("zone_prefix")] string ZonePrefix,
	[property: JsonPropertyName("core_name")] string CoreName,
	[property: JsonPropertyName("core_key")] string CoreKey,
	[property: JsonPropertyName("modifier")] string Modifier,
	[property: JsonPropertyName("variant_suffix")] string VariantSuffix,
	[property: JsonPropertyName("is_base")] bool IsBase,
	[property: JsonPropertyName("is_overlay")] bool IsOverlay,
	[property: JsonPropertyName("is_specular")] bool IsSpecular,
	[property: JsonPropertyName("is_height")] bool IsHeight);

public sealed record TextureTaxonomyEntry(
	[property: JsonPropertyName("relative_path")] string RelativePath,
	[property: JsonPropertyName("era_tag")] string EraTag,
	[property: JsonPropertyName("design_kit")] string DesignKit,
	[property: JsonPropertyName("zone_name")] string ZoneName,
	[property: JsonPropertyName("file_name")] string FileName,
	[property: JsonPropertyName("classification")] TextureClassification Classification,
	[property: JsonPropertyName("width")] int Width,
	[property: JsonPropertyName("height")] int Height,
	[property: JsonPropertyName("has_alpha")] bool HasAlpha,
	[property: JsonPropertyName("is_specular_variant")] bool IsSpecularVariant);

public sealed class TextureFamily
{
	public TextureFamily(string familyId, string zonePrefix, string textureType, string coreName, List<string> textures, HashSet<string> layerRoles, HashSet<string> variants, HashSet<string> eras)
	{
		FamilyId = familyId;
		ZonePrefix = zonePrefix;
		TextureType = textureType;
		CoreName = coreName;
		Textures = textures;
		LayerRoles = layerRoles;
		Variants = variants;
		Eras = eras;
	}

	[JsonPropertyName("family_id")]
	public string FamilyId { get; }

	[JsonPropertyName("zone_prefix")]
	public string ZonePrefix { get; }

	[JsonPropertyName("texture_type")]
	public string TextureType { get; }

	[JsonPropertyName("core_name")]
	public string CoreName { get; }

	[JsonPropertyName("textures")]
	public List<string> Textures { get; }

	[JsonPropertyName("layer_roles")]
	public HashSet<string> LayerRoles { get; }

	[JsonPropertyName("variants")]
	public HashSet<string> Variants { get; }

	[JsonPropertyName("eras")]
	public HashSet<string> Eras { get; }
}

public sealed class LayerStack
{
	public LayerStack(string zone, string textureType, HashSet<string> layers, int totalTextures, List<string> families)
	{
		Zone = zone;
		TextureType = textureType;
		Layers = layers;
		TotalTextures = totalTextures;
		Families = families;
	}

	[JsonPropertyName("zone")]
	public string Zone { get; }

	[JsonPropertyName("texture_type")]
	public string TextureType { get; }

	[JsonPropertyName("layers")]
	public HashSet<string> Layers { get; }

	[JsonPropertyName("total_textures")]
	public int TotalTextures { get; set; }

	[JsonPropertyName("families")]
	public List<string> Families { get; }
}

public sealed class CrossEraTextureGroup
{
	public CrossEraTextureGroup(string textureName, HashSet<string> eras, HashSet<string> designKits, HashSet<string> zoneNames, int eraCount)
	{
		TextureName = textureName;
		Eras = eras;
		DesignKits = designKits;
		ZoneNames = zoneNames;
		EraCount = eraCount;
	}

	[JsonPropertyName("texture_name")]
	public string TextureName { get; }

	[JsonPropertyName("eras")]
	public HashSet<string> Eras { get; }

	[JsonPropertyName("design_kits")]
	public HashSet<string> DesignKits { get; }

	[JsonPropertyName("zone_names")]
	public HashSet<string> ZoneNames { get; }

	[JsonPropertyName("era_count")]
	public int EraCount { get; set; }
}

public sealed class TextureCategory
{
	public TextureCategory(string categoryId, string textureType, string layerRole, List<string> textures, HashSet<string> designKits, HashSet<string> eras)
	{
		CategoryId = categoryId;
		TextureType = textureType;
		LayerRole = layerRole;
		Textures = textures;
		DesignKits = designKits;
		Eras = eras;
	}

	[JsonPropertyName("category_id")]
	public string CategoryId { get; }

	[JsonPropertyName("texture_type")]
	public string TextureType { get; }

	[JsonPropertyName("layer_role")]
	public string LayerRole { get; }

	[JsonPropertyName("textures")]
	public List<string> Textures { get; }

	[JsonPropertyName("design_kits")]
	public HashSet<string> DesignKits { get; }

	[JsonPropertyName("eras")]
	public HashSet<string> Eras { get; }
}

public sealed record TextureTaxonomyReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_at_utc")] DateTimeOffset GeneratedAtUtc,
	[property: JsonPropertyName("total_textures")] int TotalTextures,
	[property: JsonPropertyName("texture_families")] List<TextureFamily> TextureFamilies,
	[property: JsonPropertyName("layer_stacks")] List<LayerStack> LayerStacks,
	[property: JsonPropertyName("cross_era_groups")] List<CrossEraTextureGroup> CrossEraGroups,
	[property: JsonPropertyName("categories")] List<TextureCategory> Categories,
	[property: JsonPropertyName("naming_conventions")] Dictionary<string, object> NamingConventions);
