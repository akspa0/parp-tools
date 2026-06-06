using System.Globalization;
using System.Buffers.Binary;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.Audio;
using WowViewer.Core.Blp;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Audio;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Lit;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Lit;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Wmo;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string area = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (area)
{
	case "archive":
		RunArchive(tail);
		break;
	case "audio":
		RunAudio(tail);
		break;
	case "blp":
		RunBlp(tail);
		break;
	case "m2":
		RunM2(tail);
		break;
	case "mdx":
		RunMdx(tail);
		break;
	case "map":
		RunMap(tail);
		break;
	case "lit":
		RunLit(tail);
		break;
	case "pm4":
		RunPm4(tail);
		break;
	case "wmo":
		RunWmo(tail);
		break;
	default:
		Console.Error.WriteLine($"Unknown inspect area '{area}'.");
		ShowUsage();
		Environment.ExitCode = 1;
		break;
}

static void RunArchive(string[] args)
{
	if (args.Length == 0)
	{
		ShowArchiveUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "build-listfile-cache":
			RunArchiveBuildListfileCache(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown archive command '{command}'.");
			ShowArchiveUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunAudio(string[] args)
{
	if (args.Length == 0)
	{
		ShowAudioUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "alpha-area":
			RunAudioAlphaArea(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown audio command '{command}'.");
			ShowAudioUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunBlp(string[] args)
{
	if (args.Length == 0)
	{
		ShowBlpUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunBlpInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown blp command '{command}'.");
			ShowBlpUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunBlpInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.blp> or --archive-root <dir> with --virtual-path <path/to/file.blp>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	BlpSummary summary;
	using (Stream stream = OpenInputStream())
		summary = BlpSummaryReader.Read(stream, sourceLabel);

	PrintBlpSummary(summary);
}

static void RunM2(string[] args)
{
	if (args.Length == 0)
	{
		ShowM2Usage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunM2Inspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown m2 command '{command}'.");
			ShowM2Usage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunM2Inspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	string? profileIndexText = GetOption(args, "--profile-index", "-p");
	string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
	string? timeMsText = GetOption(args, "--time-ms", "-t");
	string? goldenOutput = GetOption(args, "--golden-output", "-g");
	string? renderFrameOutput = GetOption(args, "--render-frame-output", "--render-frame-output");
	string? visualOutput = GetOption(args, "--visual-output", "--visual-output");
	string? staticVisualOutput = GetOption(args, "--static-visual-output", "--static-visual-output");
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	int profileIndex = 0;
	int? sequenceIndex = null;
	int timeMs = 0;
	if (!string.IsNullOrWhiteSpace(profileIndexText)
		&& (!int.TryParse(profileIndexText, out profileIndex) || profileIndex < 0 || profileIndex > 99))
	{
		Console.Error.WriteLine("Error: --profile-index must be an integer in the range 0..99.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(sequenceIndexText))
	{
		if (!int.TryParse(sequenceIndexText, out int parsedSequenceIndex) || parsedSequenceIndex < 0)
		{
			Console.Error.WriteLine("Error: --sequence-index must be a non-negative integer.");
			Environment.ExitCode = 1;
			return;
		}

		sequenceIndex = parsedSequenceIndex;
	}

	if (!string.IsNullOrWhiteSpace(timeMsText) && !int.TryParse(timeMsText, out timeMs))
	{
		Console.Error.WriteLine("Error: --time-ms must be an integer.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.m2|file.mdx|file.mdl> or --archive-root <dir> with --virtual-path <path/to/file.m2|file.mdx|file.mdl>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	byte[] ReadInputBytes()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
			return archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.ReadAllBytes(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return archivedBytes;
	}

	byte[]? TryReadExactSkinBytes(string companionPath)
		=> TryReadExactCompanionBytes(companionPath);

	byte[]? TryReadExactAnimBytes(string companionPath)
		=> TryReadExactCompanionBytes(companionPath);

	byte[]? TryReadExactCompanionBytes(string companionPath)
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot))
		{
			try
			{
				return ArchiveVirtualFileReader.ReadVirtualFile(companionPath, [archiveRoot], archiveBootstrapOptions);
			}
			catch (FileNotFoundException)
			{
				return null;
			}
		}

		return File.Exists(companionPath) ? File.ReadAllBytes(companionPath) : null;
	}

	byte[] modelBytes = ReadInputBytes();
	bool isChunkedMdx = modelBytes.Length >= sizeof(uint)
		&& BinaryPrimitives.ReadUInt32LittleEndian(modelBytes.AsSpan(0, sizeof(uint))) == MdxMagic.Mdlx;
	M2ChunkedReadResult? chunkedRead = null;
	M2ModelDocument model;
	M2GeometryDocument? geometry = null;
	string? geometryError = null;
	M2Era1121EraTag detectedEra;

	if (isChunkedMdx)
	{
		using MemoryStream stream = new(modelBytes, writable: false);
		chunkedRead = M2ChunkedModelReader.ReadDetailed(stream, sourceLabel, TryReadExactCompanionBytes);
		model = chunkedRead.Model;
		detectedEra = M2Era1121EraTag.Mdlx;

		try
		{
			using MemoryStream geometryStream = new(chunkedRead.Conversion.ModelBytes, writable: false);
			geometry = M2GeometryReader.Read(geometryStream, chunkedRead.Conversion.ModelPath);
		}
		catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or ArgumentException)
		{
			geometryError = ex.Message;
		}
	}
	else
	{
		using MemoryStream dispatchStream = new(modelBytes, writable: false);
		M2DispatchResult dispatch = M2ModelReaderDispatcher.ReadDetailed(dispatchStream, sourceLabel, TryReadExactCompanionBytes);
		model = dispatch.Document;
		detectedEra = dispatch.Era;

		try
		{
			using MemoryStream geometryStream = new(modelBytes, writable: false);
			geometry = M2GeometryReader.Read(geometryStream, sourceLabel);
		}
		catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or ArgumentException)
		{
			geometryError = ex.Message;
		}
	}

	M2SkinProfileRuntimeState state = M2SkinProfileRuntime.Choose(model, profileIndex);
	byte[]? skinBytes = chunkedRead is not null
		&& string.Equals(state.Selection.CompanionPath, chunkedRead.Conversion.SkinPath, StringComparison.OrdinalIgnoreCase)
		? chunkedRead.Conversion.SkinBytes
		: TryReadExactSkinBytes(state.Selection.CompanionPath);
	if (skinBytes is not null)
	{
		using MemoryStream skinStream = new(skinBytes, writable: false);
		string skinPath = chunkedRead is not null
			&& string.Equals(state.Selection.CompanionPath, chunkedRead.Conversion.SkinPath, StringComparison.OrdinalIgnoreCase)
			? chunkedRead.Conversion.SkinPath
			: state.Selection.CompanionPath;
		M2SkinDocument skin = M2SkinReader.Read(skinStream, skinPath);
		state = M2SkinProfileRuntime.Load(state, skin);
		state = M2SkinProfileRuntime.Initialize(state);
	}

	M2StaticRenderModel? renderModel = null;
	if (geometry is not null && state.ActiveSkinProfile is not null)
		renderModel = M2StaticRenderModelBuilder.Build(geometry, state);

	M2ExternalAnimationRuntimeState? externalAnimationState = null;
	string? externalAnimationError = null;
	M2AnimatedRenderState? animatedRenderState = null;
	string? animatedRenderError = null;
	M2BonePoseState? bonePoseState = null;
	string? bonePoseError = null;
	M2SkinnedRenderModel? skinnedRenderModel = null;
	M2RenderConsumerFrameState? renderConsumerState = null;
	M2EffectRuntimeState? effectRuntimeState = null;
	M2SceneSubmissionPlan? sceneSubmissionPlan = null;
	M2RenderFrame? renderFrame = null;
	M2SoftwareVisualSnapshot? visualSnapshot = null;
	M2RuntimeGoldenFrame? goldenFrame = null;
	if (sequenceIndex is not null)
	{
		try
		{
			externalAnimationState = M2ExternalAnimationRuntime.Choose(model, sequenceIndex.Value);
			if (externalAnimationState.UsesExternalFile && !string.IsNullOrWhiteSpace(externalAnimationState.CompanionPath))
			{
				byte[]? animBytes = TryReadExactAnimBytes(externalAnimationState.CompanionPath);
				if (animBytes is not null)
				{
					using MemoryStream animStream = new(animBytes, writable: false);
					M2ExternalAnimationDocument animation = M2AnimationReader.Read(animStream, externalAnimationState.CompanionPath);
					externalAnimationState = M2ExternalAnimationRuntime.Load(externalAnimationState, animation);
				}
			}
		}
		catch (Exception ex) when (ex is InvalidDataException or InvalidOperationException or ArgumentOutOfRangeException or ArgumentException)
		{
			externalAnimationError = ex.Message;
		}
	}

	if (sequenceIndex is not null && string.IsNullOrWhiteSpace(externalAnimationError))
	{
		bool animationPayloadAvailable = externalAnimationState is null
			|| !externalAnimationState.UsesExternalFile
			|| externalAnimationState.LoadedAnimation is not null;
		if (animationPayloadAvailable)
		{
			try
			{
				if (renderModel is not null)
				{
					M2RuntimeFrameResult frameResult = M2RuntimeFramePipeline.Build(model, renderModel, sequenceIndex.Value, timeMs, externalAnimationState);
					animatedRenderState = frameResult.AnimatedState;
					bonePoseState = frameResult.BonePoseState;
					skinnedRenderModel = frameResult.SkinnedRenderModel;
					renderConsumerState = frameResult.ConsumerState;
					effectRuntimeState = frameResult.EffectRuntimeState;
					sceneSubmissionPlan = frameResult.SubmissionPlan;
					renderFrame = frameResult.RenderFrame;
					visualSnapshot = frameResult.VisualSnapshot;
					goldenFrame = frameResult.GoldenFrame;
				}
			}
			catch (Exception ex) when (ex is InvalidDataException or InvalidOperationException or ArgumentOutOfRangeException or ArgumentException or NotSupportedException)
			{
				animatedRenderError = ex.Message;
				bonePoseError = ex.Message;
			}
		}
		else
		{
			animatedRenderError = "external animation payload not loaded";
			bonePoseError = animatedRenderError;
		}
	}

	if (goldenFrame is not null)
	{
		if (!string.IsNullOrWhiteSpace(goldenOutput))
			WriteJson(goldenOutput, goldenFrame);
		if (!string.IsNullOrWhiteSpace(renderFrameOutput) && renderFrame is not null)
			WriteJson(renderFrameOutput, renderFrame);
		if (!string.IsNullOrWhiteSpace(visualOutput) && visualSnapshot is not null)
			WriteBmp(visualOutput, visualSnapshot);
		if (!string.IsNullOrWhiteSpace(staticVisualOutput) && renderModel is not null && renderConsumerState is not null && sceneSubmissionPlan is not null)
		{
			M2RenderFrame staticRenderFrame = M2RenderFrameBuilder.Build(renderModel, skinnedRenderModel: null, renderConsumerState, sceneSubmissionPlan, timeMs);
			M2SoftwareVisualSnapshot staticVisualSnapshot = M2SoftwareVisualSnapshotBuilder.Build(staticRenderFrame);
			WriteBmp(staticVisualOutput, staticVisualSnapshot);
		}
	}

	Console.WriteLine($"ERA: {detectedEra.ToDisplayString()}");
	PrintM2Summary(model, state, geometry, geometryError, renderModel, externalAnimationState, externalAnimationError, animatedRenderState, animatedRenderError, bonePoseState, bonePoseError, skinnedRenderModel, renderConsumerState, effectRuntimeState, sceneSubmissionPlan, renderFrame, visualSnapshot, goldenFrame);
	if (chunkedRead is not null)
		PrintChunkedM2ConversionSummary(chunkedRead);
}

static void RunMdx(string[] args)
{
	if (args.Length == 0)
	{
		ShowMdxUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "export-json":
			RunMdxExportJson(tail);
			break;
		case "chunk-carriers":
			RunMdxChunkCarriers(tail);
			break;
		case "inspect":
			RunMdxInspect(tail);
			break;
		case "skin-diagnostics":
			RunMdxSkinDiagnostics(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown mdx command '{command}'.");
			ShowMdxUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunMdxInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	MdxSummary summary;
	using (Stream stream = OpenInputStream())
		summary = MdxSummaryReader.Read(stream, sourceLabel);

	PrintMdxSummary(summary);
}

static void PrintM2Summary(M2ModelDocument model, M2SkinProfileRuntimeState state, M2GeometryDocument? geometry, string? geometryError, M2StaticRenderModel? renderModel, M2ExternalAnimationRuntimeState? externalAnimationState, string? externalAnimationError, M2AnimatedRenderState? animatedRenderState, string? animatedRenderError, M2BonePoseState? bonePoseState, string? bonePoseError, M2SkinnedRenderModel? skinnedRenderModel, M2RenderConsumerFrameState? renderConsumerState, M2EffectRuntimeState? effectRuntimeState, M2SceneSubmissionPlan? sceneSubmissionPlan, M2RenderFrame? renderFrame, M2SoftwareVisualSnapshot? visualSnapshot, M2RuntimeGoldenFrame? goldenFrame)
{
	string modelName = string.IsNullOrWhiteSpace(model.ModelName) ? "n/a" : model.ModelName;
	Console.WriteLine($"M2: requestedPath={model.Identity.RequestedPath} canonicalPath={model.Identity.CanonicalModelPath} canonicalized={model.Identity.WasCanonicalized} signature={model.Signature} version=0x{model.Version:X} model={modelName} boundsMin={FormatVector(model.BoundsMin)} boundsMax={FormatVector(model.BoundsMax)} boundsRadius={model.BoundsRadius:F3} skinProfiles={model.ViewCount} bones={model.BoneCount} colors={model.ColorCount} transparencyDefs={model.TextureWeightCount} textureTransforms={model.TextureTransformCount} lights={model.LightCount} ribbons={model.RibbonCount} particles={model.ParticleCount}");
	int externalCandidateCount = model.Sequences.Count(static value => value.UsesExternalAnimationFile);
	int aliasCount = model.Sequences.Count(static value => value.IsAlias);
	Console.WriteLine($"ANIM: sequences={model.SequenceCount} externalCandidates={externalCandidateCount} aliases={aliasCount} globalLoops={model.GlobalLoopCount} sequenceLookup={model.SequenceLookupCount}");
	PrintM2EffectDefinitionSummary(model);
	if (geometry is null)
	{
		string errorText = string.IsNullOrWhiteSpace(geometryError) ? "n/a" : geometryError;
		Console.WriteLine($"GEOMETRY: available=false error={errorText}");
	}
	else
	{
		Console.WriteLine($"GEOMETRY: available=true vertices={geometry.Vertices.Count} textures={geometry.Textures.Count} renderFlags={geometry.RenderFlags.Count} textureLookup={geometry.TextureLookup.Count} textureUnitLookup={geometry.TextureUnitLookup.Count} transparencyLookup={geometry.TransparencyLookup.Count} textureAnimationLookup={geometry.TextureAnimationLookup.Count} boneLookup={geometry.BoneLookup.Count}");
	}

	if (state.LoadedSkin is null)
	{
		Console.WriteLine($"SKIN: stage={state.Stage} profileIndex={state.Selection.ProfileIndex} exactPath={state.Selection.CompanionPath} loaded=false");
		if (externalAnimationState is not null || !string.IsNullOrWhiteSpace(externalAnimationError))
			PrintExternalAnimationSummary(externalAnimationState, externalAnimationError);
		PrintM2RuntimeConsumers(animatedRenderState, animatedRenderError, bonePoseState, bonePoseError, skinnedRenderModel, renderConsumerState, effectRuntimeState, sceneSubmissionPlan, renderFrame, visualSnapshot, goldenFrame);
		return;
	}

	M2SkinDocument skin = state.LoadedSkin;
	string compatibilityMode = state.ActiveSkinProfile?.UsesCompatibilityFallback == true ? " compatibilityMode=true" : string.Empty;
	Console.WriteLine($"SKIN: stage={state.Stage} profileIndex={state.Selection.ProfileIndex} exactPath={state.Selection.CompanionPath} loaded=true vertexLookup={skin.VertexLookupCount} triangleIndices={skin.TriangleIndexCount} boneEntries={skin.BoneEntryCount} submeshes={skin.SubmeshCount} batches={skin.BatchCount} globalVertexOffset={skin.GlobalVertexOffset} shadowBatches={skin.ShadowBatchCount}{compatibilityMode}");
	if (state.ActiveSkinProfile is not null)
	{
		M2ActiveSkinProfile active = state.ActiveSkinProfile;
		Console.WriteLine($"ACTIVE.SKIN: sections={active.ActiveSectionCount} sectionsWithBatches={active.SectionsWithBatchesCount} unmatchedBatches={active.UnmatchedBatchCount}");
		for (int index = 0; index < active.ActiveSections.Count; index++)
		{
			M2ActiveSkinSection section = active.ActiveSections[index];
			Console.WriteLine($"ACTIVE.SECTION[{index}]: skinSectionId={section.SkinSectionId} level={section.Level} vertexStart={section.VertexStart} vertexCount={section.VertexCount} indexStart={section.IndexStart} indexCount={section.IndexCount} boneComboIndex={section.BoneComboIndex} boneCount={section.BoneCount} boneInfluences={section.BoneInfluences} centerBoneIndex={section.CenterBoneIndex} batches={section.ActiveBatchCount}");
			for (int batchIndex = 0; batchIndex < section.Batches.Count; batchIndex++)
			{
				M2ActiveSkinBatch batch = section.Batches[batchIndex];
				Console.WriteLine($"ACTIVE.SECTION[{index}].BATCH[{batchIndex}]: batchIndex={batch.BatchIndex} flags=0x{batch.Flags:X2} priorityPlane={batch.PriorityPlane} shaderId={batch.ShaderId} geosetIndex={batch.GeosetIndex} colorIndex={batch.ColorIndex} renderFlagsIndex={batch.RenderFlagsIndex} materialIndex={batch.MaterialIndex} materialLayer={batch.MaterialLayer} textureCount={batch.TextureCount} textureComboIndex={batch.TextureComboIndex} textureCoordComboIndex={batch.TextureCoordComboIndex} transparencyComboIndex={batch.TransparencyComboIndex} textureAnimationLookupIndex={batch.TextureAnimationLookupIndex}");
			}
		}
	}
	if (renderModel is not null)
	{
		Console.WriteLine($"RENDER: compatibilitySections={renderModel.Sections.Count} structuredSections={renderModel.StructuredSections.Count} compatibilityMode={renderModel.UsesCompatibilityFallback}");
		for (int sectionIndex = 0; sectionIndex < renderModel.StructuredSections.Count; sectionIndex++)
		{
			M2StructuredRenderSection section = renderModel.StructuredSections[sectionIndex];
			Console.WriteLine($"RENDER.SECTION[{sectionIndex}]: skinSectionId={section.SkinSectionId} vertices={section.Vertices.Count} indices={section.Indices.Count} boneComboIndex={section.BoneComboIndex} boneCount={section.BoneCount} boneInfluences={section.BoneInfluences} centerBoneIndex={section.CenterBoneIndex} passes={section.PassCount}");
			for (int passIndex = 0; passIndex < section.Passes.Count; passIndex++)
			{
				M2StructuredRenderPass pass = section.Passes[passIndex];
				M2StaticRenderMaterial material = pass.Material;
				Console.WriteLine($"RENDER.SECTION[{sectionIndex}].PASS[{passIndex}]: batchIndex={material.BatchIndex} flags=0x{material.BatchFlags:X2} priorityPlane={material.PriorityPlane} shaderId={material.ShaderId} geosetIndex={material.GeosetIndex} colorIndex={material.ColorIndex} renderFlagsIndex={material.RenderFlagsIndex} materialLayer={material.MaterialLayer} textureCount={material.TextureCount} blend={material.BlendMode} renderFlags=0x{material.RenderFlags:X4} transparent={material.IsTransparent} unshaded={material.IsUnshaded} twoSided={material.IsTwoSided} effect={material.EffectRecipe.RecipeKey} projected={material.EffectRecipe.IsProjected} animated={material.EffectRecipe.IsAnimated} colorAnim={material.EffectRecipe.UsesColorAnimation} transparencyAnim={material.EffectRecipe.UsesTransparencyAnimation} textureTransformAnim={material.EffectRecipe.UsesTextureTransformAnimation} suppressCombinedAlpha={material.EffectRecipe.SuppressCombinedTransparency}");
				for (int bindingIndex = 0; bindingIndex < material.TextureBindings.Count; bindingIndex++)
				{
					M2StaticRenderTextureBinding binding = material.TextureBindings[bindingIndex];
					Console.WriteLine($"RENDER.SECTION[{sectionIndex}].PASS[{passIndex}].TEXTURE[{bindingIndex}]: lookupIndex={FormatNullableInt(binding.TextureLookupIndex)} textureId={FormatNullableUShort(binding.TextureId)} path={binding.TexturePath ?? "n/a"} replaceableId={binding.ReplaceableId} flags=0x{binding.TextureFlags:X8} coordLookupIndex={FormatNullableInt(binding.TextureCoordLookupIndex)} coordLookupValue={FormatNullableUShort(binding.TextureCoordLookupValue)} transparencyLookupIndex={FormatNullableInt(binding.TransparencyLookupIndex)} transparencyLookupValue={FormatNullableUShort(binding.TransparencyLookupValue)} textureAnimationLookupIndex={FormatNullableInt(binding.TextureAnimationLookupIndex)} textureAnimationLookupValue={FormatNullableUShort(binding.TextureAnimationLookupValue)}");
				}
			}
		}
	}
	for (int index = 0; index < skin.Submeshes.Count; index++)
	{
		M2SkinSubmesh submesh = skin.Submeshes[index];
		Console.WriteLine($"SKIN.SUBMESH[{index}]: sectionId={submesh.SkinSectionId} level={submesh.Level} vertexStart={submesh.VertexStart} vertexCount={submesh.VertexCount} indexStart={submesh.IndexStart} indexCount={submesh.IndexCount} boneComboIndex={submesh.BoneComboIndex} boneCount={submesh.BoneCount} boneInfluences={submesh.BoneInfluences} centerBoneIndex={submesh.CenterBoneIndex}");
	}
	for (int index = 0; index < skin.Batches.Count; index++)
	{
		M2SkinBatch batch = skin.Batches[index];
		Console.WriteLine($"SKIN.BATCH[{index}]: flags=0x{batch.Flags:X2} priorityPlane={batch.PriorityPlane} shaderId={batch.ShaderId} skinSectionIndex={batch.SkinSectionIndex} geosetIndex={batch.GeosetIndex} colorIndex={batch.ColorIndex} renderFlagsIndex={batch.RenderFlagsIndex} materialIndex={batch.MaterialIndex} materialLayer={batch.MaterialLayer} textureCount={batch.TextureCount} textureComboIndex={batch.TextureComboIndex} textureCoordComboIndex={batch.TextureCoordComboIndex} transparencyComboIndex={batch.TransparencyComboIndex} textureAnimationLookupIndex={batch.TextureAnimationLookupIndex}");
	}

	if (externalAnimationState is not null || !string.IsNullOrWhiteSpace(externalAnimationError))
		PrintExternalAnimationSummary(externalAnimationState, externalAnimationError);

	PrintM2RuntimeConsumers(animatedRenderState, animatedRenderError, bonePoseState, bonePoseError, skinnedRenderModel, renderConsumerState, effectRuntimeState, sceneSubmissionPlan, renderFrame, visualSnapshot, goldenFrame);
}

static void PrintChunkedM2ConversionSummary(M2ChunkedReadResult result)
{
	Console.WriteLine($"CHUNKED.MDX: sourceSignature={result.Summary.Signature} version={result.Summary.Version} chunks={result.Chunks.Count} materials={result.Summary.MaterialCount} sequences={result.Summary.SequenceCount} geosets={result.Geometry.GeosetCount} vertices={result.VertexCount} triangles={result.TriangleCount} convertedModelPath={result.Conversion.ModelPath} convertedSkinPath={result.Conversion.SkinPath}");
	for (int index = 0; index < result.Chunks.Count; index++)
	{
		M2ChunkedChunkHeader chunk = result.Chunks[index];
		string truncation = chunk.IsTruncated ? " truncated=true" : string.Empty;
		Console.WriteLine($"CHUNK[{index:D2}]: fourCC={chunk.FourCC} size=0x{chunk.Size:X} offset=0x{chunk.Offset:X}{truncation}");
	}
}

static void PrintM2RuntimeConsumers(M2AnimatedRenderState? animatedRenderState, string? animatedRenderError, M2BonePoseState? bonePoseState, string? bonePoseError, M2SkinnedRenderModel? skinnedRenderModel, M2RenderConsumerFrameState? renderConsumerState, M2EffectRuntimeState? effectRuntimeState, M2SceneSubmissionPlan? sceneSubmissionPlan, M2RenderFrame? renderFrame, M2SoftwareVisualSnapshot? visualSnapshot, M2RuntimeGoldenFrame? goldenFrame)
{
	if (animatedRenderState is not null || !string.IsNullOrWhiteSpace(animatedRenderError))
		PrintAnimatedRenderSummary(animatedRenderState, animatedRenderError);
	if (bonePoseState is not null || !string.IsNullOrWhiteSpace(bonePoseError))
		PrintBonePoseSummary(bonePoseState, bonePoseError, skinnedRenderModel);
	if (renderConsumerState is not null)
		PrintRenderConsumerSummary(renderConsumerState);
	if (effectRuntimeState is not null)
		PrintM2EffectRuntimeSummary(effectRuntimeState);
	if (sceneSubmissionPlan is not null)
		PrintSceneSubmissionSummary(sceneSubmissionPlan);
	if (renderFrame is not null)
		PrintRenderFrameSummary(renderFrame);
	if (visualSnapshot is not null)
		PrintVisualSnapshotSummary(visualSnapshot);
	if (goldenFrame is not null)
		PrintGoldenFrameSummary(goldenFrame);
}

static void PrintM2EffectDefinitionSummary(M2ModelDocument model)
{
	Console.WriteLine($"M2.EFFECT.DEFS: ribbons={model.RibbonCount} particles={model.ParticleCount}");
	int ribbonLimit = Math.Min(model.Ribbons.Count, 8);
	for (int index = 0; index < ribbonLimit; index++)
	{
		M2RibbonDefinition ribbon = model.Ribbons[index];
		Console.WriteLine($"M2.RIBBON[{index}]: bone={ribbon.BoneIndex} position={FormatVector(ribbon.Position)} textures={FormatUInt16List(ribbon.TextureIndices)} materials={FormatUInt16List(ribbon.MaterialIndices)} edgesPerSecond={ribbon.EdgesPerSecond:F3} edgeLifetime={ribbon.EdgeLifetime:F3} gravity={ribbon.Gravity:F3} rows={ribbon.TextureRows} cols={ribbon.TextureColumns} priorityPlane={ribbon.PriorityPlane} colorIndex={ribbon.RibbonColorIndex} textureTransform={ribbon.TextureTransformLookupIndex}");
	}

	if (model.Ribbons.Count > ribbonLimit)
		Console.WriteLine($"M2.RIBBON: omitted={model.Ribbons.Count - ribbonLimit}");

	int particleLimit = Math.Min(model.Particles.Count, 8);
	for (int index = 0; index < particleLimit; index++)
	{
		M2ParticleDefinition particle = model.Particles[index];
		Console.WriteLine($"M2.PARTICLE[{index}]: flags=0x{particle.Flags:X8} bone={particle.BoneIndex} texture={particle.TextureIndex} blend={particle.BlendingType} emitter={particle.EmitterType} particleType={particle.ParticleType} headOrTail={particle.HeadOrTail} rows={particle.TextureRows} cols={particle.TextureColumns} colorIndex={particle.ParticleColorIndex} geometryModel={FormatOptionalText(particle.GeometryModelPath)} recursionModel={FormatOptionalText(particle.RecursionModelPath)}");
	}

	if (model.Particles.Count > particleLimit)
		Console.WriteLine($"M2.PARTICLE: omitted={model.Particles.Count - particleLimit}");
}

static void PrintM2EffectRuntimeSummary(M2EffectRuntimeState state)
{
	Console.WriteLine($"M2.EFFECT.RUNTIME: particles={state.Particles.Count} visibleParticles={state.VisibleParticleEmitterCount} ribbons={state.Ribbons.Count} visibleRibbons={state.VisibleRibbonEmitterCount}");
	int particleLimit = Math.Min(state.Particles.Count, 8);
	for (int index = 0; index < particleLimit; index++)
	{
		M2ParticleRuntimeState particle = state.Particles[index];
		Console.WriteLine($"M2.EFFECT.RUNTIME.PARTICLE[{index}]: enabled={particle.Enabled} texture={particle.TextureIndex} blend={particle.BlendingType} effect={particle.EffectKey} batching={particle.AllowsBatching} estimatedParticles={particle.EstimatedParticleCount} vertices={particle.EstimatedVertexCount} indices={particle.EstimatedIndexCount} emissionRate={particle.EmissionRate:F3} lifespan={particle.Lifespan:F3} position={FormatVector(particle.Position)}");
	}

	if (state.Particles.Count > particleLimit)
		Console.WriteLine($"M2.EFFECT.RUNTIME.PARTICLE: omitted={state.Particles.Count - particleLimit}");

	int ribbonLimit = Math.Min(state.Ribbons.Count, 8);
	for (int index = 0; index < ribbonLimit; index++)
	{
		M2RibbonRuntimeState ribbon = state.Ribbons[index];
		Console.WriteLine($"M2.EFFECT.RUNTIME.RIBBON[{index}]: visible={ribbon.Visible} texture={ribbon.TextureSortKey} material={ribbon.MaterialSortKey} effect={ribbon.EffectKey} estimatedEdges={ribbon.EstimatedEdgeCount} vertices={ribbon.EstimatedVertexCount} indices={ribbon.EstimatedIndexCount} alpha={ribbon.Alpha:F3} color={FormatVector(ribbon.Color)} position={FormatVector(ribbon.Position)}");
	}

	if (state.Ribbons.Count > ribbonLimit)
		Console.WriteLine($"M2.EFFECT.RUNTIME.RIBBON: omitted={state.Ribbons.Count - ribbonLimit}");
}

static void PrintExternalAnimationSummary(M2ExternalAnimationRuntimeState? state, string? error)
{
	if (!string.IsNullOrWhiteSpace(error))
	{
		Console.WriteLine($"ANIM.SELECTION: error={error}");
		return;
	}

	if (state is null)
		return;

	string aliasChain = string.Join(",", state.AliasChain);
	string readyIndices = state.ReadySequenceIndices.Count == 0
		? "n/a"
		: string.Join(",", state.ReadySequenceIndices);
	Console.WriteLine($"ANIM.SELECTION: requestedSequenceIndex={state.RequestedSequenceIndex} resolvedSequenceIndex={state.ResolvedSequenceIndex} aliasChain={aliasChain} stage={state.Stage} usesExternalFile={state.UsesExternalFile} exactPath={state.CompanionPath ?? "n/a"}");
	Console.WriteLine($"ANIM.SEQUENCE: animationId={state.ResolvedSequence.AnimationId} variationIndex={state.ResolvedSequence.VariationIndex} duration={state.ResolvedSequence.Duration} flags=0x{state.ResolvedSequence.Flags:X8} frequency={state.ResolvedSequence.Frequency} replay=[{state.ResolvedSequence.ReplayMinimum},{state.ResolvedSequence.ReplayMaximum}] blend=[{state.ResolvedSequence.BlendTimeIn},{state.ResolvedSequence.BlendTimeOut}] inline={state.ResolvedSequence.UsesInlineAnimationData} alias={state.ResolvedSequence.IsAlias} externalCandidate={state.ResolvedSequence.UsesExternalAnimationFile}");
	if (state.LoadedAnimation is not null)
	{
		Console.WriteLine($"ANIM.FILE: loaded=true payloadBytes={state.LoadedAnimation.PayloadSizeBytes} chunked={state.LoadedAnimation.IsChunkedContainer} container={state.LoadedAnimation.ContainerSignature ?? "raw"} readySequenceIndices={readyIndices}");
	}
	else
	{
		Console.WriteLine($"ANIM.FILE: loaded=false readySequenceIndices={readyIndices}");
	}
}

static void PrintAnimatedRenderSummary(M2AnimatedRenderState? state, string? error)
{
	if (!string.IsNullOrWhiteSpace(error))
	{
		Console.WriteLine($"ANIM.RUNTIME: error={error}");
		return;
	}

	if (state is null)
		return;

	Console.WriteLine($"ANIM.RUNTIME: requestedSequenceIndex={state.RequestedSequenceIndex} resolvedSequenceIndex={state.ResolvedSequenceIndex} timeMs={state.TimeMs} usesExternalPayload={state.UsesExternalPayload} passStates={state.Passes.Count} lightStates={state.Lights.Count}");
	for (int passIndex = 0; passIndex < state.Passes.Count; passIndex++)
	{
		M2AnimatedRenderPassState pass = state.Passes[passIndex];
		Console.WriteLine($"ANIM.RUNTIME.PASS[{passIndex}]: sectionIndex={pass.SectionIndex} passIndex={pass.PassIndex} batchIndex={pass.BatchIndex} color={FormatVector(pass.Color)} colorAlpha={pass.ColorAlpha:F3} combinedAlpha={pass.CombinedAlpha:F3} bindings={pass.TextureBindings.Count}");
		for (int bindingIndex = 0; bindingIndex < pass.TextureBindings.Count; bindingIndex++)
		{
			M2AnimatedTextureBindingState binding = pass.TextureBindings[bindingIndex];
			Console.WriteLine($"ANIM.RUNTIME.PASS[{passIndex}].BINDING[{bindingIndex}]: stageIndex={binding.StageIndex} transparencyAlpha={binding.TransparencyAlpha:F3} translation={FormatVector(binding.Translation)} rotation={FormatQuaternion(binding.Rotation)} scaling={FormatVector(binding.Scaling)}");
		}
	}

	for (int lightIndex = 0; lightIndex < state.Lights.Count; lightIndex++)
	{
		M2AnimatedLightState light = state.Lights[lightIndex];
		Console.WriteLine($"ANIM.RUNTIME.LIGHT[{lightIndex}]: lightIndex={light.LightIndex} type={light.Type} boneIndex={light.BoneIndex} position={FormatVector(light.Position)} ambientColor={FormatVector(light.AmbientColor)} ambientIntensity={light.AmbientIntensity:F3} diffuseColor={FormatVector(light.DiffuseColor)} diffuseIntensity={light.DiffuseIntensity:F3} attenuationStart={light.AttenuationStart:F3} attenuationEnd={light.AttenuationEnd:F3} visible={light.Visible}");
	}
}

static void PrintBonePoseSummary(M2BonePoseState? state, string? error, M2SkinnedRenderModel? skinnedRenderModel)
{
	if (!string.IsNullOrWhiteSpace(error))
	{
		Console.WriteLine($"ANIM.POSE: error={error}");
		return;
	}

	if (state is null)
		return;

	Console.WriteLine($"ANIM.POSE: requestedSequenceIndex={state.RequestedSequenceIndex} resolvedSequenceIndex={state.ResolvedSequenceIndex} timeMs={state.TimeMs} usesExternalPayload={state.UsesExternalPayload} bones={state.BoneCount} skinnedVertices={skinnedRenderModel?.VertexCount ?? 0}");
	int sampleCount = Math.Min(state.Bones.Count, 8);
	for (int boneIndex = 0; boneIndex < sampleCount; boneIndex++)
	{
		M2BonePose bone = state.Bones[boneIndex];
		Console.WriteLine($"ANIM.POSE.BONE[{boneIndex}]: parent={bone.ParentBone} pivot={FormatVector(bone.Pivot)} translation={FormatVector(bone.Translation)} rotation={FormatQuaternion(bone.Rotation)} scaling={FormatVector(bone.Scaling)}");
	}
}

static void PrintRenderConsumerSummary(M2RenderConsumerFrameState state)
{
	Console.WriteLine($"RENDER.CONSUMER: passStates={state.Passes.Count} visiblePasses={state.VisiblePassCount} modelAmbient={FormatVector(state.ModelAmbient)} modelDiffuse={FormatVector(state.ModelDiffuse)}");
	for (int passIndex = 0; passIndex < state.Passes.Count; passIndex++)
	{
		M2RenderConsumerPassState pass = state.Passes[passIndex];
		M2ResolvedEffect effect = pass.ResolvedEffect;
		Console.WriteLine($"RENDER.CONSUMER.PASS[{passIndex}]: sectionIndex={pass.AnimatedPass.SectionIndex} passIndex={pass.AnimatedPass.PassIndex} batchIndex={pass.AnimatedPass.BatchIndex} effect={pass.EffectKey} effectObject={effect.EffectObjectKey} nativeFamily={effect.NativeEffectFamilyKey} diffuse={FormatVector(pass.DiffuseColor)} emissive={FormatVector(pass.EmissiveColor)} alpha={pass.Alpha:F3} receivesLighting={pass.ReceivesLighting} depthWrite={effect.DepthWrite} alphaTest={effect.AlphaTest} visible={pass.Visible} textures={pass.Textures.Count}");
	}
}

static void PrintSceneSubmissionSummary(M2SceneSubmissionPlan plan)
{
	Console.WriteLine($"SCENE.SUBMISSION: batches={plan.Batches.Count} directEntries={plan.DirectEntryCount} batchedEntries={plan.BatchedEntryCount} options=0x{(int)plan.Options:X}");
	for (int batchIndex = 0; batchIndex < plan.Batches.Count; batchIndex++)
	{
		M2SceneSubmissionBatch batch = plan.Batches[batchIndex];
		Console.WriteLine($"SCENE.SUBMISSION.BATCH[{batchIndex}]: family={batch.Family} handler={batch.HandlerName} direct={batch.IsDirect} stateScope={batch.UsesDedicatedStateScope} entries={batch.Entries.Count} model={batch.ModelKey} effect={batch.EffectKey} textureSortKey={batch.TextureSortKey} stateBucket={batch.StateBucket} vertices={batch.VertexCount} indices={batch.IndexCount}");
	}
}

static void PrintRenderFrameSummary(M2RenderFrame frame)
{
	Console.WriteLine($"RENDER.FRAME: commands={frame.CommandCount} backendVertices={frame.BackendVertexCount} backendIndices={frame.BackendIndexCount} submittedVertices={frame.SubmittedVertexCount} submittedIndices={frame.SubmittedIndexCount} hash={frame.FrameHash}");
	int commandLimit = Math.Min(frame.DrawCommands.Count, 8);
	for (int index = 0; index < commandLimit; index++)
	{
		M2RenderDrawCommand command = frame.DrawCommands[index];
		Console.WriteLine($"RENDER.FRAME.COMMAND[{index}]: family={command.Family} handler={command.HandlerName} direct={command.IsDirect} entries={command.EntryKeys.Count} effect={command.EffectKey} effectObject={FormatOptionalText(command.EffectObjectKey)} submittedVertices={command.SubmittedVertexCount} submittedIndices={command.SubmittedIndexCount} backendVertices={command.Vertices.Count} backendIndices={command.Indices.Count} textures={command.Textures.Count}");
	}

	if (frame.DrawCommands.Count > commandLimit)
		Console.WriteLine($"RENDER.FRAME.COMMAND: omitted={frame.DrawCommands.Count - commandLimit}");
}

static void PrintVisualSnapshotSummary(M2SoftwareVisualSnapshot snapshot)
{
	Console.WriteLine($"RENDER.VISUAL: size={snapshot.Width}x{snapshot.Height} litPixels={snapshot.LitPixelCount} hash={snapshot.VisualHash}");
}

static void PrintGoldenFrameSummary(M2RuntimeGoldenFrame frame)
{
	Console.WriteLine($"M2.GOLDEN: hash={frame.RuntimeHash} effects={frame.Effects.Count} batches={frame.Batches.Count} visiblePasses={frame.VisiblePassCount} skinnedVertices={frame.SkinnedVertexCount}");
}

static void WriteJson<T>(string output, T payload)
{
	string outputPath = Path.GetFullPath(output);
	string? directory = Path.GetDirectoryName(outputPath);
	if (!string.IsNullOrWhiteSpace(directory))
		Directory.CreateDirectory(directory);

	File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
	Console.WriteLine($"Wrote {outputPath}");
}

static void WriteBmp(string output, M2SoftwareVisualSnapshot snapshot)
{
	string outputPath = Path.GetFullPath(output);
	string? directory = Path.GetDirectoryName(outputPath);
	if (!string.IsNullOrWhiteSpace(directory))
		Directory.CreateDirectory(directory);

	using FileStream stream = File.Create(outputPath);
	M2SoftwareVisualSnapshotBuilder.WriteBmp(stream, snapshot);
	Console.WriteLine($"Wrote {outputPath}");
}

static string FormatNullableInt(int? value)
{
	return value?.ToString() ?? "n/a";
}

static string FormatNullableUShort(ushort? value)
{
	return value?.ToString() ?? "n/a";
}

static void RunMdxExportJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	string? output = GetOption(args, "--output", "-o");
	bool includeGeometry = HasOption(args, "--include-geometry");
	bool includeCollision = HasOption(args, "--include-collision");
	bool includeHitTest = HasOption(args, "--include-hit-test");
	bool includeTextureAnimations = HasOption(args, "--include-texture-animations");
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	MdxSummary summary;
	using (Stream stream = OpenInputStream())
		summary = MdxSummaryReader.Read(stream, sourceLabel);

	MdxGeometryFile? geometry = null;
	if (includeGeometry)
	{
		using Stream stream = OpenInputStream();
		geometry = MdxGeometryReader.Read(stream, sourceLabel);
	}

	MdxCollisionFile? collision = null;
	if (includeCollision)
	{
		using Stream stream = OpenInputStream();
		collision = MdxCollisionReader.Read(stream, sourceLabel);
	}

	MdxHitTestFile? hitTest = null;
	if (includeHitTest)
	{
		using Stream stream = OpenInputStream();
		hitTest = MdxHitTestReader.Read(stream, sourceLabel);
	}

	MdxTextureAnimationFile? textureAnimations = null;
	if (includeTextureAnimations)
	{
		using Stream stream = OpenInputStream();
		textureAnimations = MdxTextureAnimationReader.Read(stream, sourceLabel);
	}

	Dictionary<string, object?> payload = new(StringComparer.Ordinal)
	{
		["summary"] = summary,
	};

	if (geometry is not null)
		payload["geometry"] = geometry;

	if (collision is not null)
		payload["collision"] = collision;

	if (hitTest is not null)
		payload["hitTest"] = hitTest;

	if (textureAnimations is not null)
		payload["textureAnimations"] = textureAnimations;

	string json = JsonSerializer.Serialize(payload, new JsonSerializerOptions
	{
		WriteIndented = true,
		IncludeFields = true,
	});
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
}

static void RunMdxSkinDiagnostics(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	// Read summary for bone data
	MdxSummary summary;
	using (Stream stream = OpenInputStream())
		summary = MdxSummaryReader.Read(stream, sourceLabel);

	// Read full geometry for skinning data
	MdxGeometryFile geometry;
	using (Stream stream = OpenInputStream())
		geometry = MdxGeometryReader.Read(stream, sourceLabel);

	// Read bones
	MdxBoneFile bones;
	using (Stream stream = OpenInputStream())
		bones = MdxBoneReader.Read(stream, sourceLabel);

	PrintMdxSkinningDiagnostics(summary, geometry, bones);
}

static void PrintMdxSkinningDiagnostics(MdxSummary summary, MdxGeometryFile geometry, MdxBoneFile bones)
{
	Console.WriteLine($"SKIN.DIAG: model={summary.ModelName ?? "n/a"} bones={summary.BoneCount} geosets={summary.GeosetCount}");

	// Build ObjectId -> bone index mapping
	Dictionary<uint, int> objectIdToBoneIndex = new();
	for (int i = 0; i < bones.Bones.Count; i++)
		objectIdToBoneIndex[(uint)bones.Bones[i].ObjectId] = i;

	// Analyze each geoset
	for (int g = 0; g < geometry.Geosets.Count; g++)
	{
		MdxGeosetGeometry geo = geometry.Geosets[g];
		Console.WriteLine($"SKIN.GEOSET[{g}]: vertices={geo.VertexCount} vertexGroups={geo.VertexGroupCount} matrixGroups={geo.MatrixGroupCount} matrixIndices={geo.MatrixIndexCount} boneIndices={geo.BoneIndexCount} boneWeights={geo.BoneWeightCount}");

		// Analyze BIDX/BWGT data (per-vertex bone indices/weights)
		if (geo.BoneIndexCount > 0 && geo.BoneWeightCount > 0)
		{
			Console.WriteLine($"  BIDX/BWGT: count={geo.BoneIndexCount}/{geo.BoneWeightCount}");
			// BIDX/BWGT should have 4 entries per vertex if it's per-vertex data
			int expectedBidxPerVertex = geo.VertexCount * 4;
			Console.WriteLine($"  BIDX.RATIO: expected={expectedBidxPerVertex} actual={geo.BoneIndexCount} perVertex={geo.BoneIndexCount / (float)Math.Max(1, geo.VertexCount):F2}");
		}

		// Analyze GNDX/MTGC/MATS data (matrix groups)
		if (geo.MatrixGroupCount > 0 && geo.MatrixIndexCount > 0)
		{
			Console.WriteLine($"  GNDX/MTGC/MATS: groups={geo.MatrixGroupCount} totalIndices={geo.MatrixIndexCount}");

			// Calculate group offsets
			int[] groupOffsets = new int[geo.MatrixGroupCount];
			int offset = 0;
			for (int gi = 0; gi < geo.MatrixGroupCount; gi++)
			{
				groupOffsets[gi] = offset;
				offset += (int)geo.MatrixGroups[gi];
			}

			// Analyze first few vertices
			int sampleCount = Math.Min(10, geo.VertexCount);
			Console.WriteLine($"  SAMPLE.VERTEX_BONES (first {sampleCount} vertices):");
			for (int v = 0; v < sampleCount; v++)
			{
				byte groupIndex = v < geo.VertexGroups.Count ? geo.VertexGroups[v] : (byte)0;
				if (groupIndex >= geo.MatrixGroupCount)
				{
					Console.WriteLine($"    v[{v}]: group={groupIndex} INVALID_GROUP (>= {geo.MatrixGroupCount})");
					continue;
				}

				uint boneCount = geo.MatrixGroups[groupIndex];
				int matrixOffset = groupOffsets[groupIndex];

				// Get bone indices from MATS
				uint[] boneIndices = new uint[Math.Min(boneCount, 4)];
				float[] boneWeights = new float[Math.Min(boneCount, 4)];
				float weight = boneCount > 0 ? 1.0f / boneCount : 1.0f;

				for (int b = 0; b < boneIndices.Length; b++)
				{
					if (matrixOffset + b < geo.MatrixIndexCount)
					{
						uint matsValue = geo.MatrixIndices[matrixOffset + b];
						boneIndices[b] = matsValue;

						// Try to remap MATS value to bone index
						if (objectIdToBoneIndex.TryGetValue(matsValue, out int remappedIdx))
							boneIndices[b] = (uint)remappedIdx;
						else if (matsValue < (uint)bones.Bones.Count)
							; // Already a valid index
						else
							boneIndices[b] = uint.MaxValue; // Invalid

						boneWeights[b] = weight;
					}
				}

				Console.WriteLine($"    v[{v}]: group={groupIndex} bones=[{string.Join(",", boneIndices)}] weights=[{string.Join(",", boneWeights.Select(w => w.ToString("F2")))}]");
			}
		}

		// Check for mismatch between data sources
		bool hasBidx = geo.BoneIndexCount > 0;
		bool hasMats = geo.MatrixIndexCount > 0;
		if (hasBidx && hasMats)
		{
			Console.WriteLine($"  WARNING: Both BIDX ({geo.BoneIndexCount}) and MATS ({geo.MatrixIndexCount}) present - potential data conflict");
		}
	}

	// Summary of bone usage
	Console.WriteLine($"SKIN.BONES: total={bones.Bones.Count}");
	Dictionary<int, int> boneRefCount = new();
	for (int i = 0; i < bones.Bones.Count; i++)
		boneRefCount[i] = 0;

	// Count references from MATS
	for (int g = 0; g < geometry.Geosets.Count; g++)
	{
		MdxGeosetGeometry geo = geometry.Geosets[g];
		int[] groupOffsets = new int[geo.MatrixGroupCount];
		int offset = 0;
		for (int gi = 0; gi < geo.MatrixGroupCount; gi++)
		{
			groupOffsets[gi] = offset;
			offset += (int)geo.MatrixGroups[gi];
		}

		for (int v = 0; v < geo.VertexCount && v < geo.VertexGroups.Count; v++)
		{
			byte groupIndex = geo.VertexGroups[v];
			if (groupIndex >= geo.MatrixGroupCount)
				continue;

			uint boneCount = geo.MatrixGroups[groupIndex];
			int matrixOffset = groupOffsets[groupIndex];

			for (int b = 0; b < Math.Min(boneCount, 4); b++)
			{
				if (matrixOffset + b >= geo.MatrixIndexCount)
					break;

				uint matsValue = geo.MatrixIndices[matrixOffset + b];
				if (objectIdToBoneIndex.TryGetValue(matsValue, out int boneIdx) && boneIdx < bones.Bones.Count)
					boneRefCount[boneIdx]++;
				else if (matsValue < (uint)bones.Bones.Count)
					boneRefCount[(int)matsValue]++;
			}
		}
	}

	Console.WriteLine($"SKIN.BONE_REFERENCES:");
	int unreferencedBones = 0;
	for (int i = 0; i < bones.Bones.Count; i++)
	{
		if (boneRefCount[i] == 0)
			unreferencedBones++;
	}
	Console.WriteLine($"  unreferenced_bones={unreferencedBones}/{bones.Bones.Count}");
	if (unreferencedBones > 0)
	{
		Console.WriteLine($"  WARNING: {unreferencedBones} bones are not referenced by any vertex via MATS");
	}
}

static void RunMdxChunkCarriers(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	string? pathFilter = GetOption(args, "--path-filter", "-p");
	string? chunkText = GetOption(args, "--chunks", "-c") ?? GetOption(args, "--chunk", "-c");
	string? limitText = GetOption(args, "--limit", "-n");

	if (string.IsNullOrWhiteSpace(chunkText))
	{
		Console.Error.WriteLine("Error: provide --chunks <FOURCC[,FOURCC...]>.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input) && string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: provide --input <file|directory> or --archive-root <game|data dir>.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(input) && !string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: choose either --input <file|directory> or --archive-root <game|data dir>, not both.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(limitText) && (!int.TryParse(limitText, out int parsedLimit) || parsedLimit <= 0))
	{
		Console.Error.WriteLine($"Error: invalid --limit value '{limitText}'.");
		Environment.ExitCode = 1;
		return;
	}

	int? limit = string.IsNullOrWhiteSpace(limitText) ? null : int.Parse(limitText);
	IReadOnlyList<FourCC> targetChunks;
	try
	{
		targetChunks = ParseMdxChunkIds(chunkText);
	}
	catch (ArgumentException ex)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
		return;
	}

	List<string> parseFailures = [];
	List<string> readMisses = [];
	int scanned = 0;
	int matched = 0;

	Console.WriteLine($"MDX chunk carrier scan: chunks={string.Join(',', targetChunks.Select(static chunk => chunk.ToString()))} source={(archiveRoot ?? input)!}");

	if (!string.IsNullOrWhiteSpace(archiveRoot))
	{
		using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
		ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [archiveRoot], archiveBootstrapOptions);

		IEnumerable<string> candidates = bootstrap
			.AllFiles
			.Where(static path => path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase));

		if (!string.IsNullOrWhiteSpace(pathFilter))
			candidates = candidates.Where(path => path.Contains(pathFilter, StringComparison.OrdinalIgnoreCase));

		foreach (string path in candidates.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase))
		{
			if (limit.HasValue && scanned >= limit.Value)
				break;

			scanned++;
			byte[]? bytes = archiveCatalog.ReadFile(path);
			if (bytes is null)
			{
				TrackScanIssue(readMisses, $"{path}: archive read returned no bytes");
				continue;
			}

			using MemoryStream stream = new(bytes, writable: false);
			matched += PrintMdxCarrierMatch(path, stream, targetChunks, parseFailures);
		}
	}
	else
	{
		IEnumerable<string> candidates = EnumerateMdxInputPaths(input!);
		if (!string.IsNullOrWhiteSpace(pathFilter))
			candidates = candidates.Where(path => path.Contains(pathFilter, StringComparison.OrdinalIgnoreCase));

		foreach (string path in candidates.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase))
		{
			if (limit.HasValue && scanned >= limit.Value)
				break;

			scanned++;
			using FileStream stream = File.OpenRead(path);
			matched += PrintMdxCarrierMatch(path, stream, targetChunks, parseFailures);
		}
	}

	Console.WriteLine($"Scanned={scanned} matched={matched} readMisses={readMisses.Count} parseFailures={parseFailures.Count}");
	if (matched == 0)
		Console.WriteLine("No matching carriers found.");

	if (readMisses.Count > 0)
	{
		Console.WriteLine("Read misses:");
		foreach (string miss in readMisses)
			Console.WriteLine($"  {miss}");
	}

	if (parseFailures.Count > 0)
	{
		Console.WriteLine("Parse failures:");
		foreach (string failure in parseFailures)
			Console.WriteLine($"  {failure}");
	}
}

static void RunArchiveBuildListfileCache(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r") ?? GetFirstPositionalArgument(args);
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	string? cacheKey = GetOption(args, "--cache-key", "-k");
	string? cacheDirectory = GetOption(args, "--cache-dir", "-d") ?? TryFindDefaultArchiveListfileCacheDirectory();

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(cacheKey))
	{
		Console.Error.WriteLine("Error: --cache-key is required so the manifest is explicitly tied to one MPQ-era client build.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(cacheDirectory))
	{
		Console.Error.WriteLine("Error: could not resolve a cache directory; provide --cache-dir explicitly.");
		Environment.ExitCode = 1;
		return;
	}

	using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
	ArchiveCatalogBootstrapResult result = ArchiveCatalogBootstrapper.Bootstrap(
		archiveCatalog,
		[archiveRoot],
		new ArchiveCatalogBootstrapOptions(
			ExternalListfilePath: listfilePath,
			ListfileCacheKey: cacheKey,
			ListfileCacheDirectoryPath: cacheDirectory));

	Console.WriteLine("Archive listfile cache built");
	Console.WriteLine($"Archive root: {archiveRoot}");
	Console.WriteLine($"Cache key: {cacheKey}");
	Console.WriteLine($"Cache path: {result.ListfileCachePath ?? "n/a"}");
	Console.WriteLine($"Trusted internal entries: {result.InternalFiles.Count}");
	Console.WriteLine($"Supplemental external entries: {result.ExternalListfileEntries.Count}");
	Console.WriteLine($"Known file universe: {result.AllFiles.Count}");
}

static void RunAudioAlphaArea(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r") ?? GetFirstPositionalArgument(args);
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required.");
		Environment.ExitCode = 1;
		return;
	}

	string buildVersion = GetOption(args, "--build", "-b") ?? AlphaAreaAudioCatalogReader.DefaultBuildVersion;
	int? areaId = TryGetIntOption(args, "--area-id", "-a");
	int limit = TryGetIntOption(args, "--limit", "-n") ?? (areaId.HasValue ? 1 : 20);
	string? search = GetOption(args, "--search", "-s");

	using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
	ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [archiveRoot], archiveBootstrapOptions);

	AlphaAreaAudioCatalogReader reader = new();
	AlphaAreaAudioCatalog? catalog = reader.Load([archiveRoot], archiveCatalog, buildVersion);
	if (catalog is null)
	{
		Console.Error.WriteLine("Error: could not load AreaTable and AreaMIDIAmbiences from the configured archive root.");
		Environment.ExitCode = 1;
		return;
	}

	IEnumerable<AlphaAreaAudioBinding> bindings = areaId.HasValue
		? catalog.TryResolve(areaId.Value) is { } resolvedBinding ? [resolvedBinding] : Array.Empty<AlphaAreaAudioBinding>()
		: catalog.EnumerateBindings();

	if (!string.IsNullOrWhiteSpace(search))
	{
		bindings = bindings.Where(binding => MatchesSearch(binding, search));
	}

	AlphaAreaAudioAssetResolver assetResolver = new();
	List<AlphaAreaAudioBindingAssetReport> allBindingReports = assetResolver.ResolveAll(catalog.EnumerateBindings(), [archiveRoot], archiveCatalog).ToList();
	List<AlphaAreaAudioBindingAssetReport> selectedBindingReports = assetResolver
		.ResolveAll(bindings.Take(Math.Max(limit, 0)), [archiveRoot], archiveCatalog)
		.ToList();
	int resolvedMidi = catalog.EnumerateBindings().Count(static binding => binding.MidiAmbience is not null);
	int resolvedUnderwaterMidi = catalog.EnumerateBindings().Count(static binding => binding.UnderwaterMidiAmbience is not null);
	int referencedAssets = allBindingReports.Sum(static report => report.EnumerateAssets().Count(static asset => asset.IsReferenced));
	int resolvedAssets = allBindingReports.Sum(static report => report.EnumerateAssets().Count(static asset => asset.Exists));
	int archiveAssets = allBindingReports.Sum(static report => report.EnumerateAssets().Count(static asset => asset.Source == AlphaAreaAudioAssetSource.Archive));
	int diskAssets = allBindingReports.Sum(static report => report.EnumerateAssets().Count(static asset => asset.Source == AlphaAreaAudioAssetSource.Disk));

	Console.WriteLine("Alpha area audio summary");
	Console.WriteLine($"Archive root: {archiveRoot}");
	Console.WriteLine($"Build: {buildVersion}");
	Console.WriteLine($"Areas: {catalog.Areas.Count}");
	Console.WriteLine($"AreaMIDIAmbiences: {catalog.MidiAmbiences.Count}");
	Console.WriteLine($"Resolved MIDIAmbience refs: {resolvedMidi}");
	Console.WriteLine($"Resolved underwater MIDIAmbience refs: {resolvedUnderwaterMidi}");
	Console.WriteLine($"Referenced asset refs: {referencedAssets}");
	Console.WriteLine($"Resolved asset refs: {resolvedAssets}");
	Console.WriteLine($"Archive-backed asset refs: {archiveAssets}");
	Console.WriteLine($"Disk-backed asset refs: {diskAssets}");
	Console.WriteLine($"Displayed rows: {selectedBindingReports.Count}");

	foreach (AlphaAreaAudioBindingAssetReport report in selectedBindingReports)
	{
		AlphaAreaAudioBinding binding = report.Binding;
		Console.WriteLine($"area={binding.Area.Id} name='{binding.Area.AreaName}' continent={binding.Area.ContinentId} midi={binding.Area.MidiAmbienceId} underwaterMidi={binding.Area.MidiAmbienceUnderwaterId}");
		if (binding.MidiAmbience is not null)
		{
			Console.WriteLine($"  day={FormatAudioAsset(report.DaySequence)} night={FormatAudioAsset(report.NightSequence)} dls={FormatAudioAsset(report.DlsFile)} volume={binding.MidiAmbience.Volume.ToString(CultureInfo.InvariantCulture)}");
		}
		else
		{
			Console.WriteLine("  day=- night=- dls=- volume=-");
		}

		if (binding.UnderwaterMidiAmbience is not null)
		{
			Console.WriteLine($"  underwaterDay={FormatAudioAsset(report.UnderwaterDaySequence)} underwaterNight={FormatAudioAsset(report.UnderwaterNightSequence)} underwaterDls={FormatAudioAsset(report.UnderwaterDlsFile)} underwaterVolume={binding.UnderwaterMidiAmbience.Volume.ToString(CultureInfo.InvariantCulture)}");
		}
	}

	if (areaId.HasValue && selectedBindingReports.Count == 0)
	{
		Environment.ExitCode = 1;
		Console.Error.WriteLine($"Error: area id {areaId.Value} was not found in the loaded Alpha AreaTable data.");
	}
}

static void RunMap(string[] args)
{
	if (args.Length == 0)
	{
		ShowMapUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunMapInspect(tail);
			break;
		case "terrain-patch-report":
			RunMapTerrainPatchReport(tail);
			break;
		case "uniqueid-filter":
			RunMapUniqueIdFilter(tail);
			break;
		case "uniqueid-report":
			RunMapUniqueIdReport(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown map command '{command}'.");
			ShowMapUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunLit(string[] args)
{
	if (args.Length == 0)
	{
		ShowLitUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunLitInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown lit command '{command}'.");
			ShowLitUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunMapInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	bool dumpTexChunks = HasOption(args, "--dump-tex-chunks");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input map file is required.");
		Environment.ExitCode = 1;
		return;
	}

	MapFileSummary summary = MapFileSummaryReader.Read(input);
	PrintMapSummary(summary, dumpTexChunks);
}

static void RunMapTerrainPatchReport(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: terrain patch report input JSON is required.");
		Environment.ExitCode = 1;
		return;
	}

	string inputPath = Path.GetFullPath(input);
	if (!File.Exists(inputPath))
	{
		Console.Error.WriteLine($"Error: terrain patch report not found: {inputPath}");
		Environment.ExitCode = 1;
		return;
	}

	IReadOnlyList<TerrainPatchReportEntry> entries = JsonSerializer.Deserialize<List<TerrainPatchReportEntry>>(
		File.ReadAllText(inputPath),
		new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
		?? [];
	TerrainPatchReportSummary summary = BuildTerrainPatchReportSummary(inputPath, entries);

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintTerrainPatchReportSummary(summary);
}

static void RunLitInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? samplePositionText = GetOption(args, "--sample-position", string.Empty);
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (!TryParseVector3Option(samplePositionText, out Vector3 samplePosition, out string? samplePositionError))
	{
		Console.Error.WriteLine($"Error: {samplePositionError}");
		Environment.ExitCode = 1;
		return;
	}

	bool hasSamplePosition = !string.IsNullOrWhiteSpace(samplePositionText);
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <lights.lit> or --archive-root <game|data dir> with --virtual-path <world/.../lights.lit>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	LitSummary summary;
	using (Stream stream = OpenInputStream())
		summary = LitSummaryReader.Read(stream, sourceLabel);

	PrintLitSummary(summary, hasSamplePosition ? samplePosition : null);
}

static void RunMapUniqueIdReport(string[] args)
{
	IReadOnlyList<string> inputs = GetOptionValues(args, "--input", "-i");
	string? positionalInput = GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	string? build = GetOption(args, "--build", "-b");
	if (inputs.Count == 0 && !string.IsNullOrWhiteSpace(positionalInput))
		inputs = [positionalInput];

	if (inputs.Count == 0)
	{
		Console.Error.WriteLine("Error: input map source is required.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		MapUniqueIdReport report = MapUniqueIdReportSupport.Build(inputs, build);
		string outputPath = MapUniqueIdReportSupport.Write(report, output);
		MapUniqueIdReportSupport.PrintSummary(report, outputPath);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunMapUniqueIdFilter(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	int? minUniqueId = TryGetIntOption(args, "--min-uniqueid", "-min");
	int? maxUniqueId = TryGetIntOption(args, "--max-uniqueid", "-max");
	string kind = GetOption(args, "--kind", "-k") ?? "all";
	bool invert = args.Any(static arg => string.Equals(arg, "--invert", StringComparison.OrdinalIgnoreCase));
	IReadOnlyList<string> buildLabels = ParseCsvOption(GetOption(args, "--build", "-b"));

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input uniqueid report is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!minUniqueId.HasValue && !maxUniqueId.HasValue && buildLabels.Count == 0 && string.Equals(kind, "all", StringComparison.OrdinalIgnoreCase))
	{
		Console.Error.WriteLine("Error: provide at least one filter: --min-uniqueid, --max-uniqueid, --build, or --kind.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		MapUniqueIdFilterReport report = MapUniqueIdFilterSupport.Filter(
			input,
			new MapUniqueIdFilterOptions(minUniqueId, maxUniqueId, buildLabels, kind, invert));
		string outputPath = MapUniqueIdFilterSupport.Write(report, output);
		MapUniqueIdFilterSupport.PrintSummary(report, outputPath);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4(string[] args)
{
	if (args.Length == 0)
	{
		ShowPm4Usage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunPm4Inspect(tail);
			break;
		case "match":
			RunPm4Match(tail);
			break;
			case "hierarchy":
				RunPm4Hierarchy(tail);
				break;
			case "linkage":
				RunPm4Linkage(tail);
				break;
			case "mscn":
				RunPm4Mscn(tail);
				break;
			case "unknowns":
				RunPm4Unknowns(tail);
				break;
			case "mshd":
				RunPm4Mshd(tail);
				break;
		case "audit":
			RunPm4Audit(tail);
			break;
		case "audit-directory":
			RunPm4AuditDirectory(tail);
			break;
		case "cross-tile":
			RunPm4CrossTile(tail);
			break;
		case "export-json":
			RunPm4ExportJson(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown pm4 command '{command}'.");
			ShowPm4Usage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunWmo(string[] args)
{
	if (args.Length == 0)
	{
		ShowWmoUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunWmoInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown wmo command '{command}'.");
			ShowWmoUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunPm4Match(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	string? output = GetOption(args, "--output", "-o");
	string? objectOutputDirectory = GetOption(args, "--object-output-dir", "-d");
	string? maxMatchesText = GetOption(args, "--max-matches", "-n");
	string? searchRangeText = GetOption(args, "--search-range", "-s");
	int maxMatches = 8;
	float searchRange = 128f;
	if (!string.IsNullOrWhiteSpace(maxMatchesText) && (!int.TryParse(maxMatchesText, out maxMatches) || maxMatches <= 0))
	{
		Console.Error.WriteLine("Error: --max-matches must be a positive integer.");
		Environment.ExitCode = 1;
		return;
	}
	if (!string.IsNullOrWhiteSpace(searchRangeText) && (!float.TryParse(searchRangeText, out searchRange) || searchRange <= 0f))
	{
		Console.Error.WriteLine("Error: --search-range must be a positive number.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required for pm4 match so WMO/M2 assets can be read from game archives.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(placements))
	{
		if (!Pm4CoordinateService.TryParseTileCoordinates(input, out int tileX, out int tileY))
		{
			Console.Error.WriteLine("Error: could not derive tile coordinates from the PM4 filename; provide --placements <tile_obj0.adt> explicitly.");
			Environment.ExitCode = 1;
			return;
		}

		string fileName = Path.GetFileNameWithoutExtension(input);
		int lastUnderscore = fileName.LastIndexOf('_');
		int previousUnderscore = lastUnderscore > 0 ? fileName.LastIndexOf('_', lastUnderscore - 1) : -1;
		string mapName = previousUnderscore > 0 ? fileName[..previousUnderscore] : fileName;
		placements = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? string.Empty, $"{mapName}_{tileX}_{tileY}_obj0.adt");
	}

	if (!File.Exists(placements))
	{
		Console.Error.WriteLine($"Error: placement source '{placements}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MatchResult result = Pm4MatchSupport.Run(input, placements, archiveRoot, archiveBootstrapOptions, maxMatches, searchRange);
	bool wroteArtifact = false;
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, Pm4MatchSupport.ToJson(result));
		Console.WriteLine($"Wrote {outputPath}");
		wroteArtifact = true;
	}

	if (!string.IsNullOrWhiteSpace(objectOutputDirectory))
	{
		IReadOnlyList<string> writtenPaths = Pm4MatchSupport.WriteObjectArtifacts(result, objectOutputDirectory);
		Console.WriteLine($"Wrote {writtenPaths.Count} PM4 match artifact files under {Path.GetFullPath(objectOutputDirectory)}");
		wroteArtifact = true;
	}

	if (wroteArtifact)
		return;

	Pm4MatchSupport.Print(result);
}

static void RunWmoInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;
	bool dumpLights = HasOption(args, "--dump-lights");
	bool flagCorrelation = HasOption(args, "--flag-correlation");
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.wmo|file.wmo.MPQ> or --archive-root <dir> with --virtual-path <world/...wmo>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	T ReadInput<T>(Func<Stream, string, T> reader)
	{
		using Stream stream = OpenInputStream();
		return reader(stream, sourceLabel);
	}

	WowFileDetection detection;
	using (Stream detectionStream = OpenInputStream())
		detection = WowFileDetector.Detect(detectionStream, sourceLabel);

	if (detection.Kind == WowFileKind.Wmo)
	{
		WmoSummary summary = ReadInput(WmoSummaryReader.Read);
		PrintWmoSummary(summary);
		if (summary.DoodadSetEntryCount > 0 && summary.DoodadPlacementEntryCount > 0)
		{
			WmoDoodadSetRangeSummary doodadSetRangeSummary = ReadInput(WmoDoodadSetRangeSummaryReader.Read);
			PrintWmoDoodadSetRangeSummary(doodadSetRangeSummary);
		}
		if (summary.GroupInfoCount > 0)
		{
			try
			{
				WmoGroupNameReferenceSummary groupNameReferenceSummary = ReadInput(WmoGroupNameReferenceSummaryReader.Read);
				PrintWmoGroupNameReferenceSummary(groupNameReferenceSummary);
			}
			catch (InvalidDataException)
			{
			}
		}
		if (summary.DoodadPlacementEntryCount > 0 && summary.DoodadNameTableCount > 0)
		{
			WmoDoodadNameReferenceSummary doodadNameReferenceSummary = ReadInput(WmoDoodadNameReferenceSummaryReader.Read);
			PrintWmoDoodadNameReferenceSummary(doodadNameReferenceSummary);
		}
		if (summary.ReportedLightCount > 0)
		{
			try
			{
				WmoLightSummary lightSummary = ReadInput(WmoLightSummaryReader.Read);
				PrintWmoLightSummary(lightSummary);
				if (dumpLights)
				{
					IReadOnlyList<WmoLightDetail> lightDetails = ReadInput(WmoLightDetailReader.Read);
					PrintWmoLightDetails(lightDetails);
				}
			}
			catch (InvalidDataException)
			{
			}
		}
		try
		{
			WmoFogSummary fogSummary = ReadInput(WmoFogSummaryReader.Read);
			PrintWmoFogSummary(fogSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoOpaqueChunkSummary mcvpSummary = ReadInput((stream, sourcePath) => WmoOpaqueChunkSummaryReader.Read(stream, sourcePath, WmoChunkIds.Mcvp));
			PrintWmoOpaqueChunkSummary(mcvpSummary);
		}
		catch (InvalidDataException)
		{
		}
		if (summary.ReportedPortalCount > 0)
		{
			try
			{
				WmoPortalVertexSummary portalVertexSummary = ReadInput(WmoPortalVertexSummaryReader.Read);
				PrintWmoPortalVertexSummary(portalVertexSummary);
				WmoPortalInfoSummary portalInfoSummary = ReadInput(WmoPortalInfoSummaryReader.Read);
				PrintWmoPortalInfoSummary(portalInfoSummary);
				WmoPortalRefSummary portalRefSummary = ReadInput(WmoPortalRefSummaryReader.Read);
				PrintWmoPortalRefSummary(portalRefSummary);
				WmoPortalVertexRangeSummary portalVertexRangeSummary = ReadInput(WmoPortalVertexRangeSummaryReader.Read);
				PrintWmoPortalVertexRangeSummary(portalVertexRangeSummary);
				WmoPortalRefRangeSummary portalRefRangeSummary = ReadInput(WmoPortalRefRangeSummaryReader.Read);
				PrintWmoPortalRefRangeSummary(portalRefRangeSummary);
				if (summary.GroupInfoCount > 0)
				{
					WmoPortalGroupRangeSummary portalGroupRangeSummary = ReadInput(WmoPortalGroupRangeSummaryReader.Read);
					PrintWmoPortalGroupRangeSummary(portalGroupRangeSummary);
				}
			}
			catch (InvalidDataException)
			{
			}
		}
		if (summary.MaterialEntryCount > 0 || summary.GroupInfoCount > 0 || summary.DoodadSetEntryCount > 0 || summary.DoodadPlacementEntryCount > 0 || summary.ReportedPortalCount > 0 || summary.ReportedLightCount > 0)
		{
			try
			{
				WmoVisibleVertexSummary visibleVertexSummary = ReadInput(WmoVisibleVertexSummaryReader.Read);
				PrintWmoVisibleVertexSummary(visibleVertexSummary);
			}
			catch (InvalidDataException)
			{
			}
			try
			{
				WmoVisibleBlockSummary visibleBlockSummary = ReadInput(WmoVisibleBlockSummaryReader.Read);
				PrintWmoVisibleBlockSummary(visibleBlockSummary);
			}
			catch (InvalidDataException)
			{
			}
			try
			{
				WmoVisibleBlockReferenceSummary visibleBlockReferenceSummary = ReadInput(WmoVisibleBlockReferenceSummaryReader.Read);
				PrintWmoVisibleBlockReferenceSummary(visibleBlockReferenceSummary);
			}
			catch (InvalidDataException)
			{
			}
		}
		try
		{
			WmoSkyboxSummary skyboxSummary = ReadInput(WmoSkyboxSummaryReader.Read);
			PrintWmoSkyboxSummary(skyboxSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoGroupNameTableSummary groupNameSummary = ReadInput(WmoGroupNameTableSummaryReader.Read);
			PrintWmoGroupNameTableSummary(groupNameSummary);
		}
		catch (InvalidDataException)
		{
		}
		if (summary.DoodadSetEntryCount > 0)
		{
			WmoDoodadSetSummary doodadSetSummary = ReadInput(WmoDoodadSetSummaryReader.Read);
			PrintWmoDoodadSetSummary(doodadSetSummary);
		}
		if (summary.DoodadPlacementEntryCount > 0)
		{
			WmoDoodadPlacementSummary doodadPlacementSummary = ReadInput(WmoDoodadPlacementSummaryReader.Read);
			PrintWmoDoodadPlacementSummary(doodadPlacementSummary);
		}
		if (summary.DoodadNameTableCount > 0)
		{
			WmoDoodadNameTableSummary doodadNameSummary = ReadInput(WmoDoodadNameTableSummaryReader.Read);
			PrintWmoDoodadNameTableSummary(doodadNameSummary);
		}
		if (summary.TextureNameCount > 0)
		{
			WmoTextureTableSummary textureSummary = ReadInput(WmoTextureTableSummaryReader.Read);
			PrintWmoTextureTableSummary(textureSummary);
		}
		if (summary.MaterialEntryCount > 0)
		{
			WmoMaterialSummary materialSummary = ReadInput(WmoMaterialSummaryReader.Read);
			PrintWmoMaterialSummary(materialSummary);
		}
		if (summary.GroupInfoCount > 0)
		{
			WmoGroupInfoSummary groupInfoSummary = ReadInput(WmoGroupInfoSummaryReader.Read);
			PrintWmoGroupInfoSummary(groupInfoSummary);
		}
		try
		{
			WmoEmbeddedGroupSummary embeddedGroupSummary = ReadInput(WmoEmbeddedGroupSummaryReader.Read);
			PrintWmoEmbeddedGroupSummary(embeddedGroupSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoEmbeddedGroupLinkageSummary embeddedGroupLinkageSummary = ReadInput(WmoEmbeddedGroupLinkageSummaryReader.Read);
			PrintWmoEmbeddedGroupLinkageSummary(embeddedGroupLinkageSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			IReadOnlyList<WmoEmbeddedGroupDetail> embeddedGroupDetails = ReadInput(WmoEmbeddedGroupDetailReader.Read);
			PrintWmoEmbeddedGroupDetails(embeddedGroupDetails);
			if (flagCorrelation)
				PrintWmoGroupFlagCorrelationReport(embeddedGroupDetails);
		}
		catch (InvalidDataException)
		{
		}
		return;
	}

	if (detection.Kind == WowFileKind.WmoGroup)
	{
		WmoGroupSummary summary = ReadInput(WmoGroupSummaryReader.Read);
		PrintWmoGroupSummary(summary);
		if (summary.NormalCount > 0)
		{
			WmoGroupNormalSummary normalSummary = ReadInput(WmoGroupNormalSummaryReader.Read);
			PrintWmoGroupNormalSummary(normalSummary);
		}
		if (summary.VertexCount > 0)
		{
			WmoGroupVertexSummary vertexSummary = ReadInput(WmoGroupVertexSummaryReader.Read);
			PrintWmoGroupVertexSummary(vertexSummary);
		}
		if (summary.IndexCount > 0)
		{
			WmoGroupIndexSummary indexSummary = ReadInput(WmoGroupIndexSummaryReader.Read);
			PrintWmoGroupIndexSummary(indexSummary);
		}
		if (summary.DoodadRefCount > 0)
		{
			WmoGroupDoodadRefSummary doodadRefSummary = ReadInput(WmoGroupDoodadRefSummaryReader.Read);
			PrintWmoGroupDoodadRefSummary(doodadRefSummary);
		}
		if (summary.LightRefCount > 0)
		{
			WmoGroupLightRefSummary lightRefSummary = ReadInput(WmoGroupLightRefSummaryReader.Read);
			PrintWmoGroupLightRefSummary(lightRefSummary);
		}
		if (summary.VertexColorCount > 0)
		{
			WmoGroupVertexColorSummary colorSummary = ReadInput(WmoGroupVertexColorSummaryReader.Read);
			PrintWmoGroupVertexColorSummary(colorSummary);
		}
		if (summary.PrimaryUvCount > 0)
		{
			WmoGroupUvSummary uvSummary = ReadInput(WmoGroupUvSummaryReader.Read);
			PrintWmoGroupUvSummary(uvSummary);
		}
		if (summary.FaceMaterialCount > 0)
		{
			WmoGroupFaceMaterialSummary faceSummary = ReadInput(WmoGroupFaceMaterialSummaryReader.Read);
			PrintWmoGroupFaceMaterialSummary(faceSummary);
		}
		if (summary.BatchCount > 0)
		{
			WmoGroupBatchSummary batchSummary = ReadInput(WmoGroupBatchSummaryReader.Read);
			PrintWmoGroupBatchSummary(batchSummary);
		}
		if (summary.BspNodeCount > 0)
		{
			WmoGroupBspNodeSummary bspNodeSummary = ReadInput(WmoGroupBspNodeSummaryReader.Read);
			PrintWmoGroupBspNodeSummary(bspNodeSummary);
		}
		if (summary.BspFaceRefCount > 0)
		{
			WmoGroupBspFaceSummary bspFaceSummary = ReadInput(WmoGroupBspFaceSummaryReader.Read);
			PrintWmoGroupBspFaceSummary(bspFaceSummary);
		}
		if (summary.BspNodeCount > 0 && summary.BspFaceRefCount > 0)
		{
			WmoGroupBspFaceRangeSummary bspFaceRangeSummary = ReadInput(WmoGroupBspFaceRangeSummaryReader.Read);
			PrintWmoGroupBspFaceRangeSummary(bspFaceRangeSummary);
		}
		if (summary.HasLiquid)
		{
			WmoGroupLiquidSummary liquidSummary = ReadInput(WmoGroupLiquidSummaryReader.Read);
			PrintWmoGroupLiquidSummary(liquidSummary);
		}
		return;
	}

	Console.Error.WriteLine($"Error: expected WMO root or group file, but detected {detection.Kind}.");
	Environment.ExitCode = 1;
}

static void RunPm4Inspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	PrintPm4Report(report);
}

static void RunPm4Audit(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4DecodeAuditReport report = Pm4ResearchAuditAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	PrintPm4AuditReport(report);
}

static void RunPm4AuditDirectory(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4CorpusAuditReport report = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(input);
	PrintPm4CorpusAuditReport(report);
}

static void RunPm4Linkage(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4LinkageReport report = Pm4ResearchLinkageAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4LinkageReport(report);
}

static void RunPm4Hierarchy(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4TileObjectHypothesisReport report = Pm4ResearchHierarchyAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4HierarchyReport(report);
}

static void RunPm4Mscn(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MscnRelationshipReport report = Pm4ResearchMscnAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4MscnReport(report);
}

static void RunPm4Unknowns(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4UnknownsReport report = Pm4ResearchUnknownsAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4UnknownsReport(report);
}

static void RunPm4Mshd(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MshdReport report = Pm4ResearchMshdAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4MshdReport(report);
}

static void RunPm4ExportJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	string? ck24Text = GetOption(args, "--ck24", "-k");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(input);
	object report;
	if (!string.IsNullOrWhiteSpace(ck24Text))
	{
		if (!TryParseUInt32Flexible(ck24Text, out uint ck24))
		{
			Console.Error.WriteLine($"Error: invalid --ck24 value '{ck24Text}'. Use decimal or 0x-prefixed hex.");
			Environment.ExitCode = 1;
			return;
		}

		report = Pm4Ck24ForensicsAnalyzer.Analyze(document, ck24);
	}
	else
	{
		report = Pm4ResearchAnalyzer.Analyze(document);
	}

	string json = JsonSerializer.Serialize(report, new JsonSerializerOptions
	{
		WriteIndented = true,
		IncludeFields = true,
	});

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
}

static bool TryParseUInt32Flexible(string value, out uint parsed)
{
	if (value.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		return uint.TryParse(value[2..], System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out parsed);

	return uint.TryParse(value, out parsed);
}

static string? GetOption(string[] args, string longName, string shortName)
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

static IReadOnlyList<string> GetOptionValues(string[] args, string longName, string shortName)
{
	List<string> values = [];
	for (int index = 0; index < args.Length - 1; index++)
	{
		if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
			|| string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
		{
			values.Add(args[index + 1]);
			index++;
		}
	}

	return values;
}

static string? GetFirstPositionalArgument(string[] args)
{
	for (int index = 0; index < args.Length; index++)
	{
		string current = args[index];
		if (current.StartsWith('-'))
		{
			index++;
			continue;
		}

		return current;
	}

	return null;
}

static int? TryGetIntOption(string[] args, string longName, string shortName)
{
	string? value = GetOption(args, longName, shortName);
	if (string.IsNullOrWhiteSpace(value))
	{
		return null;
	}

	if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
	{
		return parsed;
	}

	Console.Error.WriteLine($"Error: option {longName} requires an integer value.");
	Environment.ExitCode = 1;
	return null;
}

static IReadOnlyList<string> ParseCsvOption(string? value)
{
	if (string.IsNullOrWhiteSpace(value))
		return [];

	return value
		.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
		.Where(static entry => !string.IsNullOrWhiteSpace(entry))
		.Distinct(StringComparer.OrdinalIgnoreCase)
		.ToArray();
}

static bool MatchesSearch(AlphaAreaAudioBinding binding, string search)
{
	return binding.Area.AreaName.Contains(search, StringComparison.OrdinalIgnoreCase)
		|| (binding.MidiAmbience?.DaySequence?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false)
		|| (binding.MidiAmbience?.NightSequence?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false)
		|| (binding.MidiAmbience?.DlsFile?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false)
		|| (binding.UnderwaterMidiAmbience?.DaySequence?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false)
		|| (binding.UnderwaterMidiAmbience?.NightSequence?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false)
		|| (binding.UnderwaterMidiAmbience?.DlsFile?.Contains(search, StringComparison.OrdinalIgnoreCase) ?? false);
}

static string FormatAudioAsset(AlphaAreaAudioAssetProbe asset)
{
	if (!asset.IsReferenced)
	{
		return "-";
	}

	return asset.Source switch
	{
		AlphaAreaAudioAssetSource.Archive => $"{asset.RequestedPath} [archive]",
		AlphaAreaAudioAssetSource.Disk => $"{asset.RequestedPath} [disk:{asset.ResolvedPath}]",
		_ => $"{asset.RequestedPath} [missing]",
	};
}

static IReadOnlyList<FourCC> ParseMdxChunkIds(string chunkText)
{
	string[] tokens = chunkText
		.Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

	if (tokens.Length == 0)
		throw new ArgumentException("--chunks requires at least one four-character chunk id.");

	List<FourCC> chunks = [];
	HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
	foreach (string token in tokens)
	{
		if (token.Length != 4)
			throw new ArgumentException($"Chunk id '{token}' must be exactly 4 ASCII characters.");

		if (!token.All(static ch => ch <= 0x7F && !char.IsWhiteSpace(ch)))
			throw new ArgumentException($"Chunk id '{token}' must contain only non-whitespace ASCII characters.");

		if (!seen.Add(token))
			continue;

		chunks.Add(FourCC.FromString(token.ToUpperInvariant()));
	}

	return chunks;
}

static IEnumerable<string> EnumerateMdxInputPaths(string input)
{
	if (File.Exists(input))
	{
		if (!input.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
			throw new FileNotFoundException($"Input file '{input}' is not an .mdx file.", input);

		yield return Path.GetFullPath(input);
		yield break;
	}

	if (!Directory.Exists(input))
		throw new DirectoryNotFoundException($"Could not find input path '{input}'.");

	foreach (string path in Directory.EnumerateFiles(input, "*.mdx", SearchOption.AllDirectories))
		yield return path;
}

static int PrintMdxCarrierMatch(string sourcePath, Stream stream, IReadOnlyList<FourCC> targetChunks, List<string> parseFailures)
{
	try
	{
		MdxSummary summary = MdxSummaryReader.Read(stream, sourcePath);
		List<string> matchedChunks = targetChunks
			.Where(target => summary.Chunks.Any(chunk => chunk.Id == target))
			.Select(static chunk => chunk.ToString())
			.ToList();

		if (matchedChunks.Count == 0)
			return 0;

		Console.WriteLine($"CARRIER: path={sourcePath} matchedChunks={string.Join(',', matchedChunks)} chunkCount={summary.ChunkCount} knownChunks={summary.KnownChunkCount} unknownChunks={summary.UnknownChunkCount}");
		return 1;
	}
	catch (Exception ex) when (ex is InvalidDataException or IOException)
	{
		TrackParseFailure(parseFailures, $"{sourcePath}: {ex.Message}");
		return 0;
	}
}

static void TrackParseFailure(List<string> parseFailures, string message)
{
	TrackScanIssue(parseFailures, message);
}

static void TrackScanIssue(List<string> issues, string message)
{
	if (issues.Count < 10)
		issues.Add(message);
}

static bool HasOption(string[] args, string name)
{
	return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
}

static bool TryBuildArchiveBootstrapOptions(string[] args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions)
{
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	string? cacheKey = GetOption(args, "--cache-key", "-k");
	string? cacheDirectory = null;

	if (!string.IsNullOrWhiteSpace(cacheKey))
	{
		cacheDirectory = GetOption(args, "--cache-dir", "-d") ?? TryFindDefaultArchiveListfileCacheDirectory();
		if (string.IsNullOrWhiteSpace(cacheDirectory))
		{
			Console.Error.WriteLine("Error: --cache-key was provided but no cache directory could be resolved; provide --cache-dir explicitly.");
			Environment.ExitCode = 1;
			archiveBootstrapOptions = new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath);
			return false;
		}
	}

	archiveBootstrapOptions = new ArchiveCatalogBootstrapOptions(
		ExternalListfilePath: listfilePath,
		ListfileCacheKey: cacheKey,
		ListfileCacheDirectoryPath: cacheDirectory);
	return true;
}

static string? TryFindDefaultListfilePath()
{
	DirectoryInfo? current = new(AppContext.BaseDirectory);
	while (current is not null)
	{
		if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
		{
			string candidate = Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt");
			return File.Exists(candidate) ? candidate : null;
		}

		current = current.Parent;
	}

	return null;
}

static string? TryFindDefaultArchiveListfileCacheDirectory()
{
	DirectoryInfo? current = new(AppContext.BaseDirectory);
	for (int depth = 0; depth < 8 && current is not null; depth++, current = current.Parent)
	{
		if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
			return Path.Combine(current.FullName, "output", "cache", "archive-listfiles");
	}

	return null;
}

static void PrintPm4Report(Pm4AnalysisReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 report");
	Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
	Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
	Console.WriteLine($"Runtime boundaries: {RuntimeBoundaries.All.Length}");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Chunks: {report.ChunkOrder.Count}");
	Console.WriteLine($"Unknown chunks: {(report.UnknownChunks.Count == 0 ? "none" : string.Join(", ", report.UnknownChunks))}");
	Console.WriteLine();
	PrintVectorSet(report.Msvt);
	PrintVectorSet(report.Mscn);
	PrintVectorSet(report.MprlPositions);
	Console.WriteLine();
	Console.WriteLine($"MPRL total={report.Mprl.TotalCount}, normal={report.Mprl.NormalCount}, terminator={report.Mprl.TerminatorCount}");
	Console.WriteLine($"MPRL floor range={report.Mprl.FloorMin?.ToString() ?? "n/a"}..{report.Mprl.FloorMax?.ToString() ?? "n/a"}");
	Console.WriteLine($"MPRL rotation range={report.Mprl.RotationMinDegrees?.ToString("F2") ?? "n/a"}..{report.Mprl.RotationMaxDegrees?.ToString("F2") ?? "n/a"}");
	if (report.Terminology.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Terminology:");
		foreach (Pm4TerminologyEntry entry in report.Terminology)
		{
			string alias = string.IsNullOrWhiteSpace(entry.LocalAlias) ? "" : $" -> local alias {entry.LocalAlias}";
			Console.WriteLine($"  {entry.RawField}{alias} ({entry.Confidence})");
			Console.WriteLine($"    {entry.Notes}");
		}
	}
	Console.WriteLine();
	Console.WriteLine("Top MSUR._0x1c-derived key24 groups:");
	if (report.TopCk24Groups.Count == 0)
	{
		Console.WriteLine("  none");
	}
	else
	{
		foreach (Pm4Ck24Summary summary in report.TopCk24Groups.Take(10))
		{
			Console.WriteLine($"  key24=0x{summary.Ck24:X6} type=0x{summary.Ck24Type:X2} low16={summary.Ck24ObjectId} surfaces={summary.SurfaceCount} indices={summary.TotalIndexCount} mscnRef={summary.DistinctMscnRefCount}");
		}
	}

	if (report.ResearchNotes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Research notes:");
		foreach (string note in report.ResearchNotes)
			Console.WriteLine($"  {note}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintVectorSet(Pm4VectorSetSummary summary)
{
	Console.WriteLine($"{summary.Name}: count={summary.Count}");
	if (summary.Bounds is null || summary.Centroid is null)
		return;

	Console.WriteLine($"  bounds min={FormatVector(summary.Bounds.Min)} max={FormatVector(summary.Bounds.Max)}");
	Console.WriteLine($"  centroid={FormatVector(summary.Centroid.Value)}");
}

static string FormatVector(System.Numerics.Vector3 value)
{
	return $"({value.X:F2}, {value.Y:F2}, {value.Z:F2})";
}

static bool TryParseVector3Option(string? text, out Vector3 value, out string? error)
{
	value = default;
	error = null;
	if (string.IsNullOrWhiteSpace(text))
		return true;

	string[] parts = text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
	if (parts.Length != 3)
	{
		error = "--sample-position must be in x,y,z form.";
		return false;
	}

	if (!float.TryParse(parts[0], NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float x)
		|| !float.TryParse(parts[1], NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float y)
		|| !float.TryParse(parts[2], NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float z))
	{
		error = "--sample-position requires three invariant-culture floating-point values, for example 100.0,200.0,15.5.";
		return false;
	}

	value = new Vector3(x, y, z);
	return true;
}

static string FormatQuaternion(System.Numerics.Quaternion value)
{
	return $"({value.X:F3}, {value.Y:F3}, {value.Z:F3}, {value.W:F3})";
}

static string FormatUInt16List(IReadOnlyList<ushort> values)
{
	return values.Count == 0 ? "[]" : $"[{string.Join(",", values)}]";
}

static string FormatOptionalText(string? value)
{
	return string.IsNullOrWhiteSpace(value) ? "n/a" : value;
}

static string FormatWmoGroupFlags(WmoGroupSummary summary)
{
	WmoGroupFlags knownFlags = summary.KnownFlags;
	List<string> labels = [];

	if ((knownFlags & WmoGroupFlags.HasBspChunks) != 0)
		labels.Add("bsp-chunks");
	if ((knownFlags & WmoGroupFlags.IsExterior) != 0)
		labels.Add("exterior");
	if ((knownFlags & WmoGroupFlags.HasVertexColorChunk) != 0)
		labels.Add("vertex-colors");
	if ((knownFlags & WmoGroupFlags.UsesExteriorLighting) != 0)
		labels.Add("exterior-lighting");
	if ((knownFlags & WmoGroupFlags.HasLightRefChunk) != 0)
		labels.Add("light-refs");
	if ((knownFlags & WmoGroupFlags.HasMpbChunks) != 0)
		labels.Add("mpb-chunks");
	if ((knownFlags & WmoGroupFlags.HasDoodadRefChunk) != 0)
		labels.Add("doodad-refs");
	if ((knownFlags & WmoGroupFlags.HasLiquidChunk) != 0)
		labels.Add("liquid");
	if ((knownFlags & WmoGroupFlags.HasMoriMorbChunks) != 0)
		labels.Add("mori-morb");
	if ((knownFlags & WmoGroupFlags.HasSecondaryVertexColorChunk) != 0)
		labels.Add("secondary-vertex-colors");
	if ((knownFlags & WmoGroupFlags.HasSecondaryUvSet) != 0)
		labels.Add("secondary-uv");
	if ((knownFlags & WmoGroupFlags.HasTertiaryUvSet) != 0)
		labels.Add("tertiary-uv");

	uint unknownMask = summary.Flags & ~(uint)WmoGroupFlags.AllKnown;
	if (unknownMask != 0)
		labels.Add($"unknown:0x{unknownMask:X8}");

	return labels.Count == 0 ? "none" : string.Join(',', labels);
}

static void PrintPm4AuditReport(Pm4DecodeAuditReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 decode audit");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Chunks: {report.ChunkCount}, recognized={report.RecognizedChunkCount}, unknown={report.UnknownChunkCount}");
	Console.WriteLine($"Trailing-bytes diagnostic: {(report.HasTrailingBytesDiagnostic ? "yes" : "no")}");
	Console.WriteLine($"Overrun diagnostic: {(report.HasOverrunDiagnostic ? "yes" : "no")}");
	Console.WriteLine();
	Console.WriteLine("Reference audits:");
	foreach (Pm4ReferenceAudit audit in report.ReferenceAudits)
	{
		Console.WriteLine($"  {audit.Name}: total={audit.TotalCount} valid={audit.ValidCount} invalid={audit.InvalidCount}");
		foreach (string example in audit.Examples.Take(3))
			Console.WriteLine($"    {example}");
	}

	Console.WriteLine();
	Console.WriteLine("Chunk audits:");
	foreach (Pm4ChunkDecodeAudit audit in report.ChunkAudits.Take(12))
	{
		Console.WriteLine($"  {audit.Signature}: chunks={audit.ChunkCount} entries={audit.EntryCount} bytes={audit.TotalBytes} strideRemainders={audit.StrideRemainderCount}");
	}

	if (report.UnknownChunkSignatures.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine($"Unknown signatures: {string.Join(", ", report.UnknownChunkSignatures)}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintPm4CorpusAuditReport(Pm4CorpusAuditReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 corpus audit");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with diagnostics: {report.FilesWithDiagnostics}");
	Console.WriteLine($"Files with unknown chunks: {report.FilesWithUnknownChunks}");
	Console.WriteLine();
	Console.WriteLine("Chunk audits:");
	foreach (Pm4CorpusChunkAudit audit in report.ChunkAudits.Take(12))
	{
		Console.WriteLine($"  {audit.Signature}: files={audit.FileCount} totalChunks={audit.TotalChunkCount} totalEntries={audit.TotalEntryCount} strideFiles={audit.FilesWithStrideRemainders}");
	}

	Console.WriteLine();
	Console.WriteLine("Reference audits:");
	foreach (Pm4CorpusReferenceAudit audit in report.ReferenceAudits)
	{
		Console.WriteLine($"  {audit.Name}: total={audit.TotalCount} invalid={audit.InvalidCount}");
		foreach (string example in audit.ExampleFailures.Take(3))
			Console.WriteLine($"    {example}");
	}

	if (report.UnknownChunkSignatures.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine($"Unknown signatures: {string.Join(", ", report.UnknownChunkSignatures)}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintPm4LinkageReport(Pm4LinkageReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 linkage report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with ref-index mismatches: {report.FilesWithRefIndexMismatches}");
	Console.WriteLine($"Files with bad MDOS refs: {report.FilesWithBadMdos}");
	Console.WriteLine($"Total ref-index mismatches: {report.TotalRefIndexMismatchCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"Identity summary: ck24={report.IdentitySummary.DistinctCk24Count} low16={report.IdentitySummary.DistinctCk24ObjectIdCount} groups={report.IdentitySummary.ObjectIdGroupsAnalyzed} reused={report.IdentitySummary.ReusedObjectIdGroupCount} crossType={report.IdentitySummary.ReusedAcrossTypeGroupCount}");

	Console.WriteLine();
	Console.WriteLine("Top mismatch families:");
	foreach (Pm4LinkageMismatchFamily family in report.TopMismatchFamilies.Take(8))
	{
		Console.WriteLine($"  {family.FamilyKey}: files={family.FileCount} entries={family.EntryCount} low16Matches={family.MatchingCk24ObjectIdEntryCount} low24Matches={family.MatchingFullCk24EntryCount}");
		Pm4LinkageMismatchExample? example = family.TopExamples.FirstOrDefault();
		if (example is not null)
		{
			string tileText = example.TileX.HasValue && example.TileY.HasValue
				? $"{example.TileX}_{example.TileY}"
				: "n/a";
			string domains = example.CandidateDomains.Count == 0
				? "none"
				: string.Join('/', example.CandidateDomains);
			Console.WriteLine($"    example: tile={tileText} ref={example.RefIndex} group=0x{example.GroupObjectId:X8} domains={domains} low16Match={(example.Low16MatchesCk24ObjectId ? "yes" : "no")} low24Match={(example.Low24MatchesCk24 ? "yes" : "no")}");
		}
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
	}

static void PrintPm4HierarchyReport(Pm4TileObjectHypothesisReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 hierarchy report");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Tile: {(report.TileX.HasValue && report.TileY.HasValue ? $"{report.TileX}_{report.TileY}" : "n/a")}");
	Console.WriteLine($"Distinct CK24 groups: {report.Ck24GroupCount}");
	Console.WriteLine($"Hypothesis objects: {report.TotalHypothesisCount}");
	Console.WriteLine();
	Console.WriteLine("Top hierarchy candidates:");
	foreach (Pm4ObjectHypothesis hypothesis in report.Objects.Take(12))
	{
		Pm4ForensicsPlacementComparison placement = hypothesis.PlacementComparison;
		string headingText = placement.MprlHeadingMeanDegrees.HasValue
			? $" heading={placement.MprlHeadingMeanDegrees.Value:F2} delta={placement.HeadingDeltaDegrees?.ToString("F2") ?? "n/a"}"
			: string.Empty;
		Console.WriteLine($"  {hypothesis.Family}#{hypothesis.FamilyObjectIndex}: ck24=0x{hypothesis.Ck24:X6} surfaces={hypothesis.SurfaceCount} indices={hypothesis.TotalIndexCount} linkGroups={hypothesis.MslkGroupObjectIds.Count} dominantGroup=0x{hypothesis.DominantLinkGroupObjectId:X} linkedMPRL={hypothesis.MprlFootprint.LinkedRefCount}/{hypothesis.MprlFootprint.LinkedInBoundsCount} mode={placement.CoordinateMode} planar=(swap={placement.PlanarTransform.SwapPlanarAxes}, invertU={placement.PlanarTransform.InvertU}, invertV={placement.PlanarTransform.InvertV}) frameYaw={placement.FrameYawDegrees:F2}{headingText}");
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(12))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintPm4MscnReport(Pm4MscnRelationshipReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 MSCN report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with MSCN: {report.FilesWithMscn}");
	Console.WriteLine($"Files with tile coordinates: {report.FilesWithTileCoordinates}");
	Console.WriteLine($"Total MSCN points: {report.TotalMscnPointCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"Coordinate space: swappedWorld={report.CoordinateSpace.SwappedWorldTileFitCount} rawWorld={report.CoordinateSpace.RawWorldTileFitCount} ambiguousWorld={report.CoordinateSpace.AmbiguousWorldTileFitCount} tileLocal={report.CoordinateSpace.TileLocalLikeCount} neither={report.CoordinateSpace.NeitherFitCount}");
	Console.WriteLine($"Dominant files: swapped={report.CoordinateSpace.FilesSwappedDominant} raw={report.CoordinateSpace.FilesRawDominant} tileLocal={report.CoordinateSpace.FilesTileLocalDominant} noDominant={report.CoordinateSpace.FilesNoDominant}");

	Console.WriteLine();
	Console.WriteLine("Cluster distributions:");
	foreach (Pm4FieldDistribution distribution in report.ClusterDistributions)
	{
		Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}");
		foreach (Pm4ValueFrequency value in distribution.TopValues.Take(4))
			Console.WriteLine($"    {value.Value} -> {value.Count}");
	}

	if (report.TopInvalidMdosClusters.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top invalid-MDOS clusters:");
		foreach (Pm4MscnClusterExample cluster in report.TopInvalidMdosClusters.Take(8))
		{
			Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} invalidMdos={cluster.InvalidMdosRefCount} distinctMscnRef={cluster.DistinctMscnRefCount} align={cluster.AlignmentMode}");
		}
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintPm4UnknownsReport(Pm4UnknownsReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 unknowns report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Non-empty files: {report.NonEmptyFileCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"MSPI interpretation: active={report.MspiInterpretation.ActiveLinkCount} indicesOnly={report.MspiInterpretation.IndicesModeOnlyCount} trianglesOnly={report.MspiInterpretation.TrianglesModeOnlyCount} both={report.MspiInterpretation.BothModesCount} neither={report.MspiInterpretation.NeitherModeCount}");
	Console.WriteLine($"LinkId patterns: total={report.LinkIdPatterns.TotalCount} sentinelTile={report.LinkIdPatterns.SentinelTileLinkCount} zero={report.LinkIdPatterns.ZeroCount} other={report.LinkIdPatterns.OtherCount}");

	Console.WriteLine();
	Console.WriteLine("Unknowns:");
	foreach (Pm4UnknownFinding finding in report.Unknowns)
	{
		Console.WriteLine($"  [{finding.Status}] {finding.Name}");
		Console.WriteLine($"    {finding.Evidence}");
	}

	if (report.TopMslkFamilies.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top MSLK families:");
		foreach (Pm4MslkFamilySummary family in report.TopMslkFamilies.Take(8))
		{
			Console.WriteLine($"  {family.FamilyKey}: entries={family.EntryCount} files={family.FileCount} msurFit={family.DirectMsurFitCount} mprlFit={family.DirectMprlFitCount} nonZeroGroup={family.NonZeroGroupObjectIdCount} mprlKeyFit={family.GroupObjectIdMatchesMprlKeyCount} sentinel={family.SentinelTileLinkCount} zeroLink={family.ZeroLinkIdCount} otherLink={family.OtherLinkIdCount}");
		}
	}

	if (report.TopMsurFamilies.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top MSUR families:");
		foreach (Pm4MsurFamilySummary family in report.TopMsurFamilies.Take(8))
		{
			Console.WriteLine($"  {family.FamilyKey}: surfaces={family.SurfaceCount} files={family.FileCount} distinctCk24={family.DistinctCk24Count} distinctType={family.DistinctCk24TypeCount} distinctMscnRef={family.DistinctMscnRefCount} incomingMslk={family.IncomingMslkCount} incomingFamilies={family.DistinctIncomingMslkFamilyCount} avgPlane={family.AveragePlaneDistance:F3}");
		}
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintPm4MshdReport(Pm4MshdReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 MSHD report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount} total, {report.FilesWithMshd} with MSHD");
	Console.WriteLine();
	Console.WriteLine("Field summaries:");
	foreach (Pm4MshdFieldSummary field in report.Fields)
	{
		string topValues = field.TopValues.Count == 0
			? "none"
			: string.Join(", ", field.TopValues.Take(4).Select(static value => $"{value.Value}->{value.Count}"));
		Console.WriteLine($"  {field.Field}: distinct={field.DistinctCount} zero={field.ZeroCount} nonZero={field.NonZeroCount} top={topValues}");
		foreach (Pm4MshdMetricCorrelation metric in field.MetricCorrelations.Take(4))
			Console.WriteLine($"    {metric.Metric}: exact={metric.ExactMatchCount} within1={metric.WithinOneCount} corr={metric.PearsonCorrelation:F3}");
	}

	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4MshdRelationshipSummary relationship in report.Relationships)
		Console.WriteLine($"  {relationship.Relationship}: {relationship.MatchCount}/{relationship.FileCount} ({relationship.Notes})");

	if (report.TilePackingHypotheses.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Tile-coordinate hypotheses:");
		foreach (Pm4MshdTilePackingSummary hypothesis in report.TilePackingHypotheses)
			Console.WriteLine($"  {hypothesis.Hypothesis}: {hypothesis.MatchCount}/{hypothesis.FileCount} ({hypothesis.Notes})");
	}

	Console.WriteLine();
	Console.WriteLine($"Tile reuse: filesWithCoords={report.TileReuse.FilesWithTileCoordinates} distinctTiles={report.TileReuse.DistinctTileCount} distinctField04={report.TileReuse.DistinctField04Count} singleTileValues={report.TileReuse.SingleTileField04Count} multiTileValues={report.TileReuse.MultiTileField04Count}");
	if (report.TileReuse.TopMultiTileField04Values.Count > 0)
	{
		foreach (Pm4MshdField04TileReuseCase reuseCase in report.TileReuse.TopMultiTileField04Values.Take(6))
			Console.WriteLine($"  Field04={reuseCase.Field04} spans {reuseCase.TileCount} tiles: {string.Join(", ", reuseCase.TileCoordinates)}");
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintMapSummary(MapFileSummary summary, bool dumpTexChunks)
{
	Console.WriteLine("WowViewer.Tool.Inspect map report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Kind: {summary.Kind}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	if (summary.Kind == MapFileKind.Wdt)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		WdtSummary wdtSummary = WdtSummaryReader.Read(stream, summary);
		Console.WriteLine($"WDT semantics: wmoBased={wdtSummary.IsWmoBased} tiles={wdtSummary.TilesWithData}/{wdtSummary.TotalTiles} mainCellBytes={wdtSummary.MainCellSizeBytes} doodadNames={wdtSummary.DoodadNameCount} wmoNames={wdtSummary.WorldModelNameCount} doodadPlacements={wdtSummary.DoodadPlacementCount} wmoPlacements={wdtSummary.WorldModelPlacementCount}");
		if (wdtSummary.MainFlags is not null)
			Console.WriteLine($"WDT MAIN flags: any={wdtSummary.MainFlags.CellsWithAnyFlags} hasAdt={wdtSummary.MainFlags.CellsWithHasAdt} allWater={wdtSummary.MainFlags.CellsWithAllWater} loaded={wdtSummary.MainFlags.CellsWithLoaded} unknown={wdtSummary.MainFlags.CellsWithUnknownFlags} asyncIds={wdtSummary.MainFlags.CellsWithAsyncId} distinct={FormatWdtMainFlags(wdtSummary.MainFlags)}");
	}
	else if (summary.Kind is MapFileKind.AdtV23 or MapFileKind.AdtV23Error)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		AdtV23Summary v23Summary = AdtV23SummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT/v23 semantics: kind={v23Summary.Kind} headerVersion={v23Summary.HeaderVersion} vertices={v23Summary.VerticesX}x{v23Summary.VerticesY} chunks={v23Summary.ChunksX}x{v23Summary.ChunksY} acnk={v23Summary.TerrainChunkCount} textures={v23Summary.TextureNameCount} objects={v23Summary.ObjectNameCount} avtx={v23Summary.HasVertexHeights} anrm={v23Summary.HasNormals} afbo={v23Summary.HasFlightBounds} acvt={v23Summary.HasVertexShading}");
	}
	else if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj or MapFileKind.AdtLod)
	{
		AdtTileFamily family = AdtTileFamilyResolver.Resolve(summary.SourcePath);
		Console.WriteLine($"ADT family: root={(family.HasRoot ? "present" : "missing")} tex0={(family.HasTex0 ? "present" : "missing")} obj0={(family.HasObj0 ? "present" : "missing")} lod={(family.HasLod ? "present" : "missing")} textureSource={FormatMapFileKind(family.TextureSourceKind)} placementSource={FormatMapFileKind(family.PlacementSourceKind)}");
	}

	if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		AdtSummary adtSummary = AdtSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT semantics: kind={adtSummary.Kind} terrainChunks={adtSummary.TerrainChunkCount} textures={adtSummary.TextureNameCount} doodadNames={adtSummary.ModelNameCount} wmoNames={adtSummary.WorldModelNameCount} doodadPlacements={adtSummary.ModelPlacementCount} wmoPlacements={adtSummary.WorldModelPlacementCount} hasMfbo={adtSummary.HasFlightBounds} hasMh2o={adtSummary.HasWater} hasMamp={adtSummary.HasTextureParams} hasMtxf={adtSummary.HasTextureFlags}");
		AdtMcnkSummary mcnkSummary = AdtMcnkSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT MCNK semantics: mcnk={mcnkSummary.McnkCount} zero={mcnkSummary.ZeroLengthMcnkCount} headerLike={mcnkSummary.HeaderLikeMcnkCount} distinctIndex={mcnkSummary.DistinctIndexCount} duplicateIndex={mcnkSummary.DuplicateIndexCount} areaIds={mcnkSummary.DistinctAreaIdCount} holes={mcnkSummary.ChunksWithHoles} liquidFlags={mcnkSummary.ChunksWithLiquidFlags} mccvFlags={mcnkSummary.ChunksWithMccvFlag} mcvt={mcnkSummary.ChunksWithMcvt} mcnr={mcnkSummary.ChunksWithMcnr} mcly={mcnkSummary.ChunksWithMcly} mcal={mcnkSummary.ChunksWithMcal} mcsh={mcnkSummary.ChunksWithMcsh} mcse={mcnkSummary.ChunksWithMcse} mcseBytes={mcnkSummary.TotalMcsePayloadBytes} mccv={mcnkSummary.ChunksWithMccv} mclq={mcnkSummary.ChunksWithMclq} mcrd={mcnkSummary.ChunksWithMcrd} mcrw={mcnkSummary.ChunksWithMcrw} totalLayers={mcnkSummary.TotalLayerCount} maxLayers={mcnkSummary.MaxLayerCount} multiLayerChunks={mcnkSummary.ChunksWithMultipleLayers}");
		if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex)
		{
			AdtMcalSummary mcalSummary = AdtMcalSummaryReader.Read(stream, summary);
			Console.WriteLine($"ADT MCAL semantics: profile={mcalSummary.DecodeProfile} mcnkWithLayers={mcalSummary.McnkWithLayerTableCount} overlayLayers={mcalSummary.OverlayLayerCount} decodedLayers={mcalSummary.DecodedLayerCount} missingPayloadLayers={mcalSummary.MissingPayloadLayerCount} decodeFailures={mcalSummary.DecodeFailureCount} compressed={mcalSummary.CompressedLayerCount} bigAlpha={mcalSummary.BigAlphaLayerCount} bigAlphaFixed={mcalSummary.BigAlphaFixedLayerCount} packed4={mcalSummary.PackedLayerCount}");
			if (dumpTexChunks && summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex)
			{
				AdtTextureFile textureFile = AdtTextureReader.Read(stream, summary);
				PrintAdtTextureFile(textureFile);
			}
		}
	}
	Console.WriteLine($"Top-level chunks: {summary.ChunkCount}");
	string chunkOrder = string.Join(", ", summary.Chunks.Take(16).Select(chunk => chunk.Id.ToString()));
	if (summary.Chunks.Count > 16)
		chunkOrder = $"{chunkOrder}, ... ({summary.Chunks.Count - 16} more)";

	Console.WriteLine($"Chunk order: {chunkOrder}");
	Console.WriteLine();
	Console.WriteLine("Chunk counts:");
	foreach (IGrouping<string, MapChunkLocation> group in summary.Chunks.GroupBy(chunk => chunk.Id.ToString()).OrderBy(group => group.Key))
	{
		Console.WriteLine($"  {group.Key}: count={group.Count()} bytes={group.Sum(chunk => (long)chunk.Size)}");
	}

	Console.WriteLine();
	Console.WriteLine("First top-level chunks:");
	foreach (MapChunkLocation chunk in summary.Chunks.Take(12))
	{
		Console.WriteLine($"  {chunk.Id}: size={chunk.Size} header={chunk.HeaderOffset} data={chunk.DataOffset}");
	}

	if (summary.Chunks.Count > 12)
		Console.WriteLine($"  ... {summary.Chunks.Count - 12} more chunks");
}

static string FormatWdtMainFlags(WdtMainFlagsSummary summary)
{
	if (summary.DistinctNonZeroValues.Count == 0)
		return "none";

	return string.Join(",", summary.DistinctNonZeroValues.Select(static value => $"0x{value.Value:x}:{value.TileCount}"));
}

static void PrintLitSummary(LitSummary summary, Vector3? samplePosition = null)
{
	Console.WriteLine("WowViewer.Tool.Inspect LIT report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: 0x{summary.VersionNumber:X8}");
	Console.WriteLine($"LIT semantics: lightCount={summary.LightCount} listEntries={summary.ListEntryCount} singlePartial={summary.UsesSinglePartialEntry} defaultFirstEntry={summary.HasDefaultFirstEntry} namedEntries={summary.NamedEntryCount} remainingPayloadBytes={summary.RemainingPayloadBytes}");
	if (summary.Entries.Count > 0)
	{
		int previewCount = Math.Min(summary.Entries.Count, 8);
		Console.WriteLine($"LIT entry preview: showing {previewCount}/{summary.Entries.Count} list entries");
		for (int index = 0; index < previewCount; index++)
		{
			LitListEntrySummary entry = summary.Entries[index];
			string label = entry.HasName ? entry.Name : "<unnamed>";
			Console.WriteLine($"LIT.ENTRY[{entry.Index}]: default={entry.IsDefaultEntry} name={label} chunk=({entry.ChunkX},{entry.ChunkY}) chunkRadius={entry.ChunkRadius} position={FormatVector(entry.Position)} radius={entry.LightRadius:F2} dropoff={entry.LightDropoff:F2} outerRadius={entry.OuterRadius:F2}");
		}

		if (summary.Entries.Count > previewCount)
			Console.WriteLine($"LIT entry preview truncated: {summary.Entries.Count - previewCount} additional entries not shown.");
	}

	if (samplePosition is Vector3 position)
	{
		IReadOnlyList<LitSpatialSampleCandidate> candidates = LitSpatialSampler.Sample(summary, position);
		Console.WriteLine($"LIT sample: query={FormatVector(position)} candidates={candidates.Count}");
		foreach (LitSpatialSampleCandidate candidate in candidates)
		{
			string label = candidate.Entry.HasName ? candidate.Entry.Name : "<unnamed>";
			Console.WriteLine($"LIT.SAMPLE[{candidate.Entry.Index}]: name={label} fallbackDefault={candidate.IsFallbackDefault} distance={candidate.Distance:F2} influence={candidate.Influence:F3} withinCore={candidate.WithinCoreRadius} withinOuter={candidate.WithinOuterRadius} entryPos={FormatVector(candidate.Entry.Position)} radius={candidate.Entry.LightRadius:F2} dropoff={candidate.Entry.LightDropoff:F2}");
		}
		Console.WriteLine("LIT sample boundary: spatial candidate selection is heuristic over list-entry radius/dropoff only; color-band payload sampling is still pending.");
	}

	Console.WriteLine("Proof boundary: parser summary plus spatial list-entry sampling only; runtime .lit color/fog ownership is still unproven.");
}

static void PrintWmoSummary(WmoSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect WMO report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"WMO semantics: materials={summary.MaterialEntryCount}/{summary.ReportedMaterialCount} groups={summary.GroupInfoCount}/{summary.ReportedGroupCount} portals={summary.ReportedPortalCount} lights={summary.ReportedLightCount} textures={summary.TextureNameCount} doodadNames={summary.DoodadNameTableCount}/{summary.ReportedDoodadNameCount} doodadPlacements={summary.DoodadPlacementEntryCount}/{summary.ReportedDoodadPlacementCount} doodadSets={summary.DoodadSetEntryCount}/{summary.ReportedDoodadSetCount} skybox={(summary.HasSkybox ? "yes" : "no")} flags=0x{summary.Flags:X8}");
	Console.WriteLine($"Bounds: min={FormatVector(summary.BoundsMin)} max={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupFlagCorrelationReport(IReadOnlyList<WmoEmbeddedGroupDetail> details)
{
	if (details.Count == 0)
		return;

	List<uint> bits = [];
	for (int bitIndex = 0; bitIndex < 32; bitIndex++)
	{
		uint bit = 1u << bitIndex;
		if (details.Any(detail => (detail.GroupSummary.Flags & bit) != 0))
			bits.Add(bit);
	}

	if (bits.Count == 0)
		return;

	Console.WriteLine($"MOGP flag correlation: groups={details.Count}");
	foreach (uint bit in bits)
	{
		int setCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0);
		int bspCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.BspNodeCount > 0 && detail.GroupSummary.BspFaceRefCount > 0);
		int lightRefCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.LightRefCount > 0);
		int doodadRefCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.DoodadRefCount > 0);
		int liquidCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.HasLiquid);
		int vertexColorCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.VertexColorCount > 0);
		int extraUvCount = details.Count(detail => (detail.GroupSummary.Flags & bit) != 0 && detail.GroupSummary.AdditionalUvSetCount > 0);
		Console.WriteLine($"MOGP.FLAG[0x{bit:X8}]: label={DescribeWmoGroupFlagBit(bit)} set={setCount}/{details.Count} bsp={bspCount}/{setCount} lightRefs={lightRefCount}/{setCount} doodadRefs={doodadRefCount}/{setCount} liquid={liquidCount}/{setCount} vertexColors={vertexColorCount}/{setCount} extraUv={extraUvCount}/{setCount}");
	}
}

static string DescribeWmoGroupFlagBit(uint bit)
{
	return bit switch
	{
		(uint)WmoGroupFlags.HasBspChunks => "bsp-chunks",
		(uint)WmoGroupFlags.IsExterior => "exterior",
		(uint)WmoGroupFlags.HasVertexColorChunk => "vertex-colors",
		(uint)WmoGroupFlags.UsesExteriorLighting => "exterior-lighting",
		(uint)WmoGroupFlags.HasLightRefChunk => "light-refs",
		(uint)WmoGroupFlags.HasMpbChunks => "mpb-chunks",
		(uint)WmoGroupFlags.HasDoodadRefChunk => "doodad-refs",
		(uint)WmoGroupFlags.HasLiquidChunk => "liquid",
		(uint)WmoGroupFlags.HasMoriMorbChunks => "mori-morb",
		(uint)WmoGroupFlags.HasSecondaryVertexColorChunk => "secondary-vertex-colors",
		(uint)WmoGroupFlags.HasSecondaryUvSet => "secondary-uv",
		(uint)WmoGroupFlags.HasTertiaryUvSet => "tertiary-uv",
		_ => "unknown",
	};
}

static string FormatMapFileKind(MapFileKind? kind)
{
	return kind?.ToString() ?? "n/a";
}

static void PrintAdtTextureFile(AdtTextureFile textureFile)
{
	Console.WriteLine($"ADT texture detail: kind={textureFile.Kind} profile={textureFile.DecodeProfile} textures={textureFile.TextureNames.Count} chunks={textureFile.Chunks.Count}");
	foreach (AdtTextureChunk chunk in textureFile.Chunks)
	{
		if (chunk.Layers.Count == 0)
			continue;

		Console.WriteLine($"MCNK(texture)[{chunk.ChunkIndex}]: xy=({chunk.ChunkX},{chunk.ChunkY}) layers={chunk.Layers.Count} alphaBytes={chunk.AlphaPayloadBytes} doNotFixAlphaMap={chunk.DoNotFixAlphaMap} decodedLayers={chunk.DecodedLayerCount}");
		foreach (AdtTextureChunkLayer layer in chunk.Layers)
		{
			string texturePath = string.IsNullOrWhiteSpace(layer.TexturePath) ? "n/a" : layer.TexturePath;
			string alphaSummary = layer.DecodedAlpha is null
				? "alpha=n/a"
				: $"alpha={layer.DecodedAlpha.Encoding} bytes={layer.DecodedAlpha.SourceBytesConsumed}";
			Console.WriteLine($"MCNK(texture)[{chunk.ChunkIndex}].LAYER[{layer.Index}]: textureId={layer.TextureId} texture={texturePath} flags=0x{layer.Flags:X8} alphaOffset={layer.AlphaOffset} effectId={layer.EffectId} {alphaSummary}");
		}
	}
}

static void PrintWmoGroupInfoSummary(WmoGroupInfoSummary summary)
{
	Console.WriteLine($"MOGI: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} entries={summary.EntryCount} distinctFlags={summary.DistinctFlagCount} nonZeroFlags={summary.NonZeroFlagCount} nameOffsetRange={summary.MinNameOffset}-{summary.MaxNameOffset} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoEmbeddedGroupSummary(WmoEmbeddedGroupSummary summary)
{
	Console.WriteLine($"MOGP(root): groups={summary.GroupCount} headerBytes={summary.MinHeaderSizeBytes}-{summary.MaxHeaderSizeBytes} groupsWithPortals={summary.GroupsWithPortals} groupsWithLiquid={summary.GroupsWithLiquid} faces={summary.TotalFaceMaterialCount} vertices={summary.TotalVertexCount} indices={summary.TotalIndexCount} normals={summary.TotalNormalCount} batches={summary.TotalBatchCount} doodadRefs={summary.TotalDoodadRefCount} lightRefs={summary.TotalLightRefCount} bspNodes={summary.TotalBspNodeCount} bspFaceRefs={summary.TotalBspFaceRefCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoEmbeddedGroupLinkageSummary(WmoEmbeddedGroupLinkageSummary summary)
{
	Console.WriteLine($"MOGI->MOGP(root): infos={summary.GroupInfoCount} groups={summary.EmbeddedGroupCount} coveredPairs={summary.CoveredPairCount} missingGroups={summary.MissingEmbeddedGroupCount} extraGroups={summary.ExtraEmbeddedGroupCount} flagMatches={summary.FlagMatchCount} boundsMatches={summary.BoundsMatchCount} maxBoundsDelta={summary.MaxBoundsDelta:F3}");
}

static void PrintWmoEmbeddedGroupDetails(IReadOnlyList<WmoEmbeddedGroupDetail> details)
{
	foreach (WmoEmbeddedGroupDetail detail in details)
	{
		PrintWmoEmbeddedGroupDetail(detail);
	}
}

static void PrintWmoLightDetails(IReadOnlyList<WmoLightDetail> details)
{
	foreach (WmoLightDetail detail in details)
	{
		PrintWmoLightDetail(detail);
	}
}

static void PrintWmoLightDetail(WmoLightDetail detail)
{
	string headerFlagsText = detail.HeaderFlagsWord is ushort headerFlagsWord
		? $"0x{headerFlagsWord:X4}"
		: "n/a";
	string rotationText = detail.Rotation is System.Numerics.Quaternion rotation
		? FormatQuaternion(rotation)
		: "n/a";
	string rotationLengthText = detail.RotationLength is float rotationLength
		? rotationLength.ToString("F3")
		: "n/a";

	Console.WriteLine($"MOLT[{detail.LightIndex}]: offset={detail.PayloadOffset} entryBytes={detail.EntrySizeBytes} type={detail.LightType} attenuated={detail.UsesAttenuation} headerFlagsWord={headerFlagsText} color=0x{detail.ColorBgra:X8} position={FormatVector(detail.Position)} intensity={detail.Intensity:F3} attenStart={detail.AttenStart:F3} attenEnd={detail.AttenEnd:F3} rotation={rotationText} rotationLen={rotationLengthText}");
}

static void PrintWmoEmbeddedGroupDetail(WmoEmbeddedGroupDetail detail)
{
	WmoGroupSummary summary = detail.GroupSummary;
	Console.WriteLine($"MOGP(root)[{detail.GroupIndex}]: offset={detail.GroupHeaderOffset} flags=0x{summary.Flags:X8} ({FormatWmoGroupFlags(summary)}) portals={summary.PortalCount}@{summary.PortalStart} faces={summary.FaceMaterialCount} vertices={summary.VertexCount} indices={summary.IndexCount} normals={summary.NormalCount} batches={summary.BatchCount}/{summary.DeclaredBatchCount} doodadRefs={summary.DoodadRefCount} lightRefs={summary.LightRefCount} bspNodes={summary.BspNodeCount} bspFaceRefs={summary.BspFaceRefCount} hasLiquid={summary.HasLiquid} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");

	if (detail.NormalSummary is not null)
		Console.WriteLine($"MONR(root)[{detail.GroupIndex}]: payloadBytes={detail.NormalSummary.PayloadSizeBytes} normals={detail.NormalSummary.NormalCount} rangeX=[{detail.NormalSummary.MinX:F3}, {detail.NormalSummary.MaxX:F3}] rangeY=[{detail.NormalSummary.MinY:F3}, {detail.NormalSummary.MaxY:F3}] rangeZ=[{detail.NormalSummary.MinZ:F3}, {detail.NormalSummary.MaxZ:F3}] lengthRange=[{detail.NormalSummary.MinLength:F3}, {detail.NormalSummary.MaxLength:F3}] avgLength={detail.NormalSummary.AverageLength:F3} nearUnit={detail.NormalSummary.NearUnitCount}");

	if (detail.VertexSummary is not null)
		Console.WriteLine($"MOVT(root)[{detail.GroupIndex}]: payloadBytes={detail.VertexSummary.PayloadSizeBytes} vertices={detail.VertexSummary.VertexCount} boundsMin={FormatVector(detail.VertexSummary.BoundsMin)} boundsMax={FormatVector(detail.VertexSummary.BoundsMax)}");

	if (detail.IndexSummary is not null)
		Console.WriteLine($"{detail.IndexSummary.ChunkId}(root)[{detail.GroupIndex}]: payloadBytes={detail.IndexSummary.PayloadSizeBytes} indices={detail.IndexSummary.IndexCount} triangles={detail.IndexSummary.TriangleCount} distinctIndices={detail.IndexSummary.DistinctIndexCount} indexRange={detail.IndexSummary.MinIndex}-{detail.IndexSummary.MaxIndex} degenerateTriangles={detail.IndexSummary.DegenerateTriangleCount}");

	if (detail.DoodadRefSummary is not null)
		Console.WriteLine($"MODR(root)[{detail.GroupIndex}]: payloadBytes={detail.DoodadRefSummary.PayloadSizeBytes} refs={detail.DoodadRefSummary.RefCount} distinctRefs={detail.DoodadRefSummary.DistinctRefCount} refRange={detail.DoodadRefSummary.MinRef}-{detail.DoodadRefSummary.MaxRef} duplicateRefs={detail.DoodadRefSummary.DuplicateRefCount}");

	if (detail.LightRefSummary is not null)
		Console.WriteLine($"MOLR(root)[{detail.GroupIndex}]: payloadBytes={detail.LightRefSummary.PayloadSizeBytes} refs={detail.LightRefSummary.RefCount} distinctRefs={detail.LightRefSummary.DistinctRefCount} refRange={detail.LightRefSummary.MinRef}-{detail.LightRefSummary.MaxRef} duplicateRefs={detail.LightRefSummary.DuplicateRefCount}");

	if (detail.VertexColorSummary is not null)
		Console.WriteLine($"MOCV(root)[{detail.GroupIndex}]: payloadBytes={detail.VertexColorSummary.PrimaryPayloadSizeBytes} primaryColors={detail.VertexColorSummary.PrimaryColorCount} rangeR=[{detail.VertexColorSummary.MinRed}, {detail.VertexColorSummary.MaxRed}] rangeG=[{detail.VertexColorSummary.MinGreen}, {detail.VertexColorSummary.MaxGreen}] rangeB=[{detail.VertexColorSummary.MinBlue}, {detail.VertexColorSummary.MaxBlue}] rangeA=[{detail.VertexColorSummary.MinAlpha}, {detail.VertexColorSummary.MaxAlpha}] avgA={detail.VertexColorSummary.AverageAlpha} extraColorSets={detail.VertexColorSummary.AdditionalColorSetCount} totalExtraColors={detail.VertexColorSummary.TotalAdditionalColorCount} maxExtraColors={detail.VertexColorSummary.MaxAdditionalColorCount}");

	if (detail.UvSummary is not null)
		Console.WriteLine($"MOTV(root)[{detail.GroupIndex}]: payloadBytes={detail.UvSummary.PrimaryPayloadSizeBytes} primaryUv={detail.UvSummary.PrimaryUvCount} rangeU=[{detail.UvSummary.MinU:F3}, {detail.UvSummary.MaxU:F3}] rangeV=[{detail.UvSummary.MinV:F3}, {detail.UvSummary.MaxV:F3}] extraUvSets={detail.UvSummary.AdditionalUvSetCount} totalExtraUv={detail.UvSummary.TotalAdditionalUvCount} maxExtraUv={detail.UvSummary.MaxAdditionalUvCount}");

	if (detail.FaceMaterialSummary is not null)
		Console.WriteLine($"MOPY(root)[{detail.GroupIndex}]: payloadBytes={detail.FaceMaterialSummary.PayloadSizeBytes} entryBytes={detail.FaceMaterialSummary.EntrySizeBytes} faces={detail.FaceMaterialSummary.FaceCount} distinctMaterials={detail.FaceMaterialSummary.DistinctMaterialIdCount} highestMaterialId={detail.FaceMaterialSummary.HighestMaterialId} hiddenFaces={detail.FaceMaterialSummary.HiddenFaceCount} flaggedFaces={detail.FaceMaterialSummary.FlaggedFaceCount}");

	if (detail.BatchSummary is not null)
		Console.WriteLine($"MOBA(root)[{detail.GroupIndex}]: payloadBytes={detail.BatchSummary.PayloadSizeBytes} entries={detail.BatchSummary.EntryCount} hasMaterialIds={detail.BatchSummary.HasMaterialIds} distinctMaterials={detail.BatchSummary.DistinctMaterialIdCount} highestMaterialId={detail.BatchSummary.HighestMaterialId} totalIndexCount={detail.BatchSummary.TotalIndexCount} firstIndexRange={detail.BatchSummary.MinFirstIndex}-{detail.BatchSummary.MaxFirstIndex} maxIndexEnd={detail.BatchSummary.MaxIndexEnd} flaggedBatches={detail.BatchSummary.FlaggedBatchCount}");

	if (detail.BspNodeSummary is not null)
		Console.WriteLine($"MOBN(root)[{detail.GroupIndex}]: payloadBytes={detail.BspNodeSummary.PayloadSizeBytes} nodes={detail.BspNodeSummary.NodeCount} leafNodes={detail.BspNodeSummary.LeafNodeCount} branchNodes={detail.BspNodeSummary.BranchNodeCount} childRefs={detail.BspNodeSummary.ChildReferenceCount} noChildRefs={detail.BspNodeSummary.NoChildReferenceCount} outOfRangeChildRefs={detail.BspNodeSummary.OutOfRangeChildReferenceCount} faceCountRange={detail.BspNodeSummary.MinFaceCount}-{detail.BspNodeSummary.MaxFaceCount} faceStartRange={detail.BspNodeSummary.MinFaceStart}-{detail.BspNodeSummary.MaxFaceStart} maxFaceEnd={detail.BspNodeSummary.MaxFaceEnd} planeDistRange=[{detail.BspNodeSummary.MinPlaneDistance:F3}, {detail.BspNodeSummary.MaxPlaneDistance:F3}]");

	if (detail.BspFaceSummary is not null)
		Console.WriteLine($"MOBR(root)[{detail.GroupIndex}]: payloadBytes={detail.BspFaceSummary.PayloadSizeBytes} refs={detail.BspFaceSummary.RefCount} distinctRefs={detail.BspFaceSummary.DistinctFaceRefCount} refRange={detail.BspFaceSummary.MinFaceRef}-{detail.BspFaceSummary.MaxFaceRef} duplicateRefs={detail.BspFaceSummary.DuplicateFaceRefCount}");

	if (detail.BspFaceRangeSummary is not null)
		Console.WriteLine($"MOBN->MOBR(root)[{detail.GroupIndex}]: nodes={detail.BspFaceRangeSummary.NodeCount} faceRefs={detail.BspFaceRangeSummary.FaceRefCount} zeroFaceNodes={detail.BspFaceRangeSummary.ZeroFaceNodeCount} coveredNodes={detail.BspFaceRangeSummary.CoveredNodeCount} outOfRangeNodes={detail.BspFaceRangeSummary.OutOfRangeNodeCount} maxFaceEnd={detail.BspFaceRangeSummary.MaxFaceEnd}");

	if (detail.LiquidSummary is not null)
		Console.WriteLine($"MLIQ(root)[{detail.GroupIndex}]: payloadBytes={detail.LiquidSummary.PayloadSizeBytes} verts={detail.LiquidSummary.XVertexCount}x{detail.LiquidSummary.YVertexCount} tiles={detail.LiquidSummary.XTileCount}x{detail.LiquidSummary.YTileCount} corner={FormatVector(detail.LiquidSummary.Corner)} materialId={detail.LiquidSummary.MaterialId} heights={detail.LiquidSummary.HeightCount} range=[{detail.LiquidSummary.MinHeight:F2}, {detail.LiquidSummary.MaxHeight:F2}] visibleTiles={detail.LiquidSummary.VisibleTileCount}/{detail.LiquidSummary.TileCount} tileFlags={detail.LiquidSummary.TileFlagByteCount} liquidType={detail.LiquidSummary.LiquidType}");
}

static void PrintWmoMaterialSummary(WmoMaterialSummary summary)
{
	Console.WriteLine($"MOMT: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} entries={summary.EntryCount} distinctShaders={summary.DistinctShaderCount} distinctBlendModes={summary.DistinctBlendModeCount} nonZeroFlags={summary.NonZeroFlagCount} maxTex1Ofs={summary.MaxTexture1Offset} maxTex2Ofs={summary.MaxTexture2Offset} maxTex3Ofs={summary.MaxTexture3Offset}");
}

static void PrintWmoTextureTableSummary(WmoTextureTableSummary summary)
{
	Console.WriteLine($"MOTX: payloadBytes={summary.PayloadSizeBytes} textures={summary.TextureCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset} extensions={summary.DistinctExtensionCount} blpEntries={summary.BlpEntryCount}");
}

static void PrintWmoDoodadNameTableSummary(WmoDoodadNameTableSummary summary)
{
	Console.WriteLine($"MODN: payloadBytes={summary.PayloadSizeBytes} names={summary.NameCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset} extensions={summary.DistinctExtensionCount} mdxEntries={summary.MdxEntryCount} m2Entries={summary.M2EntryCount}");
}

static void PrintWmoDoodadSetSummary(WmoDoodadSetSummary summary)
{
	Console.WriteLine($"MODS: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} nonEmptySets={summary.NonEmptySetCount} longestName={summary.LongestNameLength} totalDoodadRefs={summary.TotalDoodadRefs} maxStartIndex={summary.MaxStartIndex} maxRangeEnd={summary.MaxRangeEnd}");
}

static void PrintWmoDoodadPlacementSummary(WmoDoodadPlacementSummary summary)
{
	Console.WriteLine($"MODD: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctNameIndices={summary.DistinctNameIndexCount} maxNameIndex={summary.MaxNameIndex} scaleRange=[{summary.MinScale:F3}, {summary.MaxScale:F3}] alphaRange=[{summary.MinAlpha}, {summary.MaxAlpha}] boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupNameTableSummary(WmoGroupNameTableSummary summary)
{
	Console.WriteLine($"MOGN: payloadBytes={summary.PayloadSizeBytes} names={summary.NameCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset}");
}

static void PrintWmoSkyboxSummary(WmoSkyboxSummary summary)
{
	Console.WriteLine($"MOSB: payloadBytes={summary.PayloadSizeBytes} skybox={summary.SkyboxName} source=explicit-root-skybox");
}

static void PrintWmoPortalVertexSummary(WmoPortalVertexSummary summary)
{
	Console.WriteLine($"MOPV: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoPortalInfoSummary(WmoPortalInfoSummary summary)
{
	Console.WriteLine($"MOPT: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} maxStartVertex={summary.MaxStartVertex} maxVertexCount={summary.MaxVertexCount} planeDRange=[{summary.MinPlaneD:F3}, {summary.MaxPlaneD:F3}]");
}

static void PrintWmoPortalRefSummary(WmoPortalRefSummary summary)
{
	Console.WriteLine($"MOPR: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctPortals={summary.DistinctPortalIndexCount} maxGroupIndex={summary.MaxGroupIndex} sides(+/-/0)={summary.PositiveSideCount}/{summary.NegativeSideCount}/{summary.NeutralSideCount}");
}

static void PrintWmoPortalVertexRangeSummary(WmoPortalVertexRangeSummary summary)
{
	Console.WriteLine($"MOPT->MOPV: portals={summary.EntryCount} vertices={summary.VertexCount} zeroVertexPortals={summary.ZeroVertexPortalCount} coveredPortals={summary.CoveredPortalCount} outOfRangePortals={summary.OutOfRangePortalCount} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoPortalRefRangeSummary(WmoPortalRefRangeSummary summary)
{
	Console.WriteLine($"MOPR->MOPT: refs={summary.RefCount} portals={summary.PortalCount} coveredRefs={summary.CoveredRefCount} outOfRangeRefs={summary.OutOfRangeRefCount} distinctPortalRefs={summary.DistinctPortalRefCount} maxPortalIndex={summary.MaxPortalIndex}");
}

static void PrintWmoPortalGroupRangeSummary(WmoPortalGroupRangeSummary summary)
{
	Console.WriteLine($"MOPR->MOGI: refs={summary.RefCount} groups={summary.GroupCount} coveredRefs={summary.CoveredRefCount} outOfRangeRefs={summary.OutOfRangeRefCount} distinctGroupRefs={summary.DistinctGroupRefCount} maxGroupIndex={summary.MaxGroupIndex}");
}

static void PrintWmoVisibleVertexSummary(WmoVisibleVertexSummary summary)
{
	Console.WriteLine($"MOVV: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoVisibleBlockSummary(WmoVisibleBlockSummary summary)
{
	Console.WriteLine($"MOVB: payloadBytes={summary.PayloadSizeBytes} blocks={summary.BlockCount} vertexRefs={summary.TotalVertexRefs} blockSizeRange={summary.MinVerticesPerBlock}-{summary.MaxVerticesPerBlock} firstVertexRange={summary.MinFirstVertex}-{summary.MaxFirstVertex} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoVisibleBlockReferenceSummary(WmoVisibleBlockReferenceSummary summary)
{
	Console.WriteLine($"MOVB->MOVV: blocks={summary.BlockCount} vertices={summary.VisibleVertexCount} zeroVertexBlocks={summary.ZeroVertexBlockCount} coveredBlocks={summary.CoveredBlockCount} outOfRangeBlocks={summary.OutOfRangeBlockCount} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoLightSummary(WmoLightSummary summary)
{
	Console.WriteLine($"MOLT: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctTypes={summary.DistinctTypeCount} attenuated={summary.AttenuatedCount} intensityRange=[{summary.MinIntensity:F3}, {summary.MaxIntensity:F3}] attenStartRange=[{summary.MinAttenStart:F3}, {summary.MaxAttenStart:F3}] maxAttenEnd={summary.MaxAttenEnd:F3} headerFlagsWordRange=[0x{summary.MinHeaderFlagsWord:X4}, 0x{summary.MaxHeaderFlagsWord:X4}] headerFlagsWordDistinct={summary.DistinctHeaderFlagsWordCount} headerFlagsWordNonZero={summary.NonZeroHeaderFlagsWordCount} rotationEntries={summary.RotationEntryCount} nonIdentityRotations={summary.NonIdentityRotationCount} rotationLenRange=[{summary.MinRotationLength:F3}, {summary.MaxRotationLength:F3}] boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoFogSummary(WmoFogSummary summary)
{
	Console.WriteLine($"MFOG: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} nonZeroFlags={summary.NonZeroFlagCount} minSmallRadius={summary.MinSmallRadius:F3} maxLargeRadius={summary.MaxLargeRadius:F3} maxFogEnd={summary.MaxFogEnd:F3} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoOpaqueChunkSummary(WmoOpaqueChunkSummary summary)
{
	Console.WriteLine($"{summary.ChunkId}: payloadBytes={summary.PayloadSizeBytes}");
}

static void PrintWmoDoodadSetRangeSummary(WmoDoodadSetRangeSummary summary)
{
	Console.WriteLine($"MODS->MODD: sets={summary.EntryCount} placements={summary.PlacementCount} emptySets={summary.EmptySetCount} coveredSets={summary.FullyCoveredSetCount} outOfRangeSets={summary.OutOfRangeSetCount} maxRangeEnd={summary.MaxRangeEnd}");
}

static void PrintWmoGroupNameReferenceSummary(WmoGroupNameReferenceSummary summary)
{
	Console.WriteLine($"MOGI->MOGN: entries={summary.EntryCount} resolvedNames={summary.ResolvedNameCount} unresolvedNames={summary.UnresolvedNameCount} distinctResolvedNames={summary.DistinctResolvedNameCount} maxNameLength={summary.MaxResolvedNameLength}");
}

static void PrintWmoDoodadNameReferenceSummary(WmoDoodadNameReferenceSummary summary)
{
	Console.WriteLine($"MODD->MODN: entries={summary.EntryCount} resolvedNames={summary.ResolvedNameCount} unresolvedNames={summary.UnresolvedNameCount} distinctResolvedNames={summary.DistinctResolvedNameCount} maxNameLength={summary.MaxResolvedNameLength}");
}

static void PrintWmoGroupSummary(WmoGroupSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect WMO group report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"Header: bytes={summary.HeaderSizeBytes} nameOff={summary.NameOffset} descOff={summary.DescriptiveNameOffset} flags=0x{summary.Flags:X8} ({FormatWmoGroupFlags(summary)}) portals={summary.PortalCount}@{summary.PortalStart} liquid={summary.GroupLiquid}");
	Console.WriteLine($"Geometry: faces={summary.FaceMaterialCount} vertices={summary.VertexCount} indices={summary.IndexCount} normals={summary.NormalCount} primaryUv={summary.PrimaryUvCount} extraUvSets={summary.AdditionalUvSetCount} batches={summary.BatchCount}/{summary.DeclaredBatchCount} vertexColors={summary.VertexColorCount} doodadRefs={summary.DoodadRefCount} lightRefs={summary.LightRefCount} bspNodes={summary.BspNodeCount} bspFaceRefs={summary.BspFaceRefCount} hasLiquid={summary.HasLiquid}");
	Console.WriteLine($"Bounds: min={FormatVector(summary.BoundsMin)} max={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupLiquidSummary(WmoGroupLiquidSummary summary)
{
	Console.WriteLine($"MLIQ: payloadBytes={summary.PayloadSizeBytes} verts={summary.XVertexCount}x{summary.YVertexCount} tiles={summary.XTileCount}x{summary.YTileCount} corner={FormatVector(summary.Corner)} materialId={summary.MaterialId} heights={summary.HeightCount} range=[{summary.MinHeight:F2}, {summary.MaxHeight:F2}] visibleTiles={summary.VisibleTileCount}/{summary.TileCount} tileFlags={summary.TileFlagByteCount} liquidType={summary.LiquidType}");
}

static void PrintWmoGroupBatchSummary(WmoGroupBatchSummary summary)
{
	Console.WriteLine($"MOBA: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} hasMaterialIds={summary.HasMaterialIds} distinctMaterials={summary.DistinctMaterialIdCount} highestMaterialId={summary.HighestMaterialId} totalIndexCount={summary.TotalIndexCount} firstIndexRange={summary.MinFirstIndex}-{summary.MaxFirstIndex} maxIndexEnd={summary.MaxIndexEnd} flaggedBatches={summary.FlaggedBatchCount}");
}

static void PrintWmoGroupFaceMaterialSummary(WmoGroupFaceMaterialSummary summary)
{
	Console.WriteLine($"MOPY: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} faces={summary.FaceCount} distinctMaterials={summary.DistinctMaterialIdCount} highestMaterialId={summary.HighestMaterialId} hiddenFaces={summary.HiddenFaceCount} flaggedFaces={summary.FlaggedFaceCount}");
}

static void PrintWmoGroupUvSummary(WmoGroupUvSummary summary)
{
	Console.WriteLine($"MOTV: payloadBytes={summary.PrimaryPayloadSizeBytes} primaryUv={summary.PrimaryUvCount} rangeU=[{summary.MinU:F3}, {summary.MaxU:F3}] rangeV=[{summary.MinV:F3}, {summary.MaxV:F3}] extraUvSets={summary.AdditionalUvSetCount} totalExtraUv={summary.TotalAdditionalUvCount} maxExtraUv={summary.MaxAdditionalUvCount}");
}

static void PrintWmoGroupVertexColorSummary(WmoGroupVertexColorSummary summary)
{
	Console.WriteLine($"MOCV: payloadBytes={summary.PrimaryPayloadSizeBytes} primaryColors={summary.PrimaryColorCount} rangeR=[{summary.MinRed}, {summary.MaxRed}] rangeG=[{summary.MinGreen}, {summary.MaxGreen}] rangeB=[{summary.MinBlue}, {summary.MaxBlue}] rangeA=[{summary.MinAlpha}, {summary.MaxAlpha}] avgA={summary.AverageAlpha} extraColorSets={summary.AdditionalColorSetCount} totalExtraColors={summary.TotalAdditionalColorCount} maxExtraColors={summary.MaxAdditionalColorCount}");
}

static void PrintWmoGroupDoodadRefSummary(WmoGroupDoodadRefSummary summary)
{
	Console.WriteLine($"MODR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctRefCount} refRange={summary.MinRef}-{summary.MaxRef} duplicateRefs={summary.DuplicateRefCount}");
}

static void PrintWmoGroupLightRefSummary(WmoGroupLightRefSummary summary)
{
	Console.WriteLine($"MOLR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctRefCount} refRange={summary.MinRef}-{summary.MaxRef} duplicateRefs={summary.DuplicateRefCount}");
}

static void PrintWmoGroupIndexSummary(WmoGroupIndexSummary summary)
{
	Console.WriteLine($"{summary.ChunkId}: payloadBytes={summary.PayloadSizeBytes} indices={summary.IndexCount} triangles={summary.TriangleCount} distinctIndices={summary.DistinctIndexCount} indexRange={summary.MinIndex}-{summary.MaxIndex} degenerateTriangles={summary.DegenerateTriangleCount}");
}

static void PrintWmoGroupBspNodeSummary(WmoGroupBspNodeSummary summary)
{
	Console.WriteLine($"MOBN: payloadBytes={summary.PayloadSizeBytes} nodes={summary.NodeCount} leafNodes={summary.LeafNodeCount} branchNodes={summary.BranchNodeCount} childRefs={summary.ChildReferenceCount} noChildRefs={summary.NoChildReferenceCount} outOfRangeChildRefs={summary.OutOfRangeChildReferenceCount} faceCountRange={summary.MinFaceCount}-{summary.MaxFaceCount} faceStartRange={summary.MinFaceStart}-{summary.MaxFaceStart} maxFaceEnd={summary.MaxFaceEnd} planeDistRange=[{summary.MinPlaneDistance:F3}, {summary.MaxPlaneDistance:F3}]");
}

static void PrintWmoGroupBspFaceSummary(WmoGroupBspFaceSummary summary)
{
	Console.WriteLine($"MOBR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctFaceRefCount} refRange={summary.MinFaceRef}-{summary.MaxFaceRef} duplicateRefs={summary.DuplicateFaceRefCount}");
}

static void PrintWmoGroupBspFaceRangeSummary(WmoGroupBspFaceRangeSummary summary)
{
	Console.WriteLine($"MOBN->MOBR: nodes={summary.NodeCount} faceRefs={summary.FaceRefCount} zeroFaceNodes={summary.ZeroFaceNodeCount} coveredNodes={summary.CoveredNodeCount} outOfRangeNodes={summary.OutOfRangeNodeCount} maxFaceEnd={summary.MaxFaceEnd}");
}

static void PrintWmoGroupVertexSummary(WmoGroupVertexSummary summary)
{
	Console.WriteLine($"MOVT: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupNormalSummary(WmoGroupNormalSummary summary)
{
	Console.WriteLine($"MONR: payloadBytes={summary.PayloadSizeBytes} normals={summary.NormalCount} rangeX=[{summary.MinX:F3}, {summary.MaxX:F3}] rangeY=[{summary.MinY:F3}, {summary.MaxY:F3}] rangeZ=[{summary.MinZ:F3}, {summary.MaxZ:F3}] lengthRange=[{summary.MinLength:F3}, {summary.MaxLength:F3}] avgLength={summary.AverageLength:F3} nearUnit={summary.NearUnitCount}");
}

static void PrintBlpSummary(BlpSummary summary)
{
	Console.WriteLine($"BLP: format={summary.Signature} version={summary.Version?.ToString() ?? "n/a"} compression={summary.Compression} alphaBits={summary.AlphaDepthBits} pixelFormat={summary.PixelFormat} mipType={summary.MipMapTypeRaw} size={summary.Width}x{summary.Height} headerBytes={summary.HeaderSizeBytes} paletteBytes={summary.PaletteSizeBytes} jpegHeaderBytes={summary.JpegHeaderSizeBytes} mips={summary.MipMaps.Count} inBoundsMips={summary.InBoundsMipLevelCount} outOfBoundsMips={summary.OutOfBoundsMipLevelCount} maxMipEnd={summary.MaxMipEndOffset}");
	foreach (BlpMipMapEntry mipMap in summary.MipMaps)
		PrintBlpMipMap(mipMap);
}

static void PrintBlpMipMap(BlpMipMapEntry mipMap)
{
	Console.WriteLine($"MIP[{mipMap.Level}]: size={mipMap.Width}x{mipMap.Height} offset={mipMap.Offset} bytes={mipMap.SizeBytes} inBounds={mipMap.IsInBounds}");
}

static void PrintMdxSummary(MdxSummary summary)
{
	string modelName = string.IsNullOrWhiteSpace(summary.ModelName) ? "n/a" : summary.ModelName;
	string blendTime = summary.BlendTime?.ToString() ?? "n/a";
	string boundsMin = summary.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = summary.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string collisionVertices = summary.Collision?.VertexCount.ToString() ?? "0";
	string collisionTriangles = summary.Collision?.TriangleCount.ToString() ?? "0";
	Console.WriteLine($"MDX: signature={summary.Signature} version={summary.Version?.ToString() ?? "n/a"} model={modelName} blendTime={blendTime} chunks={summary.ChunkCount} knownChunks={summary.KnownChunkCount} unknownChunks={summary.UnknownChunkCount} globalSequences={summary.GlobalSequenceCount} sequences={summary.SequenceCount} geosets={summary.GeosetCount} geosetAnimations={summary.GeosetAnimationCount} bones={summary.BoneCount} lights={summary.LightCount} helpers={summary.HelperCount} attachments={summary.AttachmentCount} particleEmitters2={summary.ParticleEmitter2Count} ribbons={summary.RibbonCount} cameras={summary.CameraCount} events={summary.EventCount} hitTestShapes={summary.HitTestShapeCount} collisionVertices={collisionVertices} collisionTriangles={collisionTriangles} pivotPoints={summary.PivotPointCount} textures={summary.TextureCount} replaceableTextures={summary.ReplaceableTextureCount} materials={summary.MaterialCount} materialLayers={summary.MaterialLayerCount} boundsMin={boundsMin} boundsMax={boundsMax}");
	for (int index = 0; index < summary.Chunks.Count; index++)
		PrintMdxChunkSummary(index, summary.Chunks[index]);
	for (int index = 0; index < summary.GlobalSequences.Count; index++)
		PrintMdxGlobalSequenceSummary(summary.GlobalSequences[index]);
	for (int index = 0; index < summary.Sequences.Count; index++)
		PrintMdxSequenceSummary(summary.Sequences[index]);
	for (int index = 0; index < summary.Geosets.Count; index++)
		PrintMdxGeosetSummary(summary.Geosets[index]);
	for (int index = 0; index < summary.GeosetAnimations.Count; index++)
		PrintMdxGeosetAnimationSummary(summary.GeosetAnimations[index]);
	for (int index = 0; index < summary.Bones.Count; index++)
		PrintMdxBoneSummary(summary.Bones[index]);
	for (int index = 0; index < summary.Lights.Count; index++)
		PrintMdxLightSummary(summary.Lights[index]);
	for (int index = 0; index < summary.Helpers.Count; index++)
		PrintMdxHelperSummary(summary.Helpers[index]);
	for (int index = 0; index < summary.Attachments.Count; index++)
		PrintMdxAttachmentSummary(summary.Attachments[index]);
	for (int index = 0; index < summary.ParticleEmitters2.Count; index++)
		PrintMdxParticleEmitter2Summary(summary.ParticleEmitters2[index]);
	for (int index = 0; index < summary.Ribbons.Count; index++)
		PrintMdxRibbonEmitterSummary(summary.Ribbons[index]);
	for (int index = 0; index < summary.Cameras.Count; index++)
		PrintMdxCameraSummary(summary.Cameras[index]);
	for (int index = 0; index < summary.Events.Count; index++)
		PrintMdxEventSummary(summary.Events[index]);
	for (int index = 0; index < summary.HitTestShapes.Count; index++)
		PrintMdxHitTestShapeSummary(summary.HitTestShapes[index]);
	if (summary.Collision is not null)
		PrintMdxCollisionSummary(summary.Collision);
	for (int index = 0; index < summary.PivotPoints.Count; index++)
		PrintMdxPivotPointSummary(summary.PivotPoints[index]);
	for (int index = 0; index < summary.Textures.Count; index++)
		PrintMdxTextureSummary(summary.Textures[index]);
	for (int index = 0; index < summary.Materials.Count; index++)
		PrintMdxMaterialSummary(summary.Materials[index]);
}

static void PrintMdxChunkSummary(int index, MdxChunkSummary chunk)
{
	Console.WriteLine($"CHUNK[{index}]: id={chunk.Id} payloadBytes={chunk.PayloadSizeBytes} headerOffset={chunk.HeaderOffset} dataOffset={chunk.DataOffset} known={chunk.IsKnownChunk}");
}

static void PrintMdxTextureSummary(MdxTextureSummary texture)
{
	string path = string.IsNullOrWhiteSpace(texture.Path) ? "n/a" : texture.Path;
	Console.WriteLine($"TEXS[{texture.Index}]: replaceableId={texture.ReplaceableId} flags=0x{texture.Flags:X8} path={path}");
}

static void PrintMdxGlobalSequenceSummary(MdxGlobalSequenceSummary globalSequence)
{
	Console.WriteLine($"GLBS[{globalSequence.Index}]: duration={globalSequence.Duration}");
}

static void PrintMdxSequenceSummary(MdxSequenceSummary sequence)
{
	string name = string.IsNullOrWhiteSpace(sequence.Name) ? "n/a" : sequence.Name;
	string blendTime = sequence.BlendTime?.ToString() ?? "n/a";
	string boundsMin = sequence.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = sequence.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string boundsRadius = sequence.BoundsRadius?.ToString("F3") ?? "n/a";
	Console.WriteLine($"SEQS[{sequence.Index}]: name={name} time=[{sequence.StartTime}, {sequence.EndTime}] duration={sequence.Duration} moveSpeed={sequence.MoveSpeed:F3} flags=0x{sequence.Flags:X8} frequency={sequence.Frequency:F3} replay=[{sequence.ReplayStart}, {sequence.ReplayEnd}] blendTime={blendTime} boundsMin={boundsMin} boundsMax={boundsMax} boundsRadius={boundsRadius}");
}

static void PrintMdxGeosetSummary(MdxGeosetSummary geoset)
{
	string boundsMin = geoset.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = geoset.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string boundsRadius = geoset.BoundsRadius?.ToString("F3") ?? "n/a";
	Console.WriteLine($"GEOS[{geoset.Index}]: vertices={geoset.VertexCount} normals={geoset.NormalCount} uvSets={geoset.UvSetCount} primaryUvs={geoset.PrimaryUvCount} primitiveTypes={geoset.PrimitiveTypeCount} faceGroups={geoset.FaceGroupCount} indices={geoset.IndexCount} triangles={geoset.TriangleCount} vertexGroups={geoset.VertexGroupCount} matrixGroups={geoset.MatrixGroupCount} matrixIndices={geoset.MatrixIndexCount} boneIndices={geoset.BoneIndexCount} boneWeights={geoset.BoneWeightCount} materialId={geoset.MaterialId} selectionGroup={geoset.SelectionGroup} flags=0x{geoset.Flags:X8} animExtents={geoset.AnimationExtentCount} boundsMin={boundsMin} boundsMax={boundsMax} boundsRadius={boundsRadius}");
}

static void PrintMdxGeosetAnimationSummary(MdxGeosetAnimationSummary geosetAnimation)
{
	Vector3 staticColor = geosetAnimation.StaticColor;
	string geosetId = geosetAnimation.GeosetId == uint.MaxValue ? "none(0xFFFFFFFF)" : geosetAnimation.GeosetId.ToString();
	Console.WriteLine($"GEOA[{geosetAnimation.Index}]: geosetId={geosetId} staticAlpha={geosetAnimation.StaticAlpha:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) flags=0x{geosetAnimation.Flags:X8} usesStaticColor={geosetAnimation.UsesStaticColor} alphaTrack={FormatMdxGeosetAnimationTrack(geosetAnimation.AlphaTrack)} colorTrack={FormatMdxGeosetAnimationTrack(geosetAnimation.ColorTrack)}");
}

static void PrintMdxBoneSummary(MdxBoneSummary bone)
{
	string parentId = bone.HasParent ? bone.ParentId.ToString() : "none(-1)";
	string geosetId = bone.UsesGeoset ? bone.GeosetId.ToString() : "none(0xFFFFFFFF)";
	string geosetAnimationId = bone.UsesGeosetAnimation ? bone.GeosetAnimationId.ToString() : "none(0xFFFFFFFF)";
	Console.WriteLine($"BONE[{bone.Index}]: name={bone.Name} objectId={bone.ObjectId} parentId={parentId} flags=0x{bone.Flags:X8} geosetId={geosetId} geosetAnimId={geosetAnimationId} translationTrack={FormatMdxNodeTrack(bone.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(bone.RotationTrack)} scalingTrack={FormatMdxNodeTrack(bone.ScalingTrack)}");
}

static void PrintMdxLightSummary(MdxLightSummary light)
{
	string parentId = light.HasParent ? light.ParentId.ToString() : "none(-1)";
	Vector3 staticColor = light.StaticColor;
	Vector3 staticAmbientColor = light.StaticAmbientColor;
	Console.WriteLine($"LITE[{light.Index}]: name={light.Name} objectId={light.ObjectId} parentId={parentId} flags=0x{light.Flags:X8} type={FormatMdxLightType(light.LightType)} staticAttenStart={light.StaticAttenuationStart:F3} staticAttenEnd={light.StaticAttenuationEnd:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) staticIntensity={light.StaticIntensity:F3} staticAmbientColor=({staticAmbientColor.X:F3}, {staticAmbientColor.Y:F3}, {staticAmbientColor.Z:F3}) staticAmbientIntensity={light.StaticAmbientIntensity:F3} translationTrack={FormatMdxNodeTrack(light.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(light.RotationTrack)} scalingTrack={FormatMdxNodeTrack(light.ScalingTrack)} attenuationStartTrack={FormatMdxTrack(light.AttenuationStartTrack)} attenuationEndTrack={FormatMdxTrack(light.AttenuationEndTrack)} colorTrack={FormatMdxTrack(light.ColorTrack)} intensityTrack={FormatMdxTrack(light.IntensityTrack)} ambientColorTrack={FormatMdxTrack(light.AmbientColorTrack)} ambientIntensityTrack={FormatMdxTrack(light.AmbientIntensityTrack)} visibilityTrack={FormatMdxVisibilityTrack(light.VisibilityTrack)}");
}

static void PrintMdxHelperSummary(MdxHelperSummary helper)
{
	string parentId = helper.HasParent ? helper.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"HELP[{helper.Index}]: name={helper.Name} objectId={helper.ObjectId} parentId={parentId} flags=0x{helper.Flags:X8} translationTrack={FormatMdxNodeTrack(helper.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(helper.RotationTrack)} scalingTrack={FormatMdxNodeTrack(helper.ScalingTrack)}");
}

static void PrintMdxAttachmentSummary(MdxAttachmentSummary attachment)
{
	string parentId = attachment.HasParent ? attachment.ParentId.ToString() : "none(-1)";
	string path = string.IsNullOrWhiteSpace(attachment.Path) ? "n/a" : attachment.Path;
	Console.WriteLine($"ATCH[{attachment.Index}]: name={attachment.Name} objectId={attachment.ObjectId} parentId={parentId} flags=0x{attachment.Flags:X8} attachmentId={attachment.AttachmentId} path={path} translationTrack={FormatMdxNodeTrack(attachment.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(attachment.RotationTrack)} scalingTrack={FormatMdxNodeTrack(attachment.ScalingTrack)} visibilityTrack={FormatMdxVisibilityTrack(attachment.VisibilityTrack)}");
}

static void PrintMdxParticleEmitter2Summary(MdxParticleEmitter2Summary particleEmitter)
{
	string parentId = particleEmitter.HasParent ? particleEmitter.ParentId.ToString() : "none(-1)";
	string geometryModel = string.IsNullOrWhiteSpace(particleEmitter.GeometryModel) ? "n/a" : particleEmitter.GeometryModel;
	string recursionModel = string.IsNullOrWhiteSpace(particleEmitter.RecursionModel) ? "n/a" : particleEmitter.RecursionModel;
	Vector3 startColor = particleEmitter.StartColor;
	Vector3 middleColor = particleEmitter.MiddleColor;
	Vector3 endColor = particleEmitter.EndColor;
	Console.WriteLine($"PRE2[{particleEmitter.Index}]: name={particleEmitter.Name} objectId={particleEmitter.ObjectId} parentId={parentId} flags=0x{particleEmitter.Flags:X8} emitterType={particleEmitter.EmitterType} staticSpeed={particleEmitter.StaticSpeed:F3} staticVariation={particleEmitter.StaticVariation:F3} staticLatitude={particleEmitter.StaticLatitude:F3} staticLongitude={particleEmitter.StaticLongitude:F3} staticGravity={particleEmitter.StaticGravity:F3} staticZSource={particleEmitter.StaticZSource:F3} staticLife={particleEmitter.StaticLife:F3} staticEmissionRate={particleEmitter.StaticEmissionRate:F3} staticLength={particleEmitter.StaticLength:F3} staticWidth={particleEmitter.StaticWidth:F3} rows={particleEmitter.Rows} cols={particleEmitter.Columns} particleType={particleEmitter.ParticleType} tailLength={particleEmitter.TailLength:F3} middleTime={particleEmitter.MiddleTime:F3} startColor=({startColor.X:F3}, {startColor.Y:F3}, {startColor.Z:F3}) middleColor=({middleColor.X:F3}, {middleColor.Y:F3}, {middleColor.Z:F3}) endColor=({endColor.X:F3}, {endColor.Y:F3}, {endColor.Z:F3}) alphas=[{particleEmitter.StartAlpha},{particleEmitter.MiddleAlpha},{particleEmitter.EndAlpha}] scales=[{particleEmitter.StartScale:F3},{particleEmitter.MiddleScale:F3},{particleEmitter.EndScale:F3}] blendMode={particleEmitter.BlendMode} textureId={particleEmitter.TextureId} priorityPlane={particleEmitter.PriorityPlane} replaceableId={particleEmitter.ReplaceableId} geometryModel={geometryModel} recursionModel={recursionModel} splineCount={particleEmitter.SplineCount} squirts={particleEmitter.Squirts} translationTrack={FormatMdxNodeTrack(particleEmitter.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(particleEmitter.RotationTrack)} scalingTrack={FormatMdxNodeTrack(particleEmitter.ScalingTrack)} visibilityTrack={FormatMdxVisibilityTrack(particleEmitter.VisibilityTrack)} speedTrack={FormatMdxTrack(particleEmitter.SpeedTrack)} variationTrack={FormatMdxTrack(particleEmitter.VariationTrack)} latitudeTrack={FormatMdxTrack(particleEmitter.LatitudeTrack)} longitudeTrack={FormatMdxTrack(particleEmitter.LongitudeTrack)} gravityTrack={FormatMdxTrack(particleEmitter.GravityTrack)} lifeTrack={FormatMdxTrack(particleEmitter.LifeTrack)} emissionRateTrack={FormatMdxTrack(particleEmitter.EmissionRateTrack)} widthTrack={FormatMdxTrack(particleEmitter.WidthTrack)} lengthTrack={FormatMdxTrack(particleEmitter.LengthTrack)} zSourceTrack={FormatMdxTrack(particleEmitter.ZSourceTrack)}");
}

static void PrintMdxRibbonEmitterSummary(MdxRibbonEmitterSummary ribbon)
{
	string parentId = ribbon.HasParent ? ribbon.ParentId.ToString() : "none(-1)";
	Vector3 staticColor = ribbon.StaticColor;
	Console.WriteLine($"RIBB[{ribbon.Index}]: name={ribbon.Name} objectId={ribbon.ObjectId} parentId={parentId} flags=0x{ribbon.Flags:X8} staticHeightAbove={ribbon.StaticHeightAbove:F3} staticHeightBelow={ribbon.StaticHeightBelow:F3} staticAlpha={ribbon.StaticAlpha:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) edgeLifetime={ribbon.EdgeLifetime:F3} staticTextureSlot={ribbon.StaticTextureSlot} edgesPerSecond={ribbon.EdgesPerSecond} textureRows={ribbon.TextureRows} textureCols={ribbon.TextureColumns} materialId={ribbon.MaterialId} gravity={ribbon.Gravity:F3} translationTrack={FormatMdxNodeTrack(ribbon.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(ribbon.RotationTrack)} scalingTrack={FormatMdxNodeTrack(ribbon.ScalingTrack)} heightAboveTrack={FormatMdxTrack(ribbon.HeightAboveTrack)} heightBelowTrack={FormatMdxTrack(ribbon.HeightBelowTrack)} alphaTrack={FormatMdxTrack(ribbon.AlphaTrack)} colorTrack={FormatMdxTrack(ribbon.ColorTrack)} textureSlotTrack={FormatMdxTrack(ribbon.TextureSlotTrack)} visibilityTrack={FormatMdxVisibilityTrack(ribbon.VisibilityTrack)}");
}

static void PrintMdxCameraSummary(MdxCameraSummary camera)
{
	Vector3 pivotPoint = camera.PivotPoint;
	Vector3 targetPivotPoint = camera.TargetPivotPoint;
	Console.WriteLine($"CAMS[{camera.Index}]: name={camera.Name} pivot=({pivotPoint.X:F3}, {pivotPoint.Y:F3}, {pivotPoint.Z:F3}) fieldOfView={camera.FieldOfView:F6} farClip={camera.FarClip:F6} nearClip={camera.NearClip:F6} targetPivot=({targetPivotPoint.X:F3}, {targetPivotPoint.Y:F3}, {targetPivotPoint.Z:F3}) positionTrack={FormatMdxTrack(camera.PositionTrack)} rollTrack={FormatMdxTrack(camera.RollTrack)} visibilityTrack={FormatMdxVisibilityTrack(camera.VisibilityTrack)} targetPositionTrack={FormatMdxTrack(camera.TargetPositionTrack)}");
}

static void PrintMdxEventSummary(MdxEventSummary evnt)
{
	string parentId = evnt.HasParent ? evnt.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"EVTS[{evnt.Index}]: name={evnt.Name} objectId={evnt.ObjectId} parentId={parentId} flags=0x{evnt.Flags:X8} translationTrack={FormatMdxNodeTrack(evnt.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(evnt.RotationTrack)} scalingTrack={FormatMdxNodeTrack(evnt.ScalingTrack)} eventTrack={FormatMdxEventTrack(evnt.EventTrack)}");
}

static void PrintMdxHitTestShapeSummary(MdxHitTestShapeSummary shape)
{
	string parentId = shape.HasParent ? shape.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"HTST[{shape.Index}]: name={shape.Name} objectId={shape.ObjectId} parentId={parentId} flags=0x{shape.Flags:X8} shapeType={FormatMdxGeometryShapeType(shape.ShapeType)} shape={FormatMdxHitTestShapeGeometry(shape)} translationTrack={FormatMdxNodeTrack(shape.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(shape.RotationTrack)} scalingTrack={FormatMdxNodeTrack(shape.ScalingTrack)}");
}

static void PrintMdxCollisionSummary(MdxCollisionSummary collision)
{
	string boundsMin = collision.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = collision.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	Console.WriteLine($"CLID: vertices={collision.VertexCount} triIndices={collision.TriangleIndexCount} triangles={collision.TriangleCount} facetNormals={collision.FacetNormalCount} maxIndex={collision.MaxTriangleIndex} boundsMin={boundsMin} boundsMax={boundsMax}");
}

static void PrintMdxPivotPointSummary(MdxPivotPointSummary pivotPoint)
{
	Vector3 position = pivotPoint.Position;
	Console.WriteLine($"PIVT[{pivotPoint.Index}]: position=({position.X:F3}, {position.Y:F3}, {position.Z:F3})");
}

static void PrintMdxMaterialSummary(MdxMaterialSummary material)
{
	Console.WriteLine($"MTLS[{material.Index}]: priorityPlane={material.PriorityPlane} layers={material.LayerCount}");
	for (int layerIndex = 0; layerIndex < material.Layers.Count; layerIndex++)
		PrintMdxMaterialLayerSummary(material.Index, material.Layers[layerIndex]);
}

static void PrintMdxMaterialLayerSummary(int materialIndex, MdxMaterialLayerSummary layer)
{
	Console.WriteLine($"MTLS[{materialIndex}].LAYER[{layer.Index}]: blendMode={FormatMdxBlendMode(layer.BlendMode)} flags=0x{layer.Flags:X8} textureId={layer.TextureId} transformId={layer.TransformId} coordId={layer.CoordId} staticAlpha={layer.StaticAlpha:F3}");
}

static string FormatMdxBlendMode(uint blendMode)
{
	return blendMode switch
	{
		0 => "Load(0)",
		1 => "Transparent(1)",
		2 => "Blend(2)",
		3 => "Add(3)",
		4 => "AddAlpha(4)",
		5 => "Modulate(5)",
		6 => "Modulate2X(6)",
		_ => blendMode.ToString()
	};
}

static string FormatMdxGeosetAnimationTrack(MdxGeosetAnimationTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxNodeTrack(MdxNodeTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxTrack(MdxTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxVisibilityTrack(MdxVisibilityTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxEventTrack(MdxEventTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxGeometryShapeType(MdxGeometryShapeType shapeType)
{
	return shapeType switch
	{
		MdxGeometryShapeType.Box => "Box(0)",
		MdxGeometryShapeType.Cylinder => "Cylinder(1)",
		MdxGeometryShapeType.Sphere => "Sphere(2)",
		MdxGeometryShapeType.Plane => "Plane(3)",
		_ => ((byte)shapeType).ToString(),
	};
}

static string FormatMdxLightType(MdxLightType lightType)
{
	return lightType switch
	{
		MdxLightType.Omni => "Omni(0)",
		MdxLightType.Direct => "Direct(1)",
		MdxLightType.Ambient => "Ambient(2)",
		_ => ((uint)lightType).ToString(),
	};
}

static string FormatMdxHitTestShapeGeometry(MdxHitTestShapeSummary shape)
{
	return shape.ShapeType switch
	{
		MdxGeometryShapeType.Box when shape.Minimum is Vector3 minimum && shape.Maximum is Vector3 maximum
			=> $"boxMin=({minimum.X:F3}, {minimum.Y:F3}, {minimum.Z:F3}) boxMax=({maximum.X:F3}, {maximum.Y:F3}, {maximum.Z:F3})",
		MdxGeometryShapeType.Cylinder when shape.BasePoint is Vector3 basePoint && shape.Height is float height && shape.Radius is float radius
			=> $"base=({basePoint.X:F3}, {basePoint.Y:F3}, {basePoint.Z:F3}) height={height:F6} radius={radius:F6}",
		MdxGeometryShapeType.Sphere when shape.Center is Vector3 center && shape.Radius is float sphereRadius
			=> $"center=({center.X:F3}, {center.Y:F3}, {center.Z:F3}) radius={sphereRadius:F6}",
		MdxGeometryShapeType.Plane when shape.Length is float length && shape.Width is float width
			=> $"length={length:F6} width={width:F6}",
		_ => "n/a",
	};
}

static string FormatMdxInterpolation(uint interpolationType)
{
	return interpolationType switch
	{
		0 => "None(0)",
		1 => "Linear(1)",
		2 => "Hermite(2)",
		3 => "Bezier(3)",
		4 => "Bezier2(4)",
		_ => interpolationType.ToString()
	};
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Inspect");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-inspect audio alpha-area --archive-root <game|data dir> [--build <version>] [--area-id <id>] [--search <text>] [--limit <n>] [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect archive build-listfile-cache --archive-root <game|data dir> --cache-key <client-build> [--listfile <listfile.txt>] [--cache-dir <directory>]");
	Console.WriteLine("  wowviewer-inspect blp inspect --input <file.blp>");
	Console.WriteLine("  wowviewer-inspect blp inspect --archive-root <game|data dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect mdx inspect --input <file.mdx>");
	Console.WriteLine("  wowviewer-inspect mdx inspect --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect m2 inspect --input <file.m2|file.mdx|file.mdl> [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--golden-output <json>|-g <json>] [--render-frame-output <json>] [--visual-output <bmp>] [--static-visual-output <bmp>]");
	Console.WriteLine("  wowviewer-inspect m2 inspect --archive-root <game|data dir> --virtual-path <path/to/file.m2|file.mdx|file.mdl> [--listfile <listfile.txt>] [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--golden-output <json>|-g <json>] [--render-frame-output <json>] [--visual-output <bmp>] [--static-visual-output <bmp>]");
	Console.WriteLine("  wowviewer-inspect mdx export-json --input <file.mdx> [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  wowviewer-inspect mdx export-json --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>] [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  wowviewer-inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --input <file|directory> [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  wowviewer-inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --archive-root <game|data dir> [--listfile <listfile.txt>] [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  wowviewer-inspect mdx-render --input <file.mdx> --output <bmp> [--width <n>] [--height <n>] [--sequence <n>] [--time <ms>] [--bones]");
	Console.WriteLine("  wowviewer-inspect mdx-render --archive-root <dir> --virtual-path <path/to/file.mdx> --output <bmp> [--width <n>] [--height <n>] [--sequence <n>] [--time <ms>] [--bones]");
	Console.WriteLine("  wowviewer-inspect map inspect --input <file.wdt|file.adt|file.error>");
	Console.WriteLine("  wowviewer-inspect lit inspect --input <lights.lit>");
	Console.WriteLine("  wowviewer-inspect lit inspect --archive-root <game|data dir> --virtual-path <world/.../lights.lit> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect wmo inspect --input <file.wmo> [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect pm4 inspect --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mshd --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 audit --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit-directory --input <directory>");
	Console.WriteLine("  wowviewer-inspect pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
}

static void ShowAudioUsage()
{
	Console.WriteLine("Audio commands:");
	Console.WriteLine("  audio alpha-area --archive-root <game|data dir> [--build <version>] [--area-id <id>] [--search <text>] [--limit <n>] [--listfile <listfile.txt>]");
}

static void ShowArchiveUsage()
{
	Console.WriteLine("Archive commands:");
	Console.WriteLine("  archive build-listfile-cache --archive-root <game|data dir> --cache-key <client-build> [--listfile <listfile.txt>] [--cache-dir <directory>]");
}

static void ShowBlpUsage()
{
	Console.WriteLine("BLP commands:");
	Console.WriteLine("  blp inspect --input <file.blp>");
	Console.WriteLine("  map terrain-patch-report --input <terrain_patch_report.json> [--output <summary.json>]");
	Console.WriteLine("  blp inspect --archive-root <game|data dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]");
}

static TerrainPatchReportSummary BuildTerrainPatchReportSummary(string inputPath, IReadOnlyList<TerrainPatchReportEntry> entries)
{
	int patchedCount = entries.Count(static entry => entry.Patched);
	int copiedCount = entries.Count(static entry => entry.CopiedFromInput);
	int failedCount = entries.Count(static entry => !string.IsNullOrWhiteSpace(entry.Error));
	int mccvCount = entries.Count(static entry => !string.IsNullOrWhiteSpace(entry.OutputMccvPath));
	int guideCount = entries.Count(static entry => !string.IsNullOrWhiteSpace(entry.OutputGuideTexturePath));
	int textureMetadataCount = entries.Count(static entry => !string.IsNullOrWhiteSpace(entry.OutputTextureMetadataPath));
	int tilesetIndexCount = entries.Count(static entry => !string.IsNullOrWhiteSpace(entry.OutputTilesetIndexPath));
	int textureMaskFileCount = entries.Sum(static entry => entry.OutputTextureMaskPaths?.Count ?? 0);
	int chunkAuditCount = entries.Count(static entry => entry.ChunkChangeAudit is not null);
	int seamAuditCount = entries.Count(static entry => entry.SeamAudit is not null);

	IReadOnlyList<TerrainPatchStatusCount> statusCounts = entries
		.GroupBy(static entry => string.IsNullOrWhiteSpace(entry.TextureSupervisionStatus) ? "unspecified" : entry.TextureSupervisionStatus!, StringComparer.OrdinalIgnoreCase)
		.OrderByDescending(static group => group.Count())
		.ThenBy(static group => group.Key, StringComparer.OrdinalIgnoreCase)
		.Select(static group => new TerrainPatchStatusCount(group.Key, group.Count()))
		.ToArray();

	IReadOnlyList<TerrainPatchMissingExample> missingTextureExamples = entries
		.Where(static entry => !string.IsNullOrWhiteSpace(entry.TileName)
			&& !string.IsNullOrWhiteSpace(entry.TextureSupervisionStatus)
			&& !string.Equals(entry.TextureSupervisionStatus, "exported", StringComparison.OrdinalIgnoreCase)
			&& !string.Equals(entry.TextureSupervisionStatus, "exported-partial", StringComparison.OrdinalIgnoreCase))
		.Take(12)
		.Select(static entry => new TerrainPatchMissingExample(entry.TileName!, entry.TextureSupervisionStatus!))
		.ToArray();

	return new TerrainPatchReportSummary(
		inputPath,
		entries.Count,
		patchedCount,
		copiedCount,
		failedCount,
		mccvCount,
		guideCount,
		textureMetadataCount,
		tilesetIndexCount,
		textureMaskFileCount,
		chunkAuditCount,
		seamAuditCount,
		statusCounts,
		missingTextureExamples);
}

static void PrintTerrainPatchReportSummary(TerrainPatchReportSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect terrain patch report");
	Console.WriteLine($"Input: {summary.InputPath}");
	Console.WriteLine($"Entries: total={summary.EntryCount} patched={summary.PatchedCount} copied={summary.CopiedCount} failed={summary.FailedCount}");
	Console.WriteLine($"Guidance artifacts: mccv={summary.MccvExportCount} guideTextures={summary.GuideTextureCount}");
	Console.WriteLine($"Texture supervision: metadata={summary.TextureMetadataCount} tilesetIndex={summary.TilesetIndexCount} maskFiles={summary.TextureMaskFileCount}");
	Console.WriteLine($"Proof artifacts: chunkAudits={summary.ChunkAuditCount} seamAudits={summary.SeamAuditCount}");
	Console.WriteLine();
	Console.WriteLine("Texture supervision status counts:");
	foreach (TerrainPatchStatusCount status in summary.TextureSupervisionStatuses)
		Console.WriteLine($"  {status.Status}: {status.Count}");

	if (summary.MissingTextureExamples.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Texture supervision gaps:");
		foreach (TerrainPatchMissingExample example in summary.MissingTextureExamples)
			Console.WriteLine($"  {example.TileName}: {example.Status}");
	}
}

static void ShowLitUsage()
{
	Console.WriteLine("LIT commands:");
	Console.WriteLine("  lit inspect --input <lights.lit> [--sample-position <x,y,z>]");
	Console.WriteLine("  lit inspect --archive-root <game|data dir> --virtual-path <world/.../lights.lit> [--listfile <listfile.txt>] [--sample-position <x,y,z>]");
}

static void ShowM2Usage()
{
	Console.WriteLine("M2 commands:");
	Console.WriteLine("  m2 inspect --input <file.m2|file.mdx|file.mdl> [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--golden-output <json>|-g <json>] [--render-frame-output <json>] [--visual-output <bmp>] [--static-visual-output <bmp>]");
	Console.WriteLine("  m2 inspect --archive-root <game|data dir> --virtual-path <path/to/file.m2|file.mdx|file.mdl> [--listfile <listfile.txt>] [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--golden-output <json>|-g <json>] [--render-frame-output <json>] [--visual-output <bmp>] [--static-visual-output <bmp>]");
}

static void ShowMdxUsage()
{
	Console.WriteLine("MDX commands:");
	Console.WriteLine("  mdx inspect --input <file.mdx>");
	Console.WriteLine("  mdx inspect --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]");
	Console.WriteLine("  mdx export-json --input <file.mdx> [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  mdx export-json --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>] [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  mdx skin-diagnostics --input <file.mdx>");
	Console.WriteLine("  mdx skin-diagnostics --archive-root <game|data dir> --virtual-path <path/to/file.mdx>");
	Console.WriteLine("  mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --input <file|directory> [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --archive-root <game|data dir> [--listfile <listfile.txt>] [--path-filter <text>] [--limit <n>]");
}

static void ShowWmoUsage()
{
	Console.WriteLine("WMO commands:");
	Console.WriteLine("  wmo inspect --input <file.wmo> [--dump-lights] [--flag-correlation]");
	Console.WriteLine("  wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights] [--flag-correlation]");
}

static void ShowMapUsage()
{
	Console.WriteLine("Map commands:");
	Console.WriteLine("  map inspect --input <file.wdt|file.adt|file.error> [--dump-tex-chunks]");
	Console.WriteLine("  map uniqueid-filter --input <report.json> [--min-uniqueid <n>] [--max-uniqueid <n>] [--build <label[,label...]>] [--kind all|m2|wmo] [--invert] [--output <report.json>]");
	Console.WriteLine("  map uniqueid-report --input <file.wdt|file.adt|directory> [--input <second-source> ...] [--build <label>] [--output <report.json>]");
}

static void ShowPm4Usage()
{
	Console.WriteLine("PM4 commands:");
	Console.WriteLine("  pm4 inspect --input <file.pm4>");
	Console.WriteLine("  pm4 match --input <file.pm4> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--listfile <listfile.txt>] [--max-matches <n>] [--search-range <units>] [--output <report.json>] [--object-output-dir <directory>]");
	Console.WriteLine("  pm4 hierarchy --input <file.pm4> [--output <report.json>]");
	Console.WriteLine("  pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mshd --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 audit --input <file.pm4>");
	Console.WriteLine("  pm4 audit-directory --input <directory>");
	Console.WriteLine("  pm4 cross-tile --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
}

static void RunPm4CrossTile(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4CrossTileReport report = Pm4ResearchCrossTileAnalyzer.AnalyzeDirectory(input);

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4CrossTileReport(report);
}

static void PrintPm4CrossTileReport(Pm4CrossTileReport report)
{
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.TotalFiles} total, {report.NonEmptyFiles} non-empty");
	Console.WriteLine($"Distinct CK24 values: {report.TotalDistinctCk24}");
	Console.WriteLine($"Cross-tile CK24 (span 2+ tiles): {report.CrossTileCk24Count} ({report.CrossTileCk24Count * 100.0 / Math.Max(1, report.TotalDistinctCk24):F1}%)");
	Console.WriteLine($"Cross-tile CK24 spanning multiple Field04 buckets: {report.CrossTileCk24MultiField04Count}");
	Console.WriteLine();

	Console.WriteLine("Top cross-tile CK24 objects:");
	Console.WriteLine("  CK24     Type  ObjectId  Tiles  F04s  Surfaces  MSCNrefs");
	foreach (Pm4CrossTileCk24Record row in report.TopCrossTileCk24.Where(r => r.TileCoordinates.Count > 1).Take(25))
	{
		Console.WriteLine($"  0x{row.Ck24:X6}  0x{row.Ck24Type:X2}   {row.Ck24ObjectId,-8} {row.TileCoordinates.Count,-5}  {row.DistinctField04Count,-4}  {row.TotalSurfaces,-8}  {row.TotalMscnRefs,-8}");
	}

	Console.WriteLine();
	Console.WriteLine("Tile summary (all non-empty tiles):");
	Console.WriteLine("  Tile     CK24grps  Surfaces  MSLK     MSCN    MPRL");
	foreach (Pm4CrossTileTileSummary row in report.TileSummaries.Take(40))
	{
		Console.WriteLine($"  {row.TileCoordinate,-8} {row.Ck24GroupCount,-9} {row.SurfaceCount,-9} {row.MslkCount,-8} {row.MscnCount,-7} {row.MprlCount,-7}");
	}

	Console.WriteLine();
	foreach (string note in report.Notes)
		Console.WriteLine($"[*] {note}");
}

sealed record TerrainPatchReportEntry(
	string? SummaryPath,
	string? TileName,
	string? OutputAdtPath,
	bool Patched,
	string? OutputGlbPath,
	string? OutputMccvPath,
	string? OutputGuideTexturePath,
	string? TextureSupervisionStatus,
	string? OutputTextureMetadataPath,
	string? OutputTilesetIndexPath,
	IReadOnlyList<string>? OutputTextureMaskPaths,
	string? Error,
	bool CopiedFromInput,
	object? ChunkChangeAudit,
	object? SeamAudit);

sealed record TerrainPatchReportSummary(
	string InputPath,
	int EntryCount,
	int PatchedCount,
	int CopiedCount,
	int FailedCount,
	int MccvExportCount,
	int GuideTextureCount,
	int TextureMetadataCount,
	int TilesetIndexCount,
	int TextureMaskFileCount,
	int ChunkAuditCount,
	int SeamAuditCount,
	IReadOnlyList<TerrainPatchStatusCount> TextureSupervisionStatuses,
	IReadOnlyList<TerrainPatchMissingExample> MissingTextureExamples);

sealed record TerrainPatchStatusCount(string Status, int Count);

sealed record TerrainPatchMissingExample(string TileName, string Status);
