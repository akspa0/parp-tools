using System.Diagnostics;
using System.Globalization;
using System.Buffers.Binary;
using System.Numerics;
using System.Security.Cryptography;
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
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Wmo;
using WowViewer.Tools.Shared.Pm4Matching;

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
	case "assets":
		WowViewer.Tool.Inspect.AssetReferenceCommandSupport.Run(tail);
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
	case "light":
		RunLight(tail);
		break;
	case "pm4":
		RunPm4(tail);
		break;
	case "pd4":
		RunPd4(tail);
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
		case "scan-wmo-containers":
			WowViewer.Tool.Inspect.ArchiveWmoContainerAuditCommandSupport.Run(tail);
			break;
		case "read-text":
			WowViewer.Tool.Inspect.ArchiveReadTextCommandSupport.Run(tail);
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
	Console.WriteLine($"M2: requestedPath={model.Identity.RequestedPath} canonicalPath={model.Identity.CanonicalModelPath} canonicalized={model.Identity.WasCanonicalized} signature={model.Signature} version=0x{model.Version:X} model={modelName} boundsMin={FormatVector(model.BoundsMin)} boundsMax={FormatVector(model.BoundsMax)} boundsRadius={model.BoundsRadius:F3} skinProfiles={model.ViewCount} bones={model.BoneCount} cameras={model.CameraCount} colors={model.ColorCount} transparencyDefs={model.TextureWeightCount} textureTransforms={model.TextureTransformCount} lights={model.LightCount} ribbons={model.RibbonCount} particles={model.ParticleCount}");
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
		case "generate-blank":
			RunMapGenerateBlank(tail);
			break;
		case "patch-blank":
			RunMapPatchBlank(tail);
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
		case "profile":
		case "sample":
			RunLitProfile(tail);
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

static void RunLight(string[] args)
{
	if (args.Length == 0)
	{
		ShowLightUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();
	if (command != "profile")
	{
		Console.Error.WriteLine($"Unknown light command '{command}'.");
		ShowLightUsage();
		Environment.ExitCode = 1;
		return;
	}

	if (!TryBuildArchiveBootstrapOptions(tail, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	try
	{
		LightDbcProfileCommand.Execute(tail, archiveBootstrapOptions);
	}
	catch (Exception ex) when (ex is ArgumentException
		or IOException
		or InvalidDataException
		or UnauthorizedAccessException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunLitProfile(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? output = GetOption(args, "--output", "-o");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (!TryParseNormalizedGameTimes(args, out IReadOnlyList<float> normalizedTimes, out string? timeError))
	{
		Console.Error.WriteLine($"Error: {timeError}");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input)
		&& (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <lights.lit> or --archive-root <game|data dir> with --virtual-path <world/.../lights.lit>.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		byte[] bytes;
		string sourceKind;
		string sourceLabel;
		string? sourcePath = null;
		string? resolvedArchiveRoot = null;
		string? resolvedVirtualPath = null;

		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			resolvedArchiveRoot = Path.GetFullPath(archiveRoot);
			resolvedVirtualPath = virtualPath;
			sourceKind = "archive_virtual_file";
			sourceLabel = virtualPath;
			bytes = ArchiveVirtualFileReader.ReadVirtualFile(
				virtualPath,
				[archiveRoot],
				archiveBootstrapOptions);
		}
		else if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
		{
			sourcePath = Path.GetFullPath(input);
			sourceKind = "local_file";
			sourceLabel = sourcePath;
			bytes = File.ReadAllBytes(sourcePath);
		}
		else
		{
			sourceKind = "companion_mpq_fallback";
			sourceLabel = input!;
			sourcePath = File.Exists(input) ? Path.GetFullPath(input) : null;
			bytes = AlphaArchiveReader.ReadWithMpqFallback(input!)
				?? throw new FileNotFoundException(
					$"Could not read profile input '{input}' directly or from a companion MPQ archive.",
					input);
		}

		string sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
		LitFileProfile profile;
		using (var stream = new MemoryStream(bytes, writable: false))
			profile = LitProfileReader.Read(stream, sourceLabel);

		var source = new LitProfileSourceEvidence(
			sourceKind,
			sourceLabel,
			sourcePath,
			resolvedArchiveRoot,
			resolvedVirtualPath,
			sha256);
		LitProfileArtifact artifact = LitProfileCommandSupport.Build(profile, source, normalizedTimes);
		var jsonOptions = new JsonSerializerOptions
		{
			WriteIndented = true,
			PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
		};
		string json = JsonSerializer.Serialize(artifact, jsonOptions);

		if (string.IsNullOrWhiteSpace(output) || output == "-")
		{
			Console.WriteLine(json);
			return;
		}

		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
	}
	catch (Exception exception) when (
		exception is IOException
		or UnauthorizedAccessException
		or InvalidDataException
		or ArgumentException)
	{
		Console.Error.WriteLine($"Error: {exception.Message}");
		Environment.ExitCode = 1;
	}
}

static bool TryParseNormalizedGameTimes(
	string[] args,
	out IReadOnlyList<float> normalizedTimes,
	out string? error)
{
	IReadOnlyList<string> rawValues = GetOptionValues(args, "--game-time", "-t");
	if (rawValues.Count == 0)
	{
		normalizedTimes = [0.35f];
		error = null;
		return true;
	}

	var parsed = new List<float>();
	foreach (string rawValue in rawValues)
	{
		foreach (string token in rawValue.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
		{
			if (!float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
				|| !float.IsFinite(value)
				|| value is < 0f or > 1f)
			{
				normalizedTimes = [];
				error = $"--game-time requires normalized values in 0..1; received '{token}'.";
				return false;
			}

			parsed.Add(value);
		}
	}

	if (parsed.Count == 0)
	{
		normalizedTimes = [];
		error = "--game-time did not contain any values.";
		return false;
	}

	normalizedTimes = parsed;
	error = null;
	return true;
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

static void RunMapGenerateBlank(string[] args)
{
	string? mapName = GetOption(args, "--map-name", "-m");
	string? formatText = GetOption(args, "--format", "-f");
	string? tileXText = GetOption(args, "--tile-x", "-x");
	string? tileYText = GetOption(args, "--tile-y", "-y");
	string? outputDir = GetOption(args, "--output-dir", "-o");
	string? textureName = GetOption(args, "--texture", "-t");

	string format = formatText ?? "lk";
	if (format != "lk" && format != "alpha")
	{
		Console.Error.WriteLine("Error: --format must be 'lk' or 'alpha'.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(tileXText) || string.IsNullOrWhiteSpace(tileYText))
	{
		Console.Error.WriteLine("Error: --tile-x and --tile-y are required.");
		Console.Error.WriteLine("Usage: map generate-blank --tile-x <n> --tile-y <n> [--map-name <name>] [--format lk|alpha] [--texture <path>] [--output-dir <dir>]");
		Environment.ExitCode = 1;
		return;
	}

	if (!int.TryParse(tileXText, out int tileX) || tileX < 0 || tileX >= 64)
	{
		Console.Error.WriteLine("Error: --tile-x must be an integer in [0, 63].");
		Environment.ExitCode = 1;
		return;
	}

	if (!int.TryParse(tileYText, out int tileY) || tileY < 0 || tileY >= 64)
	{
		Console.Error.WriteLine("Error: --tile-y must be an integer in [0, 63].");
		Environment.ExitCode = 1;
		return;
	}

	string resolvedMapName = mapName ?? "testing";
	string resolvedOutputDir = outputDir ?? ".";
	string resolvedTexture = textureName ?? "tileset\\ocean\\westfallseafloor.blp";
	string mapDir = Path.Combine(resolvedOutputDir, "World", "Maps", resolvedMapName);
	Directory.CreateDirectory(mapDir);

	if (format == "alpha")
	{
		AlphaTileData tileData = BlankAdtFactory.CreateBlankAlphaTile(tileX, tileY, resolvedTexture);
		var tiles = new Dictionary<(int, int), AlphaTileData> { [(tileX, tileY)] = tileData };
		byte[] wdtBytes = AlphaWdtWriter.Build(resolvedMapName, tiles);
		string wdtPath = Path.Combine(mapDir, $"{resolvedMapName}.wdt");
		File.WriteAllBytes(wdtPath, wdtBytes);
		Console.WriteLine($"Wrote Alpha WDT: {Path.GetFullPath(wdtPath)}");
		Console.WriteLine($"  Map: {resolvedMapName}, Tile: ({tileX}, {tileY})");
		Console.WriteLine($"  256 inline MCNK chunks, 1 texture ({resolvedTexture}), flat height = 0");
	}
	else
	{
		LkAdtData adtData = BlankAdtFactory.CreateBlank(resolvedMapName, tileX, tileY, resolvedTexture);
		string adtPath = Path.Combine(mapDir, $"{resolvedMapName}_{tileX}_{tileY}.adt");
		LkAdtWriter.Write(adtPath, adtData);
		Console.WriteLine($"Wrote ADT: {Path.GetFullPath(adtPath)}");
		Console.WriteLine($"  Map: {resolvedMapName}, Tile: ({tileX}, {tileY})");
		Console.WriteLine($"  256 MCNK chunks, 0 placements, 1 texture ({resolvedTexture})");

		HashSet<(int, int)> tileSet = [(tileX, tileY)];
		LkWdtWriteOptions wdtOptions = BlankAdtFactory.CreateBlankWdtOptions();
		string wdtPath = Path.Combine(mapDir, $"{resolvedMapName}.wdt");
		LkWdtWriter.Write(wdtPath, tileSet, wdtOptions);
		Console.WriteLine($"Wrote WDT: {Path.GetFullPath(wdtPath)}");
		Console.WriteLine($"  WDT flags: MPHD=0x00000000 (no MCCV/big-alpha/MTXF/MAID/MCLV flags), tile ({tileX},{tileY}) flagged as HasAdt");

		WdlHeightTile wdlTile = BlankAdtFactory.CreateBlankWdlTile(tileX, tileY);
		string wdlPath = Path.Combine(mapDir, $"{resolvedMapName}.wdl");
		WdlWriter.Write(wdlPath, [wdlTile]);
		Console.WriteLine($"Wrote WDL: {Path.GetFullPath(wdlPath)}");
		Console.WriteLine($"  Flat height = 0 for tile ({tileX},{tileY})");
	}
}

static void RunMapPatchBlank(string[] args)
{
	string? obj0Path = GetOption(args, "--placements", "-p");
	string? mapName = GetOption(args, "--map-name", "-m");
	string? tileXText = GetOption(args, "--tile-x", "-x");
	string? tileYText = GetOption(args, "--tile-y", "-y");
	string? outputDir = GetOption(args, "--output-dir", "-o");
	string? textureName = GetOption(args, "--texture", "-t");
	string? uidMode = GetOption(args, "--unique-id-mode", "-u");
	string? sourceWdt = GetOption(args, "--source-wdt", "-w");
	string? terrainSource = GetOption(args, "--terrain-source", "-s");

	if (string.IsNullOrWhiteSpace(obj0Path))
	{
		Console.Error.WriteLine("Error: --placements <obj0.adt> is required.");
		Console.Error.WriteLine("Usage: map patch-blank --placements <obj0.adt> --tile-x <n> --tile-y <n> [--map-name <name>] [--texture <path>] [--output-dir <dir>] [--unique-id-mode preserve|synthetic] [--source-wdt <path>] [--terrain-source <adt>]");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(tileXText) || string.IsNullOrWhiteSpace(tileYText))
	{
		Console.Error.WriteLine("Error: --tile-x and --tile-y are required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!int.TryParse(tileXText, out int tileX) || tileX < 0 || tileX >= 64)
	{
		Console.Error.WriteLine("Error: --tile-x must be an integer in [0, 63].");
		Environment.ExitCode = 1;
		return;
	}

	if (!int.TryParse(tileYText, out int tileY) || tileY < 0 || tileY >= 64)
	{
		Console.Error.WriteLine("Error: --tile-y must be an integer in [0, 63].");
		Environment.ExitCode = 1;
		return;
	}

	if (!File.Exists(obj0Path))
	{
		Console.Error.WriteLine($"Error: placement source '{obj0Path}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	string resolvedMapName = mapName ?? "testing";
	string resolvedOutputDir = outputDir ?? ".";
	string resolvedTexture = textureName ?? "tileset\\ocean\\westfallseafloor.blp";
	string mapDir = Path.Combine(resolvedOutputDir, "World", "Maps", resolvedMapName);
	Directory.CreateDirectory(mapDir);

	AdtPlacementCatalog catalog = AdtPlacementReader.Read(obj0Path);

	UniqueIdSource uidSource = uidMode?.ToLowerInvariant() switch
	{
		"synthetic" => UniqueIdSource.SyntheticSequential,
		_ => UniqueIdSource.PreserveFromCatalog,
	};

	LkAdtData blankAdt;
	if (!string.IsNullOrWhiteSpace(terrainSource) && File.Exists(terrainSource))
	{
		byte[] adtBytes = File.ReadAllBytes(terrainSource);
		string? tex0Path = Path.Combine(Path.GetDirectoryName(terrainSource)!, Path.GetFileNameWithoutExtension(terrainSource) + "_tex0.adt");
		string? obj0Path2 = Path.Combine(Path.GetDirectoryName(terrainSource)!, Path.GetFileNameWithoutExtension(terrainSource) + "_obj0.adt");
		byte[]? tex0Bytes = File.Exists(tex0Path) ? File.ReadAllBytes(tex0Path) : null;
		byte[]? obj0Bytes = File.Exists(obj0Path2) ? File.ReadAllBytes(obj0Path2) : null;
		var terrainAdt = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tileX, tileY);
		blankAdt = new LkAdtData
		{
			MapName = resolvedMapName,
			TileX = terrainAdt.TileX,
			TileY = terrainAdt.TileY,
			TextureNames = terrainAdt.TextureNames,
			ModelNames = terrainAdt.ModelNames,
			WorldModelNames = terrainAdt.WorldModelNames,
			ModelPlacements = terrainAdt.ModelPlacements,
			WorldModelPlacements = terrainAdt.WorldModelPlacements,
			Chunks = terrainAdt.Chunks,
			MhdrFlags = terrainAdt.MhdrFlags,
			MfboFlightBounds = terrainAdt.MfboFlightBounds,
		};
		Console.WriteLine($"Read terrain from: {Path.GetFullPath(terrainSource)}");
		Console.WriteLine($"  Textures: {blankAdt.TextureNames.Count}, Chunks: {blankAdt.Chunks.Count}");
	}
	else
	{
		blankAdt = BlankAdtFactory.CreateBlank(resolvedMapName, tileX, tileY, resolvedTexture);
	}
	LkAdtData patchedAdt = BlankAdtFactory.WithPlacements(blankAdt, catalog, uidSource);

	string adtPath = Path.Combine(mapDir, $"{resolvedMapName}_{tileX}_{tileY}.adt");
	LkAdtWriter.Write(adtPath, patchedAdt);
	Console.WriteLine($"Wrote patched ADT: {Path.GetFullPath(adtPath)}");
	Console.WriteLine($"  Map: {resolvedMapName}, Tile: ({tileX}, {tileY})");
	Console.WriteLine($"  M2 placements: {patchedAdt.ModelPlacements.Count}, WMO placements: {patchedAdt.WorldModelPlacements.Count}");
	Console.WriteLine($"  M2 names: {patchedAdt.ModelNames.Count}, WMO names: {patchedAdt.WorldModelNames.Count}");

	HashSet<(int, int)> tileSet = [(tileX, tileY)];
	LkWdtWriteOptions wdtOptions = BlankAdtFactory.CreateBlankWdtOptions();
	string wdtPath = Path.Combine(mapDir, $"{resolvedMapName}.wdt");

	const int WdtMainDataOffset = 60;
	const int MainEntrySize = 8;

	if (!string.IsNullOrWhiteSpace(sourceWdt) && File.Exists(sourceWdt))
	{
		byte[] wdtBytes = File.ReadAllBytes(sourceWdt);
		int entryOffset = WdtMainDataOffset + (tileY * 64 + tileX) * MainEntrySize;
		if (entryOffset + 4 <= wdtBytes.Length)
		{
			uint existingFlags = BitConverter.ToUInt32(wdtBytes, entryOffset);
			wdtBytes[entryOffset] = 1;
			wdtBytes[entryOffset + 1] = 0;
			wdtBytes[entryOffset + 2] = 0;
			wdtBytes[entryOffset + 3] = 0;
			File.WriteAllBytes(wdtPath, wdtBytes);
			Console.WriteLine($"Merged tile ({tileX},{tileY}) flags=0x{existingFlags:X8} into source WDT");
		}
		else
		{
			Console.Error.WriteLine($"Warning: source WDT too small to patch tile ({tileX},{tileY})");
		}
	}
	else
	{
		LkWdtWriter.Write(wdtPath, tileSet, wdtOptions);
		Console.WriteLine($"Wrote new minimal WDT: {Path.GetFullPath(wdtPath)}");
	}

	WdlHeightTile wdlTile = BlankAdtFactory.CreateBlankWdlTile(tileX, tileY);
	string wdlPath = Path.Combine(mapDir, $"{resolvedMapName}.wdl");
	WdlWriter.Write(wdlPath, [wdlTile]);
	Console.WriteLine($"Wrote WDL: {Path.GetFullPath(wdlPath)}");
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
	case "export-segments":
		RunPm4ExportSegments(tail);
		break;
	case "export-asset-signals":
		RunPm4ExportAssetSignals(tail);
		break;
	case "match-assets":
		RunPm4MatchAssets(tail);
		break;
	case "synthesize-placements":
		RunPm4SynthesizePlacements(tail);
		break;
	case "dump-collision":
		Pm4CollisionDumper.Run(tail);
		break;
	case "correlate-models":
		RunPm4CorrelateModels(tail);
		break;
	case "sweep-correlate":
		RunPm4SweepCorrelate(tail);
		break;
	case "match":
		RunPm4Match(tail);
		break;
	case "match-report":
		RunPm4MatchReport(tail);
		break;
	case "manifest":
		RunPm4Manifest(tail);
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
		case "connective-geometry":
			RunPm4ConnectiveGeometry(tail);
			break;
		case "bounds-audit":
			RunPm4BoundsAudit(tail);
			break;
		case "yaw-evidence":
			RunPm4YawEvidence(tail);
			break;
		case "doodad-split":
			RunPm4DoodadSplit(tail);
			break;
		case "component-identity":
			RunPm4ComponentIdentity(tail);
			break;
		case "mprr":
			RunPm4Mprr(tail);
			break;
		case "export-json":
			RunPm4ExportJson(tail);
			break;
		case "export-obj":
			RunPm4ExportObj(tail);
			break;
		case "bond-stats":
			RunPm4BondStats(tail);
			break;
case "fingerprint-scan":
		RunPm4FingerprintScan(tail);
		break;
	case "identify-models":
		RunPm4IdentifyModels(tail);
		break;
	case "tile-reports":
		RunPm4TileReports(tail);
		break;
	case "generate-from-wmo":
		RunPm4GenerateFromWmo(tail);
		break;
	case "validate-generator":
		RunPm4ValidateGenerator(tail);
		break;
	case "validate-generator-geometry":
		RunPm4ValidateGeneratorGeometry(tail);
		break;
	case "extract-wmo-cache":
		RunPm4ExtractWmoCache(tail);
		break;
	case "match-groups-to-wmos":
		RunPm4MatchGroupsToWmos(tail);
		break;
	case "extract-wmo-pattern":
		RunPm4ExtractWmoPattern(tail);
		break;
	case "test-generator":
		RunPm4TestGenerator();
		break;
	case "analyze-simplification":
		RunPm4AnalyzeSimplification(tail);
		break;
	case "build-wmo-fingerprint-db":
		RunPm4BuildWmoFingerprintDb(tail);
		break;
	case "extract-pm4-fingerprints":
		RunPm4ExtractPm4Fingerprints(tail);
		break;
	case "match-fingerprints":
		RunPm4MatchFingerprints(tail);
		break;
	case "validate-matches":
		RunPm4ValidateMatches(tail);
		break;
	case "build-wmo-surface-db":
		RunPm4BuildWmoSurfaceDb(tail);
		break;
	case "extract-pm4-surfaces":
		RunPm4ExtractPm4Surfaces(tail);
		break;
	case "match-surfaces":
		RunPm4MatchSurfaces(tail);
		break;
		default:
			Console.Error.WriteLine($"Unknown pm4 command '{command}'.");
			ShowPm4Usage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunPd4(string[] args)
{
	if (args.Length == 0)
	{
		ShowPd4Usage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunPd4Inspect(tail);
			break;
		case "export-obj":
			RunPm4ExportObj(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown pd4 command '{command}'.");
			ShowPd4Usage();
			Environment.ExitCode = 1;
			break;
	}
}

static void ShowPd4Usage()
{
	Console.WriteLine("PD4 commands:");
	Console.WriteLine("  pd4 inspect --input <file.pd4>");
}

static void RunPd4Inspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i");
	string? positionalInput = GetFirstPositionalArgument(args);
	input = input ?? positionalInput
		?? throw new InvalidOperationException("--input <file.pd4> is required.");

	if (!File.Exists(input))
	{
		Console.Error.WriteLine($"Error: file not found: {input}");
		Environment.ExitCode = 1;
		return;
	}

	var doc = Pd4ResearchReader.ReadFile(input);

	Console.WriteLine("PD4 Report");
	Console.WriteLine($"Input: {doc.SourcePath ?? input}");
	Console.WriteLine($"Version: {Pm4VersionFormatter.Format(doc.Version)}");
	Console.WriteLine($"MCRC: 0x{doc.Mcrc:X8}");
	Console.WriteLine($"Chunks: {doc.Chunks.Count}");
	Console.WriteLine($"Unknown chunks: {string.Join(", ", doc.Chunks
		.Where(c => c.Signature is not ("MVER" or "MCRC" or "MSHD" or "MSPV" or "MSPI" or "MSCN" or "MSLK" or "MSVT" or "MSVI" or "MSUR"))
		.Select(c => $"{c.Signature}:{c.Size}"))}");

	Console.WriteLine($"\nMSPV: count={doc.KnownChunks.Mspv.Count}");
	Console.WriteLine($"MSPI: count={doc.KnownChunks.Mspi.Count}");
	Console.WriteLine($"MSCN: count={doc.KnownChunks.Mscn.Count}");
	Console.WriteLine($"MSLK: count={doc.KnownChunks.Mslk.Count}");
	Console.WriteLine($"MSVI: count={doc.KnownChunks.Msvi.Count}");

	var msvt = doc.KnownChunks.Msvt;
	Console.WriteLine($"\nMSVT: count={msvt.Count}");
	if (msvt.Count > 0)
	{
		var min = new Vector3(msvt.Min(v => v.X), msvt.Min(v => v.Y), msvt.Min(v => v.Z));
		var max = new Vector3(msvt.Max(v => v.X), msvt.Max(v => v.Y), msvt.Max(v => v.Z));
		var ctr = (min + max) * 0.5f;
		Console.WriteLine($"  bounds min=({min.X:F2}, {min.Y:F2}, {min.Z:F2}) max=({max.X:F2}, {max.Y:F2}, {max.Z:F2})");
		Console.WriteLine($"  centroid=({ctr.X:F2}, {ctr.Y:F2}, {ctr.Z:F2})");
	}

	var msur = doc.KnownChunks.Msur;
	Console.WriteLine($"\nMSUR: count={msur.Count}");
	if (msur.Count > 0)
	{
		var flags = msur.GroupBy(e => e.Flags).OrderByDescending(g => g.Count()).Take(5);
		Console.WriteLine($"  Top flags: {string.Join(", ", flags.Select(g => $"0x{g.Key:X2}×{g.Count()}"))}");
		var indexCounts = msur.GroupBy(e => e.IndexCount).OrderByDescending(g => g.Count()).Take(5);
		Console.WriteLine($"  Top indexCounts: {string.Join(", ", indexCounts.Select(g => $"{g.Key}×{g.Count()}"))}");
		var zeroCount = msur.Count(e => e.Zero != 0);
		Console.WriteLine($"  Non-zero _0x1C fields: {zeroCount}/{msur.Count}");
		var refDistinct = msur.Select(e => e.RefIndex).Distinct().Count();
		Console.WriteLine($"  Distinct RefIndex values: {refDistinct}");

		Console.WriteLine($"\nFirst 5 MSUR entries:");
		for (int i = 0; i < Math.Min(5, msur.Count); i++)
			Console.WriteLine($"  [{i}] {msur[i]}");
	}

	if (doc.Diagnostics.Count > 0)
	{
		Console.WriteLine($"\nDiagnostics ({doc.Diagnostics.Count}):");
		foreach (var d in doc.Diagnostics)
			Console.WriteLine($"  {d}");
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

static void RunPm4MatchReport(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	string? maxMatchesText = GetOption(args, "--max-matches", "-n");
	string? searchRangeText = GetOption(args, "--search-range", "-s");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

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
		Console.Error.WriteLine("Error: --archive-root is required for pm4 match-report so WMO/M2 assets can be read from game archives.");
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
		string derivedMapName = previousUnderscore > 0 ? fileName[..previousUnderscore] : fileName;
		placements = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? string.Empty, $"{derivedMapName}_{tileX}_{tileY}_obj0.adt");
	}

	if (!File.Exists(placements))
	{
		Console.Error.WriteLine($"Error: placement source '{placements}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MatchResult result = Pm4MatchSupport.Run(input, placements, archiveRoot, archiveBootstrapOptions, maxMatches, searchRange);
	string markdown = FormatPm4MatchReport(result);

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);
		File.WriteAllText(outputPath, markdown);
		Console.WriteLine($"Wrote {outputPath}");
	}
	else
	{
		Console.Write(markdown);
	}
}

static string FormatPm4MatchReport(Pm4MatchResult result)
{
	var sb = new System.Text.StringBuilder();
	sb.AppendLine($"# PM4 Match Report: {Path.GetFileName(result.Pm4Path)}");
	sb.AppendLine();
	sb.AppendLine($"- **PM4**: `{result.Pm4Path}`");
	sb.AppendLine($"- **Placements**: `{result.PlacementPath}`");
	sb.AppendLine($"- **Archive**: `{result.ArchiveRoot}`");
	sb.AppendLine($"- **Tile**: ({result.TileX}, {result.TileY})");
	sb.AppendLine($"- **PM4 Objects**: {result.Pm4ObjectCount}");
	sb.AppendLine($"- **WMO Placements**: {result.WmoPlacementCount}");
	sb.AppendLine($"- **M2 Placements**: {result.M2PlacementCount}");
	sb.AppendLine($"- **Search Range**: {result.SearchRange} units");
	sb.AppendLine();

	if (result.Notes.Count > 0)
	{
		sb.AppendLine("## Notes");
		foreach (string note in result.Notes)
			sb.AppendLine($"- {note}");
		sb.AppendLine();
	}

	if (result.Pm4ObjectMatches.Count > 0)
	{
		sb.AppendLine("## PM4 Object Matches");
		sb.AppendLine();
		sb.AppendLine("| # | CK24 | Type | Part | Surface Count | Footprint Area | Candidate Count | Exported |");
		sb.AppendLine("|---|------|------|------|---------------|-----------------|-----------------|----------|");
		for (int i = 0; i < result.Pm4ObjectMatches.Count; i++)
		{
			Pm4ObjectMatch obj = result.Pm4ObjectMatches[i];
			sb.AppendLine($"| {i + 1} | 0x{obj.Ck24:X8} | {obj.Ck24Type} | {obj.ObjectPartId} | {obj.SurfaceCount} | {obj.FootprintArea:F1} | {obj.NearbyCandidateCount} | {obj.ExportedCandidateCount} |");
		}
		sb.AppendLine();
	}

	WritePlacementSection(sb, "WMO Placements (MODF)", result.WmoPlacements);
	WritePlacementSection(sb, "M2 Placements (MDDF)", result.M2Placements);

	return sb.ToString();
}

static void WritePlacementSection(System.Text.StringBuilder sb, string title, IReadOnlyList<Pm4PlacementMatchPlacement> placements)
{
	if (placements.Count == 0)
		return;

	sb.AppendLine($"## {title}");
	sb.AppendLine();
	sb.AppendLine("| # | UniqueID | Model Path | Position | Rotation | Scale | Bounds Min | Bounds Max | Asset | Candidates |");
	sb.AppendLine("|---|---------|------------|----------|----------|-------|------------|------------|-------|------------|");

	for (int i = 0; i < placements.Count; i++)
	{
		Pm4PlacementMatchPlacement p = placements[i];
		string assetCol = p.AssetResolved ? (p.AssetSource ?? "yes") : "not found";
		sb.AppendLine($"| {i + 1} | {p.UniqueId} | `{p.ModelPath}` | ({p.PlacementPosition.X:F1}, {p.PlacementPosition.Y:F1}, {p.PlacementPosition.Z:F1}) | ({p.PlacementRotation.X:F1}, {p.PlacementRotation.Y:F1}, {p.PlacementRotation.Z:F1}) | {p.PlacementScale:F3} | ({p.WorldBoundsMin.X:F1}, {p.WorldBoundsMin.Y:F1}, {p.WorldBoundsMin.Z:F1}) | ({p.WorldBoundsMax.X:F1}, {p.WorldBoundsMax.Y:F1}, {p.WorldBoundsMax.Z:F1}) | {assetCol} | {p.CandidateCount} |");
	}

	sb.AppendLine();

	for (int i = 0; i < placements.Count; i++)
	{
		Pm4PlacementMatchPlacement p = placements[i];
		if (p.Matches.Count == 0)
			continue;

		sb.AppendLine($"### Placement {i + 1}: `{p.ModelPath}` (UID {p.UniqueId}) — PM4 Candidates");
		sb.AppendLine();
		sb.AppendLine("| # | CK24 | Type | Part | Surfaces | Height | Planar Gap | Vert Gap | Center Dist | Footprint Dist | Overlap |");
		sb.AppendLine("|---|------|------|------|----------|--------|------------|----------|-------------|-----------------|---------|");

		for (int j = 0; j < p.Matches.Count; j++)
		{
			Pm4PlacementMatchCandidate c = p.Matches[j];
			sb.AppendLine($"| {j + 1} | 0x{c.Ck24:X8} | {c.Ck24Type} | {c.ObjectPartId} | {c.SurfaceCount} | {c.AverageSurfaceHeight:F1} | {c.PlanarGap:F1} | {c.VerticalGap:F1} | {c.CenterDistance:F1} | {c.FootprintDistance:F1} | {c.FootprintOverlapRatio:P0} |");
		}

		sb.AppendLine();
	}
}

static void RunPm4Manifest(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? outputDir = GetOption(args, "--output", "-o");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file or directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required.");
		Environment.ExitCode = 1;
		return;
	}

	string resolvedOutputDir = outputDir ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? ".", "manifests");
	Directory.CreateDirectory(resolvedOutputDir);

	string[] pm4Files = File.Exists(input)
		? [input]
		: Directory.GetFiles(input, "*.pm4", SearchOption.TopDirectoryOnly);

	int ok = 0, err = 0;
	foreach (string pm4File in pm4Files.OrderBy(Path.GetFileName))
	{
		if (!Pm4CoordinateService.TryParseTileCoordinates(pm4File, out int tileX, out int tileY))
		{
			err++;
			continue;
		}

		string fileName = Path.GetFileNameWithoutExtension(pm4File);
		int lastUnderscore = fileName.LastIndexOf('_');
		int prevUnderscore = lastUnderscore > 0 ? fileName.LastIndexOf('_', lastUnderscore - 1) : -1;
		string mapName = prevUnderscore > 0 ? fileName[..prevUnderscore] : fileName;
		string resolvedPlacements = placements ?? Path.Combine(
			Path.GetDirectoryName(Path.GetFullPath(pm4File)) ?? ".",
			$"{mapName}_{tileX}_{tileY}_obj0.adt");

		if (!File.Exists(resolvedPlacements))
		{
			err++;
			continue;
		}

		Pm4MatchResult match = Pm4MatchSupport.Run(pm4File, resolvedPlacements, archiveRoot, archiveBootstrapOptions, 8, 128f);

		var manifest = new
		{
			tile = $"{tileX}_{tileY}",
			source = Path.GetFileName(pm4File),
			placementSource = Path.GetFileName(resolvedPlacements),
			pm4ObjectCount = match.Pm4ObjectCount,
			wmoPlacementCount = match.WmoPlacementCount,
			m2PlacementCount = match.M2PlacementCount,
			searchRange = match.SearchRange,
			pm4Objects = match.Pm4ObjectMatches.Select(o => new
			{
				ck24 = $"0x{o.Ck24:X6}",
				ck24Type = $"0x{o.Ck24Type:X2}",
				ck24ObjectId = o.Ck24ObjectId,
				surfaceCount = o.SurfaceCount,
				linkGroupObjectId = o.LinkGroupObjectId,
				boundsMin = new { x = o.BoundsMin.X, y = o.BoundsMin.Y, z = o.BoundsMin.Z },
				boundsMax = new { x = o.BoundsMax.X, y = o.BoundsMax.Y, z = o.BoundsMax.Z },
				nearbyCandidateCount = o.NearbyCandidateCount,
				topCandidates = o.PossibleMatches.Take(3).Select(c => new
				{
					kind = c.Kind,
					modelPath = c.ModelPath,
					uniqueId = c.UniqueId,
					assetResolved = c.AssetResolved,
					centerDistance = c.CenterDistance,
					footprintOverlap = c.FootprintOverlapRatio,
					anchorPlanarGap = c.AnchorPlanarGap,
				})
			}),
			m2Placements = match.M2Placements.Select(p => new
			{
				modelPath = p.ModelPath,
				uniqueId = p.UniqueId,
				assetResolved = p.AssetResolved,
				candidateCount = p.CandidateCount,
				position = new { x = p.PlacementPosition.X, y = p.PlacementPosition.Y, z = p.PlacementPosition.Z },
				rotation = new { x = p.PlacementRotation.X, y = p.PlacementRotation.Y, z = p.PlacementRotation.Z },
				scale = p.PlacementScale,
				worldBoundsMin = new { x = p.WorldBoundsMin.X, y = p.WorldBoundsMin.Y, z = p.WorldBoundsMin.Z },
				worldBoundsMax = new { x = p.WorldBoundsMax.X, y = p.WorldBoundsMax.Y, z = p.WorldBoundsMax.Z },
				topPm4Candidates = p.Matches.Take(3).Select(m => new
				{
					ck24 = $"0x{m.Ck24:X6}",
					ck24Type = $"0x{m.Ck24Type:X2}",
					centerDistance = m.CenterDistance,
					planarOverlap = m.PlanarOverlapRatio,
					volumeOverlap = m.VolumeOverlapRatio,
					verticalGap = m.VerticalGap,
				})
			}),
			wmoPlacements = match.WmoPlacements.Select(p => new
			{
				modelPath = p.ModelPath,
				uniqueId = p.UniqueId,
				assetResolved = p.AssetResolved,
				candidateCount = p.CandidateCount,
			}),
			notes = match.Notes,
		};

		string json = System.Text.Json.JsonSerializer.Serialize(manifest, new System.Text.Json.JsonSerializerOptions
		{
			WriteIndented = true,
			IncludeFields = true
		});

		string outPath = Path.Combine(resolvedOutputDir, $"{mapName}_{tileX}_{tileY}_manifest.json");
		File.WriteAllText(outPath, json);
		ok++;
	}

	Console.WriteLine($"Manifests: {ok} written, {err} skipped (no _obj0.adt).");
	Console.WriteLine($"Output: {resolvedOutputDir}");
}



static void RunPm4MatchAssets(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? assetCorpus = GetOption(args, "--asset-corpus", "-c");
	string? output = GetOption(args, "--output", "-o");
	string? maxCandidatesText = GetOption(args, "--max-candidates", "-n");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	int maxCandidates = 10;
	if (!string.IsNullOrWhiteSpace(maxCandidatesText) && (!int.TryParse(maxCandidatesText, out maxCandidates) || maxCandidates <= 0))
	{
		Console.Error.WriteLine("Error: --max-candidates must be a positive integer.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!File.Exists(input))
	{
		Console.Error.WriteLine($"Error: PM4 input '{input}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(assetCorpus) && !string.IsNullOrWhiteSpace(placements))
	{
		Console.Error.WriteLine("Error: choose either --asset-corpus <report.json> or --placements <tile_obj0.adt>, not both.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(assetCorpus) && !File.Exists(assetCorpus))
	{
		Console.Error.WriteLine($"Error: asset corpus '{assetCorpus}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required for pm4 match-assets so WMO/M2 assets can be read from the staged client.");
		Environment.ExitCode = 1;
		return;
	}

	if (!Pm4CoordinateService.TryParseTileCoordinates(input, out int tileX, out int tileY))
	{
		Console.Error.WriteLine("Error: could not derive tile coordinates from the PM4 filename.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && string.IsNullOrWhiteSpace(placements))
	{
		string fileName = Path.GetFileNameWithoutExtension(input);
		int lastUnderscore = fileName.LastIndexOf('_');
		int previousUnderscore = lastUnderscore > 0 ? fileName.LastIndexOf('_', lastUnderscore - 1) : -1;
		string mapName = previousUnderscore > 0 ? fileName[..previousUnderscore] : fileName;
		placements = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? string.Empty, $"{mapName}_{tileX}_{tileY}_obj0.adt");
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && !File.Exists(placements))
	{
		Console.Error.WriteLine($"Error: placement source '{placements}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		Pm4SegmentExportRun exportRun = Pm4SegmentExportService.Export(input);
		Pm4SegmentExportFile file = AssertSinglePm4ExportFile(exportRun, input);
		Pm4AssetReferenceBuildResult assetBuild;
		string assetReferenceSource;
		if (!string.IsNullOrWhiteSpace(assetCorpus))
		{
			assetBuild = Pm4AssetSignalCorpusSupport.LoadFromManifest(assetCorpus);
			assetReferenceSource = assetCorpus;
		}
		else
		{
			assetBuild = Pm4AssetReferenceSupport.BuildFromPlacements(placements!, archiveRoot!, archiveBootstrapOptions, tileX, tileY);
			assetReferenceSource = placements!;
		}

		IReadOnlyList<Pm4SegmentMatchResult> matchResults = Pm4AssetMatchScorer.ScoreSegments(file.Segments, assetBuild.Assets, maxCandidates);
		IReadOnlyList<Pm4ReplacementPlacementProposal> placementProposals = Pm4ReplacementPlacementSynthesizer.Synthesize(matchResults, assetBuild.Assets);
		Pm4MatchRunManifest manifest = BuildPm4AssetMatchManifest(
			exportRun,
			matchResults,
			placementProposals,
			assetReferenceSource,
			assetBuild.Warnings,
			"match-assets");

		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			WritePm4Report(manifest, outputPath);
			Console.WriteLine($"Matched {manifest.SegmentCount} PM4 segments against {assetBuild.Assets.Count} validation asset references.");
			Console.WriteLine($"Synthesized {placementProposals.Count} placement proposals from the ranked candidates.");
			return;
		}

		PrintPm4AssetMatchRun(manifest, assetBuild.Assets.Count);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4CorrelateModels(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	string? pm4VPath = GetOption(args, "--pm4-vpath", "-pv");
	string? adtVPath = GetOption(args, "--adt-vpath", "-av");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	bool inputIsFile = !string.IsNullOrWhiteSpace(input) && File.Exists(input);
	bool placementsIsFile = !string.IsNullOrWhiteSpace(placements) && File.Exists(placements);

	if (!inputIsFile && string.IsNullOrWhiteSpace(pm4VPath))
	{
		Console.Error.WriteLine("Error: --input <file.pm4> must exist, or provide --pm4-vpath <archive-path>.");
		Environment.ExitCode = 1;
		return;
	}

	if (!placementsIsFile && string.IsNullOrWhiteSpace(adtVPath))
	{
		Console.Error.WriteLine("Error: --placements <file.adt> must exist, or provide --adt-vpath <archive-path>.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		Pm4CorrelateModelsResult result = Pm4CorrelateModelsSupport.Correlate(
			input!, placements!, archiveRoot!, archiveBootstrapOptions, pm4VPath, adtVPath);

		Console.WriteLine($"PM4 Correlation Report: {Path.GetFileName(input)}");
		Console.WriteLine($"Tile: ({result.TileX}, {result.TileY})");
		Console.WriteLine($"CK24 groups: {result.Ck24Groups.Count}");
		Console.WriteLine($"Placements: {result.PlacementSummaries.Count}");
		Console.WriteLine($"Correlations: {result.Correlations.Count}");

		Console.WriteLine();
		Console.WriteLine("## CK24 Groups (by index count, desc)");

		Console.WriteLine("| # | CK24 | Type | ObjID | Surfaces | Indices | Verts | PM4 Bounds | WoW Bounds |");
		Console.WriteLine("|---|------|------|-------|----------|---------|-------|------------|------------|");
		for (int i = 0; i < Math.Min(result.Ck24Groups.Count, 20); i++)
		{
			Pm4Ck24GroupSummary g = result.Ck24Groups[i];
			string pm4Bounds = $"({g.Pm4BoundsMin.X:F0},{g.Pm4BoundsMin.Y:F0},{g.Pm4BoundsMin.Z:F0})-({g.Pm4BoundsMax.X:F0},{g.Pm4BoundsMax.Y:F0},{g.Pm4BoundsMax.Z:F0})";
			string wowBounds = $"({g.WowBoundsMin.X:F0},{g.WowBoundsMin.Y:F0},{g.WowBoundsMin.Z:F0})-({g.WowBoundsMax.X:F0},{g.WowBoundsMax.Y:F0},{g.WowBoundsMax.Z:F0})";
			Console.WriteLine($"| {i + 1} | 0x{g.Ck24:X6} | 0x{g.Ck24Type:X2} | {g.Ck24ObjectId} | {g.SurfaceCount} | {g.TotalIndexCount} | {g.VertexCount} | {pm4Bounds} | {wowBounds} |");
		}
		if (result.Ck24Groups.Count > 20)
			Console.WriteLine($"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ({result.Ck24Groups.Count - 20} more)");

		Console.WriteLine();
		Console.WriteLine("## Placement Collision Summaries");

		Console.WriteLine("| # | UID | Kind | Path | Groups | Verts | Faces | Local Bounds | World Bounds |");
		Console.WriteLine("|---|-----|------|------|--------|-------|-------|-------------|--------------|");
		for (int i = 0; i < result.PlacementSummaries.Count; i++)
		{
			Pm4PlacementCollisionSummary p = result.PlacementSummaries[i];
			string shortPath = p.ModelPath.Length > 50 ? "..." + p.ModelPath[^47..] : p.ModelPath;
			string localBounds = $"({p.LocalBoundsMin.X:F0},{p.LocalBoundsMin.Y:F0},{p.LocalBoundsMin.Z:F0})-({p.LocalBoundsMax.X:F0},{p.LocalBoundsMax.Y:F0},{p.LocalBoundsMax.Z:F0})";
			string worldBounds = $"({p.WorldBoundsMin.X:F0},{p.WorldBoundsMin.Y:F0},{p.WorldBoundsMin.Z:F0})-({p.WorldBoundsMax.X:F0},{p.WorldBoundsMax.Y:F0},{p.WorldBoundsMax.Z:F0})";
			Console.WriteLine($"| {i + 1} | {p.UniqueId} | {p.AssetKind} | {shortPath} | {p.GroupCount} | {p.TotalCollisionVertices} | {p.TotalCollisionFaces} | {localBounds} | {worldBounds} |");
		}

		Console.WriteLine();
		Console.WriteLine("## Correlations (top 30 by WoW overlap)");

		Console.WriteLine("| # | UID | Path | Kind | CK24 | Type | WoW Overlap | PM4 Overlap | WoW Dist | PM4 Dist |");
		Console.WriteLine("|---|-----|------|------|------|------|-------------|-------------|----------|----------|");
		for (int i = 0; i < Math.Min(result.Correlations.Count, 30); i++)
		{
			Pm4CorrelationEntry c = result.Correlations[i];
			string shortPath = c.ModelPath.Length > 40 ? "..." + c.ModelPath[^37..] : c.ModelPath;
			Console.WriteLine($"| {i + 1} | {c.UniqueId} | {shortPath} | {c.AssetKind} | 0x{c.Ck24:X6} | 0x{c.Ck24Type:X2} | {c.WowBoundsOverlap:F3} | {c.Pm4BoundsOverlap:F3} | {c.WowCenterDistance:F0} | {c.Pm4CenterDistance:F0} |");
		}
		if (result.Correlations.Count > 30)
			Console.WriteLine($"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ({result.Correlations.Count - 30} more)");

		if (result.Warnings.Count > 0)
		{
			Console.WriteLine();
			Console.WriteLine("## Warnings");
			foreach (string warning in result.Warnings)
				Console.WriteLine($"- {warning}");
		}

		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
			string json = JsonSerializer.Serialize(result, Pm4MatchSupport.CreateJsonOptions());
			File.WriteAllText(outputPath, json);
			Console.WriteLine($"Wrote JSON to {outputPath}");
		}
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4SweepCorrelate(string[] args)
{
	string? mapDir = GetOption(args, "--map-dir", "-d");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	string? limitText = GetOption(args, "--limit", "-n");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	int limit = int.MaxValue;
	if (!string.IsNullOrWhiteSpace(limitText) && (!int.TryParse(limitText, out limit) || limit <= 0))
	{
		Console.Error.WriteLine("Error: --limit must be a positive integer.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(mapDir) || !Directory.Exists(mapDir))
	{
		Console.Error.WriteLine("Error: --map-dir <directory> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	string[] pm4Files = Directory.GetFiles(mapDir, "*.pm4");
	Array.Sort(pm4Files, StringComparer.OrdinalIgnoreCase);

	List<string> csvLines = [];
	csvLines.Add("tileX,tileY,pm4Path,pm4Bytes,ck24Groups,placements,correlations,topModelPath,topOverlap,topDist");

	int processed = 0;
	int skippedNoData = 0;
	int skippedNoAdt = 0;
	int failed = 0;
	Stopwatch sw = Stopwatch.StartNew();

	for (int i = 0; i < pm4Files.Length && processed < limit; i++)
	{
		string pm4Path = pm4Files[i];
		FileInfo fi = new(pm4Path);
		if (fi.Length == 0)
		{
			skippedNoData++;
			continue;
		}

		if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
			continue;

		string baseName = Path.GetFileNameWithoutExtension(pm4Path);
		string adtPath = Path.Combine(mapDir, baseName + "_obj0.adt");
		if (!File.Exists(adtPath))
		{
			skippedNoAdt++;
			continue;
		}

		try
		{
			Pm4CorrelateModelsResult result = Pm4CorrelateModelsSupport.Correlate(
				pm4Path, adtPath, archiveRoot, archiveBootstrapOptions, null, null);

			string topModel = "";
			double topOverlap = 0;
			double topDist = 0;
			if (result.Correlations.Count > 0)
			{
				Pm4CorrelationEntry top = result.Correlations[0];
				topModel = Path.GetFileName(top.ModelPath);
				topOverlap = top.WowBoundsOverlap;
				topDist = top.WowCenterDistance;
			}

			csvLines.Add($"{tileX},{tileY},{baseName},{fi.Length},{result.Ck24Groups.Count},{result.PlacementSummaries.Count},{result.Correlations.Count},\"{topModel}\",{topOverlap:F4},{topDist:F1}");
			processed++;

			if (processed % 20 == 0 || processed == 1)
			{
				double elapsed = sw.Elapsed.TotalSeconds;
				double rate = processed / elapsed;
				int remaining = Math.Min(pm4Files.Length - i - 1, limit - processed);
				Console.Error.WriteLine($"  [{processed}/{limit}] tile ({tileX},{tileY}) ck24={result.Ck24Groups.Count} corr={result.Correlations.Count} ... {rate:F1}/s, ~{remaining / rate:F0}s remaining");
			}
		}
		catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
		{
			failed++;
			csvLines.Add($"{tileX},{tileY},{baseName},{fi.Length},ERROR,\"{ex.Message}\"");
		}
	}

	sw.Stop();
	double totalSec = sw.Elapsed.TotalSeconds;

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
		File.WriteAllLines(outputPath, csvLines);
		Console.WriteLine($"Wrote sweep CSV to {outputPath}");
	}

	Console.WriteLine($"Sweep complete: {processed} tiles processed, {skippedNoData} skipped (0-byte), {skippedNoAdt} skipped (no _obj0.adt), {failed} failed, {totalSec:F1}s ({processed / totalSec:F1}/s)");

	if (processed > 0)
	{
		Console.WriteLine();
		Console.WriteLine("## Summary Stats");
		Console.WriteLine($"  Tiles with correlations: {csvLines.Skip(1).Count(l => l.Split(',').Length >= 7 && int.TryParse(l.Split(',')[6], out int c) && c > 0)}");
		Console.WriteLine($"  Total correlations: {csvLines.Skip(1).Sum(l => l.Split(',').Length >= 7 ? (int.TryParse(l.Split(',')[6], out int c) ? c : 0) : 0)}");
	}
}

static void RunPm4ExtractWmoCache(string[] args)
{
	string? mapDir = GetOption(args, "--map-dir", "-d");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? cacheDir = GetOption(args, "--cache-dir", "-c") ?? Path.Combine(Path.GetDirectoryName(mapDir) ?? ".", "wmo-cache");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(mapDir) || !Directory.Exists(mapDir))
	{
		Console.Error.WriteLine("Error: --map-dir <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}
	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	Directory.CreateDirectory(cacheDir);
	HashSet<string> neededWmos = new(StringComparer.OrdinalIgnoreCase);

	// Scan all ADTs for WMO placements
	string[] adtFiles = Directory.GetFiles(mapDir, "*.adt");
	int scannedAdts = 0;
	Stopwatch sw = Stopwatch.StartNew();

	foreach (string adtPath in adtFiles)
	{
		string name = Path.GetFileName(adtPath);
		if (name.Contains("_obj0") || name.Contains("_tex0") || name.Contains("_lod")) continue;

		try
		{
			var placements = AdtPlacementReader.Read(adtPath);
			foreach (var wmo in placements.WorldModelPlacements)
			{
				string normalized = wmo.ModelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
				if (!string.IsNullOrWhiteSpace(normalized))
					neededWmos.Add(normalized);
			}
			scannedAdts++;
		}
		catch { }
	}

	Console.WriteLine($"Scanned {scannedAdts} ADTs, found {neededWmos.Count} unique WMO root paths.");

	// Extract each WMO
	int extracted = 0;
	int failed = 0;
	int total = neededWmos.Count;

	foreach (string wmoVPath in neededWmos.OrderBy(static p => p))
	{
		try
		{
			byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
				wmoVPath, [archiveRoot], archiveBootstrapOptions);

			string cachePath = Path.Combine(cacheDir, wmoVPath.Replace('/', '\\'));
			string? cacheDirPart = Path.GetDirectoryName(cachePath);
			if (!string.IsNullOrWhiteSpace(cacheDirPart))
				Directory.CreateDirectory(cacheDirPart);
			File.WriteAllBytes(cachePath, rootBytes);

			// Extract group files
			using MemoryStream ms = new(rootBytes, writable: false);
			var summary = WmoSummaryReader.Read(ms, wmoVPath);
			string rootStem = Path.Combine(
				Path.GetDirectoryName(cachePath) ?? "",
				Path.GetFileNameWithoutExtension(cachePath));

			for (int gi = 0; gi < summary.ReportedGroupCount; gi++)
			{
				string groupVPath = wmoVPath.Replace(".wmo", $"_{gi:D3}.wmo", StringComparison.OrdinalIgnoreCase);
				try
				{
					byte[] groupBytes = ArchiveVirtualFileReader.ReadVirtualFile(
						groupVPath, [archiveRoot], archiveBootstrapOptions);
					string groupCachePath = Path.Combine(cacheDir, groupVPath.Replace('/', '\\'));
					string? groupDirPart = Path.GetDirectoryName(groupCachePath);
					if (!string.IsNullOrWhiteSpace(groupDirPart))
						Directory.CreateDirectory(groupDirPart);
					File.WriteAllBytes(groupCachePath, groupBytes);
				}
				catch { }
			}

			extracted++;
		}
		catch (Exception ex)
		{
			failed++;
		}

		if (extracted % 50 == 0)
		{
			double elapsed = sw.Elapsed.TotalSeconds;
			Console.WriteLine($"  [{extracted}/{total}] extracted, {failed} failed, {elapsed:F1}s, ~{(double)(total - extracted) / (extracted / elapsed):F0}s remaining");
		}
	}

	sw.Stop();
	Console.WriteLine($"Extraction complete: {extracted} WMOs extracted, {failed} failed, {sw.Elapsed.TotalSeconds:F1}s");
	Console.WriteLine($"Cache directory: {Path.GetFullPath(cacheDir)}");
}

static void RunPm4ExtractWmoPattern(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? cacheDir = GetOption(args, "--cache-dir", "-c");
	string? pattern = GetOption(args, "--pattern", "-p");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required.");
		Environment.ExitCode = 1;
		return;
	}
	if (string.IsNullOrWhiteSpace(pattern))
	{
		Console.Error.WriteLine("Error: --pattern <substring> is required (e.g., 'ulduar').");
		Environment.ExitCode = 1;
		return;
	}

	var session = ArchiveCatalogSessionCache.GetOrCreate([archiveRoot], archiveBootstrapOptions);
	var allFiles = session.ArchiveCatalog.GetAllKnownFiles();
	string lowerPattern = pattern.ToLowerInvariant();

	var matchingWmos = allFiles
		.Where(f => f.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
			&& f.Contains(lowerPattern, StringComparison.OrdinalIgnoreCase))
		.OrderBy(static f => f)
		.ToList();

	Console.WriteLine($"Found {matchingWmos.Count} WMOs matching '{pattern}'.");

	if (!string.IsNullOrWhiteSpace(cacheDir))
		Directory.CreateDirectory(cacheDir);

	int extracted = 0, failed = 0;
	Stopwatch sw = Stopwatch.StartNew();
	object lockObj = new();

	Parallel.ForEach(matchingWmos, wmoVPath =>
	{
		try
		{
			byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
				wmoVPath.Replace('\\', '/').ToLowerInvariant(),
				[archiveRoot], archiveBootstrapOptions);

			if (!string.IsNullOrWhiteSpace(cacheDir))
			{
				string cachePath = Path.Combine(cacheDir, wmoVPath.Replace('/', '\\'));
				string? dir = Path.GetDirectoryName(cachePath);
				if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
				File.WriteAllBytes(cachePath, bytes);
			}

			lock (lockObj) { extracted++; }
		}
		catch
		{
			lock (lockObj) { failed++; }
		}
	});

	sw.Stop();
	Console.WriteLine($"Extracted {extracted} WMOs ({failed} failed) in {sw.Elapsed.TotalSeconds:F1}s");
}

static void RunPm4MatchGroupsToWmos(string[] args)
{
	string? pm4Path = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? wmoCacheDir = GetOption(args, "--wmo-cache", "-w");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(pm4Path) || !File.Exists(pm4Path))
	{
		Console.Error.WriteLine("Error: --input <file.pm4> is required.");
		Environment.ExitCode = 1;
		return;
	}
	if (string.IsNullOrWhiteSpace(wmoCacheDir) || !Directory.Exists(wmoCacheDir))
	{
		Console.Error.WriteLine("Error: --wmo-cache <dir> is required.");
		Environment.ExitCode = 1;
		return;
	}

	var doc = Pm4ResearchReader.ReadFile(pm4Path);
	var msvt = doc.KnownChunks.Msvt;
	var msvi = doc.KnownChunks.Msvi;

	List<Dictionary<string, object?>> groups = [];
	foreach (var ck24Group in doc.KnownChunks.Msur
		.Where(static s => s.Ck24 != 0 && s.IndexCount >= 3)
		.GroupBy(static s => s.Ck24))
	{
		uint ck24 = ck24Group.Key;
		var surfaces = ck24Group.ToList();
		byte type = surfaces[0].Ck24Type;
		ushort objId = surfaces[0].Ck24ObjectId;
		int totalIndices = surfaces.Sum(static s => s.IndexCount);

		Vector3 pm4Min = new(float.MaxValue), pm4Max = new(float.MinValue, float.MinValue, float.MinValue);
		foreach (var s in surfaces)
		{
			int first = checked((int)s.MsviFirstIndex);
			int end = Math.Min(first + s.IndexCount, msvi.Count);
			for (int i = first; i < end; i++)
			{
				int vi = checked((int)msvi[i]);
				if ((uint)vi < (uint)msvt.Count)
				{
					pm4Min = Vector3.Min(pm4Min, msvt[vi]);
					pm4Max = Vector3.Max(pm4Max, msvt[vi]);
				}
			}
		}

		Vector3 size = pm4Max - pm4Min;
		float[] dims = [size.X, size.Y, size.Z];
		Array.Sort(dims);

		groups.Add(new Dictionary<string, object?>
		{
			["ck24"] = $"0x{ck24:X6}",
			["type"] = $"0x{type:X2}",
			["objId"] = (int)objId,
			["surfaces"] = surfaces.Count,
			["verts"] = 0,
			["indices"] = totalIndices,
			["size"] = new double[] { Math.Round(dims[0], 1), Math.Round(dims[1], 1), Math.Round(dims[2], 1) },
			["pm4Bounds"] = new { min = new double[] { Math.Round(pm4Min.X, 1), Math.Round(pm4Min.Y, 1), Math.Round(pm4Min.Z, 1) }, max = new double[] { Math.Round(pm4Max.X, 1), Math.Round(pm4Max.Y, 1), Math.Round(pm4Max.Z, 1) } },
		});
	}

	groups.Sort((a, b) => ((int)b["surfaces"]!).CompareTo((int)a["surfaces"]!));

	Console.WriteLine($"\n=== PM4 Groups on {Path.GetFileName(pm4Path)} ===");
	Console.WriteLine($"Total CK24 groups: {groups.Count}");
	Console.WriteLine($"\n{"CK24",-12} {"Type",-6} {"ObjID",-6} {"Surf",-6} {"Indices",-8} {"Size (sorted)",-20} {"PM4 Bounds"}");
	Console.WriteLine(new string('-', 100));
	foreach (var g in groups)
	{
		var s = (double[])g["size"]!;
		var b = (dynamic)g["pm4Bounds"]!;
		Console.WriteLine($"{g["ck24"],-12} {g["type"],-6} {g["objId"],-6} {g["surfaces"],-6} {g["indices"],-8} ({s[0],-6:F1}x{s[1],-6:F1}x{s[2],-6:F1}) ({b.min[0]:F1},{b.min[1]:F1},{b.min[2]:F1})-({b.max[0]:F1},{b.max[1]:F1},{b.max[2]:F1})");
	}

	// Scan WMO cache for matching local bounds
	Console.WriteLine($"\n=== Scanning WMO cache for matching bounds ===");
	var wmoRoots = Directory.GetFiles(wmoCacheDir, "*.wmo", SearchOption.AllDirectories)
		.Where(f => !System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileNameWithoutExtension(f), @"_\d{3}$"))
		.ToArray();
	Console.WriteLine($"WMO roots in cache: {wmoRoots.Length}");

	List<Dictionary<string, object?>> wmoEntries = [];
	int scanned = 0;
	Stopwatch sw = Stopwatch.StartNew();
	object scanLock = new();

	Parallel.ForEach(wmoRoots, wmoPath =>
	{
		try
		{
			using FileStream fs = File.OpenRead(wmoPath);
			var summary = WmoSummaryReader.Read(fs, wmoPath);
			Vector3 size = summary.BoundsMax - summary.BoundsMin;
			float[] dims = [size.X, size.Y, size.Z];
			Array.Sort(dims);

			lock (scanLock)
			{
				wmoEntries.Add(new Dictionary<string, object?>
				{
					["path"] = wmoPath,
					["groups"] = summary.ReportedGroupCount,
					["size"] = new double[] { Math.Round(dims[0], 1), Math.Round(dims[1], 1), Math.Round(dims[2], 1) },
				});
				scanned++;
				if (scanned % 100 == 0)
					Console.Error.Write($"\r  Scanned {scanned}/{wmoRoots.Length} WMOs...");
			}
		}
		catch { }
	});
	sw.Stop();
	Console.Error.WriteLine($"\r  Scanned {scanned}/{wmoRoots.Length} WMOs in {sw.Elapsed.TotalSeconds:F1}s");

	// For each PM4 group, find the top 5 WMO matches by sorted dimension similarity
	Console.WriteLine($"\n=== Top WMO matches for each PM4 group ===");
	Console.WriteLine($"\n{"CK24",-12} {"Surf",-6} {"Size",-22} {"Best WMO Match",-60} {"WMO Size",-22} {"Score",-6}");
	Console.WriteLine(new string('-', 130));
	foreach (var g in groups)
	{
		var pm4Size = (double[])g["size"]!;
		double bestScore = 0;
		string bestWmo = "";

		foreach (var w in wmoEntries)
		{
			var wSize = (double[])w["size"]!;
			double r0 = Math.Min(pm4Size[0], wSize[0]) / (double)Math.Max(pm4Size[0], wSize[0]);
			double r1 = Math.Min(pm4Size[1], wSize[1]) / (double)Math.Max(pm4Size[1], wSize[1]);
			double r2 = Math.Min(pm4Size[2], wSize[2]) / (double)Math.Max(pm4Size[2], wSize[2]);
			double score = (r0 + r1 + r2) / 3.0;
			if (score > bestScore)
			{
				bestScore = score;
				bestWmo = (string)w["path"]!;
			}
		}

		string shortPath = bestWmo.Length > 55 ? "..." + bestWmo[^52..] : bestWmo;
		Console.WriteLine($"{g["ck24"],-12} {g["surfaces"],-6} ({pm4Size[0],-6:F1}x{pm4Size[1],-6:F1}x{pm4Size[2],-6:F1}) {shortPath,-60} {bestScore,-6:F3}");
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
		var result = new Dictionary<string, object?> { ["groups"] = groups, ["wmoMatches"] = wmoEntries };
		File.WriteAllText(outputPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"\nWrote {outputPath}");
	}
}

static void RunPm4ValidateGenerator(string[] args)
{
	string? mapDir = GetOption(args, "--map-dir", "-d");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	string? wmoCacheDir = GetOption(args, "--wmo-cache", "-w");
	int? limit = TryParseInt(GetOption(args, "--limit", "-n"));
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(mapDir) || !Directory.Exists(mapDir))
	{
		Console.Error.WriteLine("Error: --map-dir <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	string[] pm4Files = Directory.GetFiles(mapDir, "*.pm4");
	Array.Sort(pm4Files, StringComparer.OrdinalIgnoreCase);

	List<Dictionary<string, object?>> tileResults = [];
	int processed = 0;
	int skippedNoData = 0;
	int skippedNoAdt = 0;
	int skippedNoPlacements = 0;
	int failed = 0;
	Stopwatch sw = Stopwatch.StartNew();

	for (int i = 0; i < pm4Files.Length; i++)
	{
		if (limit.HasValue && processed >= limit.Value) break;

		string pm4Path = pm4Files[i];
		FileInfo fi = new(pm4Path);
		if (fi.Length == 0) { skippedNoData++; continue; }

		string pm4Name = Path.GetFileNameWithoutExtension(pm4Path);
		if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
			continue;

		string adtPath = Path.Combine(mapDir, $"development_{tileX}_{tileY}.adt");
		if (!File.Exists(adtPath))
		{
			skippedNoAdt++;
			continue;
		}

		try
		{
			var doc = Pm4ResearchReader.ReadFile(pm4Path);
			bool hasCk24Surfaces = doc.KnownChunks.Msur.Any(s => s.Ck24 != 0 && s.IndexCount >= 3);
			if (!hasCk24Surfaces) { skippedNoPlacements++; continue; }

			var placements = AdtPlacementReader.Read(adtPath);
			if (placements.WorldModelPlacements.Count == 0) { skippedNoPlacements++; continue; }

			List<Dictionary<string, object?>> placementResults = [];
			int genTotalSurfaces = 0;
			int genTotalVerts = 0;
			int matchedPlacements = 0;

			foreach (var wmoPlacement in placements.WorldModelPlacements)
			{
				try
				{
					string normalizedPath = wmoPlacement.ModelPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();

					byte[] rootBytes;
					if (!string.IsNullOrWhiteSpace(wmoCacheDir))
					{
						string cachePath = Path.Combine(wmoCacheDir, normalizedPath.Replace('/', '\\'));
						if (File.Exists(cachePath))
							rootBytes = File.ReadAllBytes(cachePath);
						else
							rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(normalizedPath, [archiveRoot], archiveBootstrapOptions);
					}
					else
					{
						rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(normalizedPath, [archiveRoot], archiveBootstrapOptions);
					}

					Func<string, byte[]?> readGroupBytes = vp =>
					{
						if (!string.IsNullOrWhiteSpace(wmoCacheDir))
						{
							string cachePath = Path.Combine(wmoCacheDir, vp.Replace('/', '\\'));
							if (File.Exists(cachePath))
								return File.ReadAllBytes(cachePath);
						}
						try { return ArchiveVirtualFileReader.ReadVirtualFile(vp, [archiveRoot], archiveBootstrapOptions); }
						catch { return null; }
					};

					using MemoryStream ms = new(rootBytes, writable: false);
					WmoRenderDocument renderDoc = WmoRenderDocumentReader.Read(ms, wmoPlacement.ModelPath, readGroupBytes);

					matchedPlacements++;

					Pm4GenerationData genData = Pm4Generator.GenerateFromWmo(
						renderDoc,
						wmoPlacement.Position,
						wmoPlacement.Rotation,
						1f,
						ck24Type: 0x43,
						ck24ObjectId: (ushort)(matchedPlacements & 0xFFFF));

					if (genData.Msur.Count == 0)
						continue;

					genTotalSurfaces += genData.Msur.Count;
					genTotalVerts += genData.Msvt.Count;

					int wmoFaceCount = renderDoc.Groups.Sum(static g => g.Mesh.Indices.Count / 3);
					placementResults.Add(new Dictionary<string, object?>
					{
						["uid"] = wmoPlacement.UniqueId,
						["model"] = Path.GetFileName(wmoPlacement.ModelPath),
						["wmoFaces"] = wmoFaceCount,
						["genSurfaces"] = genData.Msur.Count,
						["genVerts"] = genData.Msvt.Count,
						["genIndices"] = genData.Msvi.Count,
					});
				}
				catch (FileNotFoundException) { continue; }
				catch (InvalidDataException) { continue; }
			}

			int realSurfaces = doc.KnownChunks.Msur.Count(s => s.Ck24 != 0 && s.IndexCount >= 3);
			if (matchedPlacements > 0)
			{
				tileResults.Add(new Dictionary<string, object?>
				{
					["tileX"] = tileX,
					["tileY"] = tileY,
					["pm4File"] = Path.GetFileName(pm4Path),
					["adtFile"] = Path.GetFileName(adtPath),
					["realSurfaces"] = realSurfaces,
					["genTotalSurfaces"] = genTotalSurfaces,
					["genTotalVerts"] = genTotalVerts,
					["matchedPlacements"] = matchedPlacements,
					["totalPlacements"] = placements.WorldModelPlacements.Count,
					["surfacesMatchPct"] = realSurfaces > 0
						? Math.Round((double)Math.Min(genTotalSurfaces, realSurfaces) / Math.Max(genTotalSurfaces, realSurfaces) * 100, 2)
						: 0.0,
					["placements"] = placementResults,
				});
			}

			processed++;
			if (processed % 10 == 0 || processed == 1)
				Console.Error.WriteLine($"  [{processed}] tile ({tileX},{tileY}) realSurfaces={realSurfaces} genSurfaces={genTotalSurfaces} matched={matchedPlacements}");
		}
		catch (Exception ex)
		{
			failed++;
		}
	}

	sw.Stop();

	double avgMatch = tileResults.Count > 0 ? tileResults.Average(static t => (double)t["surfacesMatchPct"]!) : 0;
	int totalReal = tileResults.Sum(static t => (int)t["realSurfaces"]!);
	int totalGen = tileResults.Sum(static t => (int)t["genTotalSurfaces"]!);
	int totalPlacements = tileResults.Sum(static t => (int)t["matchedPlacements"]!);

	Console.WriteLine($"\n=== Generator Validation Complete ===");
	Console.WriteLine($"Tiles processed: {processed}");
	Console.WriteLine($"Tiles with matches: {tileResults.Count}");
	Console.WriteLine($"Skipped (0-byte PM4): {skippedNoData}");
	Console.WriteLine($"Skipped (no ADT): {skippedNoAdt}");
	Console.WriteLine($"Skipped (no placements): {skippedNoPlacements}");
	Console.WriteLine($"Failed: {failed}");
	Console.WriteLine($"Total real surfaces across all tiles: {totalReal}");
	Console.WriteLine($"Total generated surfaces: {totalGen}");
	Console.WriteLine($"Total matched WMO placements: {totalPlacements}");
	Console.WriteLine($"Average surface count match: {avgMatch:F1}%");
	Console.WriteLine($"Overall surface ratio (gen/real): {(totalReal > 0 ? (double)totalGen / totalReal : 0):F3}");
	Console.WriteLine($"Time: {sw.Elapsed.TotalSeconds:F1}s");

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
		File.WriteAllText(outputPath, JsonSerializer.Serialize(new Dictionary<string, object?>
		{
			["summary"] = new Dictionary<string, object>
			{
				["tilesProcessed"] = processed,
				["tilesWithMatches"] = tileResults.Count,
				["totalRealSurfaces"] = totalReal,
				["totalGenSurfaces"] = totalGen,
				["totalMatchedPlacements"] = totalPlacements,
				["avgMatchPct"] = Math.Round(avgMatch, 1),
				["overallRatio"] = Math.Round(totalReal > 0 ? (double)totalGen / totalReal : 0, 3),
				["timeSeconds"] = Math.Round(sw.Elapsed.TotalSeconds, 1),
			},
			["tiles"] = tileResults,
		}, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
	}
}

static void RunPm4ValidateGeneratorGeometry(string[] args)
{
	string? pm4Path = GetOption(args, "--pm4", "-p") ?? GetFirstPositionalArgument(args);
	string? adtPath = GetOption(args, "--adt", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	string? binSizeText = GetOption(args, "--bin-size", "-b");
	string? areaBinSizeText = GetOption(args, "--area-bin-size", "-ab");
	string? normalAlignmentBinSizeText = GetOption(args, "--normal-alignment-bin-size", "-na");
	string? planarOffsetBinSizeText = GetOption(args, "--planar-offset-bin-size", "-po");

	if (string.IsNullOrWhiteSpace(pm4Path) || !File.Exists(pm4Path))
	{
		Console.Error.WriteLine("Error: --pm4 <file.pm4> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(adtPath) || !File.Exists(adtPath))
	{
		Console.Error.WriteLine("Error: --adt <tile_obj0.adt> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <staged client dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions bootstrapOptions))
		return;

	float binSize = float.TryParse(binSizeText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float bs) ? bs : 1.0f;
	float areaBinSize = float.TryParse(areaBinSizeText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float absv) ? absv : 1.0f;
	float normalAlignmentBinSize = float.TryParse(normalAlignmentBinSizeText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float nas) ? nas : 0.0f;
	float planarOffsetBinSize = float.TryParse(planarOffsetBinSizeText, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float pos) ? pos : 0.0f;

	Console.WriteLine($"Validating PM4 generator geometry for: {Path.GetFullPath(pm4Path)}");
	Console.WriteLine($"  ADT placements: {Path.GetFullPath(adtPath)}");
	Console.WriteLine($"  Archive root: {Path.GetFullPath(archiveRoot)}");
	Console.WriteLine($"  Bins: edge={binSize}, area={areaBinSize}, normalAlign={normalAlignmentBinSize}, planarOffset={planarOffsetBinSize}");

	Pm4GeneratorValidationResult result = Pm4GeneratorValidationSupport.ValidateTile(
		pm4Path, adtPath, archiveRoot, bootstrapOptions,
		binSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize,
		progress: msg => Console.WriteLine($"  {msg}"));

	Console.WriteLine($"\n=== Generator Geometry Validation Report ===");
	Console.WriteLine($"Real CK24 groups: {result.RealCk24GroupCount}");
	Console.WriteLine($"ADT WMO placements: {result.AdtWmoPlacementCount}");
	Console.WriteLine($"Generated groups with geometry: {result.GeneratedGroupCount}");
	Console.WriteLine($"Matched groups (score >= 0.50): {result.MatchedGroupCount}");
	Console.WriteLine($"Mean symmetric score: {result.MeanSymmetricScore:F3}");
	Console.WriteLine($"Mean PM4 coverage: {result.MeanPm4Coverage:F3}");
	Console.WriteLine($"Mean WMO coverage: {result.MeanWmoCoverage:F3}");

	Console.WriteLine($"\nTop 20 matched/subthreshold validations:");
	Console.WriteLine($"  {"WMO",-50} {"GenTris",-8} {"RealCK24",-10} {"Score",-8} {"PM4Cov",-8} {"WMOCov",-8} {"Status",-12}");
	foreach (Pm4GeneratorGroupValidation g in result.GroupValidations
		.Where(static g => g.Status is "matched" or "subthreshold")
		.OrderByDescending(static g => g.SymmetricScore ?? 0)
		.Take(20))
	{
		string shortWmo = g.WmoPath.Length > 48 ? "..." + g.WmoPath[^47..] : g.WmoPath;
		string ck24 = g.MatchedRealCk24.HasValue ? $"0x{g.MatchedRealCk24.Value:X6}" : "n/a";
		Console.WriteLine($"  {shortWmo,-50} {g.GeneratedTriangleCount,8} {ck24,-10} {g.SymmetricScore ?? 0,8:F3} {g.Pm4Coverage ?? 0,8:F3} {g.WmoCoverage ?? 0,8:F3} {g.Status,-12}");
	}

	if (result.Warnings.Count > 0)
	{
		Console.WriteLine($"\nWarnings ({result.Warnings.Count}):");
		foreach (string warning in result.Warnings.Take(20))
			Console.WriteLine($"  {warning}");
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote validation report to {outputPath}");
	}
}

static void RunPm4GenerateFromWmo(string[] args)
{
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	string? wmoPath = GetOption(args, "--wmo-path", "-w") ?? GetOption(args, "--wmo-root");
	string? positionText = GetOption(args, "--position", "-p");
	string? rotationText = GetOption(args, "--rotation", "-r");
	string? tileText = GetOption(args, "--tile", "-t");
	string? archiveRoot = GetOption(args, "--archive-root", "-a");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(wmoPath))
	{
		Console.Error.WriteLine("Error: --wmo-path <virtual-wmo-path> is required (e.g. World/wmo/northrend/wintergrasp/buildings/guardtower/guardtower_intact.wmo).");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(positionText))
	{
		Console.Error.WriteLine("Error: --position <x,y,z> is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!TryParseVector3(positionText, out Vector3 position))
	{
		Console.Error.WriteLine("Error: --position must be three comma-separated floats <x,y,z>.");
		Environment.ExitCode = 1;
		return;
	}

	Vector3 rotation = Vector3.Zero;
	if (!string.IsNullOrWhiteSpace(rotationText) && !TryParseVector3(rotationText, out rotation))
	{
		Console.Error.WriteLine("Error: --rotation must be three comma-separated floats <rx,ry,rz>.");
		Environment.ExitCode = 1;
		return;
	}

	int tileX = 0, tileY = 0;
	if (!string.IsNullOrWhiteSpace(tileText))
	{
		string[] parts = tileText.Split(',', 'x');
		if (parts.Length != 2 || !int.TryParse(parts[0], out tileX) || !int.TryParse(parts[1], out tileY))
		{
			Console.Error.WriteLine("Error: --tile must be <x,y>.");
			Environment.ExitCode = 1;
			return;
		}
	}

	try
	{
		string normalizedPath = wmoPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
		byte[] wmoBytes = ArchiveVirtualFileReader.ReadVirtualFile(
			normalizedPath, [archiveRoot], archiveBootstrapOptions);

		Func<string, byte[]?> assetReader = virtualPath =>
		{
			try
			{
				return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], archiveBootstrapOptions);
			}
			catch { return null; }
		};

		WmoRenderDocument renderDoc = WmoRenderDocumentReader.Read(
			new MemoryStream(wmoBytes, writable: false), normalizedPath, assetReader);

		Pm4GenerationData genData = Pm4Generator.GenerateFromWmo(
			renderDoc, position, rotation, scale: 1f,
			ck24Type: 0x43, ck24ObjectId: 1, regionId: 0);

		if (genData.Msur.Count == 0)
		{
			Console.Error.WriteLine("Error: WMO has no collision geometry.");
			Environment.ExitCode = 1;
			return;
		}

		byte[] pm4Bytes = Pm4BinaryWriter.Write(genData);

		if (!string.IsNullOrWhiteSpace(output))
		{
			Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
			File.WriteAllBytes(output, pm4Bytes);
			Console.WriteLine($"Wrote PM4 ({pm4Bytes.Length} bytes) to {output}");
		}
		else
		{
			string defaultName = Path.GetFileNameWithoutExtension(wmoPath) + ".pm4";
			string defaultPath = Path.Combine(Directory.GetCurrentDirectory(), defaultName);
			File.WriteAllBytes(defaultPath, pm4Bytes);
			Console.WriteLine($"Wrote PM4 ({pm4Bytes.Length} bytes) to {defaultPath}");
		}

		Console.WriteLine($"  Vertices: {genData.Msvt.Count}");
		Console.WriteLine($"  Indices: {genData.Msvi.Count}");
		Console.WriteLine($"  Surfaces: {genData.Msur.Count}");
	}
	catch (Exception ex) when (ex is not NullReferenceException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4TestGenerator()
{
	var verts = new List<Vector3>
	{
		new(100, 100, 0),
		new(200, 100, 10),
		new(150, 200, 20),
		new(100, 100, 30),
	};
	var indices = new List<ushort> { 0, 1, 2, 0, 2, 3 };

	var genData = Pm4Generator.GenerateFromCollisionMesh(
		verts, indices,
		placementPosition: Vector3.Zero,
		placementRotationDegrees: Vector3.Zero,
		scale: 1f,
		ck24Type: 0x43, ck24ObjectId: 1,
		regionId: 0);

	byte[] pm4Bytes = Pm4BinaryWriter.Write(genData);
	Console.WriteLine($"Generated PM4: {pm4Bytes.Length} bytes");
	Console.WriteLine($"  Vertices: {genData.Msvt.Count}");
	Console.WriteLine($"  Indices: {genData.Msvi.Count}");
	Console.WriteLine($"  Surfaces: {genData.Msur.Count}");

	var doc = Pm4ResearchReader.Read(pm4Bytes, "generated.pm4");
	Console.WriteLine($"\nRead back: version={doc.Version}, chunks={doc.Chunks.Count}");
	Console.WriteLine($"  MSVT: {doc.KnownChunks.Msvt.Count} vertices");
	Console.WriteLine($"  MSVI: {doc.KnownChunks.Msvi.Count} indices");
	Console.WriteLine($"  MSUR: {doc.KnownChunks.Msur.Count} surfaces");
	Console.WriteLine($"  MSCN: {doc.KnownChunks.Mscn.Count} points");
	Console.WriteLine($"  MPRL: {doc.KnownChunks.Mprl.Count} entries");
	Console.WriteLine($"  MSLK: {doc.KnownChunks.Mslk.Count} links");

	if (doc.KnownChunks.Msur.Count > 0)
	{
		var first = doc.KnownChunks.Msur[0];
		Console.WriteLine($"\nFirst MSUR: Ck24=0x{first.Ck24:X6} Type=0x{first.Ck24Type:X2} ObjID={first.Ck24ObjectId} IndexCount={first.IndexCount}");
	}

	if (doc.Diagnostics.Count > 0)
	{
		Console.WriteLine($"\nDiagnostics ({doc.Diagnostics.Count}):");
		foreach (var d in doc.Diagnostics)
			Console.WriteLine($"  {d}");
	}
	else
	{
		Console.WriteLine("\nNo diagnostics — clean round-trip!");
	}
}

static bool TryParseVector3(string text, out Vector3 result)
{
	result = Vector3.Zero;
	if (string.IsNullOrWhiteSpace(text))
		return false;

	string[] parts = text.Split(',');
	if (parts.Length != 3)
		return false;

	if (!float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out float x) ||
		!float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float y) ||
		!float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
		return false;

	result = new Vector3(x, y, z);
	return true;
}

static int? TryParseInt(string? text)
{
	if (string.IsNullOrWhiteSpace(text)) return null;
	if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
		return value;
	return null;
}

static void RunPm4SynthesizePlacements(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? assetCorpus = GetOption(args, "--asset-corpus", "-c");
	string? output = GetOption(args, "--output", "-o");
	string? targetTilesText = GetOption(args, "--target-tiles", "-t");
	string? maxCandidatesText = GetOption(args, "--max-candidates", "-n");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	int maxCandidates = 10;
	if (!string.IsNullOrWhiteSpace(maxCandidatesText) && (!int.TryParse(maxCandidatesText, out maxCandidates) || maxCandidates <= 0))
	{
		Console.Error.WriteLine("Error: --max-candidates must be a positive integer.");
		Environment.ExitCode = 1;
		return;
	}

	IReadOnlyList<string> targetTiles = ParseCsvOption(targetTilesText);
	if (targetTiles.Count == 0)
	{
		Console.Error.WriteLine("Error: provide at least one tile in --target-tiles <x_y[,x_y...]>");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!File.Exists(input))
	{
		Console.Error.WriteLine($"Error: PM4 input '{input}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(assetCorpus) && !string.IsNullOrWhiteSpace(placements))
	{
		Console.Error.WriteLine("Error: choose either --asset-corpus <report.json> or --placements <tile_obj0.adt>, not both.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(assetCorpus) && !File.Exists(assetCorpus))
	{
		Console.Error.WriteLine($"Error: asset corpus '{assetCorpus}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required for pm4 synthesize-placements so WMO/M2 assets can be read from the staged client.");
		Environment.ExitCode = 1;
		return;
	}

	if (!Pm4CoordinateService.TryParseTileCoordinates(input, out int tileX, out int tileY))
	{
		Console.Error.WriteLine("Error: could not derive tile coordinates from the PM4 filename.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && string.IsNullOrWhiteSpace(placements))
	{
		string fileName = Path.GetFileNameWithoutExtension(input);
		int lastUnderscore = fileName.LastIndexOf('_');
		int previousUnderscore = lastUnderscore > 0 ? fileName.LastIndexOf('_', lastUnderscore - 1) : -1;
		string mapName = previousUnderscore > 0 ? fileName[..previousUnderscore] : fileName;
		placements = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? string.Empty, $"{mapName}_{tileX}_{tileY}_obj0.adt");
	}

	if (string.IsNullOrWhiteSpace(assetCorpus) && !File.Exists(placements))
	{
		Console.Error.WriteLine($"Error: placement source '{placements}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		Pm4SegmentExportRun exportRun = Pm4SegmentExportService.Export(input);
		Pm4SegmentExportFile file = AssertSinglePm4ExportFile(exportRun, input);
		Pm4AssetReferenceBuildResult assetBuild;
		string assetReferenceSource;
		if (!string.IsNullOrWhiteSpace(assetCorpus))
		{
			assetBuild = Pm4AssetSignalCorpusSupport.LoadFromManifest(assetCorpus);
			assetReferenceSource = assetCorpus;
		}
		else
		{
			assetBuild = Pm4AssetReferenceSupport.BuildFromPlacements(placements!, archiveRoot!, archiveBootstrapOptions, tileX, tileY);
			assetReferenceSource = placements!;
		}

		IReadOnlyList<Pm4SegmentMatchResult> matchResults = Pm4AssetMatchScorer.ScoreSegments(file.Segments, assetBuild.Assets, maxCandidates);
		IReadOnlyList<Pm4ReplacementPlacementProposal> placementProposals = Pm4ReplacementPlacementSynthesizer.Synthesize(matchResults, assetBuild.Assets, targetTiles);
		Pm4MatchRunManifest manifest = BuildPm4AssetMatchManifest(
			exportRun,
			matchResults,
			placementProposals,
			assetReferenceSource,
			assetBuild.Warnings,
			"synthesize-placements");

		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			WritePm4Report(manifest, outputPath);
			Console.WriteLine($"Synthesized {placementProposals.Count} placement proposals from {manifest.SegmentCount} PM4 segments using {assetBuild.Assets.Count} asset references.");
			return;
		}

		PrintPm4AssetMatchRun(manifest, assetBuild.Assets.Count);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4ExportAssetSignals(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	string? kind = GetOption(args, "--kind", "-k");
	string? pathFilter = GetOption(args, "--path-filter", "-f");
	string? seedPlacements = GetOption(args, "--seed-placements", "-s");
	string? limitText = GetOption(args, "--limit", "-n");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (!Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine($"Error: archive root '{archiveRoot}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	int? limit = null;
	if (!string.IsNullOrWhiteSpace(limitText))
	{
		if (!int.TryParse(limitText, out int parsedLimit) || parsedLimit <= 0)
		{
			Console.Error.WriteLine("Error: --limit must be a positive integer.");
			Environment.ExitCode = 1;
			return;
		}

		limit = parsedLimit;
	}

	try
	{
		Pm4AssetSignalCorpusManifest manifest = Pm4AssetSignalCorpusSupport.BuildFromArchive(archiveRoot, archiveBootstrapOptions, kind, pathFilter, limit, seedPlacements);

		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			string? directory = Path.GetDirectoryName(outputPath);
			if (!string.IsNullOrWhiteSpace(directory))
				Directory.CreateDirectory(directory);

			File.WriteAllText(outputPath, JsonSerializer.Serialize(manifest, Pm4MatchSupport.CreateJsonOptions()));
			Console.WriteLine($"Wrote {outputPath}");
			Console.WriteLine($"Exported {manifest.AssetCount} durable asset signals from {manifest.ClientBuild}.");
			return;
		}

		PrintPm4AssetSignalCorpus(manifest);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException or ArgumentOutOfRangeException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunPm4ExportSegments(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file or directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		Pm4SegmentExportRun exportRun = Pm4SegmentExportService.Export(input);
		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			Pm4MatchRunManifest manifest = BuildPm4SegmentExportManifest(exportRun);
			WritePm4Report(manifest, outputPath);
			Console.WriteLine($"Exported {exportRun.SegmentCount} PM4 segments from {exportRun.FileCount} file(s).");
			return;
		}

		PrintPm4SegmentExportRun(exportRun);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
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

static void RunPm4AnalyzeSimplification(string[] args)
{
	string? pm4Path = GetOption(args, "--pm4", "-p") ?? GetFirstPositionalArgument(args);
	string? adtPath = GetOption(args, "--adt", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? output = GetOption(args, "--output", "-o");
	bool includeFullGeometry = HasOption(args, "--full-geometry");
	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions archiveBootstrapOptions))
		return;

	if (string.IsNullOrWhiteSpace(pm4Path) || !File.Exists(pm4Path))
	{
		Console.Error.WriteLine("Error: --pm4 <file.pm4> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}
	if (string.IsNullOrWhiteSpace(adtPath) || !File.Exists(adtPath))
	{
		Console.Error.WriteLine("Error: --adt <file.adt> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}
	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	try
	{
		Pm4CorrelateModelsResult correlation = Pm4CorrelateModelsSupport.Correlate(
			pm4Path, adtPath, archiveRoot, archiveBootstrapOptions, null, null);

		Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(pm4Path);
		List<Dictionary<string, object?>> comparisons = [];

		var strongCorrelations = correlation.Correlations
			.Where(c => c.WowBoundsOverlap > 0.8 && c.AssetKind == "wmo")
			.ToList();

		Console.WriteLine($"Strong correlations (WoW overlap > 0.8): {strongCorrelations.Count}");

		var wmoPlacements = AdtPlacementReader.Read(adtPath).WorldModelPlacements;

		foreach (Pm4CorrelationEntry corr in strongCorrelations)
		{
			var placement = wmoPlacements.FirstOrDefault(p => p.UniqueId == corr.UniqueId);
			if (placement is null) continue;

			Pm4Ck24GeometryExport pm4Geo = Pm4CorrelateModelsSupport.ExportCk24GroupGeometry(
				document, corr.Ck24);
			Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY);
			WmoMeshInPm4Space wmoGeo = Pm4CorrelateModelsSupport.ReadWmoInPm4Space(
				placement, archiveRoot, archiveBootstrapOptions, tileX, tileY);

			double vertRatio = pm4Geo.VertexCount > 0
				? (double)wmoGeo.WmoPm4Verts.Count / pm4Geo.VertexCount : 0;
			double faceRatio = pm4Geo.SurfaceCount > 0
				? (double)wmoGeo.FaceMaterials.Count / pm4Geo.SurfaceCount : 0;

			var indexCountDist = pm4Geo.Pm4IndexCounts
				.GroupBy(static c => c)
				.Select(static g => new { indexCount = (int)g.Key, count = g.Count() })
				.OrderByDescending(static x => x.count)
				.ToList();

			Dictionary<string, object?> entry = new()
			{
				["ck24"] = $"0x{corr.Ck24:X6}",
				["ck24Type"] = $"0x{corr.Ck24Type:X2}",
				["ck24ObjectId"] = (int)(corr.Ck24 & 0xFFFF),
				["uniqueId"] = corr.UniqueId,
				["modelPath"] = corr.ModelPath,
				["wowOverlap"] = corr.WowBoundsOverlap,
				["pm4Surfaces"] = pm4Geo.SurfaceCount,
				["pm4Vertices"] = pm4Geo.VertexCount,
				["pm4IndexCountDist"] = indexCountDist,
				["wmoFaces"] = wmoGeo.FaceMaterials.Count,
				["wmoVertices"] = wmoGeo.WmoPm4Verts.Count,
				["wmoGroups"] = wmoGeo.GroupCount,
				["vertRatio"] = Math.Round(vertRatio, 3),
				["faceRatio"] = Math.Round(faceRatio, 3),
			};

			if (includeFullGeometry)
			{
				entry["pm4Vertices"] = pm4Geo.Pm4Vertices.Select(v => new[] { v.X, v.Y, v.Z }).ToList();
				entry["pm4Indices"] = pm4Geo.Pm4CornerIndices.ToList();
				entry["wmoPm4Verts"] = wmoGeo.WmoPm4Verts.Select(v => new[] { v.X, v.Y, v.Z }).ToList();
				entry["wmoIndices"] = wmoGeo.Indices.ToList();
				entry["wmoFaceFlags"] = wmoGeo.FaceMaterials.Select(f => new { f.FaceIndex, f.Flags, f.MaterialId }).ToList();
			}

			comparisons.Add(entry);
		}

		Dictionary<string, object?> summary = new()
		{
			["pm4File"] = Path.GetFileName(pm4Path),
			["adtFile"] = Path.GetFileName(adtPath),
			["tile"] = new { x = correlation.TileX, y = correlation.TileY },
			["totalCk24Groups"] = correlation.Ck24Groups.Count,
			["totalPlacements"] = correlation.PlacementSummaries.Count,
			["warnings"] = correlation.Warnings.ToList(),
			["comparisons"] = comparisons,
		};

		string json = JsonSerializer.Serialize(summary, new JsonSerializerOptions
		{
			WriteIndented = true,
			IncludeFields = true,
		});

		if (!string.IsNullOrWhiteSpace(output))
		{
			string outputPath = Path.GetFullPath(output);
			string? dir = Path.GetDirectoryName(outputPath);
			if (!string.IsNullOrWhiteSpace(dir))
				Directory.CreateDirectory(dir);
			File.WriteAllText(outputPath, json);
			Console.WriteLine($"Wrote {outputPath}");
			return;
		}

		Console.WriteLine(json);
	}
	catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException or DirectoryNotFoundException or FileNotFoundException)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
	}
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

static void RunPm4BondStats(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(input);
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

	PrintPm4BondStatsReport(report);
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

static void RunPm4ExportObj(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4/PD4 file or directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (File.Exists(input))
	{
		string objText = ExportPm4FileToObj(input);
		string outputPath = !string.IsNullOrWhiteSpace(output)
			? Path.GetFullPath(output)
			: Path.ChangeExtension(input, ".obj");

		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		File.WriteAllText(outputPath, objText, System.Text.Encoding.UTF8);
		Console.WriteLine($"Wrote 3D OBJ mesh: {outputPath}");
	}
	else if (Directory.Exists(input))
	{
		string outDir = !string.IsNullOrWhiteSpace(output) ? Path.GetFullPath(output) : input;
		Directory.CreateDirectory(outDir);

		var files = Directory.EnumerateFiles(input, "*.*", SearchOption.AllDirectories)
			.Where(f => f.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".pd4", StringComparison.OrdinalIgnoreCase))
			.ToList();

		if (files.Count == 0)
		{
			Console.WriteLine($"No .pm4 or .pd4 files found under {input}");
			return;
		}

		int writtenCount = 0;
		foreach (string filePath in files)
		{
			try
			{
				string objText = ExportPm4FileToObj(filePath);
				string relativePath = Path.GetRelativePath(input, filePath);
				string targetObjPath = Path.Combine(outDir, Path.ChangeExtension(relativePath, ".obj"));
				string? targetDir = Path.GetDirectoryName(targetObjPath);
				if (!string.IsNullOrWhiteSpace(targetDir))
					Directory.CreateDirectory(targetDir);

				File.WriteAllText(targetObjPath, objText, System.Text.Encoding.UTF8);
				writtenCount++;
			}
			catch (Exception ex)
			{
				Console.Error.WriteLine($"Failed to export '{filePath}': {ex.Message}");
			}
		}

		Console.WriteLine($"Exported {writtenCount}/{files.Count} PM4/PD4 files as OBJ meshes under {outDir}");
	}
	else
	{
		Console.Error.WriteLine($"Error: input path '{input}' does not exist.");
		Environment.ExitCode = 1;
	}
}

static string ExportPm4FileToObj(string filePath)
{
	Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(filePath);
	var sb = new System.Text.StringBuilder();
	string sourceName = Path.GetFileName(filePath);
	sb.AppendLine($"# OBJ exported from {sourceName}");
	sb.AppendLine($"# PM4/PD4 version: {doc.Version}");

	var vertices = doc.KnownChunks.Msvt;
	var indices = doc.KnownChunks.Msvi;
	var surfaces = doc.KnownChunks.Msur;

	sb.AppendLine($"# Vertices: {vertices.Count}, Surfaces: {surfaces.Count}");

	for (int i = 0; i < vertices.Count; i++)
	{
		Vector3 v = vertices[i];
		sb.AppendLine(CultureInfo.InvariantCulture, $"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
	}

	int triangleCount = 0;
	foreach (var surface in surfaces)
	{
		int firstIndex = (int)surface.MsviFirstIndex;
		int indexCount = surface.IndexCount;
		if (indexCount < 3 || firstIndex < 0 || firstIndex + indexCount > indices.Count)
			continue;

		uint i0 = indices[firstIndex];
		if (i0 >= vertices.Count)
			continue;

		for (int idx = firstIndex + 1; idx + 1 < firstIndex + indexCount; idx++)
		{
			uint i1 = indices[idx];
			uint i2 = indices[idx + 1];
			if (i1 >= vertices.Count || i2 >= vertices.Count)
				continue;

			sb.AppendLine(CultureInfo.InvariantCulture, $"f {i0 + 1} {i1 + 1} {i2 + 1}");
			triangleCount++;
		}
	}

	var mspv = doc.KnownChunks.Mspv;
	var mspi = doc.KnownChunks.Mspi;
	var mslk = doc.KnownChunks.Mslk;
	if (mspv.Count > 0 && mspi.Count > 0)
	{
		sb.AppendLine($"# MSPV Path Vertices: {mspv.Count}");
		int mspvOffset = vertices.Count;
		for (int i = 0; i < mspv.Count; i++)
		{
			Vector3 v = mspv[i];
			sb.AppendLine(CultureInfo.InvariantCulture, $"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
		}

		foreach (var link in mslk)
		{
			if (link.MspiFirstIndex < 0 || link.MspiIndexCount < 2)
				continue;

			int firstMspi = link.MspiFirstIndex;
			int countMspi = link.MspiIndexCount;
			if (firstMspi + countMspi > mspi.Count)
				continue;

			sb.Append("l");
			for (int k = 0; k < countMspi; k++)
			{
				uint pIdx = mspi[firstMspi + k];
				if (pIdx < mspv.Count)
					sb.Append(CultureInfo.InvariantCulture, $" {mspvOffset + pIdx + 1}");
			}

			sb.AppendLine();
		}
	}

	return sb.ToString();
}

static bool TryParseUInt32Flexible(string value, out uint parsed)
{
	if (value.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		return uint.TryParse(value[2..], System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out parsed);

	return uint.TryParse(value, out parsed);
}

static string? GetOption(string[] args, string longName, string? shortName = null)
{
	for (int index = 0; index < args.Length - 1; index++)
	{
		if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
			|| (shortName is not null && string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase)))
		{
			return args[index + 1];
		}
	}

	return null;
}

static bool HasFlag(string[] args, string longName)
{
	foreach (string arg in args)
	{
		if (string.Equals(arg, longName, StringComparison.OrdinalIgnoreCase))
			return true;
	}

	return false;
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
	Console.WriteLine($"Version: {Pm4VersionFormatter.Format(report.Version)}");
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

	if (report.TopInvalidMscnRefClusters.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top invalid-MSCN-ref clusters:");
		foreach (Pm4MscnClusterExample cluster in report.TopInvalidMscnRefClusters.Take(8))
		{
			Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} invalidMscnRef={cluster.InvalidMscnRefCount} distinctMscnRef={cluster.DistinctMscnRefCount} align={cluster.AlignmentMode}");
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

static void RunPm4Mprr(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MprrReport report = Pm4MprrAnalyzer.AnalyzeDirectory(input);
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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 MPRR analysis");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Files with MPRR: {report.FilesWithMprr}");
	Console.WriteLine($"Entries: {report.TotalEntries}  sentinels: {report.SentinelEntries}  non-sentinel: {report.NonSentinelEntries}");
	Console.WriteLine($"Sentinel-delimited runs: {report.TotalRuns}");
	Console.WriteLine();

	Console.WriteLine("STRUCTURAL — files where run count == chunk entry count:");
	foreach (Pm4MprrRunCountMatch match in report.RunCountMatches)
		Console.WriteLine($"  {match.Domain,-6} {match.FilesMatching,5}/{match.FilesTotal,-5} ({match.MatchFraction:P1})");
	Console.WriteLine();

	Console.WriteLine("Value1 bound fits by domain:");
	foreach (Pm4MprrDomainFit fit in report.Value1DomainFits)
		Console.WriteLine($"  {fit.Domain,-6} fits={fit.Fits,12} misses={fit.Misses,12} ({fit.FitFraction:P1})");
	Console.WriteLine();

	Console.WriteLine("Value2 bound fits by domain:");
	foreach (Pm4MprrDomainFit fit in report.Value2DomainFits)
		Console.WriteLine($"  {fit.Domain,-6} fits={fit.Fits,12} misses={fit.Misses,12} ({fit.FitFraction:P1})");
	Console.WriteLine();

	Console.WriteLine($"Value1 < its own run length : {report.Value1WithinRunLengthFraction:P1}");
	Console.WriteLine($"Value1 < its own run index  : {report.Value1WithinRunIndexFraction:P1}");
	Console.WriteLine();

	Console.WriteLine("Run length histogram (top):");
	foreach (Pm4ValueFrequency bucket in report.RunLengthHistogram)
		Console.WriteLine($"  len={bucket.Value,-6} runs={bucket.Count}");
	Console.WriteLine();

	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

static void RunPm4BoundsAudit(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (HasFlag(args, "--by-region"))
	{
		RunPm4BoundsAuditByRegion(args, input, output);
		return;
	}

	Pm4BoundsAuditReport report = Pm4BoundsAuditAnalyzer.AnalyzeDirectory(input);
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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 tile-bounds audit");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Files with geometry: {report.FilesWithGeometry}, overflowing their tile: {report.FilesOverflowing}");
	Console.WriteLine($"Vertices: {report.VerticesOutside}/{report.VerticesTotal} outside tile bounds ({report.OutsideFraction:P3})");
	Console.WriteLine();

	Pm4BoundsSideSummary side = report.SideSummary;
	Console.WriteLine("Spill by side (yards summed over tiles / tiles affected):");
	Console.WriteLine($"  -X {side.TotalNegX,12:F1} / {side.TilesNegX,4}");
	Console.WriteLine($"  +X {side.TotalPosX,12:F1} / {side.TilesPosX,4}");
	Console.WriteLine($"  -Z {side.TotalNegZ,12:F1} / {side.TilesNegZ,4}");
	Console.WriteLine($"  +Z {side.TotalPosZ,12:F1} / {side.TilesPosZ,4}");
	Console.WriteLine();

	Console.WriteLine("Worst tiles by spill:");
	foreach (Pm4TileBoundsRecord tile in report.WorstTiles)
	{
		Console.WriteLine($"  {tile.FileName} tile=({tile.TileX},{tile.TileY}) outside={tile.VerticesOutside}/{tile.VertexCount} ({tile.OutsideFraction:P1})");
		Console.WriteLine($"    x {tile.MinX:F1}..{tile.MaxX:F1}  z {tile.MinZ:F1}..{tile.MaxZ:F1}  spill -X={tile.SpillNegX:F1} +X={tile.SpillPosX:F1} -Z={tile.SpillNegZ:F1} +Z={tile.SpillPosZ:F1}");
	}

	Console.WriteLine();
	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

/// <summary>
/// `pm4 bounds-audit --by-region` — groups MSVT by MSHD.Field04 and reports, per region, the frame
/// the placement fitter resolves and how far it moves geometry off the ADT-verified canonical one.
/// </summary>
static void RunPm4BoundsAuditByRegion(string[] args, string input, string? output)
{
	string? placementsDirectory = GetOption(args, "--placements");
	string resolvedInput = Pm4CoordinateService.ResolveMapDirectory(input);
	string resolvedPlacements = string.IsNullOrWhiteSpace(placementsDirectory)
		? resolvedInput
		: Pm4CoordinateService.ResolveMapDirectory(placementsDirectory);

	Dictionary<string, IReadOnlyList<Vector2>>? referencePoints =
		LoadAdtReferencePlacements(resolvedInput, resolvedPlacements, out int pairedFiles, out int missingFiles);

	Pm4RegionFrameAuditReport report = Pm4RegionFrameAuditAnalyzer.AnalyzeDirectory(resolvedInput, referencePoints);

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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 region frame audit");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Files with geometry: {report.FilesWithGeometry}, objects: {report.ObjectCount}, regions: {report.DistinctRegionCount}");
	Console.WriteLine($"ADT placement pairs: {pairedFiles} paired, {missingFiles} without a companion _obj0.adt");
	Console.WriteLine();

	Console.WriteLine("Is the frame region-scoped?");
	Console.WriteLine($"  Regions spanning >1 file        : {report.MultiFileRegionCount}");
	Console.WriteLine($"  ...of those with mixed frames   : {report.MultiFileRegionsWithMixedFrames}");
	Console.WriteLine($"  Objects on the canonical frame  : {report.ObjectsOnCanonicalFrame}");
	Console.WriteLine($"  Objects off the canonical frame : {report.ObjectsOffCanonicalFrame}");
	Console.WriteLine($"  Files off their raw filename band: {report.FilesOffRawBand} (baseline; expected 0)");
	Console.WriteLine();

	Console.WriteLine("Resolved frame families (corpus):");
	foreach (Pm4FrameFamilyCount family in report.CorpusFrames)
		Console.WriteLine($"  {family.Frame,-22} {family.ObjectCount,8} objects");
	Console.WriteLine();

	Console.WriteLine("Whole-tile displacement caused by the resolved frame (corpus):");
	foreach (Pm4TileOffsetFamilyCount family in report.CorpusTileOffsets.Take(16))
		Console.WriteLine($"  ({family.OffsetX,3},{family.OffsetY,3}) {family.ObjectCount,8} objects");
	Console.WriteLine();

	if (report.ReferencePlacements > 0)
	{
		Console.WriteLine($"ADT placements inside their PM4's footprint: {report.ReferencePlacementsInside}/{report.ReferencePlacements} ({report.ReferenceAgreement:P1})");
		Console.WriteLine();
	}

	Console.WriteLine("Worst regions by off-canonical objects:");
	foreach (Pm4RegionFrameSummary region in report.Regions
		.OrderByDescending(static region => region.Frames.Count)
		.ThenByDescending(static region => region.ObjectCount)
		.Take(20))
	{
		string label = region.IsSharedBucket ? " [shared bucket]" : region.IsEmptyStubRegion ? " [empty stub]" : string.Empty;
		Console.WriteLine($"  region={region.RegionId,-6}{label} files={region.FileCount} objects={region.ObjectCount} frames={region.Frames.Count} homogeneous={region.IsFrameHomogeneous}");
		foreach (Pm4FrameFamilyCount family in region.Frames.Take(4))
			Console.WriteLine($"      {family.Frame,-22} {family.ObjectCount,7}");
	}

	Console.WriteLine();
	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

/// <summary>
/// Reads each PM4's companion <c>_obj0.adt</c> and returns its MDDF/MODF positions in ADT placement
/// space, keyed by PM4 file name. Lives in the tool because <c>WowViewer.Core.IO</c> already
/// references <c>WowViewer.Core.PM4</c>, so the analyzer cannot read ADTs itself.
/// </summary>
static Dictionary<string, IReadOnlyList<Vector2>>? LoadAdtReferencePlacements(
	string pm4Directory,
	string placementsDirectory,
	out int pairedFiles,
	out int missingFiles)
{
	pairedFiles = 0;
	missingFiles = 0;

	if (!Directory.Exists(placementsDirectory))
		return null;

	Dictionary<string, IReadOnlyList<Vector2>> byFile = new(StringComparer.OrdinalIgnoreCase);

	foreach (string pm4Path in Directory.EnumerateFiles(pm4Directory, "*.pm4", SearchOption.TopDirectoryOnly))
	{
		string probe = Path.Combine(placementsDirectory, Path.GetFileName(pm4Path));
		string? obj0Path = Pm4CoordinateService.TryGetObj0PathForPm4(probe);
		if (obj0Path is null)
		{
			missingFiles++;
			continue;
		}

		AdtPlacementCatalog catalog;
		try
		{
			catalog = AdtPlacementReader.Read(obj0Path);
		}
		catch (Exception ex) when (ex is InvalidDataException or IOException)
		{
			missingFiles++;
			continue;
		}

		List<Vector2> points = [];
		foreach (AdtModelPlacement placement in catalog.ModelPlacements)
			points.Add(new Vector2(placement.Position.X, placement.Position.Y));
		foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
			points.Add(new Vector2(placement.Position.X, placement.Position.Y));

		if (points.Count == 0)
			continue;

		pairedFiles++;
		byFile[Path.GetFileName(pm4Path)] = points;
	}

	return byFile.Count == 0 ? null : byFile;
}

/// <summary>
/// `pm4 yaw-evidence` — decides whether the placement fitter's per-object yaw correction helps or
/// hurts, by scoring each object against the world bounding box of the WMO placement it stands in.
/// </summary>
static void RunPm4YawEvidence(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	string? placementsDirectory = GetOption(args, "--placements");
	string resolvedInput = Pm4CoordinateService.ResolveMapDirectory(input);
	string resolvedPlacements = string.IsNullOrWhiteSpace(placementsDirectory)
		? resolvedInput
		: Pm4CoordinateService.ResolveMapDirectory(placementsDirectory);

	Dictionary<string, IReadOnlyList<Pm4PlacementBox>> boxes = LoadAdtPlacementBoxes(resolvedInput, resolvedPlacements);
	if (boxes.Count == 0)
	{
		Console.Error.WriteLine(
			$"Error: no MODF world-model placements found under '{resolvedPlacements}'. "
			+ "This test needs WMO bounding boxes; MDDF doodad positions cannot score a rotation.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4YawEvidenceReport report = Pm4YawEvidenceAnalyzer.AnalyzeDirectory(resolvedInput, boxes);

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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 yaw-correction evidence");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Files with WMO boxes: {report.FilesScored}, objects seen: {report.ObjectsSeen}");
	Console.WriteLine($"  matched to a WMO box : {report.ObjectsMatched}");
	Console.WriteLine($"  unmatched            : {report.ObjectsUnmatched} (doodad collision has no box)");
	Console.WriteLine();

	Console.WriteLine("Of the matched objects:");
	Console.WriteLine($"  carry a yaw correction        : {report.ObjectsWithYaw}");
	Console.WriteLine($"  ...box can see a rotation     : {report.ObjectsDecidable}");
	Console.WriteLine($"  ...box cannot (excluded)      : {report.ObjectsWithoutPower}");
	Console.WriteLine();

	Console.WriteLine("Decidable objects — mean fraction of vertices inside their WMO box:");
	Console.WriteLine($"  canonical, no yaw        : {report.MeanInsideCanonical:P1}");
	Console.WriteLine($"  canonical + fitted yaw   : {report.MeanInsideYawOnly:P1}");
	Console.WriteLine($"  full resolved solution   : {report.MeanInsideResolved:P1}");
	Console.WriteLine($"  45 deg control (wrong)   : {report.MeanInsideControl45:P1}");
	Console.WriteLine();

	Console.WriteLine($"  yaw helps : {report.YawHelps}");
	Console.WriteLine($"  yaw hurts : {report.YawHurts}");
	Console.WriteLine($"  tie       : {report.Ties}");
	Console.WriteLine();
	Console.WriteLine($"VERDICT: {report.Verdict}");
	Console.WriteLine();

	if (report.WorstObjects.Count > 0)
	{
		Console.WriteLine("Objects the yaw moves furthest out of their box:");
		foreach (Pm4YawEvidenceObjectRecord record in report.WorstObjects)
		{
			Console.WriteLine($"  {record.FileName} ck24={record.Ck24} yaw={record.YawCorrectionDegrees,7:F1} deg  "
				+ $"inside {record.InsideCanonical:P0} -> {record.InsideYawOnly:P0} (control {record.InsideControl45:P0})  {record.Verdict}");
			Console.WriteLine($"      {record.ModelPath}");
		}

		Console.WriteLine();
	}

	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

/// <summary>
/// Reads MODF world-model placements from each PM4's companion <c>_obj0.adt</c> as bounding boxes in
/// ADT placement space, keyed by PM4 file name.
/// </summary>
static Dictionary<string, IReadOnlyList<Pm4PlacementBox>> LoadAdtPlacementBoxes(
	string pm4Directory,
	string placementsDirectory)
{
	Dictionary<string, IReadOnlyList<Pm4PlacementBox>> byFile = new(StringComparer.OrdinalIgnoreCase);
	if (!Directory.Exists(placementsDirectory))
		return byFile;

	foreach (string pm4Path in Directory.EnumerateFiles(pm4Directory, "*.pm4", SearchOption.TopDirectoryOnly))
	{
		string probe = Path.Combine(placementsDirectory, Path.GetFileName(pm4Path));
		string? obj0Path = Pm4CoordinateService.TryGetObj0PathForPm4(probe);
		if (obj0Path is null)
			continue;

		AdtPlacementCatalog catalog;
		try
		{
			catalog = AdtPlacementReader.Read(obj0Path);
		}
		catch (Exception ex) when (ex is InvalidDataException or IOException)
		{
			continue;
		}

		List<Pm4PlacementBox> boxes = [];
		foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
		{
			boxes.Add(new Pm4PlacementBox(
				MathF.Min(placement.BoundsMin.X, placement.BoundsMax.X),
				MathF.Min(placement.BoundsMin.Y, placement.BoundsMax.Y),
				MathF.Max(placement.BoundsMin.X, placement.BoundsMax.X),
				MathF.Max(placement.BoundsMin.Y, placement.BoundsMax.Y),
				placement.ModelPath,
				placement.UniqueId));
		}

		if (boxes.Count > 0)
			byFile[Path.GetFileName(pm4Path)] = boxes;
	}

	return byFile;
}

/// <summary>
/// `pm4 doodad-split` — tests whether CK24 0 is the bucket M2 doodad collision falls into, and
/// screens candidate fields for the per-doodad identity inside it.
/// </summary>
static void RunPm4DoodadSplit(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	string? placementsDirectory = GetOption(args, "--placements");
	string resolvedInput = Pm4CoordinateService.ResolveMapDirectory(input);
	string resolvedPlacements = string.IsNullOrWhiteSpace(placementsDirectory)
		? resolvedInput
		: Pm4CoordinateService.ResolveMapDirectory(placementsDirectory);

	Dictionary<string, Pm4TilePlacements> placements = LoadTilePlacements(resolvedInput, resolvedPlacements);
	if (placements.Count == 0)
	{
		Console.Error.WriteLine($"Error: no companion _obj0.adt placements found under '{resolvedPlacements}'.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4DoodadSplitReport report = Pm4DoodadSplitAnalyzer.AnalyzeDirectory(resolvedInput, placements);

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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 doodad/object split");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Tiles with ADT ground truth: {report.TilesScored}, objects scored: {report.ObjectsScored}");
	Console.WriteLine();

	Console.WriteLine("                          sits on an MDDF doodad | inside a MODF world model");
	Console.WriteLine($"  CK24 == 0  ({report.ZeroBucketObjects,6} objects)   {report.ZeroBucketOnDoodadFraction,10:P1}   |   {report.ZeroBucketInWorldModelFraction,10:P1}");
	Console.WriteLine($"  CK24 != 0  ({report.NonZeroObjects,6} objects)   {report.NonZeroOnDoodadFraction,10:P1}   |   {report.NonZeroInWorldModelFraction,10:P1}");
	Console.WriteLine();

	Pm4Ck24WmoCorrespondence c = report.WmoCorrespondence;
	Console.WriteLine("Does a keyed (non-zero) CK24 count as one WMO instance?  [the falsifiable test]");
	Console.WriteLine($"  tiles tested                                   : {c.TilesTested}");
	Console.WriteLine($"  tiles with NO WMO placements                   : {c.WmoFreeTiles}");
	Console.WriteLine($"  ...of those, tiles WITH a keyed object         : {c.WmoFreeTilesWithKeyedObjects}   <- must be 0");
	Console.WriteLine($"  tiles where keyed count == WMO count exactly   : {c.TilesWithExactCountMatch}");
	Console.WriteLine($"  tiles within +/-1                              : {c.TilesWithinOne}");
	Console.WriteLine($"  totals: {c.TotalKeyedObjects} keyed objects vs {c.TotalWorldModelPlacements} WMO placements");
	Console.WriteLine();
	Console.WriteLine($"  tiles with at least one CK24 0 bucket          : {c.TilesWithAnyZeroBucket}");
	Console.WriteLine($"  ...of those, with EXACTLY one                  : {c.TilesWithExactlyOneZeroBucket}");
	Console.WriteLine();
	Console.WriteLine($"VERDICT: {report.Verdict}");
	Console.WriteLine();

	Console.WriteLine("Candidate per-doodad identity fields (cardinality vs the tile's MDDF count):");
	foreach (Pm4DoodadSeparatorFit fit in report.SeparatorFits)
	{
		Console.WriteLine($"  {fit.Field,-42} tiles={fit.TilesTested,4}  mean={fit.MeanRatioToDoodadCount,9:F2}x  median={fit.MedianRatioToDoodadCount,8:F2}x  exact={fit.TilesMatchingExactly}");
	}

	Console.WriteLine();
	Console.WriteLine("Sample CK24 0 objects that landed on a doodad:");
	foreach (Pm4DoodadSplitObjectRecord record in report.MatchedDoodadSamples.Take(12))
	{
		Console.WriteLine($"  {record.FileName} region={record.RegionId,-5} surfaces={record.SurfaceCount,-4} dist={record.NearestDoodadDistance,7:F1}  groupIds={record.DistinctGroupObjectIds} anchors={record.AnchorOnlyLinks}");
		Console.WriteLine($"      {record.NearestDoodadPath}");
	}

	Console.WriteLine();
	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

/// <summary>
/// Reads each PM4's companion <c>_obj0.adt</c> and splits its placements by asset class: MDDF
/// doodad positions and MODF world-model boxes, both already in ADT placement space.
/// </summary>
static Dictionary<string, Pm4TilePlacements> LoadTilePlacements(string pm4Directory, string placementsDirectory)
{
	Dictionary<string, Pm4TilePlacements> byFile = new(StringComparer.OrdinalIgnoreCase);
	if (!Directory.Exists(placementsDirectory))
		return byFile;

	foreach (string pm4Path in Directory.EnumerateFiles(pm4Directory, "*.pm4", SearchOption.TopDirectoryOnly))
	{
		string probe = Path.Combine(placementsDirectory, Path.GetFileName(pm4Path));
		string? obj0Path = Pm4CoordinateService.TryGetObj0PathForPm4(probe);
		if (obj0Path is null)
			continue;

		AdtPlacementCatalog catalog;
		try
		{
			catalog = AdtPlacementReader.Read(obj0Path);
		}
		catch (Exception ex) when (ex is InvalidDataException or IOException)
		{
			continue;
		}

		List<Pm4NamedPoint> doodads = [];
		foreach (AdtModelPlacement placement in catalog.ModelPlacements)
		{
			doodads.Add(new Pm4NamedPoint(
				placement.Position.X,
				placement.Position.Y,
				placement.Position.Z,
				placement.ModelPath,
				placement.UniqueId));
		}

		List<Pm4PlacementBox> boxes = [];
		foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
		{
			boxes.Add(new Pm4PlacementBox(
				MathF.Min(placement.BoundsMin.X, placement.BoundsMax.X),
				MathF.Min(placement.BoundsMin.Y, placement.BoundsMax.Y),
				MathF.Max(placement.BoundsMin.X, placement.BoundsMax.X),
				MathF.Max(placement.BoundsMin.Y, placement.BoundsMax.Y),
				placement.ModelPath,
				placement.UniqueId));
		}

		if (doodads.Count > 0 || boxes.Count > 0)
			byFile[Path.GetFileName(pm4Path)] = new Pm4TilePlacements(doodads, boxes);
	}

	return byFile;
}

/// <summary>
/// `pm4 component-identity` — splits the CK24 0 remainder into geometric components, checks them
/// against real doodad placements, and scores which field reproduces them.
/// </summary>
static void RunPm4ComponentIdentity(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	string? placementsDirectory = GetOption(args, "--placements");
	string resolvedInput = Pm4CoordinateService.ResolveMapDirectory(input);
	string resolvedPlacements = string.IsNullOrWhiteSpace(placementsDirectory)
		? resolvedInput
		: Pm4CoordinateService.ResolveMapDirectory(placementsDirectory);

	int maxTiles = 0;
	if (int.TryParse(GetOption(args, "--max-tiles"), out int parsedMaxTiles) && parsedMaxTiles > 0)
		maxTiles = parsedMaxTiles;

	Dictionary<string, Pm4TilePlacements> placements = LoadTilePlacements(resolvedInput, resolvedPlacements);
	if (placements.Count == 0)
	{
		Console.Error.WriteLine($"Error: no companion _obj0.adt placements found under '{resolvedPlacements}'.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4ComponentIdentityReport report = Pm4ComponentIdentityAnalyzer.AnalyzeDirectory(resolvedInput, placements, maxTiles);

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

	Console.WriteLine("WowViewer.Tool.Inspect PM4 CK24-0 component identity");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Tiles scored: {report.TilesScored}, components found: {report.ComponentCount}");
	Console.WriteLine();

	Console.WriteLine("Q1 — are components the right unit? (geometry vs real doodad placements)");
	Console.WriteLine($"  components landing on an MDDF placement : {report.ComponentsOnDoodad} ({report.ComponentsOnDoodadFraction:P1})");
	Console.WriteLine($"  components per MDDF placement           : {report.ComponentsPerDoodadPlacement:F3}");
	Console.WriteLine($"  VERDICT: {report.Verdict}");
	Console.WriteLine();

	Console.WriteLine("Q2 — which field is constant within a component AND unique between them?");
	Console.WriteLine("  field                   pure    purity   distinct  distinctness  no-links  vals/tile");
	foreach (Pm4FieldSeparatorScore score in report.SeparatorScores)
	{
		Console.WriteLine($"  {score.Field,-22} {score.PureComponents,6}   {score.Purity,7:P1}   {score.DistinctComponents,8}   {score.Distinctness,10:P1}   {score.AbsentFraction,7:P1}   {score.DistinctValuesPerTileMedian,8}");
	}

	Console.WriteLine();
	Console.WriteLine("  purity alone proves nothing — a field that never varies is perfectly pure and");
	Console.WriteLine("  identifies nothing. An identity needs high purity AND high distinctness.");
	Console.WriteLine();

	Console.WriteLine("Closest component/doodad matches:");
	foreach (Pm4ComponentRecord record in report.ClosestMatches.Take(14))
	{
		Console.WriteLine($"  {record.FileName} region={record.RegionId,-5} surf={record.SurfaceCount,-4} extent={record.ExtentX,6:F1}x{record.ExtentY,6:F1}x{record.ExtentZ,6:F1}  dist={record.NearestDoodadDistance,6:F2}  groupIds={record.DistinctGroupObjectIds}");
		Console.WriteLine($"      {record.NearestDoodadPath}");
	}

	Console.WriteLine();
	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

static void RunPm4ConnectiveGeometry(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file or directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4ConnectiveGeometryReport report = File.Exists(input)
		? Pm4ConnectiveGeometryAnalyzer.AnalyzeFile(input)
		: Pm4ConnectiveGeometryAnalyzer.AnalyzeDirectory(input);

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

	PrintPm4ConnectiveGeometryReport(report);
}

static void PrintPm4ConnectiveGeometryReport(Pm4ConnectiveGeometryReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 connective-geometry report");
	Console.WriteLine($"Input: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}, non-empty={report.NonEmptyFileCount}");
	Console.WriteLine();

	Pm4MslkWindowPopulation population = report.WindowPopulation;
	Console.WriteLine("MSLK path-window population:");
	Console.WriteLine($"  total MSLK entries      = {population.TotalMslkEntries}");
	Console.WriteLine($"  active windows          = {population.ActiveWindows}");
	Console.WriteLine($"  MspiFirstIndex < 0      = {population.NegativeFirstIndexEntries}");
	Console.WriteLine($"  MspiIndexCount == 0     = {population.ZeroCountEntries}");
	Console.WriteLine($"  total window indices    = {population.TotalWindowIndices}");
	Console.WriteLine($"  mean indices per window = {population.MeanIndicesPerWindow:F3}");
	Console.WriteLine($"  window size range       = {population.MinWindowSize}..{population.MaxWindowSize}");
	Console.WriteLine();

	Console.WriteLine("Window size histogram (top buckets):");
	foreach (Pm4WindowSizeBucket bucket in report.SizeHistogram)
		Console.WriteLine($"  size={bucket.Size,-4} windows={bucket.WindowCount,-8} {bucket.Fraction:P2}");
	Console.WriteLine();

	PrintPm4WindowTopology("Topology evidence (all families)", report.Topology);

	PrintPm4FaceOrientation(report.SurfaceNormalOrientation);
	PrintPm4FaceOrientation(report.PathWindowOrientation);

	Pm4StreamCoincidenceSummary coincidence = report.StreamCoincidence;
	Console.WriteLine($"MSPV <-> MSVT vertex coincidence (epsilon={coincidence.Epsilon}):");
	Console.WriteLine($"  MSPV on MSVT = {coincidence.MspvPointsCoincidentWithMsvt}/{coincidence.MspvPointsTested} ({coincidence.CoincidentFraction:P2})");
	Console.WriteLine($"  MSVT on MSPV = {coincidence.MsvtPointsCoincidentWithMspv}/{coincidence.MsvtPointsTested} ({coincidence.MsvtCoincidentFraction:P2})");
	Console.WriteLine();

	Console.WriteLine("Per-family (TypeFlags/Subtype):");
	foreach (Pm4WindowFamilySummary family in report.Families.Take(12))
	{
		Console.WriteLine($"  {family.FamilyKey}");
		Console.WriteLine($"    files={family.FileCount} entries={family.TotalEntries} active={family.ActiveWindows} negFirst={family.NegativeFirstIndexEntries}");
		Console.WriteLine($"    meanSize={family.MeanWindowSize:F2} modalSize={family.ModalWindowSize} mult3={family.MultipleOfThreeFraction:P1} closed={family.ClosedFraction:P1}");
		Console.WriteLine($"    degenerateTriples={family.Topology.DegenerateTripleFraction:P1} collinear={family.Topology.CollinearWindows} coplanar={family.Topology.CoplanarWindows}");
		Console.WriteLine($"    topSizes: {string.Join(", ", family.TopSizes.Select(static size => $"{size.Size}x{size.WindowCount}"))}");
	}
	Console.WriteLine();

	Pm4MscnLinkageSummary mscn = report.MscnLinkage;
	Console.WriteLine("MSUR._0x18 -> MSCN linkage:");
	Console.WriteLine($"  files with MSCN        = {mscn.FilesWithMscn}");
	Console.WriteLine($"  fits={mscn.MsurToMscnFits} misses={mscn.MsurToMscnMisses}");
	Console.WriteLine($"  MSCN points            = {mscn.TotalMscnPoints}");
	Console.WriteLine($"  distinct referenced    = {mscn.DistinctMscnReferenced} ({mscn.ReferencedFraction:P2})");
	Console.WriteLine($"  never referenced       = {mscn.MscnPointsUnreferenced}");
	Console.WriteLine($"  MSCN/MSVT count ratio  = {mscn.MscnToMsvtRatio:F3}");
	Console.WriteLine();

	Console.WriteLine("Notes:");
	foreach (string note in report.Notes)
		Console.WriteLine($"  - {note}");
}

static void PrintPm4FaceOrientation(Pm4FaceOrientationSummary orientation)
{
	Console.WriteLine($"{orientation.Name} orientation:");
	Console.WriteLine($"  faces measured          = {orientation.FacesMeasured}");
	Console.WriteLine($"  dominant axis counts    = X:{orientation.DominantX} Y:{orientation.DominantY} Z:{orientation.DominantZ}");
	Console.WriteLine($"  mean |normal|           = X:{orientation.MeanAbsNormalX:F3} Y:{orientation.MeanAbsNormalY:F3} Z:{orientation.MeanAbsNormalZ:F3}");
	Console.WriteLine($"  near axis-aligned       = {orientation.NearAxisAligned}");
	Console.WriteLine($"  perp. to reference axis = {orientation.NearPerpendicularToDominantAxis}");
	Console.WriteLine();
}

static void PrintPm4WindowTopology(string title, Pm4WindowTopologyEvidence topology)
{
	Console.WriteLine($"{title}:");
	Console.WriteLine($"  windows measured        = {topology.WindowsMeasured}");
	Console.WriteLine($"  closed (first==last)    = {topology.ClosedWindows}");
	Console.WriteLine($"  length multiple of 3    = {topology.MultipleOfThreeWindows}");
	Console.WriteLine($"  duplicate vertices      = {topology.WindowsWithDuplicateVertices}");
	Console.WriteLine($"  collinear windows       = {topology.CollinearWindows}");
	Console.WriteLine($"  coplanar windows        = {topology.CoplanarWindows}");
	Console.WriteLine($"  triples tested          = {topology.TriplesTested}");
	Console.WriteLine($"  degenerate triples      = {topology.DegenerateTriples} ({topology.DegenerateTripleFraction:P2})");
	Console.WriteLine();
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

static void PrintPm4BondStatsReport(Pm4BondStatsReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 bond-stats report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Total non-zero surfaces: {report.TotalSurfaceCount}");
	Console.WriteLine($"Zero-CK24 surfaces excluded: {report.ZeroCk24SurfaceCount}");
	Console.WriteLine($"Distinct CK24 values: {report.DistinctCk24Values}");
	Console.WriteLine($"Distinct CK24 types: {report.DistinctCk24Types}");
	Console.WriteLine();
	Console.WriteLine("Cross-tabulation:");
	Console.WriteLine($"  Total (high,low) pairs: {report.CrossTabulation.TotalPairs}");
	Console.WriteLine($"  Distinct high-byte values: {report.CrossTabulation.DistinctHighByteValues}");
	Console.WriteLine($"  Distinct low-byte values: {report.CrossTabulation.DistinctLowByteValues}");
	Console.WriteLine();
	Console.WriteLine("Top (high,low) pairs by surface count:");
	foreach (var pair in report.CrossTabulation.TopPairsByCount.Take(12))
	{
		string types = string.Join(",", pair.AssociatedCk24Types.Select(t => $"0x{t:X2}"));
		Console.WriteLine($"  (0x{pair.HighByte:X2}, 0x{pair.LowByte:X2}) surfaces={pair.SurfaceCount} files={pair.FileCount} types=[{types}]");
	}

	Console.WriteLine();
	Console.WriteLine("Per-file summaries:");
	foreach (var entry in report.PerFileEntries)
	{
		string tileText = entry.TileX.HasValue && entry.TileY.HasValue
			? $"{entry.TileX}_{entry.TileY}"
			: Path.GetFileNameWithoutExtension(entry.SourcePath);
		Console.WriteLine($"  {tileText}: surfaces={entry.SurfaceCount} types={entry.DistinctCk24Types}");
	}

	Console.WriteLine();
	Console.WriteLine("Type bucket breakdown:");
	foreach (var entry in report.PerFileEntries)
	{
		if (entry.TypeBucketBreakdown.Count == 0)
			continue;
		string tileText = entry.TileX.HasValue && entry.TileY.HasValue
			? $"{entry.TileX}_{entry.TileY}"
			: Path.GetFileNameWithoutExtension(entry.SourcePath);
		foreach (var bucket in entry.TypeBucketBreakdown)
		{
			Console.WriteLine($"  {tileText} type=0x{bucket.Ck24Type:X2} ({bucket.TypeLabel}): surfaces={bucket.SurfaceCount} distinctHigh={bucket.DistinctHighBytes} distinctLow={bucket.DistinctLowBytes} distinctCombined={bucket.DistinctCombinedIds}");
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
	Console.WriteLine("  wowviewer-inspect map generate-blank --tile-x <n> --tile-y <n> [--map-name <name>] [--format lk|alpha] [--texture <path>] [--output-dir <dir>]");
	Console.WriteLine("  wowviewer-inspect lit inspect --input <lights.lit>");
	Console.WriteLine("  wowviewer-inspect lit inspect --archive-root <game|data dir> --virtual-path <world/.../lights.lit> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect lit profile --input <lights.lit> [--game-time <0..1>[,<0..1>...]]... [--output <profile.json>]");
	Console.WriteLine("  wowviewer-inspect lit profile --archive-root <game|data dir> --virtual-path <world/.../lights.lit> [--listfile <listfile.txt>] [--game-time <0..1>[,<0..1>...]]... [--output <profile.json>]");
	Console.WriteLine("  wowviewer-inspect light profile --archive-root <staged-client> --build <exact-build> --map-id <id> (--world-position <x,y,z> | --renderer-position <x,y,z>) [--map-origin <units>] [--game-time <normalized:0..1|raw:0..2880>[,...]]... [--dbd-dir <definitions>] [--output <profile.json>]");
	Console.WriteLine("  wowviewer-inspect wmo inspect --input <file.wmo> [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect pm4 inspect --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 export-segments --input <file.pm4|directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 export-asset-signals --archive-root <staged client dir> [--seed-placements <tile_obj0.adt|directory>] [--kind all|wmo|m2] [--path-filter <text>] [--limit <n>] [--listfile <listfile.txt>] [--output <corpus.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 match-assets --input <file.pm4> [--asset-corpus <corpus.json> | --archive-root <staged client dir> [--placements <tile_obj0.adt>]] [--listfile <listfile.txt>] [--max-candidates <n>] [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 synthesize-placements --input <file.pm4> --target-tiles <x_y[,x_y...]> [--asset-corpus <corpus.json> | --archive-root <staged client dir> [--placements <tile_obj0.adt>]] [--listfile <listfile.txt>] [--max-candidates <n>] [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mshd --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 audit --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit-directory --input <directory>");
	Console.WriteLine("  wowviewer-inspect pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
}

static Pm4SegmentExportFile AssertSinglePm4ExportFile(Pm4SegmentExportRun exportRun, string input)
{
	if (exportRun.Files.Count != 1)
		throw new InvalidOperationException($"Expected a single PM4 export file for '{input}', but found {exportRun.Files.Count}.");

	return exportRun.Files[0];
}

static Pm4MatchRunManifest BuildPm4SegmentExportManifest(Pm4SegmentExportRun exportRun)
{
	List<Pm4MatchReportSegment> segments = exportRun.Files
		.SelectMany(static file => file.Segments)
		.OrderBy(static segment => segment.Segment.TileCoordinates[0], StringComparer.Ordinal)
		.ThenBy(static segment => segment.Segment.SegmentId, StringComparer.Ordinal)
		.Select(segment => BuildPm4MatchReportSegment(segment, null, null))
		.ToList();

	int ineligibleCount = segments.Count(static segment => string.IsNullOrWhiteSpace(segment.ExpectedAssetKind));

	return new Pm4MatchRunManifest(
		exportRun.RunId,
		exportRun.InputPath,
		exportRun.SegmentCount,
		segments,
		AssetReferenceCorpus: null,
		SegmentSignalCorpus: Pm4SegmentSignalExtractor.CurrentSignalVersion,
		MatchedCount: 0,
		AmbiguousCount: 0,
		UnresolvedCount: 0,
		IneligibleCount: ineligibleCount,
		Warnings: exportRun.Warnings);
}

static Pm4MatchRunManifest BuildPm4AssetMatchManifest(
	Pm4SegmentExportRun exportRun,
	IReadOnlyList<Pm4SegmentMatchResult> matchResults,
	IReadOnlyList<Pm4ReplacementPlacementProposal> placementProposals,
	string assetReferenceCorpus,
	IReadOnlyList<string> warnings,
	string commandName)
{
	Dictionary<string, Pm4ReplacementPlacementProposal> placementBySegmentId = placementProposals
		.GroupBy(static proposal => proposal.SegmentId, StringComparer.Ordinal)
		.ToDictionary(static group => group.Key, static group => group.First(), StringComparer.Ordinal);
	List<Pm4MatchReportSegment> segments = matchResults
		.OrderBy(static result => result.Segment.Segment.TileCoordinates[0], StringComparer.Ordinal)
		.ThenBy(static result => result.Segment.Segment.SegmentId, StringComparer.Ordinal)
		.Select(result => BuildPm4MatchReportSegment(
			result.Segment,
			result,
			placementBySegmentId.TryGetValue(result.Segment.Segment.SegmentId, out Pm4ReplacementPlacementProposal? proposal) ? proposal : null))
		.ToList();

	List<string> allWarnings = exportRun.Warnings.Concat(warnings).ToList();
	return new Pm4MatchRunManifest(
		$"{exportRun.RunId}:{commandName}:{Path.GetFileNameWithoutExtension(assetReferenceCorpus)}",
		exportRun.InputPath,
		segments.Count,
		segments,
		Path.GetFullPath(assetReferenceCorpus),
		Pm4SegmentSignalExtractor.CurrentSignalVersion,
		segments.Count(static segment => string.Equals(segment.Status, "matched", StringComparison.Ordinal)),
		segments.Count(static segment => string.Equals(segment.Status, "ambiguous", StringComparison.Ordinal)),
		segments.Count(static segment => string.Equals(segment.Status, "unresolved", StringComparison.Ordinal)),
	segments.Count(static segment => string.Equals(segment.Status, "ineligible", StringComparison.Ordinal)),
	allWarnings);
}

static Pm4MatchReportSegment BuildPm4MatchReportSegment(
	Pm4BuiltObjectSegment segment,
	Pm4SegmentMatchResult? matchResult,
	Pm4ReplacementPlacementProposal? placementProposal)
{
	IReadOnlyList<string> linkGroupIds = segment.Segment.LinkGroupIds
		.Select(static groupId => $"0x{groupId:X}")
		.ToList();
	IReadOnlyList<string> confidenceFlags = FormatConfidenceFlags(segment.Segment.ConfidenceFlags);
	string? expectedAssetKind = matchResult?.ExpectedAssetKind ?? PredictAssetKind(segment.Segment.Ck24Type);
	IReadOnlyList<Pm4MatchReportCandidate> candidates = matchResult is null
		? Array.Empty<Pm4MatchReportCandidate>()
		: matchResult.Candidates.Select(ToReportCandidate).ToList();

	IReadOnlyDictionary<string, Pm4MatchReportBounds>? typedBounds = segment.Signal.TypedBounds is { Count: > 0 }
		? segment.Signal.TypedBounds.ToDictionary(
			static kv => $"0x{kv.Key:X2}",
			static kv => ToReportBounds(kv.Value)!)!
		: null;

	return new Pm4MatchReportSegment(
		segment.Segment.SegmentId,
		$"0x{segment.Segment.Ck24:X6}",
		segment.Segment.Ck24Type,
		segment.Segment.Ck24ObjectId,
		segment.Segment.TileCoordinates,
		segment.Segment.Field04Values,
		expectedAssetKind,
		matchResult is null ? null : FormatMatchStatus(matchResult.Status),
		matchResult?.ReviewRequired ?? segment.Segment.ConfidenceFlags != Pm4SegmentConfidenceFlags.None,
		matchResult?.Rationale ?? BuildExportRationale(segment, expectedAssetKind),
		confidenceFlags.Count > 0 ? confidenceFlags : null,
		segment.Segment.SurfaceCount,
		segment.Segment.TotalIndexCount,
		linkGroupIds.Count > 0 ? linkGroupIds : null,
		segment.Segment.DominantLinkGroupId != 0u ? $"0x{segment.Segment.DominantLinkGroupId:X}" : null,
		segment.CoordinateMode.ToString(),
		segment.AxisConvention.ToString(),
		segment.FrameYawDegrees,
		ToReportBounds(segment.Signal.Bounds),
		ToReportVector3(segment.CorrelationState.Center),
		segment.Signal.FootprintHull.Select(static point => new Pm4MatchVector2(point.X, point.Y)).ToList(),
		segment.CorrelationState.FootprintArea,
		new Pm4MatchReportHeightStats(
			segment.Signal.HeightStats.MinimumPlaneDistance,
			segment.Signal.HeightStats.MaximumPlaneDistance,
			segment.Signal.HeightStats.AveragePlaneDistance),
		new Pm4MatchReportTopologyStats(
			segment.Signal.TopologyStats.SurfaceCount,
			segment.Signal.TopologyStats.TotalIndexCount,
			segment.Signal.TopologyStats.AnchorPointCount,
			segment.Signal.TopologyStats.AnchorNormalCount),
		new Pm4MatchReportAnchorSignals(
			segment.Signal.AnchorSignals.LinkedPositionRefCount,
			segment.Signal.AnchorSignals.NormalHeadingCount,
			segment.Signal.AnchorSignals.TerminatorCount,
			segment.Signal.AnchorSignals.FloorMinimum,
			segment.Signal.AnchorSignals.FloorMaximum,
			segment.Signal.AnchorSignals.HeadingMinimumDegrees,
			segment.Signal.AnchorSignals.HeadingMaximumDegrees,
			segment.Signal.AnchorSignals.HeadingMeanDegrees),
		segment.Signal.SurfaceFamilyHistogram,
		typedBounds,
		candidates,
		placementProposal is null ? null : ToReportPlacementProposal(placementProposal));
}

static void WritePm4Report(Pm4MatchRunManifest manifest, string outputPath)
{
	string ext = Path.GetExtension(outputPath) ?? "";
	if (string.Equals(ext, ".json", StringComparison.OrdinalIgnoreCase))
	{
		Pm4MatchReportWriter.WriteToFile(manifest, outputPath);
		Console.WriteLine($"Wrote {outputPath}");
		string markdownPath = Path.ChangeExtension(outputPath, ".md");
		Pm4MatchReportFormatter.WriteMarkdownToFile(manifest, markdownPath);
		Console.WriteLine($"Wrote {markdownPath}");
	}
	else
	{
		if (!string.Equals(ext, ".md", StringComparison.OrdinalIgnoreCase))
			outputPath = Path.ChangeExtension(outputPath, ".md");
		Pm4MatchReportFormatter.WriteMarkdownToFile(manifest, outputPath);
		Console.WriteLine($"Wrote {outputPath}");
	}
}

static void PrintPm4SegmentExportRun(Pm4SegmentExportRun exportRun)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 segment export");
	Console.WriteLine($"Input: {exportRun.InputPath}");
	Console.WriteLine($"RunId: {exportRun.RunId}");
	Console.WriteLine($"Files: {exportRun.FileCount} Segments: {exportRun.SegmentCount} Warnings: {exportRun.Warnings.Count}");

	foreach (Pm4SegmentExportFile file in exportRun.Files.Take(8))
	{
		Console.WriteLine($"  tile={file.TileX}_{file.TileY} segments={file.SegmentCount} source={Path.GetFileName(file.SourcePath)}");
		foreach (Pm4BuiltObjectSegment segment in file.Segments.Take(4))
		{
			string? expectedAssetKind = PredictAssetKind(segment.Segment.Ck24Type);
			string flagText = segment.Segment.ConfidenceFlags == Pm4SegmentConfidenceFlags.None
				? "none"
				: string.Join(",", FormatConfidenceFlags(segment.Segment.ConfidenceFlags));
			Console.WriteLine(
				$"    {segment.Segment.SegmentId} kind={expectedAssetKind ?? "ineligible"} ck24=0x{segment.Segment.Ck24:X6} area={segment.CorrelationState.FootprintArea:F1} bounds={FormatVector(segment.CorrelationState.BoundsMin)}..{FormatVector(segment.CorrelationState.BoundsMax)} anchors={segment.Signal.AnchorSignals.LinkedPositionRefCount} flags={flagText}");
		}
	}

	if (exportRun.Warnings.Count > 0)
	{
		foreach (string warning in exportRun.Warnings)
			Console.WriteLine($"Warning: {warning}");
	}
}

static void PrintPm4AssetSignalCorpus(Pm4AssetSignalCorpusManifest manifest)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 durable asset corpus");
	Console.WriteLine($"Archive root: {manifest.ArchiveRoot}");
	Console.WriteLine($"Client build: {manifest.ClientBuild}");
	Console.WriteLine($"RunId: {manifest.RunId}");
	Console.WriteLine($"Assets: {manifest.AssetCount} warnings={manifest.Warnings.Count}");

	foreach (Pm4AssetReferenceSignalRecord asset in manifest.Assets.Take(12))
	{
		Vector3 span = asset.Bounds is null
			? Vector3.Zero
			: asset.Bounds.Max - asset.Bounds.Min;
		Console.WriteLine($"  {asset.AssetKind} {asset.AssetPath} span=({span.X:F1},{span.Y:F1},{span.Z:F1}) area={asset.FootprintArea:F1}");
	}

	if (manifest.Warnings.Count > 0)
	{
		foreach (string warning in manifest.Warnings.Take(12))
			Console.WriteLine($"Warning: {warning}");
	}
}

static void PrintPm4AssetMatchRun(Pm4MatchRunManifest manifest, int assetReferenceCount)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 asset match report");
	Console.WriteLine($"PM4: {manifest.InputPm4Root}");
	Console.WriteLine($"Asset references: {manifest.AssetReferenceCorpus}");
	Console.WriteLine($"RunId: {manifest.RunId}");
	Console.WriteLine($"Segments: {manifest.SegmentCount} Assets: {assetReferenceCount} matched={manifest.MatchedCount ?? 0} ambiguous={manifest.AmbiguousCount ?? 0} unresolved={manifest.UnresolvedCount ?? 0} ineligible={manifest.IneligibleCount ?? 0} proposals={manifest.Segments.Count(static segment => segment.PlacementProposal is not null)}");

	foreach (Pm4MatchReportSegment segment in manifest.Segments
		.Where(static segment => !string.Equals(segment.Status, "ineligible", StringComparison.Ordinal))
		.OrderBy(static segment => segment.Status, StringComparer.Ordinal)
		.ThenBy(static segment => segment.SegmentId, StringComparer.Ordinal)
		.Take(12))
	{
		Console.WriteLine($"  {segment.SegmentId} kind={segment.ExpectedAssetKind ?? "n/a"} status={segment.Status ?? "exported"} area={segment.FootprintArea ?? 0d:F1} flags={(segment.ConfidenceFlags is { Count: > 0 } ? string.Join(",", segment.ConfidenceFlags) : "none")}");
		foreach (Pm4MatchReportCandidate candidate in segment.Candidates.Take(3))
			Console.WriteLine($"    rank={candidate.Rank} score={candidate.OverallScore:F3} {candidate.AssetKind} {candidate.AssetPath}");
		if (segment.PlacementProposal is not null)
			Console.WriteLine($"    proposal={segment.PlacementProposal.ProposalId} pos={FormatMatchVector3(segment.PlacementProposal.WorldPosition)} scale={segment.PlacementProposal.WorldScale?.ToString("F2", CultureInfo.InvariantCulture) ?? "n/a"} review={segment.PlacementProposal.ReviewRequired}");
	}

	if (manifest.Warnings is { Count: > 0 })
	{
		foreach (string warning in manifest.Warnings)
			Console.WriteLine($"Warning: {warning}");
	}
}

static Pm4MatchReportCandidate ToReportCandidate(Pm4AssetMatchCandidate candidate)
{
	return new Pm4MatchReportCandidate(
		candidate.AssetId,
		candidate.AssetPath,
		candidate.AssetKind,
		candidate.Rank,
		candidate.OverallScore,
		FormatMatchStatus(candidate.Status),
		candidate.ScoreBreakdown,
		candidate.Rationale);
}

static Pm4MatchReportPlacementProposal ToReportPlacementProposal(Pm4ReplacementPlacementProposal proposal)
{
	return new Pm4MatchReportPlacementProposal(
		proposal.ProposalId,
		proposal.AssetId,
		proposal.TargetTileCoordinates,
		proposal.WorldPosition is null ? null : ToReportVector3(proposal.WorldPosition.Value),
		proposal.WorldRotation is null ? null : ToReportRotation(proposal.WorldRotation.Value),
		proposal.WorldScale,
		proposal.Confidence,
		proposal.ReviewRequired,
		proposal.Provenance);
}

static Pm4MatchReportBounds? ToReportBounds(Pm4Bounds3? bounds)
{
	return bounds is null
		? null
		: new Pm4MatchReportBounds(ToReportVector3(bounds.Min), ToReportVector3(bounds.Max));
}

static Pm4MatchVector3 ToReportVector3(Vector3 value)
{
	return new Pm4MatchVector3(value.X, value.Y, value.Z);
}

static Pm4MatchRotation ToReportRotation(Vector3 rotationDegrees)
{
	return new Pm4MatchRotation(rotationDegrees.Z, rotationDegrees.X, rotationDegrees.Y);
}

static IReadOnlyList<string> BuildExportRationale(Pm4BuiltObjectSegment segment, string? expectedAssetKind)
{
	List<string> rationale =
	[
		$"segment exported from PM4 with {segment.Segment.SurfaceCount} surfaces and {segment.Segment.TotalIndexCount} indices.",
	];

	if (expectedAssetKind is null)
		rationale.Add($"ck24Type 0x{segment.Segment.Ck24Type:X2} is not currently classified as WMO/M2-matchable.");
	else
		rationale.Add($"ck24Type 0x{segment.Segment.Ck24Type:X2} is currently treated as {expectedAssetKind}-matchable.");

	if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ZeroCk24Seed))
		rationale.Add("segment came from the zero-CK24 fallback seed path.");
	if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.UsedConnectivityFallback))
		rationale.Add("segment required connectivity fallback splitting.");
	if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.MissingPositionRefs))
		rationale.Add("segment is missing linked position refs.");
	if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.MultipleLinkGroupIds))
		rationale.Add("segment spans multiple link-group ids.");
	if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ReusedLow16ObjectId))
		rationale.Add("segment reuses a low16 object id across multiple CK24 families.");

	return rationale;
}

static IReadOnlyList<string> FormatConfidenceFlags(Pm4SegmentConfidenceFlags flags)
{
	if (flags == Pm4SegmentConfidenceFlags.None)
		return Array.Empty<string>();

	List<string> values = [];
	foreach (Pm4SegmentConfidenceFlags flag in Enum.GetValues<Pm4SegmentConfidenceFlags>())
	{
		if (flag == Pm4SegmentConfidenceFlags.None || !flags.HasFlag(flag))
			continue;

		values.Add(flag switch
		{
			Pm4SegmentConfidenceFlags.ZeroCk24Seed => "zero-ck24-seed",
			Pm4SegmentConfidenceFlags.UsedConnectivityFallback => "connectivity-fallback",
			Pm4SegmentConfidenceFlags.MultipleLinkGroupIds => "multiple-link-groups",
			Pm4SegmentConfidenceFlags.MissingPositionRefs => "missing-position-refs",
			Pm4SegmentConfidenceFlags.ReusedLow16ObjectId => "reused-low16-object-id",
			Pm4SegmentConfidenceFlags.SpansMultipleField04Values => "multiple-field04",
			Pm4SegmentConfidenceFlags.HasUnlinkedSurfaces => "unlinked-surfaces",
			_ => flag.ToString(),
		});
	}

	return values;
}

static string FormatMatchStatus(Pm4AssetMatchStatus status)
{
	return status switch
	{
		Pm4AssetMatchStatus.Matched => "matched",
		Pm4AssetMatchStatus.Ambiguous => "ambiguous",
		Pm4AssetMatchStatus.Unresolved => "unresolved",
		Pm4AssetMatchStatus.Ineligible => "ineligible",
		_ => status.ToString().ToLowerInvariant(),
	};
}

static string? PredictAssetKind(byte ck24Type)
{
	return ck24Type switch
	{
		0x42 or 0x43 => "wmo",
		0x40 or 0x41 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "m2",
		_ => null,
	};
}

static string FormatMatchVector3(Pm4MatchVector3? value)
{
	return value is null
		? "n/a"
		: $"({value.X:F2},{value.Y:F2},{value.Z:F2})";
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
	Console.WriteLine("  archive scan-wmo-containers --archive-root <game dir>");
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
	Console.WriteLine("  lit profile --input <lights.lit> [--game-time <0..1>[,<0..1>...]]... [--output <profile.json>]");
	Console.WriteLine("  lit profile --archive-root <game|data dir> --virtual-path <world/.../lights.lit> [--listfile <listfile.txt>] [--game-time <0..1>[,<0..1>...]]... [--output <profile.json>]");
	Console.WriteLine("  lit sample is an alias for lit profile; default --game-time is 0.35.");
}

static void ShowLightUsage()
{
	Console.WriteLine("Light DBC commands:");
	Console.WriteLine("  light profile --archive-root <staged-client> --build <exact-build> --map-id <id> --world-position <x,y,z> [--game-time <normalized:0..1|raw:0..2880>[,...]]... [--dbd-dir <definitions>] [--output <profile.json>]");
	Console.WriteLine("  light profile --archive-root <staged-client> --build <exact-build> --map-id <id> --renderer-position <x,y,z> [--map-origin <units>] [--game-time <normalized:0..1|raw:0..2880>[,...]]... [--dbd-dir <definitions>] [--output <profile.json>]");
	Console.WriteLine("  Bare times in 0..1 are normalized; bare times above 1 are raw 0..2880 units. Prefix raw: to disambiguate raw values 0 or 1. Default time is normalized:0.35.");
	Console.WriteLine($"  Default map origin: {LightDbcProfileCommand.DefaultMapOrigin.ToString("R", CultureInfo.InvariantCulture)}.");
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
	Console.WriteLine("  map generate-blank --tile-x <n> --tile-y <n> [--map-name <name>] [--format lk|alpha] [--texture <path>] [--output-dir <dir>]");
	Console.WriteLine("  map patch-blank --placements <obj0.adt> --tile-x <n> --tile-y <n> [--map-name <name>] [--texture <path>] [--output-dir <dir>]");
}

static void ShowPm4Usage()
{
	Console.WriteLine("PM4 commands:");
	Console.WriteLine("  pm4 inspect --input <file.pm4>");
	Console.WriteLine("  pm4 export-segments --input <file.pm4|directory> [--output <report.json>]");
	Console.WriteLine("  pm4 export-asset-signals --archive-root <staged client dir> [--seed-placements <tile_obj0.adt|directory>] [--kind all|wmo|m2] [--path-filter <text>] [--limit <n>] [--listfile <listfile.txt>] [--output <corpus.json>]");
	Console.WriteLine("  pm4 match-assets --input <file.pm4> [--asset-corpus <corpus.json> | --archive-root <staged client dir> [--placements <tile_obj0.adt>]] [--listfile <listfile.txt>] [--max-candidates <n>] [--output <report.json>]");
	Console.WriteLine("  pm4 synthesize-placements --input <file.pm4> --target-tiles <x_y[,x_y...]> [--asset-corpus <corpus.json> | --archive-root <staged client dir> [--placements <tile_obj0.adt>]] [--listfile <listfile.txt>] [--max-candidates <n>] [--output <report.json>]");
	Console.WriteLine("  pm4 match --input <file.pm4> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--listfile <listfile.txt>] [--max-matches <n>] [--search-range <units>] [--output <report.json>] [--object-output-dir <directory>]");
	Console.WriteLine("  pm4 match-report --input <file.pm4> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--max-matches <n>] [--search-range <units>] [--output <report.md>]");
	Console.WriteLine("  pm4 manifest --input <file.pm4|directory> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--output <output-dir>]");
	Console.WriteLine("  pm4 hierarchy --input <file.pm4> [--output <report.json>]");
	Console.WriteLine("  pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mshd --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 audit --input <file.pm4>");
	Console.WriteLine("  pm4 audit-directory --input <directory>");
	Console.WriteLine("  pm4 cross-tile --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 bond-stats --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
	Console.WriteLine("  pm4 correlate-models --input <file.pm4> --placements <file.adt> --archive-root <dir> [--output <report.json>] [--pm4-vpath <archive-path>] [--adt-vpath <archive-path>]");
	Console.WriteLine("  pm4 sweep-correlate --map-dir <directory> --archive-root <dir> [--output <summary.csv>] [--limit <n>]");
	Console.WriteLine("  pm4 fingerprint-scan --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 identify-models --fingerprints <fingerprints.json> --archive-root <staged client dir> [--min-score <0.0-1.0>] [--max-matches <n>] [--output <report.json>]");
	Console.WriteLine("  pm4 tile-reports --fingerprints <fingerprints.json> --identity <identity.json> --pm4-dir <directory> [--output-dir <dir>] [--tiles <x_y[,x_y...]>]");
	Console.WriteLine("  pm4 generate-from-wmo --wmo-root <file.wmo> --position <x,y,z> --rotation <rx,ry,rz> --tile <x,y> --archive-root <dir> [--output <out.pm4>]");
	Console.WriteLine("  pm4 validate-generator-geometry --pm4 <file.pm4> --adt <tile_obj0.adt> --archive-root <staged client dir> [--bin-size <1.0>] [--area-bin-size <1.0>] [--normal-alignment-bin-size <0.0>] [--planar-offset-bin-size <0.0>] [--output <report.json>]");
	Console.WriteLine("  pm4 build-wmo-fingerprint-db --archive-root <staged client dir> [--listfile <listfile.txt>] [--limit <n>] [--output <db.json>]");
	Console.WriteLine("  pm4 extract-pm4-fingerprints --input <directory> [--output <fp.json>] [--tiles <x_y[,x_y...]>]");
	Console.WriteLine("  pm4 match-fingerprints --pm4-fingerprints <fp.json> --wmo-db <db.json> [--min-score <0.0-1.0>] [--max-candidates <n>] [--output <matches.json>]");
	Console.WriteLine("  pm4 validate-matches --matches <matches.json> --adt-dir <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 build-wmo-surface-db --archive-root <staged client dir> [--listfile <listfile.txt>] [--bin-size <1.0>] [--area-bin-size <1.0>] [--normal-alignment-bin-size <0.1>] [--planar-offset-bin-size <1.0>] [--limit <n>] [--output <db.json>]");
	Console.WriteLine("  pm4 extract-pm4-surfaces --input <directory> [--bin-size <1.0>] [--area-bin-size <1.0>] [--normal-alignment-bin-size <0.1>] [--planar-offset-bin-size <1.0>] [--output <fp.json>]");
	Console.WriteLine("  pm4 match-surfaces --pm4-surfaces <fp.json> --wmo-surface-db <db.json> [--min-score <0.0-1.0>] [--output <matches.json>]");
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

static void RunPm4FingerprintScan(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input) || !Directory.Exists(input))
	{
		Console.Error.WriteLine("Error: --input <directory> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(input);
	string[] pm4Files = Directory.GetFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly);
	Console.WriteLine($"Scanning {pm4Files.Length} PM4 files...");

	var fingerprints = new List<Dictionary<string, object>>();
	int processed = 0;
	int coordLocal = 0;
	int coordWorld = 0;

	foreach (string pm4File in pm4Files.OrderBy(Path.GetFileName))
	{
		try
		{
			Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(pm4File);
			IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
			IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;
			IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;

			if (!Pm4CoordinateService.TryParseTileCoordinates(pm4File, out int tileX, out int tileY))
				continue;

			bool isLikelyTileLocal = Pm4PlacementMath.IsLikelyTileLocal(msvt);
			if (isLikelyTileLocal) coordLocal++; else coordWorld++;

			var groups = msur
				.Where(static s => s.Ck24 != 0)
				.GroupBy(static s => s.Ck24)
				.OrderByDescending(static g => g.Sum(static s => s.IndexCount));

			foreach (IGrouping<uint, Pm4MsurEntry> group in groups)
			{
				List<Pm4MsurEntry> surfaces = group.ToList();
				HashSet<int> vertexIndices = [];
				foreach (Pm4MsurEntry surface in surfaces)
				{
					int first = checked((int)surface.MsviFirstIndex);
					int end = Math.Min(first + surface.IndexCount, msvi.Count);
					for (int i = first; i < end; i++)
					{
						int vi = checked((int)msvi[i]);
						if ((uint)vi < (uint)msvt.Count)
							vertexIndices.Add(vi);
					}
				}

				Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
				Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);
				foreach (int vi in vertexIndices)
				{
					min = Vector3.Min(min, msvt[vi]);
					max = Vector3.Max(max, msvt[vi]);
				}

				Pm4CoordinateMode coordMode = isLikelyTileLocal ? Pm4CoordinateMode.TileLocal : Pm4CoordinateMode.WorldSpace;
				Pm4AxisConvention axisConvention = Pm4AxisConvention.XYPlaneZUp;
				Pm4PlanarTransform defaultTransform = Pm4PlacementContract.GetDefaultPlanarTransform(coordMode);

				Vector3 wowMin = Pm4PlacementMath.ConvertPm4VertexToWorld(min, tileX, tileY, coordMode, axisConvention, defaultTransform);
				Vector3 wowMax = Pm4PlacementMath.ConvertPm4VertexToWorld(max, tileX, tileY, coordMode, axisConvention, defaultTransform);
				Vector3 wowBoundsMin = Vector3.Min(wowMin, wowMax);
				Vector3 wowBoundsMax = Vector3.Max(wowMin, wowMax);

				float sizeX = wowBoundsMax.X - wowBoundsMin.X;
				float sizeY = wowBoundsMax.Y - wowBoundsMin.Y;
				float sizeZ = wowBoundsMax.Z - wowBoundsMin.Z;
				float[] sizes = [sizeX, sizeY, sizeZ];
				Array.Sort(sizes);

				fingerprints.Add(new Dictionary<string, object>
				{
					["tile"] = $"{tileX}_{tileY}",
					["ck24"] = $"0x{group.Key:X6}",
					["type"] = $"0x{surfaces[0].Ck24Type:X2}",
					["objectId"] = surfaces[0].Ck24ObjectId,
					["surfaces"] = surfaces.Count,
					["indices"] = surfaces.Sum(static s => s.IndexCount),
					["vertices"] = vertexIndices.Count,
					["coordMode"] = isLikelyTileLocal ? 0 : 1,
					["wowBoundsMin"] = $"({wowBoundsMin.X:F1},{wowBoundsMin.Y:F1},{wowBoundsMin.Z:F1})",
					["wowBoundsMax"] = $"({wowBoundsMax.X:F1},{wowBoundsMax.Y:F1},{wowBoundsMax.Z:F1})",
					["sortedSize"] = $"{sizes[0]:F0}x{sizes[1]:F0}x{sizes[2]:F0}",
				});
			}

			processed++;
			if (processed % 100 == 0)
				Console.WriteLine($"  Processed {processed}/{pm4Files.Length}...");
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error reading {Path.GetFileName(pm4File)}: {ex.Message}");
		}
	}

	Console.WriteLine($"Processed {processed} PM4 files, {fingerprints.Count} CK24 groups total.");
	Console.WriteLine($"Coordinate modes: TileLocal={coordLocal}, WorldSpace={coordWorld}");

	var byFingerprint = fingerprints
		.GroupBy(f => $"{f["surfaces"]}_{f["indices"]}_{f["vertices"]}")
		.OrderByDescending(g => g.Count())
		.ToList();

	Console.WriteLine($"\nDistinct fingerprints (surf_idx_vert): {byFingerprint.Count}");
	Console.WriteLine("\nTop 20 most common fingerprints:");
	Console.WriteLine("  Fingerprint                     Count  Types           Sample CK24");
	foreach (var g in byFingerprint.Take(20))
	{
		var types = string.Join(",", g.Select(f => f["type"]).Distinct());
		var samples = string.Join(", ", g.Select(f => $"{f["ck24"]}").Distinct().Take(3));
		string sortedSizes = g.First()["sortedSize"]?.ToString() ?? "?";
		Console.WriteLine($"  {g.Key,-35} {g.Count(),5}  {types,-15} {samples}  {sortedSizes}");
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);
		string json = JsonSerializer.Serialize(fingerprints, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote {fingerprints.Count} fingerprints to {outputPath}");
	}
}

static void RunPm4IdentifyModels(string[] args)
{
	string? fingerprintsPath = GetOption(args, "--fingerprints", "-f");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? minScoreText = GetOption(args, "--min-score", "-s");
	string? maxMatchesText = GetOption(args, "--max-matches", "-m");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(fingerprintsPath) || !File.Exists(fingerprintsPath))
	{
		Console.Error.WriteLine("Error: --fingerprints <path> is required and must point to an existing fingerprint-scan JSON file.");
		Console.Error.WriteLine("  Run 'pm4 fingerprint-scan --input <directory> --output <file.json>' first.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <staged client dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions bootstrapOptions))
		return;

	double minScore = 0.5;
	if (!string.IsNullOrWhiteSpace(minScoreText) && !double.TryParse(minScoreText, out minScore))
	{
		Console.Error.WriteLine("Error: --min-score must be a number between 0 and 1.");
		Environment.ExitCode = 1;
		return;
	}

	int maxMatches = 3;
	if (!string.IsNullOrWhiteSpace(maxMatchesText))
		int.TryParse(maxMatchesText, out maxMatches);

	Console.WriteLine($"Loading fingerprints from: {Path.GetFullPath(fingerprintsPath)}");
	string fingerprintsJson = File.ReadAllText(fingerprintsPath);
	List<Dictionary<string, JsonElement>>? rawFingerprints = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(fingerprintsJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (rawFingerprints is null || rawFingerprints.Count == 0)
	{
		Console.Error.WriteLine("Error: fingerprint file contains no entries.");
		Environment.ExitCode = 1;
		return;
	}

	Console.WriteLine($"Loaded {rawFingerprints.Count} fingerprint entries.");

	List<Pm4FingerprintGroup> fingerprintGroups = Pm4CorrelateModelsSupport.BuildFingerprintGroups(rawFingerprints);
	Console.WriteLine($"Grouped into {fingerprintGroups.Count} distinct fingerprints.");
	Console.WriteLine($"  Top 5 by instance count:");
	foreach (Pm4FingerprintGroup fg in fingerprintGroups.Take(5))
		Console.WriteLine($"    ({fg.Surfaces},{fg.Indices},{fg.Vertices}) type={fg.Ck24Type:X2} instances={fg.ObjectIds.Count} dims={fg.MergedSortedDim0:F0}x{fg.MergedSortedDim1:F0}x{fg.MergedSortedDim2:F0}");

	Console.WriteLine($"\nScanning WMO archive for local bounds...");
	List<WmoLocalBoundsEntry> wmoBounds = Pm4CorrelateModelsSupport.ScanWmoLocalBounds(archiveRoot, bootstrapOptions);

	Console.WriteLine($"\nMatching {fingerprintGroups.Count} fingerprint groups against {wmoBounds.Count} WMOs (minScore={minScore:F2})...");
	List<Pm4IdentityMatch> matches = Pm4CorrelateModelsSupport.MatchFingerprintsToWmos(fingerprintGroups, wmoBounds, minScore);

	Console.WriteLine($"\nFound {matches.Count} matches with score >= {minScore:F2}");
	Console.WriteLine("\nTop matches:");
	Console.WriteLine($"  {"Fingerprint",-35} {"Type",-6} {"PM4 Dims",-18} {"WMO",-50} {"WMO Dims",-18} {"Score",-8} {"Ratio",-8}");
	Console.WriteLine($"  {new string('-', 35)} {new string('-', 6)} {new string('-', 18)} {new string('-', 50)} {new string('-', 18)} {new string('-', 8)} {new string('-', 8)}");

	foreach (Pm4IdentityMatch match in matches.Take(Math.Max(maxMatches * fingerprintGroups.Count, 50)))
	{
		string shortPath = match.WmoPath.Length > 48 ? "..." + match.WmoPath[^45..] : match.WmoPath;
		Console.WriteLine($"  {match.Fingerprint,-35} 0x{match.Ck24Type:X2}   {match.Pm4SortedDim0,5:F0}x{match.Pm4SortedDim1:F0}x{match.Pm4SortedDim2:F0}  {shortPath,-50} {match.WmoSortedDim0,5:F0}x{match.WmoSortedDim1:F0}x{match.WmoSortedDim2:F0}  {match.Score,8:F3} {match.DimensionRatio,8:F3}");
	}

	var typeSummary = matches.GroupBy(static m => m.Ck24Type)
		.Select(static g => (Type: g.Key, Count: g.Count()))
		.OrderByDescending(static x => x.Count)
		.ToList();
	Console.WriteLine("\nMatch count by CK24 type:");
	foreach (var ts in typeSummary)
		Console.WriteLine($"  0x{ts.Type:X2}: {ts.Count}");

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			FingerprintGroups = fingerprintGroups,
			WmoCount = wmoBounds.Count,
			Matches = matches,
			MinScore = minScore
		};

		string outputJson = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, outputJson);
		Console.WriteLine($"\nWrote {matches.Count} matches to {outputPath}");
	}
}

static string? GetJsonString(Dictionary<string, JsonElement> dict, string key)
{
	foreach (var kv in dict)
	{
		if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase) && kv.Value.ValueKind == JsonValueKind.String)
			return kv.Value.GetString();
	}
	return null;
}

static int GetJsonInt(Dictionary<string, JsonElement> dict, string key, int defaultValue = 0)
{
	foreach (var kv in dict)
	{
		if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase) && kv.Value.ValueKind == JsonValueKind.Number)
			return kv.Value.GetInt32();
	}
	return defaultValue;
}

static double GetJsonDouble(Dictionary<string, JsonElement> dict, string key, double defaultValue = 0)
{
	foreach (var kv in dict)
	{
		if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase) && kv.Value.ValueKind == JsonValueKind.Number)
			return kv.Value.GetDouble();
	}
	return defaultValue;
}

static void RunPm4TileReports(string[] args)
{
	string? fingerprintsPath = GetOption(args, "--fingerprints", "-f");
	string? identityPath = GetOption(args, "--identity", "-d");
	string? pm4Dir = GetOption(args, "--pm4-dir", "-p");
	string? outputDir = GetOption(args, "--output-dir", "-o");
	string? tilesFilter = GetOption(args, "--tiles", "-t");

	if (string.IsNullOrWhiteSpace(fingerprintsPath) || !File.Exists(fingerprintsPath))
	{
		Console.Error.WriteLine("Error: --fingerprints <path> is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(identityPath) || !File.Exists(identityPath))
	{
		Console.Error.WriteLine("Error: --identity <path> is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(pm4Dir) || !Directory.Exists(pm4Dir))
	{
		Console.Error.WriteLine("Error: --pm4-dir <directory> is required.");
		Environment.ExitCode = 1;
		return;
	}

	outputDir ??= Path.Combine(Directory.GetCurrentDirectory(), "pm4-tile-reports");
	Directory.CreateDirectory(outputDir);

	HashSet<string>? tileFilter = null;
	if (!string.IsNullOrWhiteSpace(tilesFilter))
	{
		tileFilter = new HashSet<string>(tilesFilter.Split(',', StringSplitOptions.RemoveEmptyEntries), StringComparer.OrdinalIgnoreCase);
		Console.WriteLine($"Filtering to {tileFilter.Count} tiles: {string.Join(", ", tileFilter.Take(10))}{(tileFilter.Count > 10 ? "..." : "")}");
	}

	Console.WriteLine($"Loading fingerprints from: {Path.GetFullPath(fingerprintsPath)}");
	string fingerprintsJson = File.ReadAllText(fingerprintsPath);
	List<Dictionary<string, JsonElement>>? rawFingerprints = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(fingerprintsJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (rawFingerprints is null || rawFingerprints.Count == 0)
	{
		Console.Error.WriteLine("Error: fingerprint file contains no entries.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {rawFingerprints.Count} fingerprint entries.");

	Console.WriteLine($"Loading identity matches from: {Path.GetFullPath(identityPath)}");
	string identityJson = File.ReadAllText(identityPath);
	using JsonDocument identityDoc = JsonDocument.Parse(identityJson);
	List<Dictionary<string, JsonElement>>? rawIdentity = identityDoc.RootElement.TryGetProperty("Matches", out JsonElement matchesElem) ? matchesElem.EnumerateArray().Select(e => e.EnumerateObject().ToDictionary(p => p.Name, p => p.Value)).ToList() : null;
	if (rawIdentity is null || rawIdentity.Count == 0)
	{
		Console.Error.WriteLine("Warning: identity file contains no matches. Reports will lack WMO identifications.");
		rawIdentity = new List<Dictionary<string, JsonElement>>();
	}
	Console.WriteLine($"Loaded {rawIdentity.Count} identity matches.");

	Dictionary<string, List<Dictionary<string, JsonElement>>> fingerprintsByTile = new(StringComparer.OrdinalIgnoreCase);
	foreach (Dictionary<string, JsonElement> entry in rawFingerprints)
	{
		string? tile = GetJsonString(entry, "Tile");
		if (tile is null) continue;
		if (tileFilter is not null && !tileFilter.Contains(tile))
			continue;
		if (!fingerprintsByTile.ContainsKey(tile))
			fingerprintsByTile[tile] = new List<Dictionary<string, JsonElement>>();
		fingerprintsByTile[tile].Add(entry);
	}

	Console.WriteLine($"Generating reports for {fingerprintsByTile.Count} tiles...");

	// Build identity lookup using fingerprint as key (case-insensitive)
	Dictionary<string, List<Dictionary<string, JsonElement>>> identityByFingerprint = new(StringComparer.OrdinalIgnoreCase);
	foreach (Dictionary<string, JsonElement> entry in rawIdentity)
	{
		string? fp = GetJsonString(entry, "Fingerprint");
		if (fp is null) continue;
		if (!identityByFingerprint.ContainsKey(fp))
			identityByFingerprint[fp] = new List<Dictionary<string, JsonElement>>();
		identityByFingerprint[fp].Add(entry);
	}

	int reportCount = 0;
	foreach (KeyValuePair<string, List<Dictionary<string, JsonElement>>> tileEntry in fingerprintsByTile.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
	{
		string tile = tileEntry.Key;
		List<Dictionary<string, JsonElement>> entries = tileEntry.Value;

		int totalCk24 = entries.Count;
		int matched = entries.Count(e =>
		{
			int s = GetJsonInt(e, "Surfaces"), i = GetJsonInt(e, "Indices"), v = GetJsonInt(e, "Vertices");
			string t = GetJsonString(e, "Type") ?? "0";
			string fp = $"{s}_{i}_{v}_{t.TrimStart("0x".AsSpan())}";
			return identityByFingerprint.ContainsKey(fp);
		});

		Dictionary<string, int> typeCounts = new();
		foreach (Dictionary<string, JsonElement> e in entries)
		{
			string? typeStr = GetJsonString(e, "Type");
			if (typeStr is not null)
			{
				typeCounts[typeStr] = typeCounts.GetValueOrDefault(typeStr, 0) + 1;
			}
		}

		List<(string Fingerprint, string Type, int Surfaces, int Indices, int Vertices, string? WmoPath, double Score)> unmatchedGroups = new();
		List<(string Fingerprint, string Type, int Surfaces, int Indices, int Vertices, string WmoPath, double Score)> matchedGroups = new();

		foreach (Dictionary<string, JsonElement> e in entries)
		{
			int surf = GetJsonInt(e, "Surfaces");
			int idx = GetJsonInt(e, "Indices");
			int vert = GetJsonInt(e, "Vertices");
			string type = GetJsonString(e, "Type") ?? "?";
			string fp = $"{surf}_{idx}_{vert}_{type.TrimStart("0x".AsSpan())}";

			if (identityByFingerprint.TryGetValue(fp, out List<Dictionary<string, JsonElement>>? matches) && matches.Count > 0)
			{
				string wmoPath = GetJsonString(matches[0], "WmoPath") ?? "unknown";
				double score = GetJsonDouble(matches[0], "Score");
				matchedGroups.Add((fp, type, surf, idx, vert, wmoPath, score));
			}
			else
			{
				unmatchedGroups.Add((fp, type, surf, idx, vert, null, 0));
			}
		}

		string reportPath = Path.Combine(outputDir, $"tile_{tile}.md");
		using (StreamWriter sw = new StreamWriter(reportPath))
		{
			sw.WriteLine($"# PM4 Tile Report: {tile}");
			sw.WriteLine();
			sw.WriteLine($"- **Total CK24 groups**: {totalCk24}");
			sw.WriteLine($"- **Matched to WMO**: {matched} ({(totalCk24 > 0 ? matched * 100.0 / totalCk24 : 0):F1}%)");
			sw.WriteLine($"- **Unmatched**: {totalCk24 - matched}");
			sw.WriteLine();

			sw.WriteLine("## Type Distribution");
			sw.WriteLine();
			sw.WriteLine("| Type | Count |");
			sw.WriteLine("|------|-------|");
			foreach (KeyValuePair<string, int> tc in typeCounts.OrderByDescending(k => k.Value))
				sw.WriteLine($"| {tc.Key} | {tc.Value} |");
			sw.WriteLine();

			if (matchedGroups.Count > 0)
			{
				sw.WriteLine("## Matched Models (WMO)");
				sw.WriteLine();
				sw.WriteLine("| Fingerprint | Type | S/I/V | WMO Path | Score |");
				sw.WriteLine("|-------------|------|-------|----------|-------|");
				foreach (var m in matchedGroups.OrderByDescending(m => m.Score))
					sw.WriteLine($"| {m.Fingerprint} | {m.Type} | {m.Surfaces}/{m.Indices}/{m.Vertices} | {m.WmoPath} | {m.Score:F3} |");
				sw.WriteLine();
			}

			if (unmatchedGroups.Count > 0)
			{
				sw.WriteLine("## Unmatched Groups");
				sw.WriteLine();
				sw.WriteLine("| Fingerprint | Type | S/I/V |");
				sw.WriteLine("|-------------|------|-------|");
				foreach (var u in unmatchedGroups.OrderByDescending(u => u.Surfaces))
					sw.WriteLine($"| {u.Fingerprint} | {u.Type} | {u.Surfaces}/{u.Indices}/{u.Vertices} |");
				sw.WriteLine();
			}

			sw.WriteLine("---");
			sw.WriteLine($"*Generated from pm4-fingerprints-full.json + pm4-model-identity-full.json*");
		}

		reportCount++;
		if (reportCount % 20 == 0)
			Console.WriteLine($"  Generated {reportCount} reports...");
	}

	Console.WriteLine($"\nWrote {reportCount} tile reports to: {Path.GetFullPath(outputDir)}");
}

static void RunPm4BuildWmoFingerprintDb(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? listfilePath = GetOption(args, "--listfile", "-l");
	string? limitText = GetOption(args, "--limit", "-n");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <staged client dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions bootstrapOptions))
		return;

	int limit = 0;
	if (!string.IsNullOrWhiteSpace(limitText))
		int.TryParse(limitText, out limit);

	Console.WriteLine($"Building WMO fingerprint database from: {Path.GetFullPath(archiveRoot)}");

	List<string> wmoPaths = EnumerateWmoRoots(archiveRoot, bootstrapOptions, listfilePath);

	if (limit > 0 && wmoPaths.Count > limit)
	{
		Console.WriteLine($"Limiting to first {limit} WMOs (of {wmoPaths.Count} found).");
		wmoPaths = wmoPaths.Take(limit).ToList();
	}

	Console.WriteLine($"Found {wmoPaths.Count} WMO root files.");
	if (wmoPaths.Count == 0)
	{
		Console.Error.WriteLine("Error: no WMO root files found. Try --listfile <path> for listfile-based enumeration.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4FingerprintDatabase database = Pm4FingerprintBuildSupport.BuildDatabase(
		archiveRoot,
		bootstrapOptions,
		wmoPaths,
		progress: static msg => Console.WriteLine(msg));

	Console.WriteLine($"\nDatabase built: {database.WmoCount} WMO roots, {database.Records.Count} total fingerprints (root + group).");

	int rootCount = database.Records.Count(static r => r.SourceLabel == "wmo-root-merged");
	int groupCount = database.Records.Count(static r => r.SourceLabel.StartsWith("wmo-group-"));
	Console.WriteLine($"  Root fingerprints: {rootCount}");
	Console.WriteLine($"  Group fingerprints: {groupCount}");

	IReadOnlyList<Pm4FingerprintRecord> sortedByDim = database.Records
		.Where(static r => r.SortedDim2 > 0)
		.OrderByDescending(static r => r.SortedDim2)
		.Take(10)
		.ToList();
	Console.WriteLine("\nTop 10 fingerprints by largest dimension:");
	Console.WriteLine($"  {"Path",-50} {"Dims",-18} {"Hull",-6} {"Area",-10}");
	foreach (Pm4FingerprintRecord r in sortedByDim)
	{
		string shortPath = r.AssetPath.Length > 48 ? "..." + r.AssetPath[^45..] : r.AssetPath;
		Console.WriteLine($"  {shortPath,-50} {r.SortedDim0,5:F0}x{r.SortedDim1,5:F0}x{r.SortedDim2,5:F0}  {r.NormalizedFootprintHull.Count,-6} {r.FootprintArea,8:F0}");
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		string json = JsonSerializer.Serialize(database, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote fingerprint database to {outputPath}");
	}
}

static void RunPm4ExtractPm4Fingerprints(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	string? tilesFilter = GetOption(args, "--tiles", "-t");

	if (string.IsNullOrWhiteSpace(input) || !Directory.Exists(input))
	{
		Console.Error.WriteLine("Error: --input <directory> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	HashSet<string>? filterTiles = null;
	if (!string.IsNullOrWhiteSpace(tilesFilter))
		filterTiles = new HashSet<string>(tilesFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));

	string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(input);
	string[] pm4Files = Directory.GetFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly);
	Console.WriteLine($"Scanning {pm4Files.Length} PM4 files...");

	List<Pm4FingerprintRecord> fingerprints = [];
	int processed = 0;
	int groupsFound = 0;
	int skipped = 0;

	foreach (string pm4File in pm4Files.OrderBy(Path.GetFileName))
	{
		if (!Pm4CoordinateService.TryParseTileCoordinates(pm4File, out int tileX, out int tileY))
		{
			skipped++;
			continue;
		}

		if (filterTiles is not null && !filterTiles.Contains($"{tileX}_{tileY}"))
			continue;

		try
		{
			Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(pm4File);
			IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
			IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;
			IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;

			var groups = msur
				.Where(static s => s.Ck24 != 0 && s.IndexCount >= 3)
				.GroupBy(static s => s.Ck24)
				.OrderByDescending(static g => g.Sum(static s => s.IndexCount));

			foreach (IGrouping<uint, Pm4MsurEntry> group in groups)
			{
				List<Pm4MsurEntry> surfaces = group.ToList();
				HashSet<int> vertexIndices = [];
				foreach (Pm4MsurEntry surface in surfaces)
				{
					int first = checked((int)surface.MsviFirstIndex);
					int end = Math.Min(first + surface.IndexCount, msvi.Count);
					for (int i = first; i < end; i++)
					{
						int vi = checked((int)msvi[i]);
						if ((uint)vi < (uint)msvt.Count)
							vertexIndices.Add(vi);
					}
				}

				if (vertexIndices.Count < 3)
					continue;

				List<Vector3> verts = new(vertexIndices.Count);
				List<int> indices = [];
				Dictionary<int, int> globalToLocal = [];

				foreach (int vi in vertexIndices.OrderBy(static i => i))
				{
					globalToLocal[vi] = verts.Count;
					verts.Add(msvt[vi]);
				}

				foreach (Pm4MsurEntry surface in surfaces)
				{
					int first = checked((int)surface.MsviFirstIndex);
					int end = Math.Min(first + surface.IndexCount, msvi.Count);
					for (int i = first; i < end; i++)
					{
						int vi = checked((int)msvi[i]);
						if (globalToLocal.TryGetValue(vi, out int localIdx))
							indices.Add(localIdx);
					}
				}

				Dictionary<byte, int> typeFlags = [];
				foreach (Pm4MsurEntry s in surfaces)
				{
					if (typeFlags.ContainsKey(s.GroupKey))
						typeFlags[s.GroupKey]++;
					else
						typeFlags[s.GroupKey] = 1;
				}

				byte ck24Type = surfaces[0].Ck24Type;
				uint ck24 = group.Key;
				string assetId = $"tile{tileX}_{tileY}_ck24_0x{ck24:X6}";
				string assetPath = Path.GetFileName(pm4File);

				Pm4FingerprintRecord? fp = Pm4FingerprintExtractor.ExtractFromGeometry(
					verts,
					indices,
					surfaceCount: surfaces.Count,
					ck24Type: ck24Type,
					typeFlagsProfile: typeFlags,
					assetId: assetId,
					assetPath: assetPath,
					assetKind: ck24Type switch
					{
						0x42 or 0x43 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "wmo",
						0x40 or 0x41 => "m2",
						_ => "unknown",
					},
					groupCount: 1,
					sourceLabel: $"pm4-tile-{tileX}_{tileY}");

				if (fp is not null)
				{
					fingerprints.Add(fp);
					groupsFound++;
				}
			}

			processed++;
			if (processed % 100 == 0)
				Console.WriteLine($"  Processed {processed}/{pm4Files.Length}...");
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error reading {Path.GetFileName(pm4File)}: {ex.Message}");
			skipped++;
		}
	}

	Console.WriteLine($"\nProcessed {processed} PM4 files, {groupsFound} CK24 group fingerprints extracted, {skipped} skipped.");

	var byType = fingerprints.GroupBy(static f => f.Ck24Type)
		.Select(static g => (Type: g.Key, Count: g.Count()))
		.OrderByDescending(static x => x.Count)
		.ToList();
	Console.WriteLine("\nFingerprints by CK24 type:");
	foreach (var t in byType)
		Console.WriteLine($"  0x{t.Type:X2}: {t.Count}");

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			BuildDate = DateTime.UtcNow.ToString("o"),
			SourceDirectory = resolvedDirectory,
			TotalFiles = pm4Files.Length,
			ProcessedFiles = processed,
			TotalFingerprints = fingerprints.Count,
			Fingerprints = fingerprints,
		};

		string json = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote {fingerprints.Count} fingerprints to {outputPath}");
	}
}

static List<string> EnumerateWmoRoots(
	string archiveRoot,
	ArchiveCatalogBootstrapOptions bootstrapOptions,
	string? listfilePath)
{
	HashSet<string> wmoPaths = new(StringComparer.OrdinalIgnoreCase);

	try
	{
		ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate([archiveRoot], bootstrapOptions);
		IReadOnlyList<string> allFiles = session.ArchiveCatalog.GetAllKnownFiles();
		foreach (string f in allFiles)
		{
			if (f.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) && !f.Contains('_'))
				wmoPaths.Add(f.Replace('\\', '/').TrimStart('/').ToLowerInvariant());
		}
		Console.WriteLine($"  Archive catalog found {wmoPaths.Count} WMO roots.");
	}
	catch (Exception ex)
	{
		Console.WriteLine($"  Archive catalog enumeration failed: {ex.Message}");
	}

	if (!string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath))
	{
		int listfileCount = 0;
		foreach (string line in File.ReadLines(listfilePath))
		{
			string trimmed = line.Trim();
			if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#') || trimmed.StartsWith("//"))
				continue;

			string normalized = trimmed.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
			if (normalized.EndsWith(".wmo") && !normalized.Contains('_'))
			{
				if (wmoPaths.Add(normalized))
					listfileCount++;
			}
		}
		Console.WriteLine($"  Listfile added {listfileCount} new WMO roots (total: {wmoPaths.Count}).");
	}
	else if (wmoPaths.Count < 500 && string.IsNullOrWhiteSpace(listfilePath))
	{
		Console.WriteLine("  Warning: archive catalog found <500 WMOs. Consider providing --listfile for full coverage.");
	}

	return wmoPaths.OrderBy(static f => f, StringComparer.OrdinalIgnoreCase).ToList();
}

static void RunPm4MatchFingerprints(string[] args)
{
	string? pm4Path = GetOption(args, "--pm4-fingerprints", "-p");
	string? wmoDbPath = GetOption(args, "--wmo-db", "-w");
	string? minScoreText = GetOption(args, "--min-score", "-s");
	string? maxCandidatesText = GetOption(args, "--max-candidates", "-m");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(pm4Path) || !File.Exists(pm4Path))
	{
		Console.Error.WriteLine("Error: --pm4-fingerprints <path> is required and must exist.");
		Console.Error.WriteLine("  Run 'pm4 extract-pm4-fingerprints --input <dir> --output <file.json>' first.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(wmoDbPath) || !File.Exists(wmoDbPath))
	{
		Console.Error.WriteLine("Error: --wmo-db <path> is required and must exist.");
		Console.Error.WriteLine("  Run 'pm4 build-wmo-fingerprint-db --archive-root <dir> --output <file.json>' first.");
		Environment.ExitCode = 1;
		return;
	}

	double minScore = 0.45;
	if (!string.IsNullOrWhiteSpace(minScoreText) && !double.TryParse(minScoreText, out minScore))
	{
		Console.Error.WriteLine("Error: --min-score must be a number between 0 and 1.");
		Environment.ExitCode = 1;
		return;
	}

	int maxCandidates = 10;
	if (!string.IsNullOrWhiteSpace(maxCandidatesText))
		int.TryParse(maxCandidatesText, out maxCandidates);

	Console.WriteLine($"Loading PM4 fingerprints from: {Path.GetFullPath(pm4Path)}");
	string pm4Json = File.ReadAllText(pm4Path);
	Pm4FingerprintExtractOutput? pm4Output = JsonSerializer.Deserialize<Pm4FingerprintExtractOutput>(pm4Json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (pm4Output is null || pm4Output.Fingerprints is null || pm4Output.Fingerprints.Count == 0)
	{
		Console.Error.WriteLine("Error: PM4 fingerprint file contains no entries.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {pm4Output.Fingerprints.Count} PM4 fingerprints.");

	Console.WriteLine($"Loading WMO fingerprint database from: {Path.GetFullPath(wmoDbPath)}");
	string wmoJson = File.ReadAllText(wmoDbPath);
	Pm4FingerprintDatabase? wmoDb = JsonSerializer.Deserialize<Pm4FingerprintDatabase>(wmoJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (wmoDb is null || wmoDb.Records.Count == 0)
	{
		Console.Error.WriteLine("Error: WMO fingerprint database contains no entries.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {wmoDb.WmoCount} WMO roots, {wmoDb.Records.Count} total fingerprints.");

	Pm4FingerprintMatchOptions options = new(MinScore: minScore, MaxCandidates: maxCandidates);
	Console.WriteLine($"\nMatching {pm4Output.Fingerprints.Count} PM4 fingerprints against {wmoDb.WmoRecords.Count} WMO fingerprints (minScore={minScore:F2})...");

	IReadOnlyList<Pm4FingerprintMatchResult> results = Pm4FingerprintMatcher.Match(
		pm4Output.Fingerprints,
		wmoDb,
		options);

	int matched = results.Count(static r => r.Status == Pm4FingerprintMatchStatus.Matched);
	int ambiguous = results.Count(static r => r.Status == Pm4FingerprintMatchStatus.Ambiguous);
	int unresolved = results.Count(static r => r.Status == Pm4FingerprintMatchStatus.Unresolved);
	int ineligible = results.Count(static r => r.Status == Pm4FingerprintMatchStatus.Ineligible);

	Console.WriteLine($"\nResults: {matched} matched, {ambiguous} ambiguous, {unresolved} unresolved, {ineligible} ineligible");

	var matchedResults = results.Where(static r => r.Status == Pm4FingerprintMatchStatus.Matched).Take(20).ToList();
	if (matchedResults.Count > 0)
	{
		Console.WriteLine("\nTop 20 matched:");
		Console.WriteLine($"  {"PM4 Fingerprint",-40} {"Dims",-18} {"WMO Candidate",-50} {"Score",-8} {"FP Overlap",-10}");
		foreach (Pm4FingerprintMatchResult r in matchedResults)
		{
			if (r.Candidates.Count == 0)
				continue;
			Pm4FingerprintMatchCandidate c = r.Candidates[0];
			string shortPm4 = r.Pm4FingerprintId.Length > 38 ? "..." + r.Pm4FingerprintId[^37..] : r.Pm4FingerprintId;
			string shortWmo = c.Candidate.AssetPath.Length > 48 ? "..." + c.Candidate.AssetPath[^47..] : c.Candidate.AssetPath;
			Console.WriteLine($"  {shortPm4,-40} {r.SortedDim0,5:F0}x{r.SortedDim1,5:F0}x{r.SortedDim2,5:F0}  {shortWmo,-50} {c.OverallScore,8:F3} {c.FootprintOverlapRatio,10:F3}");
		}
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			MatchDate = DateTime.UtcNow.ToString("o"),
			Pm4FingerprintCount = pm4Output.Fingerprints.Count,
			WmoFingerprintCount = wmoDb.Records.Count,
			MinScore = minScore,
			Matched = matched,
			Ambiguous = ambiguous,
			Unresolved = unresolved,
			Ineligible = ineligible,
			Results = results,
		};

		string json = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote {results.Count} match results to {outputPath}");
	}
}

static void RunPm4ValidateMatches(string[] args)
{
	string? matchesPath = GetOption(args, "--matches", "-m");
	string? adtDir = GetOption(args, "--adt-dir", "-a");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(matchesPath) || !File.Exists(matchesPath))
	{
		Console.Error.WriteLine("Error: --matches <path> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(adtDir) || !Directory.Exists(adtDir))
	{
		Console.Error.WriteLine("Error: --adt-dir <directory> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	Console.WriteLine($"Loading match results from: {Path.GetFullPath(matchesPath)}");
	string matchesJson = File.ReadAllText(matchesPath);
	Pm4SurfaceMatchOutput? matchOutput = JsonSerializer.Deserialize<Pm4SurfaceMatchOutput>(matchesJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (matchOutput is null || matchOutput.Results is null || matchOutput.Results.Count == 0)
	{
		Console.Error.WriteLine("Error: match file contains no results.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {matchOutput.Results.Count} match results.");

	Dictionary<string, HashSet<string>> tileToWmoPaths = new(StringComparer.OrdinalIgnoreCase);
	string[] adtFiles = Directory.GetFiles(adtDir, "*_obj0.adt", SearchOption.TopDirectoryOnly);
	Console.WriteLine($"Reading WMO placements from {adtFiles.Length} ADT obj0 files...");

	foreach (string adtFile in adtFiles)
	{
		string fileName = Path.GetFileNameWithoutExtension(adtFile);
		string tileKey = ExtractTileKeyFromAdtName(fileName);
		if (string.IsNullOrEmpty(tileKey))
			continue;

		try
		{
			AdtPlacementCatalog placements = AdtPlacementReader.Read(adtFile);
			HashSet<string> wmoPaths = new(StringComparer.OrdinalIgnoreCase);
			foreach (AdtWorldModelPlacement wmo in placements.WorldModelPlacements)
			{
				string normalized = wmo.ModelPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
				wmoPaths.Add(normalized);
			}
			if (wmoPaths.Count > 0)
				tileToWmoPaths[tileKey] = wmoPaths;
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"  Error reading {Path.GetFileName(adtFile)}: {ex.Message}");
		}
	}

	Console.WriteLine($"Loaded ADT placements for {tileToWmoPaths.Count} tiles.");

	int totalEvaluated = 0;
	int totalWithAdt = 0;
	int precisionAt1 = 0;
	int precisionAt3 = 0;
	int totalMatched = 0;
	int totalAmbiguous = 0;
	int totalUnresolved = 0;
	int totalIneligible = 0;

	Dictionary<string, int> failureCategories = new()
	{
		["no_adt_for_tile"] = 0,
		["no_candidates"] = 0,
		["top1_not_in_adt"] = 0,
		["top3_not_in_adt"] = 0,
		["ambiguous_not_in_adt"] = 0,
		["ineligible"] = 0,
	};

	List<(string pm4Id, string tile, string status, string topMatch, string adtWmos, bool correct)> detailRows = new();

	foreach (SurfaceMatchResult result in matchOutput.Results)
	{
		totalEvaluated++;

		if (result.Status == "Ineligible")
		{
			totalIneligible++;
			failureCategories["ineligible"]++;
			continue;
		}

		string tileKey = ExtractTileKeyFromPm4Id(result.Pm4FingerprintId);
		if (string.IsNullOrEmpty(tileKey) || !tileToWmoPaths.TryGetValue(tileKey, out HashSet<string>? adtWmos))
		{
			failureCategories["no_adt_for_tile"]++;
			continue;
		}

		totalWithAdt++;

		if (result.Candidates.Count == 0)
		{
			failureCategories["no_candidates"]++;
			totalUnresolved++;
			continue;
		}

		string adtWmosStr = string.Join("|", adtWmos.Take(5));
		bool top1InAdt = false;
		bool top3InAdt = false;

		for (int i = 0; i < Math.Min(3, result.Candidates.Count); i++)
		{
			string candidatePath = result.Candidates[i].Candidate.AssetPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
			if (adtWmos.Contains(candidatePath))
			{
				if (i == 0) top1InAdt = true;
				top3InAdt = true;
			}
		}

		switch (result.Status)
		{
			case "Matched":
				totalMatched++;
				if (top1InAdt) precisionAt1++;
				else failureCategories["top1_not_in_adt"]++;
				if (top3InAdt) precisionAt3++;
				else if (!top1InAdt) failureCategories["top3_not_in_adt"]++;
				detailRows.Add((result.Pm4FingerprintId, tileKey, "Matched", result.Candidates[0].Candidate.AssetPath, adtWmosStr, top1InAdt));
				break;
			case "Ambiguous":
				totalAmbiguous++;
				if (top1InAdt) precisionAt1++;
				if (top3InAdt) precisionAt3++;
				if (!top3InAdt) failureCategories["ambiguous_not_in_adt"]++;
				detailRows.Add((result.Pm4FingerprintId, tileKey, "Ambiguous", result.Candidates[0].Candidate.AssetPath, adtWmosStr, top1InAdt));
				break;
			case "Unresolved":
				totalUnresolved++;
				if (top3InAdt) precisionAt3++;
				failureCategories["top1_not_in_adt"]++;
				detailRows.Add((result.Pm4FingerprintId, tileKey, "Unresolved", result.Candidates.Count > 0 ? result.Candidates[0].Candidate.AssetPath : "none", adtWmosStr, top1InAdt));
				break;
		}
	}

	double p1 = totalWithAdt > 0 ? (double)precisionAt1 / totalWithAdt : 0;
	double p3 = totalWithAdt > 0 ? (double)precisionAt3 / totalWithAdt : 0;

	Console.WriteLine($"\n=== Validation Report ===");
	Console.WriteLine($"Total match results: {totalEvaluated}");
	Console.WriteLine($"  Matched: {totalMatched}, Ambiguous: {totalAmbiguous}, Unresolved: {totalUnresolved}, Ineligible: {totalIneligible}");
	Console.WriteLine($"Tiles with ADT ground truth: {tileToWmoPaths.Count}");
	Console.WriteLine($"Results with ADT ground truth: {totalWithAdt}");
	Console.WriteLine();
	Console.WriteLine($"=== Precision ===");
	Console.WriteLine($"  Precision@1 (top-1 in ADT placement list): {precisionAt1}/{totalWithAdt} = {p1:P1}");
	Console.WriteLine($"  Precision@3 (top-3 in ADT placement list): {precisionAt3}/{totalWithAdt} = {p3:P1}");
	Console.WriteLine();
	Console.WriteLine($"=== Failure Categories ===");
	foreach (var kv in failureCategories.OrderByDescending(static kv => kv.Value))
		if (kv.Value > 0)
			Console.WriteLine($"  {kv.Key}: {kv.Value}");
	Console.WriteLine();
	Console.WriteLine($"=== Sample Correct Matches (top 15) ===");
	Console.WriteLine($"  {"PM4 Id",-35} {"Tile",-8} {"Status",-10} {"Top Match",-50} {"Correct",-7}");
	foreach (var row in detailRows.Where(static r => r.correct).Take(15))
	{
		string shortPm4 = row.pm4Id.Length > 33 ? "..." + row.pm4Id[^32..] : row.pm4Id;
		string shortMatch = row.topMatch.Length > 48 ? "..." + row.topMatch[^47..] : row.topMatch;
		Console.WriteLine($"  {shortPm4,-35} {row.tile,-8} {row.status,-10} {shortMatch,-50} {"YES",-7}");
	}
	Console.WriteLine();
	Console.WriteLine($"=== Sample Incorrect Matches (top 15) ===");
	Console.WriteLine($"  {"PM4 Id",-35} {"Tile",-8} {"Status",-10} {"Top Match",-50} {"ADT WMOs",-50}");
	foreach (var row in detailRows.Where(static r => !r.correct && r.status == "Matched").Take(15))
	{
		string shortPm4 = row.pm4Id.Length > 33 ? "..." + row.pm4Id[^32..] : row.pm4Id;
		string shortMatch = row.topMatch.Length > 48 ? "..." + row.topMatch[^47..] : row.topMatch;
		Console.WriteLine($"  {shortPm4,-35} {row.tile,-8} {row.status,-10} {shortMatch,-50} {row.adtWmos,-50}");
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			ValidationDate = DateTime.UtcNow.ToString("o"),
			TotalResults = totalEvaluated,
			ResultsWithAdt = totalWithAdt,
			PrecisionAt1 = p1,
			PrecisionAt3 = p3,
			PrecisionAt1Count = precisionAt1,
			PrecisionAt3Count = precisionAt3,
			FailureCategories = failureCategories,
			DetailRows = detailRows,
		};

		string json = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote validation report to {outputPath}");
	}
}

static string ExtractTileKeyFromPm4Id(string pm4FingerprintId)
{
	int tileStart = pm4FingerprintId.IndexOf("tile", StringComparison.OrdinalIgnoreCase);
	if (tileStart < 0)
		return string.Empty;

	int ck24Start = pm4FingerprintId.IndexOf("_ck24_", StringComparison.OrdinalIgnoreCase);
	if (ck24Start < 0)
		return string.Empty;

	return pm4FingerprintId.Substring(tileStart + 4, ck24Start - tileStart - 4);
}

static string ExtractTileKeyFromAdtName(string fileNameWithoutExtension)
{
	int obj0Index = fileNameWithoutExtension.IndexOf("_obj0", StringComparison.OrdinalIgnoreCase);
	if (obj0Index <= 0)
		return string.Empty;

	string prefix = fileNameWithoutExtension[..obj0Index];
	string[] parts = prefix.Split('_');
	if (parts.Length < 3)
		return string.Empty;

	return $"{parts[^2]}_{parts[^1]}";
}

static void RunPm4BuildWmoSurfaceDb(string[] args)
{
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? listfilePath = GetOption(args, "--listfile", "-l");
	string? binSizeText = GetOption(args, "--bin-size", "-b");
	string? areaBinSizeText = GetOption(args, "--area-bin-size", "-ab");
	string? normalAlignmentBinSizeText = GetOption(args, "--normal-alignment-bin-size", "-na");
	string? planarOffsetBinSizeText = GetOption(args, "--planar-offset-bin-size", "-po");
	string? limitText = GetOption(args, "--limit", "-n");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root <staged client dir> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (!TryBuildArchiveBootstrapOptions(args, out ArchiveCatalogBootstrapOptions bootstrapOptions))
		return;

	float binSize = 1.0f;
	if (!string.IsNullOrWhiteSpace(binSizeText))
		float.TryParse(binSizeText, out binSize);

	float areaBinSize = 1.0f;
	if (!string.IsNullOrWhiteSpace(areaBinSizeText))
		float.TryParse(areaBinSizeText, out areaBinSize);

	float normalAlignmentBinSize = 0.0f;
	if (!string.IsNullOrWhiteSpace(normalAlignmentBinSizeText))
		float.TryParse(normalAlignmentBinSizeText, out normalAlignmentBinSize);

	float planarOffsetBinSize = 0.0f;
	if (!string.IsNullOrWhiteSpace(planarOffsetBinSizeText))
		float.TryParse(planarOffsetBinSizeText, out planarOffsetBinSize);

	int limit = 0;
	if (!string.IsNullOrWhiteSpace(limitText))
		int.TryParse(limitText, out limit);

	Console.WriteLine($"Building WMO surface correlation database from: {Path.GetFullPath(archiveRoot)} (binSize={binSize}, areaBinSize={areaBinSize}, normalAlignmentBinSize={normalAlignmentBinSize}, planarOffsetBinSize={planarOffsetBinSize})");

	List<string> wmoPaths = EnumerateWmoRoots(archiveRoot, bootstrapOptions, listfilePath);

	if (limit > 0 && wmoPaths.Count > limit)
		wmoPaths = wmoPaths.Take(limit).ToList();

	Console.WriteLine($"Found {wmoPaths.Count} WMO root files.");

	SurfaceCorrelationDatabase database = Pm4SurfaceBuildSupport.BuildSurfaceDatabase(
		archiveRoot, bootstrapOptions, wmoPaths, binSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize,
		progress: static msg => Console.WriteLine(msg));

	Console.WriteLine($"\nDatabase: {database.WmoCount} WMO roots, {database.Records.Count} surface fingerprints.");
	int totalTriangles = database.Records.Sum(static r => r.TriangleCount);
	Console.WriteLine($"  Total triangles across all fingerprints: {totalTriangles}");

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);
		string json = JsonSerializer.Serialize(database, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote surface database to {outputPath} ({new FileInfo(outputPath).Length / 1024 / 1024:F1} MB)");
	}
}

static void RunPm4ExtractPm4Surfaces(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? output = GetOption(args, "--output", "-o");
	string? binSizeText = GetOption(args, "--bin-size", "-b");
	string? areaBinSizeText = GetOption(args, "--area-bin-size", "-ab");
	string? normalAlignmentBinSizeText = GetOption(args, "--normal-alignment-bin-size", "-na");
	string? planarOffsetBinSizeText = GetOption(args, "--planar-offset-bin-size", "-po");

	if (string.IsNullOrWhiteSpace(input) || !Directory.Exists(input))
	{
		Console.Error.WriteLine("Error: --input <directory> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	float binSize = 1.0f;
	if (!string.IsNullOrWhiteSpace(binSizeText))
		float.TryParse(binSizeText, out binSize);

	float areaBinSize = 1.0f;
	if (!string.IsNullOrWhiteSpace(areaBinSizeText))
		float.TryParse(areaBinSizeText, out areaBinSize);

	float normalAlignmentBinSize = 0.0f;
	if (!string.IsNullOrWhiteSpace(normalAlignmentBinSizeText))
		float.TryParse(normalAlignmentBinSizeText, out normalAlignmentBinSize);

	float planarOffsetBinSize = 0.0f;
	if (!string.IsNullOrWhiteSpace(planarOffsetBinSizeText))
		float.TryParse(planarOffsetBinSizeText, out planarOffsetBinSize);

	List<SurfaceCorrelationFingerprint> fingerprints = Pm4SurfaceBuildSupport.ExtractPm4SurfaceFingerprints(
		input, binSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize, progress: static msg => Console.WriteLine(msg));

	Console.WriteLine($"\nTotal: {fingerprints.Count} PM4 surface fingerprints.");
	int totalTriangles = fingerprints.Sum(static f => f.TriangleCount);
	Console.WriteLine($"  Total triangles: {totalTriangles}");

	var byType = fingerprints.GroupBy(static f => f.Ck24Type)
		.Select(static g => (Type: g.Key, Count: g.Count(), Triangles: g.Sum(static f => f.TriangleCount)))
		.OrderByDescending(static x => x.Count)
		.ToList();
	Console.WriteLine("\nBy CK24 type:");
	foreach (var t in byType)
		Console.WriteLine($"  0x{t.Type:X2}: {t.Count} fingerprints, {t.Triangles} triangles");

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			BuildDate = DateTime.UtcNow.ToString("o"),
			BinSize = binSize,
			AreaBinSize = areaBinSize,
			NormalAlignmentBinSize = normalAlignmentBinSize,
			PlanarOffsetBinSize = planarOffsetBinSize,
			TotalFingerprints = fingerprints.Count,
			TotalTriangles = totalTriangles,
			Fingerprints = fingerprints,
		};

		string json = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote {fingerprints.Count} surface fingerprints to {outputPath} ({new FileInfo(outputPath).Length / 1024 / 1024:F1} MB)");
	}
}

static void RunPm4MatchSurfaces(string[] args)
{
	string? pm4Path = GetOption(args, "--pm4-surfaces", "-p");
	string? wmoDbPath = GetOption(args, "--wmo-surface-db", "-w");
	string? minScoreText = GetOption(args, "--min-score", "-s");
	string? output = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(pm4Path) || !File.Exists(pm4Path))
	{
		Console.Error.WriteLine("Error: --pm4-surfaces <path> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(wmoDbPath) || !File.Exists(wmoDbPath))
	{
		Console.Error.WriteLine("Error: --wmo-surface-db <path> is required and must exist.");
		Environment.ExitCode = 1;
		return;
	}

	double minScore = 0.50;
	if (!string.IsNullOrWhiteSpace(minScoreText))
		double.TryParse(minScoreText, out minScore);

	Console.WriteLine($"Loading PM4 surface fingerprints from: {Path.GetFullPath(pm4Path)}");
	string pm4Json = File.ReadAllText(pm4Path);
	Pm4SurfaceExtractOutput? pm4Output = JsonSerializer.Deserialize<Pm4SurfaceExtractOutput>(pm4Json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (pm4Output is null || pm4Output.Fingerprints is null || pm4Output.Fingerprints.Count == 0)
	{
		Console.Error.WriteLine("Error: PM4 surface file contains no entries.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {pm4Output.Fingerprints.Count} PM4 surface fingerprints ({pm4Output.TotalTriangles} total triangles).");

	Console.WriteLine($"Loading WMO surface database from: {Path.GetFullPath(wmoDbPath)}");
	string wmoJson = File.ReadAllText(wmoDbPath);
	SurfaceCorrelationDatabase? wmoDb = JsonSerializer.Deserialize<SurfaceCorrelationDatabase>(wmoJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (wmoDb is null || wmoDb.Records.Count == 0)
	{
		Console.Error.WriteLine("Error: WMO surface database contains no entries.");
		Environment.ExitCode = 1;
		return;
	}
	Console.WriteLine($"Loaded {wmoDb.WmoCount} WMO roots, {wmoDb.Records.Count} surface fingerprints.");

	int wmoTotalTriangles = wmoDb.Records.Sum(static r => r.TriangleCount);
	Console.WriteLine($"  WMO total triangles: {wmoTotalTriangles}");

	SurfaceMatchOptions options = new(MinScore: minScore);
	Console.WriteLine($"\nMatching {pm4Output.Fingerprints.Count} PM4 surfaces against {wmoDb.Records.Count} WMO surfaces (minScore={minScore:F2})...");

	IReadOnlyList<SurfaceMatchResult> results = Pm4SurfaceCorrelationMatcher.Match(
		pm4Output.Fingerprints, wmoDb, options);

	int matched = results.Count(static r => r.Status == "Matched");
	int ambiguous = results.Count(static r => r.Status == "Ambiguous");
	int unresolved = results.Count(static r => r.Status == "Unresolved");
	int ineligible = results.Count(static r => r.Status == "Ineligible");

	Console.WriteLine($"\nResults: {matched} matched, {ambiguous} ambiguous, {unresolved} unresolved, {ineligible} ineligible");

	var topMatches = results.Where(static r => r.Status == "Matched")
		.OrderByDescending(static r => r.Candidates.Count > 0 ? r.Candidates[0].SymmetricScore : 0)
		.Take(30)
		.ToList();

	if (topMatches.Count > 0)
	{
		Console.WriteLine($"\nTop 30 matched:");
		Console.WriteLine($"  {"PM4 Id",-35} {"Tris",-6} {"WMO Candidate",-50} {"Score",-8} {"PM4 Cov",-8} {"WMO Cov",-8}");
		foreach (SurfaceMatchResult r in topMatches)
		{
			if (r.Candidates.Count == 0) continue;
			SurfaceMatchCandidate c = r.Candidates[0];
			string shortPm4 = r.Pm4FingerprintId.Length > 33 ? "..." + r.Pm4FingerprintId[^32..] : r.Pm4FingerprintId;
			string shortWmo = c.Candidate.AssetPath.Length > 48 ? "..." + c.Candidate.AssetPath[^47..] : c.Candidate.AssetPath;
			Console.WriteLine($"  {shortPm4,-35} {r.TriangleCount,6} {shortWmo,-50} {c.SymmetricScore,8:F3} {c.Pm4Coverage,8:F2} {c.WmoCoverage,8:F2}");
		}
	}

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? dir = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(dir))
			Directory.CreateDirectory(dir);

		var outputModel = new
		{
			MatchDate = DateTime.UtcNow.ToString("o"),
			Pm4FingerprintCount = pm4Output.Fingerprints.Count,
			WmoFingerprintCount = wmoDb.Records.Count,
			MinScore = minScore,
			Matched = matched,
			Ambiguous = ambiguous,
			Unresolved = unresolved,
			Ineligible = ineligible,
			Results = results,
		};

		string json = JsonSerializer.Serialize(outputModel, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(outputPath, json);
		Console.WriteLine($"\nWrote {results.Count} match results to {outputPath}");
	}
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

sealed record Pm4FingerprintExtractOutput(
	string BuildDate,
	string SourceDirectory,
	int TotalFiles,
	int ProcessedFiles,
	int TotalFingerprints,
	IReadOnlyList<Pm4FingerprintRecord> Fingerprints);

sealed record Pm4FingerprintMatchOutput(
	string MatchDate,
	int Pm4FingerprintCount,
	int WmoFingerprintCount,
	double MinScore,
	int Matched,
	int Ambiguous,
	int Unresolved,
	int Ineligible,
	IReadOnlyList<Pm4FingerprintMatchResult> Results);

sealed record Pm4SurfaceExtractOutput(
	string BuildDate,
	float BinSize,
	float AreaBinSize,
	float NormalAlignmentBinSize,
	float PlanarOffsetBinSize,
	int TotalFingerprints,
	int TotalTriangles,
	IReadOnlyList<SurfaceCorrelationFingerprint> Fingerprints);

sealed record Pm4SurfaceMatchOutput(
	string MatchDate,
	int Pm4FingerprintCount,
	int WmoFingerprintCount,
	double MinScore,
	int Matched,
	int Ambiguous,
	int Unresolved,
	int Ineligible,
	IReadOnlyList<SurfaceMatchResult> Results);
