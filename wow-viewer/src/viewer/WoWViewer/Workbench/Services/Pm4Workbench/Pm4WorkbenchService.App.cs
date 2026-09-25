using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// Pm4WorkbenchService: members moved from ViewerApp.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class Pm4WorkbenchService
{
    private float _pm4TranslationStepUnits = 10f;
    private float _pm4RotationStepDegrees = 90f;
    private float _pm4ScaleStepUnits = 0.1f;
    private Pm4WorkbenchTab? _pendingPm4WorkbenchTab;
    private Pm4ObjectMatchReport? _pm4ObjectMatchReport;
    private Pm4ObjectMatchObject? _selectedPm4ObjectMatch;
    private (int tileX, int tileY, uint ck24, int objectPart)? _selectedPm4ObjectMatchKey;
    private int _selectedPm4ObjectMatchCacheMaxMatches = -1;
    private readonly List<(int tileX, int tileY, uint ck24, int objectPart)> _pm4ObjectCollection = new();
    private int _selectedPm4ObjectMatchObjectIndex = -1;
    private int _selectedPm4ObjectMatchCandidateIndex;
    private Pm4WmoMatchResult? _pm4WmoGroupMatchResult;
    private string _pm4WmoMatchStatus = "";
    private Pm4WmoCorrelationReport? _pm4WmoCorrelationReport;
    private int _pm4WmoCorrelationMaxMatchesPerPlacement = 8;
    private int _selectedPm4WmoCorrelationPlacementIndex = -1;
    private int _selectedPm4WmoCorrelationMatchIndex;
    private bool _pm4WmoCorrelationNearOnly = true;
    private string _pm4WmoCorrelationModelFilter = string.Empty;
    private string _pm4SceneFilter = "";
}
