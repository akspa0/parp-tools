using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WoWRollback.AnalysisModule;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.Cli.Commands;

/// <summary>
/// Debug command to visualize a single ADT tile with terrain and placements.
/// No transforms, no complexity - just raw data for coordinate debugging.
/// </summary>
public static class DebugSingleAdtCommand
{
    public static async Task ExecuteAsync(int tileX, int tileY, string clientPath, string outDir, string mapName)
    {
        Console.WriteLine($"[DebugSingleAdt] Tile: ({tileX}, {tileY})");
        Console.WriteLine($"[DebugSingleAdt] Client: {clientPath}");
        Console.WriteLine($"[DebugSingleAdt] Output: {outDir}");
        Console.WriteLine($"[DebugSingleAdt] Map: {mapName}");

        Directory.CreateDirectory(outDir);

        // Setup archive source
        var mpqs = ArchiveLocator.LocateMpqs(clientPath);
        using var archiveSource = new PrioritizedArchiveSource(clientPath, mpqs);

        // Extract terrain mesh using AdtMeshExtractor
        Console.WriteLine($"[DebugSingleAdt] Extracting terrain mesh and placements...");
        var extractor = new AdtMeshExtractor();
        var result = extractor.ExtractFromArchive(archiveSource, mapName, outDir, exportGlb: true, exportObj: true, maxTiles: 1);
        
        Console.WriteLine($"[DebugSingleAdt] Mesh extraction complete: {result.TilesProcessed} tiles");

        // For now, create empty placements JSON (mesh extractor already extracted everything)
        var placements = new List<object>();
        Console.WriteLine($"[DebugSingleAdt] Placements will be loaded from analysis data");

        // Write placements JSON
        var placementsPath = Path.Combine(outDir, "placements.json");
        var json = System.Text.Json.JsonSerializer.Serialize(placements, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        await File.WriteAllTextAsync(placementsPath, json);
        Console.WriteLine($"[DebugSingleAdt] Placements saved: {placementsPath}");

        // Create minimal viewer HTML
        var viewerHtml = CreateViewerHtml();
        var viewerPath = Path.Combine(outDir, "viewer.html");
        await File.WriteAllTextAsync(viewerPath, viewerHtml);
        Console.WriteLine($"[DebugSingleAdt] Viewer created: {viewerPath}");

        Console.WriteLine($"[DebugSingleAdt] === Complete ===");
        Console.WriteLine($"[DebugSingleAdt] Open: {viewerPath}");
    }

    private static string CreateViewerHtml()
    {
        return """
<!DOCTYPE html>
<html>
<head>
    <title>ADT Debug Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; font-family: monospace; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <div>ADT Debug Viewer</div>
        <div>Gray = Terrain | Red = M2 | Blue = WMO</div>
        <div>Mouse: Rotate | Scroll: Zoom</div>
    </div>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

        // Setup scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87CEEB);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50000);
        camera.position.set(0, 500, 500);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);

        // Grid
        const gridHelper = new THREE.GridHelper(2000, 20);
        scene.add(gridHelper);

        // Axes
        const axesHelper = new THREE.AxesHelper(500);
        scene.add(axesHelper);

        // Load terrain (find the GLB file in mesh directory)
        const loader = new GLTFLoader();
        fetch('mesh/')
            .then(r => r.text())
            .then(html => {
                // Find .glb file in directory listing
                const match = html.match(/tile_\d+_\d+\.glb/);
                if (match) {
                    const glbFile = 'mesh/' + match[0];
                    console.log('Loading:', glbFile);
                    loader.load(glbFile, (gltf) => {
                        // NO TRANSFORMS - use raw coordinates
                        scene.add(gltf.scene);
                        console.log('Terrain loaded');
                    });
                } else {
                    console.error('No GLB file found');
                }
            })
            .catch(err => {
                console.error('Failed to load mesh directory:', err);
                // Try direct path as fallback
                loader.load('mesh/tile_30_23.glb', (gltf) => {
                    scene.add(gltf.scene);
                    console.log('Terrain loaded (fallback)');
                });
            });

        // Load placements
        fetch('placements.json')
            .then(r => r.json())
            .then(placements => {
                console.log(`Loaded ${placements.length} placements`);
                
                const m2Geom = new THREE.SphereGeometry(5, 8, 8);
                const m2Mat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                const wmoMat = new THREE.MeshBasicMaterial({ color: 0x0000ff });

                placements.forEach(p => {
                    const mat = p.type === 'M2' ? m2Mat : wmoMat;
                    const mesh = new THREE.Mesh(m2Geom, mat);
                    
                    // NO TRANSFORMS - use raw coordinates
                    mesh.position.set(p.x, p.z, p.y); // Note: WoW Y/Z swap for Three.js
                    scene.add(mesh);
                });
            });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
""";
    }
}
