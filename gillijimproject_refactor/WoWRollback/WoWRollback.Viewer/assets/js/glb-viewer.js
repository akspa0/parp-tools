import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

const viewport = document.getElementById('viewport');
const fileInput = document.getElementById('fileInput');
const chkDouble = document.getElementById('chkDouble');
const chkWire = document.getElementById('chkWire');
const chkGrid = document.getElementById('chkGrid');
const chkAxes = document.getElementById('chkAxes');
const chkRot90 = document.getElementById('chkRot90');
const chkInvertZ = document.getElementById('chkInvertZ');
const scaleRange = document.getElementById('scaleRange');
const scaleNum = document.getElementById('scaleNum');
const scaleReset = document.getElementById('scaleReset');
const btnTop = document.getElementById('btnTop');
const btnThreeQ = document.getElementById('btnThreeQ');
const btnResetView = document.getElementById('btnResetView');

let renderer, scene, camera, controls, grid, axes;
// transformGroup applies global transforms (rotate/invertZ); group applies global scale and holds models
let transformGroup = new THREE.Group();
let group = new THREE.Group();
const statusEl = document.getElementById('dropHint');
let scaleDenom = parseInt(localStorage.getItem('glbViewerScaleDenom') || '64', 10);

init();

function init() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight - 48);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121212);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / (window.innerHeight - 48), 0.1, 200000);
  camera.position.set(200, 200, 200);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = false;
  controls.minPolarAngle = 0.001;                // keep above horizon
  controls.maxPolarAngle = Math.PI / 2 - 0.01;   // don't orbit below ground
  controls.rotateSpeed = 0.9;
  controls.zoomSpeed = 1.0;
  controls.panSpeed = 0.8;
  controls.update();

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
  scene.add(hemiLight);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(300, 1000, 300);
  dir.castShadow = false;
  scene.add(dir);

  grid = new THREE.GridHelper(5000, 50, 0x444444, 0x222222);
  scene.add(grid);
  axes = new THREE.AxesHelper(500);
  axes.visible = false;
  scene.add(axes);

  scene.add(transformGroup);
  transformGroup.add(group);

  window.addEventListener('resize', onResize);
  fileInput.addEventListener('change', onFiles);
  document.addEventListener('dragover', e => { e.preventDefault(); });
  document.addEventListener('drop', onDrop);
  chkDouble.addEventListener('change', updateMaterialFlags);
  chkWire.addEventListener('change', updateMaterialFlags);
  chkGrid.addEventListener('change', () => grid.visible = chkGrid.checked);
  chkAxes.addEventListener('change', () => axes.visible = chkAxes.checked);
  if (chkRot90) chkRot90.addEventListener('change', updateTransform);
  if (chkInvertZ) chkInvertZ.addEventListener('change', () => { updateTransform(); updateMaterialFlags(); });
  if (btnTop) btnTop.addEventListener('click', presetTopView);
  if (btnThreeQ) btnThreeQ.addEventListener('click', presetThreeQuarterView);
  if (btnResetView) btnResetView.addEventListener('click', () => fitCameraToGroup(transformGroup, camera, controls));

  // Scale controls
  if (scaleRange && scaleNum) {
    scaleRange.value = String(scaleDenom);
    scaleNum.value = String(scaleDenom);
    const applyScale = () => {
      scaleDenom = Math.max(1, Math.min(4096, parseInt(scaleNum.value || scaleRange.value, 10) || 64));
      scaleRange.value = String(scaleDenom);
      scaleNum.value = String(scaleDenom);
      const s = 1 / scaleDenom;
      group.scale.set(s, s, s);
      localStorage.setItem('glbViewerScaleDenom', String(scaleDenom));
      fitCameraToGroup(transformGroup, camera, controls);
      resizeHelpersToGroup();
      if (statusEl) statusEl.textContent = `Scale 1/${scaleDenom}`;
    };
    scaleRange.addEventListener('input', () => { scaleNum.value = scaleRange.value; applyScale(); });
    scaleNum.addEventListener('change', applyScale);
    if (scaleReset) scaleReset.addEventListener('click', () => { scaleNum.value = '64'; scaleRange.value = '64'; applyScale(); });
    // initial apply
    applyScale();
  }

function updateTransform() {
  const rotY = (chkRot90 && chkRot90.checked) ? Math.PI / 2 : 0;
  transformGroup.rotation.set(0, rotY, 0);
  const invZ = (chkInvertZ && chkInvertZ.checked) ? -1 : 1;
  transformGroup.scale.set(1, 1, invZ);
  fitCameraToGroup(transformGroup, camera, controls);
  resizeHelpersToGroup();
}

function presetThreeQuarterView() {
  const box = new THREE.Box3().setFromObject(transformGroup);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  let d = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5;
  camera.position.set(center.x + d, center.y + d, center.z + d);
  controls.target.copy(center);
  controls.update();
}

function presetTopView() {
  const box = new THREE.Box3().setFromObject(transformGroup);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.z);
  const fov = camera.fov * (Math.PI / 180);
  let d = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.2;
  camera.position.set(center.x, center.y + d, center.z);
  controls.target.copy(center);
  controls.update();
}

  animate();
}

function onResize() {
  camera.aspect = window.innerWidth / (window.innerHeight - 48);
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight - 48);
}

async function onFiles(e) {
  const files = Array.from(e.target.files);
  if (statusEl) statusEl.textContent = `Selected ${files.length} file(s). Loading...`;
  await loadFiles(files);
}

async function onDrop(e) {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files).filter(f => f.name.match(/\.(glb|gltf)$/i));
  if (statusEl) statusEl.textContent = `Dropped ${files.length} file(s). Loading...`;
  await loadFiles(files);
}

async function loadFiles(files) {
  clearScene();
  const loader = new GLTFLoader();

  // Load sequentially to keep memory reasonable
  for (const f of files) {
    const url = URL.createObjectURL(f);
    try {
      const gltf = await loader.loadAsync(url);
      const root = gltf.scene;
      normalizeMaterials(root);
      group.add(root);
    } catch (err) {
      console.error('Failed to load', f.name, err);
      if (statusEl) statusEl.textContent = `Error loading ${f.name}: ${err?.message ?? err}`;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  fitCameraToGroup(transformGroup, camera, controls);
  if (statusEl) statusEl.textContent = `Loaded ${group.children.length} model(s).`;
  resizeHelpersToGroup();
  // Set a stable initial angle
  presetThreeQuarterView();
}

function clearScene() {
  while (group.children.length) group.remove(group.children[0]);
}

function normalizeMaterials(object3d) {
  object3d.traverse(o => {
    if (o.isMesh && o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
        m.side = chkDouble.checked ? THREE.DoubleSide : THREE.FrontSide;
        m.wireframe = chkWire.checked;
      }
    }
  });
}

function updateMaterialFlags() {
  group.traverse(o => {
    if (o.isMesh && o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        const wantDouble = (chkDouble && chkDouble.checked) || (chkInvertZ && chkInvertZ.checked);
        m.side = wantDouble ? THREE.DoubleSide : THREE.FrontSide;
        m.wireframe = chkWire.checked;
        m.needsUpdate = true;
      }
    }
  });
}

function fitCameraToGroup(object3D, camera, controls) {
  const box = new THREE.Box3().setFromObject(object3D);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2)));
  cameraZ *= 1.5; // padding

  camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
  camera.near = cameraZ / 100;
  camera.far = cameraZ * 1000;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  const diag = Math.sqrt(size.x*size.x + size.y*size.y + size.z*size.z);
  controls.minDistance = Math.max(0.01, diag * 0.001);
  controls.maxDistance = Math.max(10, diag * 10);
  controls.update();
}

function resizeHelpersToGroup() {
  const box = new THREE.Box3().setFromObject(transformGroup);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.z) * 1.2 || 1000;
  if (grid) {
    scene.remove(grid);
    grid = new THREE.GridHelper(span, 50, 0x444444, 0x222222);
    grid.visible = chkGrid.checked;
    scene.add(grid);
  }
  if (axes) {
    axes.scale.set(span/500, span/500, span/500);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
