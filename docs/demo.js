// casoCAD — static browser demo
// All objects live only in browser memory. No scene file is loaded.
// The real casoCAD app runs as a Python/PySide6 desktop application.

let THREE, OrbitControls;
try {
  // Resolved via the <script type="importmap"> in index.html. The addon imports
  // a bare "three" specifier, so the import map (not a raw URL) is required.
  THREE = await import("three");
  ({ OrbitControls } = await import("three/addons/controls/OrbitControls.js"));
} catch (err) {
  console.warn("casoCAD demo: Three.js CDN unavailable —", err);
  document.getElementById("cdn-fallback")?.classList.remove("hidden");
  setStatus("CDN unavailable", true);
  setLog("Three.js failed to load from unpkg.com.");
  // Stop here; the rest of the demo needs Three.js.
  throw err;
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const treeEl = $("#scene-tree");
const propsEl = $("#properties");

function setStatus(text, busy = false) {
  $("#status-text").textContent = text;
  $("#status-dot").classList.toggle("busy", busy);
}
function setLog(text) {
  const t = new Date().toLocaleTimeString([], { hour12: false });
  $("#log-strip").textContent = `[${t}] ${text}`;
}

// ---------------------------------------------------------------------------
// Scene state (demo-only object registry)
// ---------------------------------------------------------------------------
const objects = []; // { id, name, kind, color, mesh, props }
let nextId = 1;
let selectedId = null;
let shellOpaque = false;
let boundaryVisible = true;
let meshOverlay = null;

const KIND = {
  domain: { label: "Box Domain", tag: "domain", color: 0x4aa8ff },
  obstacle: { label: "Cylinder Obstacle", tag: "obstacle", color: 0xe0604a },
  boundary: { label: "Boundary Tag", tag: "boundary", color: 0xe0a44a },
};

// ---------------------------------------------------------------------------
// Three.js setup
// ---------------------------------------------------------------------------
const viewport = $("#viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e12);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(7, 5.5, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.5, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(6, 10, 7);
scene.add(key);
const fill = new THREE.DirectionalLight(0x99bbff, 0.35);
fill.position.set(-6, 4, -5);
scene.add(fill);

// Grid + axes (always present — viewport is never empty)
const grid = new THREE.GridHelper(20, 20, 0x37424f, 0x222a33);
grid.position.y = 0;
scene.add(grid);

const axes = new THREE.AxesHelper(2.4);
axes.position.y = 0.001;
scene.add(axes);

// Container for demo CAD objects (so clear is easy)
const cadGroup = new THREE.Group();
scene.add(cadGroup);

// ---------------------------------------------------------------------------
// Object factories
// ---------------------------------------------------------------------------
const DOMAIN_SIZE = { x: 6, y: 3, z: 4 };

function makeDomain() {
  const geo = new THREE.BoxGeometry(DOMAIN_SIZE.x, DOMAIN_SIZE.y, DOMAIN_SIZE.z);
  const mat = new THREE.MeshStandardMaterial({
    color: KIND.domain.color,
    transparent: true,
    opacity: shellOpaque ? 0.9 : 0.12,
    metalness: 0.0,
    roughness: 0.7,
    depthWrite: shellOpaque,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, DOMAIN_SIZE.y / 2, 0);

  // Crisp wireframe edges so the domain reads as a flow box
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: KIND.domain.color })
  );
  mesh.add(edges);
  mesh.userData.edges = edges;
  return mesh;
}

function makeObstacle() {
  // Cylinder spanning the domain height, styled as a cutout obstacle.
  const r = 0.7;
  const h = DOMAIN_SIZE.y;
  const geo = new THREE.CylinderGeometry(r, r, h, 48, 1, false);
  const mat = new THREE.MeshStandardMaterial({
    color: KIND.obstacle.color,
    metalness: 0.1,
    roughness: 0.45,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(-0.5, h / 2, 0);

  // "Cutout" hint: a faint negative-space ring at the base
  const ringGeo = new THREE.RingGeometry(r * 0.98, r * 1.25, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: KIND.obstacle.color,
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(-0.5, 0.01, 0);
  mesh.userData.ring = ring;
  cadGroup.add(ring);
  return mesh;
}

function makeBoundary(index) {
  // Colored indicator plane on a face of the domain (inlet/outlet style).
  // Cycle across faces so multiple tags don't overlap.
  const faces = [
    { name: "Inlet (−X)", pos: [-DOMAIN_SIZE.x / 2, DOMAIN_SIZE.y / 2, 0], rot: [0, Math.PI / 2, 0], w: DOMAIN_SIZE.z, h: DOMAIN_SIZE.y, color: 0x4cc38a },
    { name: "Outlet (+X)", pos: [DOMAIN_SIZE.x / 2, DOMAIN_SIZE.y / 2, 0], rot: [0, -Math.PI / 2, 0], w: DOMAIN_SIZE.z, h: DOMAIN_SIZE.y, color: 0xe0604a },
    { name: "Wall (+Z)", pos: [0, DOMAIN_SIZE.y / 2, DOMAIN_SIZE.z / 2], rot: [0, 0, 0], w: DOMAIN_SIZE.x, h: DOMAIN_SIZE.y, color: 0xe0a44a },
    { name: "Wall (−Z)", pos: [0, DOMAIN_SIZE.y / 2, -DOMAIN_SIZE.z / 2], rot: [0, Math.PI, 0], w: DOMAIN_SIZE.x, h: DOMAIN_SIZE.y, color: 0xb07adb },
    { name: "Floor (−Y)", pos: [0, 0.01, 0], rot: [-Math.PI / 2, 0, 0], w: DOMAIN_SIZE.x, h: DOMAIN_SIZE.z, color: 0x4aa8ff },
  ];
  const f = faces[index % faces.length];
  const geo = new THREE.PlaneGeometry(f.w * 0.96, f.h * 0.96);
  const mat = new THREE.MeshBasicMaterial({
    color: f.color,
    transparent: true,
    opacity: 0.28,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(...f.pos);
  mesh.rotation.set(...f.rot);

  const border = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: f.color })
  );
  mesh.add(border);
  mesh.userData.faceName = f.name;
  mesh.userData.faceColor = f.color;
  return mesh;
}

// ---------------------------------------------------------------------------
// Object registry operations
// ---------------------------------------------------------------------------
function addObject(kind) {
  const def = KIND[kind];
  let mesh, name, props;

  if (kind === "domain") {
    if (objects.some((o) => o.kind === "domain")) {
      setLog("A box domain already exists in this demo scene.");
      return;
    }
    mesh = makeDomain();
    name = "FlowDomain";
    props = {
      "SDF role": "domain (positive)",
      "Primitive": "box",
      "Size X": `${DOMAIN_SIZE.x.toFixed(2)} m`,
      "Size Y": `${DOMAIN_SIZE.y.toFixed(2)} m`,
      "Size Z": `${DOMAIN_SIZE.z.toFixed(2)} m`,
    };
  } else if (kind === "obstacle") {
    if (!objects.some((o) => o.kind === "domain")) {
      setLog("Add a Box Domain before placing an obstacle.");
      return;
    }
    const n = objects.filter((o) => o.kind === "obstacle").length + 1;
    mesh = makeObstacle();
    name = `Obstacle.${String(n).padStart(2, "0")}`;
    props = {
      "SDF role": "subtraction (negative)",
      "Primitive": "cylinder",
      "Radius": "0.70 m",
      "Height": `${DOMAIN_SIZE.y.toFixed(2)} m`,
      "Boolean": "domain − cylinder",
    };
  } else {
    if (!objects.some((o) => o.kind === "domain")) {
      setLog("Add a Box Domain before tagging boundaries.");
      return;
    }
    const n = objects.filter((o) => o.kind === "boundary").length;
    mesh = makeBoundary(n);
    name = `bc_${mesh.userData.faceName.split(" ")[0].toLowerCase()}_${n + 1}`;
    mesh.visible = boundaryVisible;
    props = {
      "Type": "boundary region",
      "Face": mesh.userData.faceName,
      "Condition": "tag-only (demo)",
    };
  }

  cadGroup.add(mesh);
  const obj = {
    id: nextId++,
    name,
    kind,
    color: kind === "boundary" ? mesh.userData.faceColor : def.color,
    mesh,
    props,
  };
  objects.push(obj);

  refreshTree();
  refreshDomainInfo();
  updateEmptyState();
  select(obj.id);
  setLog(`Added ${def.label} → “${name}”.`);
  setStatus("Ready");
}

function removeMesh(obj) {
  cadGroup.remove(obj.mesh);
  if (obj.mesh.userData.ring) cadGroup.remove(obj.mesh.userData.ring);
  obj.mesh.traverse?.((n) => {
    n.geometry?.dispose?.();
    if (Array.isArray(n.material)) n.material.forEach((m) => m.dispose?.());
    else n.material?.dispose?.();
  });
}

function clearScene() {
  for (const obj of objects) removeMesh(obj);
  objects.length = 0;
  selectedId = null;
  clearMeshOverlay();
  refreshTree();
  refreshProps();
  refreshDomainInfo();
  updateEmptyState();
  setLog("Scene cleared. Empty scene restored.");
  setStatus("Ready");
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
function select(id) {
  selectedId = id;
  // visual highlight
  for (const obj of objects) {
    const isSel = obj.id === id;
    const edges = obj.mesh.userData.edges;
    if (edges) edges.material.color.set(isSel ? 0xffffff : obj.color);
    if (obj.kind === "obstacle") {
      obj.mesh.material.emissive?.set(isSel ? 0x331008 : 0x000000);
    }
  }
  refreshTree();
  refreshProps();
}

function refreshProps() {
  const obj = objects.find((o) => o.id === selectedId);
  if (!obj) {
    propsEl.innerHTML = '<div class="empty-sel">No selection</div>';
    return;
  }
  const rows = [
    `<div class="kv"><span>Name</span><span>${obj.name}</span></div>`,
    `<div class="kv"><span>Kind</span><span>${KIND[obj.kind].label}</span></div>`,
  ];
  for (const [k, v] of Object.entries(obj.props)) {
    rows.push(`<div class="kv"><span>${k}</span><span>${v}</span></div>`);
  }
  propsEl.innerHTML = rows.join("");
}

// ---------------------------------------------------------------------------
// Tree / panels rendering
// ---------------------------------------------------------------------------
function refreshTree() {
  $("#tree-count").textContent = String(objects.length);
  if (objects.length === 0) {
    treeEl.innerHTML = '<li class="tree-empty">No objects</li>';
    return;
  }
  treeEl.innerHTML = objects
    .map((o) => {
      const sel = o.id === selectedId ? " selected" : "";
      const hex = "#" + o.color.toString(16).padStart(6, "0");
      return `<li class="tree-item${sel}" data-id="${o.id}">
        <span class="tree-swatch" style="background:${hex}"></span>
        <span class="tree-label">${o.name}</span>
        <span class="tree-kind">${KIND[o.kind].tag}</span>
      </li>`;
    })
    .join("");
}

function refreshDomainInfo() {
  const hasDomain = objects.some((o) => o.kind === "domain");
  $("#dom-type").textContent = hasDomain ? "Box (rectangular)" : "—";
  $("#dom-bounds").textContent = hasDomain
    ? `${DOMAIN_SIZE.x}×${DOMAIN_SIZE.y}×${DOMAIN_SIZE.z} m`
    : "—";
  $("#dom-obst").textContent = String(objects.filter((o) => o.kind === "obstacle").length);
  $("#dom-bnd").textContent = String(objects.filter((o) => o.kind === "boundary").length);

  const enable = hasDomain;
  $("#btn-mesh").disabled = !enable;
  $("#btn-export").disabled = !enable;
}

function updateEmptyState() {
  $("#vp-empty").classList.toggle("hidden", objects.length > 0);
}

// ---------------------------------------------------------------------------
// Viewport actions
// ---------------------------------------------------------------------------
function fitView() {
  const box = new THREE.Box3();
  let has = false;
  cadGroup.traverse((n) => {
    if (n.isMesh) {
      box.expandByObject(n);
      has = true;
    }
  });
  if (!has) {
    camera.position.set(7, 5.5, 8);
    controls.target.set(0, 0.5, 0);
    controls.update();
    setLog("Fit view → empty scene framing.");
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = 0.5 * Math.max(size.x, size.y, size.z);
  const dist = (radius / Math.sin((camera.fov * Math.PI) / 180 / 2)) * 1.5;
  const dir = new THREE.Vector3(1, 0.8, 1.1).normalize();
  camera.position.copy(center.clone().add(dir.multiplyScalar(dist)));
  controls.target.copy(center);
  controls.update();
  setLog("Fit view → framed scene contents.");
}

function toggleOpacity(btn) {
  shellOpaque = !shellOpaque;
  for (const obj of objects.filter((o) => o.kind === "domain")) {
    obj.mesh.material.opacity = shellOpaque ? 0.9 : 0.12;
    obj.mesh.material.depthWrite = shellOpaque;
    obj.mesh.material.needsUpdate = true;
  }
  btn.classList.toggle("active", shellOpaque);
  setLog(`Shell opacity: ${shellOpaque ? "solid" : "transparent"}.`);
}

function toggleBoundary(btn) {
  boundaryVisible = !boundaryVisible;
  for (const obj of objects.filter((o) => o.kind === "boundary")) {
    obj.mesh.visible = boundaryVisible;
  }
  btn.classList.toggle("active", boundaryVisible); // active = regions shown
  setLog(`Boundary regions: ${boundaryVisible ? "shown" : "hidden"}.`);
}

// ---------------------------------------------------------------------------
// Mesh preview (fake lattice overlay) + export (log only)
// ---------------------------------------------------------------------------
function clearMeshOverlay() {
  if (meshOverlay) {
    cadGroup.remove(meshOverlay);
    meshOverlay.geometry?.dispose?.();
    meshOverlay.material?.dispose?.();
    meshOverlay = null;
  }
}

function meshPreview(btn) {
  if (!objects.some((o) => o.kind === "domain")) {
    setLog("Mesh preview needs a domain. Add a Box Domain first.");
    return;
  }
  if (meshOverlay) {
    clearMeshOverlay();
    btn?.classList.remove("active");
    setLog("Mesh preview hidden.");
    return;
  }
  setStatus("Generating preview…", true);

  // Build a lightweight lattice of points clipped to the domain, skipping
  // points that fall inside the cylindrical obstacle — purely illustrative.
  const obstacles = objects.filter((o) => o.kind === "obstacle");
  const pts = [];
  const nx = 13, ny = 7, nz = 9;
  for (let i = 0; i <= nx; i++) {
    for (let j = 0; j <= ny; j++) {
      for (let k = 0; k <= nz; k++) {
        const x = -DOMAIN_SIZE.x / 2 + (i / nx) * DOMAIN_SIZE.x;
        const y = (j / ny) * DOMAIN_SIZE.y;
        const z = -DOMAIN_SIZE.z / 2 + (k / nz) * DOMAIN_SIZE.z;
        // crude obstacle carve-out (cylinder at x=-0.5,z=0,r=0.7)
        let inside = false;
        for (const o of obstacles) {
          const dx = x - o.mesh.position.x;
          const dz = z - o.mesh.position.z;
          if (dx * dx + dz * dz < 0.72 * 0.72) { inside = true; break; }
        }
        if (!inside) pts.push(x, y, z);
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.PointsMaterial({ color: 0x6fe3c4, size: 0.05, sizeAttenuation: true });
  meshOverlay = new THREE.Points(geo, mat);
  cadGroup.add(meshOverlay);

  btn?.classList.add("active");
  const cells = pts.length / 3;
  setLog(`Mesh preview: ${cells} lattice nodes (illustrative, not a real mesh).`);
  setStatus("Ready");
}

function exportArrow() {
  if (!objects.some((o) => o.kind === "domain")) {
    setLog("Nothing to export — add a Box Domain first.");
    return;
  }
  const o = objects.filter((x) => x.kind === "obstacle").length;
  const b = objects.filter((x) => x.kind === "boundary").length;
  setStatus("Export (demo)", true);
  setLog(`Demo export: would write Arrow/CFD case (1 domain, ${o} obstacle(s), ${b} BC tag(s)). No solver run.`);
  setTimeout(() => setStatus("Ready"), 900);
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------
document.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act) {
    const btn = e.target.closest("[data-act]");
    switch (act) {
      case "add-box": addObject("domain"); break;
      case "add-cyl": addObject("obstacle"); break;
      case "add-bnd": addObject("boundary"); break;
      case "clear": clearScene(); break;
      case "fit": fitView(); break;
      case "toggle-opacity": toggleOpacity(btn); break;
      case "toggle-boundary": toggleBoundary(btn); break;
      case "mesh": meshPreview($('#vp-toolbar [data-act="mesh"]')); break;
      case "export": exportArrow(); break;
    }
    return;
  }
  const item = e.target.closest(".tree-item");
  if (item) select(Number(item.dataset.id));
});

// ---------------------------------------------------------------------------
// Resize + render loop
// ---------------------------------------------------------------------------
function resize() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  if (w === 0 || h === 0) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(viewport);

let frames = 0, lastFpsT = performance.now();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);

  frames++;
  const now = performance.now();
  if (now - lastFpsT >= 1000) {
    $("#fps-readout").textContent = `${frames} fps · ${objects.length} obj`;
    frames = 0;
    lastFpsT = now;
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
resize();
refreshTree();
refreshProps();
refreshDomainInfo();
updateEmptyState();
// init the boundary-toggle button as active (regions visible by default)
$('#vp-toolbar [data-act="toggle-boundary"]')?.classList.add("active");
animate();
setStatus("Ready");
setLog("Viewport ready. Empty scene — grid and axes shown.");
