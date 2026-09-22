/* global THREE */

const viewport = document.getElementById('viewport');
const loading = document.getElementById('loading');
const loadStatus = document.getElementById('loadStatus');
const summary = document.getElementById('summary');
const tooltip = document.getElementById('tooltip');
const renderStats = document.getElementById('renderStats');

let renderer;
let scene;
let camera;
let controls;
let manifest;
let outerRadius = 20;
let autoRotate = false;
let lastPointerEvent = null;
let pickPending = false;

const boundaryObjects = [];
const pmtObjects = [];
const supportObjects = [];
const knownStructureObjects = [];
const clippingPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.30;
raycaster.layers.set(1);
const pointer = new THREE.Vector2();
const pmtPickers = [];

const unitY = new THREE.Vector3(0, 1, 0);
const position = new THREE.Vector3();
const axis = new THREE.Vector3();
const scale = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const matrix = new THREE.Matrix4();

function material(color, opacity, side = THREE.FrontSide) {
  return new THREE.MeshPhongMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    side,
    depthWrite: opacity >= 1,
    clippingPlanes: [],
    shininess: 35,
  });
}

function setInstanceTransform(mesh, instance, row, cylinder = false) {
  position.set(row[0], row[1], row[2]);
  axis.set(row[3], row[4], row[5]).normalize();
  quaternion.setFromUnitVectors(unitY, axis);
  if (cylinder) scale.set(row[6], row[7], row[6]);
  else scale.set(row[6], row[7], row[6]);
  matrix.compose(position, quaternion, scale);
  mesh.setMatrixAt(instance, matrix);
}

function makeHemisphereGeometry() {
  const geometry = new THREE.SphereGeometry(
    1, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2,
  );
  geometry.computeVertexNormals();
  return geometry;
}

function makeDiscGeometry() {
  const geometry = new THREE.CylinderGeometry(1, 1, 1, 18, 1, false);
  geometry.computeVertexNormals();
  return geometry;
}

function makeAnnularCylinderGeometry(
  innerRadius, outerRadius, height, segments = 64,
) {
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const normals = [];
  const indices = [];
  const half = height / 2;

  function addQuad(points, normal) {
    const first = positions.length / 3;
    for (const point of points) {
      positions.push(point[0], point[1], point[2]);
      normals.push(normal[0], normal[1], normal[2]);
    }
    indices.push(
      first, first + 1, first + 2,
      first, first + 2, first + 3,
    );
  }

  for (let index = 0; index < segments; index += 1) {
    const angle0 = (index / segments) * Math.PI * 2;
    const angle1 = ((index + 1) / segments) * Math.PI * 2;
    const c0 = Math.cos(angle0);
    const s0 = Math.sin(angle0);
    const c1 = Math.cos(angle1);
    const s1 = Math.sin(angle1);
    const outer0 = [outerRadius * c0, outerRadius * s0];
    const outer1 = [outerRadius * c1, outerRadius * s1];
    const inner0 = [innerRadius * c0, innerRadius * s0];
    const inner1 = [innerRadius * c1, innerRadius * s1];
    const mid = (angle0 + angle1) / 2;
    const radial = [Math.cos(mid), Math.sin(mid), 0];

    addQuad([
      [outer0[0], outer0[1], -half],
      [outer1[0], outer1[1], -half],
      [outer1[0], outer1[1], half],
      [outer0[0], outer0[1], half],
    ], radial);
    addQuad([
      [inner1[0], inner1[1], -half],
      [inner0[0], inner0[1], -half],
      [inner0[0], inner0[1], half],
      [inner1[0], inner1[1], half],
    ], radial.map((value) => -value));
    addQuad([
      [inner0[0], inner0[1], half],
      [outer0[0], outer0[1], half],
      [outer1[0], outer1[1], half],
      [inner1[0], inner1[1], half],
    ], [0, 0, 1]);
    addQuad([
      [inner1[0], inner1[1], -half],
      [outer1[0], outer1[1], -half],
      [outer0[0], outer0[1], -half],
      [inner0[0], inner0[1], -half],
    ], [0, 0, -1]);
  }
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute(
    'normal',
    new THREE.Float32BufferAttribute(normals, 3),
  );
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();
  return geometry;
}

async function fetchRows(group) {
  const response = await fetch(group.url);
  if (!response.ok) {
    throw new Error(
      'Failed to load ' + group.url + ': ' + response.status,
    );
  }
  const values = new Float32Array(await response.arrayBuffer());
  if (values.length !== group.count * group.stride) {
    throw new Error(`Invalid instance buffer ${group.url}`);
  }
  return values;
}

function addControl(containerId, object, group, checked = true) {
  const container = document.getElementById(containerId);
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  input.addEventListener('change', () => { object.visible = input.checked; });
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = group.color;
  const text = document.createElement('span');
  text.textContent = group.name;
  const count = document.createElement('span');
  count.className = 'count';
  if (group.count !== undefined) {
    count.textContent = group.count.toLocaleString();
  } else if (group.radius !== undefined) {
    count.textContent = 'R=' + group.radius.toFixed(3) + ' m';
  } else {
    count.textContent = group.detail || '';
  }
  label.append(input, swatch, text, count);
  container.append(label);
}

function addBoundaries() {
  for (const boundary of manifest.boundaries) {
    outerRadius = Math.max(outerRadius, boundary.radius);
    const geometry = new THREE.SphereGeometry(boundary.radius, 64, 32);
    const mesh = new THREE.Mesh(
      geometry,
      material(boundary.color, 0.10, THREE.DoubleSide),
    );
    mesh.name = boundary.name;
    mesh.visible = boundary.visible;
    mesh.renderOrder = -1;
    scene.add(mesh);
    boundaryObjects.push(mesh);
    addControl('boundaryControls', mesh, boundary, boundary.visible);
  }
}

async function addPmtGroup(group, sharedGeometry) {
  const rows = await fetchRows(group);
  const mesh = new THREE.InstancedMesh(
    sharedGeometry,
    material(group.color, 1.0, THREE.DoubleSide),
    group.count,
  );
  mesh.name = group.name;
  mesh.userData = { group, rows };
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  for (let index = 0; index < group.count; index += 1) {
    const offset = index * group.stride;
    setInstanceTransform(
      mesh,
      index,
      rows.subarray(offset, offset + group.stride),
    );
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  pmtObjects.push(mesh);

  const pickerGeometry = new THREE.BufferGeometry();
  const centres = new Float32Array(group.count * 3);
  for (let index = 0; index < group.count; index += 1) {
    const source = index * group.stride;
    const target = index * 3;
    centres[target] = rows[source];
    centres[target + 1] = rows[source + 1];
    centres[target + 2] = rows[source + 2];
  }
  pickerGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(centres, 3),
  );
  const picker = new THREE.Points(
    pickerGeometry,
    new THREE.PointsMaterial({
      size: 0.001, transparent: true, opacity: 0, depthWrite: false,
    }),
  );
  picker.userData = { group, rows, displayMesh: mesh };
  picker.layers.set(1);
  scene.add(picker);
  pmtPickers.push(picker);
  addControl('pmtControls', mesh, group);
}

async function addSupportGroup(group, sharedGeometry) {
  const rows = await fetchRows(group);
  const mesh = new THREE.InstancedMesh(
    sharedGeometry,
    material(group.color, 0.95),
    group.count,
  );
  mesh.name = group.name;
  mesh.userData = { group, rows };
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  for (let index = 0; index < group.count; index += 1) {
    const offset = index * group.stride;
    setInstanceTransform(
      mesh, index, rows.subarray(offset, offset + group.stride), true,
    );
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  supportObjects.push(mesh);
  addControl('supportControls', mesh, group);
}

function addChimneySegment(segment) {
  const height = segment.z_max - segment.z_min;
  const geometry = makeAnnularCylinderGeometry(
    segment.inner_radius,
    segment.outer_radius,
    height,
  );
  const mesh = new THREE.Mesh(
    geometry,
    material(segment.color, 0.72, THREE.DoubleSide),
  );
  mesh.name = segment.name;
  mesh.position.z = (segment.z_min + segment.z_max) / 2;
  mesh.userData = { group: segment };
  scene.add(mesh);
  knownStructureObjects.push(mesh);
  const details = {
    ...segment,
    detail: segment.z_min.toFixed(3)
      + '-'
      + segment.z_max.toFixed(3)
      + ' m',
  };
  addControl('knownStructureControls', mesh, details);
}

async function addAcrylicBowlGroup(group) {
  const rows = await fetchRows(group);
  const geometry = new THREE.CylinderGeometry(
    group.tip_ratio, 1, 2, 20, 1, false,
  );
  const mesh = new THREE.InstancedMesh(
    geometry,
    material(group.color, 0.72, THREE.DoubleSide),
    group.count,
  );
  mesh.name = group.name;
  mesh.userData = { group, rows };
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  for (let index = 0; index < group.count; index += 1) {
    const offset = index * group.stride;
    setInstanceTransform(
      mesh,
      index,
      rows.subarray(offset, offset + group.stride),
      true,
    );
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  knownStructureObjects.push(mesh);
  addControl('knownStructureControls', mesh, group);
}

function setupScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080b0e);
  scene.fog = new THREE.FogExp2(0x080b0e, 0.006);

  camera = new THREE.PerspectiveCamera(42, 1, 0.02, 250);
  camera.up.set(0, 0, 1);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.localClippingEnabled = true;
  viewport.prepend(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.8;
  controls.panSpeed = 0.6;

  scene.add(new THREE.HemisphereLight(0xaedff2, 0x182028, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(20, -10, 30);
  scene.add(key);
  const axes = new THREE.AxesHelper(4);
  axes.material.depthTest = false;
  axes.renderOrder = 3;
  scene.add(axes);
}

function resetCamera() {
  camera.position.set(
    outerRadius * 1.65,
    -outerRadius * 1.65,
    outerRadius * 0.9,
  );
  controls.target.set(0, 0, 0);
  controls.update();
}

function focusTopStructures() {
  camera.position.set(5.0, -6.5, 21.5);
  controls.target.set(0, 0, 18.2);
  controls.update();
}

function setupUi() {
  document.getElementById('resetCamera').addEventListener('click', resetCamera);
  document.getElementById('focusTop').addEventListener(
    'click',
    focusTopStructures,
  );
  const rotateButton = document.getElementById('autoRotate');
  rotateButton.addEventListener('click', () => {
    autoRotate = !autoRotate;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 0.7;
    rotateButton.classList.toggle('active', autoRotate);
  });

  const boundaryOpacity = document.getElementById('boundaryOpacity');
  boundaryOpacity.addEventListener('input', () => {
    const value = Number(boundaryOpacity.value);
    document.getElementById('boundaryOpacityValue').value = value.toFixed(2);
    for (const object of boundaryObjects) object.material.opacity = value;
  });

  const pmtOpacity = document.getElementById('pmtOpacity');
  pmtOpacity.addEventListener('input', () => {
    const value = Number(pmtOpacity.value);
    document.getElementById('pmtOpacityValue').value = value.toFixed(2);
    for (const object of pmtObjects) {
      const transparent = value < 1;
      const modeChanged = object.material.transparent !== transparent;
      object.material.opacity = value;
      object.material.transparent = transparent;
      object.material.depthWrite = !transparent;
      if (modeChanged) object.material.needsUpdate = true;
    }
  });

  const structureOpacity = document.getElementById('structureOpacity');
  structureOpacity.addEventListener('input', () => {
    const value = Number(structureOpacity.value);
    document.getElementById('structureOpacityValue').value = value.toFixed(2);
    for (const object of knownStructureObjects) {
      const transparent = value < 1;
      const modeChanged = object.material.transparent !== transparent;
      object.material.opacity = value;
      object.material.transparent = transparent;
      object.material.depthWrite = !transparent;
      if (modeChanged) object.material.needsUpdate = true;
    }
  });

  const clipEnabled = document.getElementById('clipEnabled');
  const clipPosition = document.getElementById('clipPosition');
  clipPosition.min = (-outerRadius).toFixed(1);
  clipPosition.max = outerRadius.toFixed(1);
  function updateClipping(recompile) {
    const enabled = clipEnabled.checked;
    const value = Number(clipPosition.value);
    clipPosition.disabled = !enabled;
    document.getElementById('clipValue').value = `${value.toFixed(1)} m`;
    clippingPlane.constant = value;
    const objects = [
      ...boundaryObjects,
      ...pmtObjects,
      ...supportObjects,
      ...knownStructureObjects,
    ];
    for (const object of objects) {
      object.material.clippingPlanes = enabled ? [clippingPlane] : [];
      if (recompile) object.material.needsUpdate = true;
    }
  }
  clipEnabled.addEventListener('change', () => updateClipping(true));
  clipPosition.addEventListener('input', () => updateClipping(false));
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function inspectPointer(event) {
  if (!lastPointerEvent) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const visiblePickers = pmtPickers.filter(
    (item) => item.userData.displayMesh.visible,
  );
  const intersections = raycaster.intersectObjects(visiblePickers, false);
  if (!intersections.length) {
    tooltip.style.display = 'none';
    return;
  }
  const hit = intersections[0];
  const { group, rows } = hit.object.userData;
  const offset = hit.index * group.stride;
  const copyNumber = Math.round(rows[offset + 8]);
  tooltip.innerHTML = `<strong>${group.name}</strong><br>`
    + `CopyNo ${copyNumber}<br>`
    + `x ${rows[offset].toFixed(3)} m<br>`
    + `y ${rows[offset + 1].toFixed(3)} m<br>`
    + `z ${rows[offset + 2].toFixed(3)} m`;
  tooltip.style.display = 'block';
  const left = Math.min(
    event.clientX - rect.left + 14,
    rect.width - 190,
  );
  const top = Math.min(
    event.clientY - rect.top + 14,
    rect.height - 125,
  );
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}

function queuePick(event) {
  lastPointerEvent = event;
  if (pickPending) return;
  pickPending = true;
  window.setTimeout(() => {
    pickPending = false;
    inspectPointer(lastPointerEvent);
  }, 45);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function init() {
  if (typeof THREE === 'undefined') {
    throw new Error(
      'Three.js failed to load. Check network access to unpkg.',
    );
  }
  setupScene();
  const response = await fetch('/geometry.json');
  if (!response.ok) {
    throw new Error(
      'Geometry manifest failed: ' + response.status,
    );
  }
  manifest = await response.json();
  loadStatus.textContent = 'Building GPU instances...';
  addBoundaries();

  const pmtGeometry = manifest.pmt_shape === 'front_head'
    ? makeHemisphereGeometry() : makeDiscGeometry();
  const supportGeometry = new THREE.CylinderGeometry(
    1, 1, 2, 12, 1, false,
  );
  for (const segment of manifest.chimney_segments) {
    addChimneySegment(segment);
  }
  await Promise.all([
    ...manifest.pmt_groups.map((group) => addPmtGroup(group, pmtGeometry)),
    ...manifest.support_groups.map(
      (group) => addSupportGroup(group, supportGeometry),
    ),
    ...manifest.acrylic_bowl_groups.map(
      (group) => addAcrylicBowlGroup(group),
    ),
  ]);

  if (!manifest.support_groups.length) {
    document.getElementById('supportSection').style.display = 'none';
  }
  if (
    !manifest.chimney_segments.length
    && !manifest.acrylic_bowl_groups.length
  ) {
    document.getElementById('knownStructureSection').style.display = 'none';
  }
  summary.textContent = manifest.detector_type + ' | '
    + manifest.pmt_count.toLocaleString() + ' PMTs | '
    + manifest.support_count.toLocaleString() + ' supports | '
    + manifest.chimney_segment_count.toLocaleString()
    + ' chimney segments | '
    + manifest.acrylic_bowl_count.toLocaleString() + ' bowls';
  setupUi();
  resetCamera();
  resize();
  window.addEventListener('resize', resize);
  renderer.domElement.addEventListener('pointermove', queuePick);
  renderer.domElement.addEventListener('pointerleave', () => {
    lastPointerEvent = null;
    tooltip.style.display = 'none';
  });
  loading.classList.add('hidden');
  animate();

  window.setInterval(() => {
    const info = renderer.info.render;
    renderStats.textContent = info.calls
      + ' draw calls | '
      + info.triangles.toLocaleString()
      + ' triangles';
  }, 500);
}

init().catch((error) => {
  console.error(error);
  loadStatus.textContent = error.message;
  loadStatus.style.color = '#ef8354';
});
