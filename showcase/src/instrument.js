import * as THREE from "three";

export function createInstrument(container, isPaused) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "low-power",
    });
  } catch {
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setClearColor(0x11120f, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  container.classList.add("scene-ready");
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x15180f, 0.035);
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(10, 12, 16);
  camera.lookAt(0, 0, -2.5);
  scene.add(new THREE.AmbientLight(0xd5dfc1, 1.6));
  const light = new THREE.DirectionalLight(0xfff4da, 4);
  light.position.set(-5, 12, 8);
  scene.add(light);
  const rim = new THREE.DirectionalLight(0xb0db64, 2);
  rim.position.set(4, 8, -8);
  scene.add(rim);
  const instrument = new THREE.Group();
  scene.add(instrument);
  instrument.rotation.y = -0.15;
  const keyGeometry = new THREE.BoxGeometry(0.32, 0.24, 2.35);
  const darkGeometry = new THREE.BoxGeometry(0.19, 0.32, 1.4);
  const whiteMaterial = new THREE.MeshStandardMaterial({
    color: 0xeae8da,
    roughness: 0.32,
    metalness: 0.12,
  });
  const blackMaterial = new THREE.MeshStandardMaterial({
    color: 0x151811,
    roughness: 0.35,
    metalness: 0.25,
  });
  const keys = [];
  for (let i = 0; i < 28; i++) {
    const key = new THREE.Mesh(keyGeometry, whiteMaterial.clone());
    key.position.set((i - 13.5) * 0.355, 0, 3.3);
    instrument.add(key);
    keys.push(key);
    if ([0, 1, 3, 4, 5].includes(i % 7) && i < 27) {
      const black = new THREE.Mesh(darkGeometry, blackMaterial);
      black.position.set((i - 13.5) * 0.355 + 0.1775, 0.21, 2.9);
      instrument.add(black);
    }
  }
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(10.25, 0.3, 2.7),
    new THREE.MeshStandardMaterial({ color: 0x272d1e, roughness: 0.65 }),
  );
  base.position.set(0, -0.26, 3.25);
  instrument.add(base);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x68784a,
    transparent: true,
    opacity: 0.2,
  });
  const linePoints = [];
  for (let i = 0; i <= 28; i++) {
    const x = (i - 14) * 0.355;
    linePoints.push(x, -0.02, -11, x, -0.02, 2.1);
  }
  for (let z = -11; z <= 2; z += 1.1)
    linePoints.push(-5, -0.02, z, 5, -0.02, z);
  const grid = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(linePoints, 3),
    ),
    lineMaterial,
  );
  instrument.add(grid);
  const notes = [];
  const colors = [0xd4f57b, 0xb2cb8c, 0xe2b998, 0x91a777];
  const random = (i) => {
    const n = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return n - Math.floor(n);
  };
  const noteGeometry = new THREE.BoxGeometry(0.26, 0.12, 1);
  for (let i = 0; i < 52; i++) {
    const pitch = Math.floor(random(i + 1) * 28);
    const material = new THREE.MeshStandardMaterial({
      color: colors[i % 4],
      emissive: colors[i % 4],
      emissiveIntensity: 0.15,
      roughness: 0.28,
      metalness: 0.12,
      transparent: true,
      opacity: 0.85,
    });
    const mesh = new THREE.Mesh(noteGeometry, material);
    const length = 0.35 + random(i + 70) * 1.4;
    mesh.scale.z = length;
    mesh.position.set((pitch - 13.5) * 0.355, 0.22, -random(i + 120) * 14);
    instrument.add(mesh);
    notes.push({ mesh, pitch, start: mesh.position.z, length });
  }
  const playhead = new THREE.Mesh(
    new THREE.BoxGeometry(10, 0.012, 0.025),
    new THREE.MeshBasicMaterial({
      color: 0xe4ffb0,
      transparent: true,
      opacity: 0.8,
    }),
  );
  playhead.position.set(0, 0.27, 2.02);
  instrument.add(playhead);
  const target = new THREE.Vector2();
  container.addEventListener("pointermove", (event) => {
    const rect = container.getBoundingClientRect();
    target.set(
      (event.clientX - rect.left) / rect.width - 0.5,
      (event.clientY - rect.top) / rect.height - 0.5,
    );
  });
  container.addEventListener("pointerleave", () => target.set(0, 0));
  const resize = () => {
    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.fov = width < 500 ? 44 : 34;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  };
  new ResizeObserver(resize).observe(container);
  resize();
  let visible = true;
  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
  }).observe(container);
  let last = 0,
    elapsed = 0;
  renderer.setAnimationLoop((now) => {
    const dt = Math.min((now - last) / 1000, 0.05);
    if (now - last < 32) return;
    last = now;
    if (!visible || document.hidden || isPaused()) return;
    elapsed += dt;
    instrument.rotation.y = THREE.MathUtils.lerp(
      instrument.rotation.y,
      -0.15 + target.x * 0.13,
      0.025,
    );
    instrument.rotation.x = THREE.MathUtils.lerp(
      instrument.rotation.x,
      target.y * 0.025,
      0.025,
    );
    keys.forEach((key) => {
      key.material.color.setHex(0xeae8da);
      key.position.y = 0;
    });
    notes.forEach(({ mesh, pitch, start, length }) => {
      const z = ((start + elapsed * 0.72 + 14) % 15) - 12;
      mesh.position.z = z;
      mesh.material.opacity = Math.min(1, (z + 12) / 3) * 0.85;
      if (z > 1.7 && z < 2.2 + length / 2) {
        keys[pitch].material.color.setHex(0xcce99c);
        keys[pitch].position.y = -0.035;
      }
      mesh.visible = z < 2.3;
    });
    renderer.render(scene, camera);
  });
}
