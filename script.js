;import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const canvas = document.getElementById("c");
const hud = document.getElementById("hud");
const imagePopup = document.getElementById("imagePopup");
const popupImg = document.getElementById("popupImg");
const closePopupBtn = document.getElementById("closePopup");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.05,
    5000
);
camera.position.set(0, 0, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.35;
controls.zoomSpeed = 0.7;
controls.enablePan = true;
controls.autoRotate = false;

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const keyLight = new THREE.PointLight(0xffffff, 1.2);
keyLight.position.set(15, 12, 10);
scene.add(keyLight);

const loader = new PLYLoader();
const cameraGroup = new THREE.Group();
scene.add(cameraGroup);

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const clock = new THREE.Clock();

const viewerState = {
    cloudCenter: new THREE.Vector3(),
    cloudRadius: 5,
    cameraData: [],
    cameraSpheres: [],
    pointCloudBounds: null,
    cameraBounds: null,
    pointsObject: null,
    showCameraMarkers: true,
    savedView: null,
    groundY: 0,
    imageZoomEnabled: true
};

const CameraModes = Object.freeze({
    ORBIT: "orbit",
    STABILIZED: "stabilized"
});
let currentMode = CameraModes.ORBIT;
const CAMERA_ALIGNMENT_OFFSET = new THREE.Vector3(0, 0, 0);

const stabilizedKeys = {
    forward: 0,
    right: 0,
    pressed: {
        w: false,
        s: false,
        a: false,
        d: false,
        arrowup: false,
        arrowdown: false,
        arrowleft: false,
        arrowright: false
    }
};
const upAxis = new THREE.Vector3(0, 1, 0);
const tempVec = new THREE.Vector3();

function computeBoundsFromVectors(vectors) {
    if (!vectors.length) {
        return null;
    }
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    vectors.forEach((vec) => {
        min.min(vec);
        max.max(vec);
    });
    return { min, max };
}

function getBoundsCenter(bounds) {
    if (!bounds) {
        return null;
    }
    return bounds.min.clone().add(bounds.max).multiplyScalar(0.5);
}

function updateSceneBounds() {
    const sources = [];
    if (viewerState.pointCloudBounds) {
        sources.push(viewerState.pointCloudBounds);
    }
    if (viewerState.cameraBounds) {
        sources.push(viewerState.cameraBounds);
    }
    if (!sources.length) {
        return;
    }

    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    sources.forEach((source) => {
        min.min(source.min);
        max.max(source.max);
    });

    viewerState.cloudCenter.copy(min.clone().add(max).multiplyScalar(0.5));
    const diag = max.clone().sub(min).length();
    viewerState.cloudRadius = Math.max(diag * 0.5, 1.0);
}

function updateCameraAlignment() {
    if (!viewerState.cameraData.length) {
        viewerState.cameraBounds = null;
        return;
    }

    const baseBounds = computeBoundsFromVectors(
        viewerState.cameraData.map((cam) => cam.worldPosition)
    );
    if (!viewerState.pointCloudBounds || !baseBounds) {
        viewerState.cameraData.forEach((cam) => {
            cam.position.copy(cam.worldPosition);
        });
        viewerState.cameraBounds = computeBoundsFromVectors(
            viewerState.cameraData.map((cam) => cam.position)
        );
        return;
    }

    const cloudCenter = getBoundsCenter(viewerState.pointCloudBounds);
    const cameraCenter = getBoundsCenter(baseBounds);
    const offset = cloudCenter.clone().sub(cameraCenter).add(CAMERA_ALIGNMENT_OFFSET);

    viewerState.cameraData.forEach((cam) => {
        cam.position.copy(cam.worldPosition).add(offset);
        cam.position.y = viewerState.groundY;
    });
    viewerState.cameraBounds = computeBoundsFromVectors(
        viewerState.cameraData.map((cam) => cam.position)
    );
}

function updateHud() {
    if (!hud) return;
    const camCount = viewerState.cameraData.length;
    const markerStatus = viewerState.showCameraMarkers ? "visible" : "hidden";
    const lines = [
        `<strong>Mode:</strong> ${
            currentMode === CameraModes.ORBIT ? "Orbit (press 2 to stabilize)" : "Stabilized (press 1 to orbit)"
        }`,
        "1 — Orbit mode",
        "2 — Stabilized view",
        "F — Frame the point cloud",
        `H — Toggle camera markers (${markerStatus})`,
        "C — Close image / restore view",
        `Camera markers loaded: ${camCount}`
    ];
    hud.innerHTML = lines.join("<br/>");
}

function focusOnPointCloud(multiplier = 2.6) {
    const radius = Math.max(viewerState.cloudRadius, 1.5);
    const center = viewerState.cloudCenter.clone();
    const offset = new THREE.Vector3(0, 0, radius * multiplier);

    camera.position.copy(center.clone().add(offset));
    camera.near = Math.max(radius * 0.01, 0.05);
    camera.far = radius * 40;
    camera.updateProjectionMatrix();

    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

function clearGroup(group) {
    for (let i = group.children.length - 1; i >= 0; i -= 1) {
        const child = group.children[i];
        group.remove(child);
        if (child.geometry) {
            child.geometry.dispose();
        }
        if (child.material) {
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            mats.forEach((mat) => {
                if (!mat) return;
                if (mat.map) {
                    mat.map.dispose();
                }
                if (mat.dispose) {
                    mat.dispose();
                }
            });
        }
    }
}

function rebuildCameraGroup() {
    viewerState.cameraSpheres = [];
    clearGroup(cameraGroup);
    if (!viewerState.showCameraMarkers || !viewerState.cameraData.length) {
        updateHud();
        return;
    }

    const markerRadius = Math.max(viewerState.cloudRadius * 0.015, 0.025);
    const maxIndex = Math.max(viewerState.cameraData.length - 1, 1);

    viewerState.cameraData.forEach((cam, idx) => {
        const t = idx / maxIndex;
        const color = new THREE.Color().setHSL(0.03 + 0.02 * (1 - t), 0.7, 0.55 + 0.25 * t);
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(markerRadius, 16, 16),
            new THREE.MeshBasicMaterial({ color })
        );
        sphere.position.copy(cam.position);
        sphere.userData = cam;
        viewerState.cameraSpheres.push(sphere);
        cameraGroup.add(sphere);

        const label = createLabelSprite(cam.label || cam.imagePath || `cam_${idx}`);
        const labelOffset = markerRadius * 2.0;
        label.position.copy(cam.position).add(new THREE.Vector3(0, labelOffset, 0));
        cameraGroup.add(label);
    });
    updateHud();
}

function toggleCameraMarkers(forceValue) {
    const nextValue =
        typeof forceValue === "boolean" ? forceValue : !viewerState.showCameraMarkers;
    viewerState.showCameraMarkers = nextValue;
    if (!nextValue && imagePopup) {
        hidePopupAndRestore();
    }
    rebuildCameraGroup();
    updateHud();
}

function createLabelSprite(text) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "36px 'Inter', 'Segoe UI', sans-serif";
    ctx.fillStyle = "#ffe3e3";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    const scale = Math.max(viewerState.cloudRadius * 0.07, 0.3);
    sprite.scale.set(scale, (scale * canvas.height) / canvas.width, 1);
    return sprite;
}

function storeCurrentView() {
    viewerState.savedView = {
        position: camera.position.clone(),
        quaternion: camera.quaternion.clone(),
        target: controls.target.clone(),
        mode: currentMode
    };
}

function restoreSavedView() {
    if (!viewerState.savedView) {
        return;
    }
    camera.position.copy(viewerState.savedView.position);
    camera.quaternion.copy(viewerState.savedView.quaternion);
    controls.target.copy(viewerState.savedView.target);
    controls.update();
    setCameraMode(viewerState.savedView.mode);
    viewerState.savedView = null;
}

function setCameraMode(mode) {
    if (mode === currentMode) {
        return;
    }
    currentMode = mode;
    const stabilized = currentMode === CameraModes.STABILIZED;
    controls.enabled = !stabilized;
    if (stabilized) {
        focusOnPointCloud();
        camera.lookAt(viewerState.cloudCenter);
    }
    updateHud();
}

function recomputeStabilizedAxes() {
    const pressed = stabilizedKeys.pressed;
    stabilizedKeys.forward =
        (pressed.w || pressed.arrowup ? 1 : 0) + (pressed.s || pressed.arrowdown ? -1 : 0);
    stabilizedKeys.right =
        (pressed.d || pressed.arrowright ? 1 : 0) +
        (pressed.a || pressed.arrowleft ? -1 : 0);
}

function handleStabilizedMovementKey(key, isDown) {
    if (!(key in stabilizedKeys.pressed)) {
        return false;
    }
    stabilizedKeys.pressed[key] = isDown;
    recomputeStabilizedAxes();
    return true;
}

function handleKeydown(event) {
    if (event.repeat) {
        return;
    }
    const key = event.key.toLowerCase();
    if (handleStabilizedMovementKey(key, true)) {
        if (currentMode === CameraModes.STABILIZED) {
            event.preventDefault();
        }
        return;
    }
    if (key === "1") {
        setCameraMode(CameraModes.ORBIT);
    } else if (key === "2") {
        setCameraMode(CameraModes.STABILIZED);
    } else if (key === "f") {
        focusOnPointCloud();
    } else if (key === "h") {
        toggleCameraMarkers();
    } else if (key === "c") {
        if (imagePopup && imagePopup.classList.contains("visible")) {
            hidePopupAndRestore();
        }
    }
}
window.addEventListener("keydown", handleKeydown);

window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (handleStabilizedMovementKey(key, false) && currentMode === CameraModes.STABILIZED) {
        event.preventDefault();
    }
});

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onWindowResize);

function updateStabilizedMotion(delta) {
    if (currentMode !== CameraModes.STABILIZED) {
        return;
    }
    const move = new THREE.Vector3();
    const speed = Math.max(viewerState.cloudRadius * 0.5, 0.5);
    const forwardDir = camera.getWorldDirection(new THREE.Vector3());
    forwardDir.y = 0;
    if (forwardDir.lengthSq() < 1e-5) {
        forwardDir.set(0, 0, -1);
    }
    forwardDir.normalize();
    const rightDir = new THREE.Vector3().crossVectors(upAxis, forwardDir).normalize();

    move.addScaledVector(forwardDir, stabilizedKeys.forward);
    move.addScaledVector(rightDir, stabilizedKeys.right);
    if (move.lengthSq() > 0) {
        move.normalize();
        camera.position.addScaledVector(move, speed * delta);
    }
    camera.lookAt(viewerState.cloudCenter);
    controls.target.copy(viewerState.cloudCenter);
}

function convertCameraEntry(entry, index) {
    if (
        !entry ||
        !Array.isArray(entry.rotation) ||
        entry.rotation.length !== 3 ||
        !Array.isArray(entry.translation)
    ) {
        return null;
    }

    const rot = entry.rotation;
    const trans = entry.translation;
    const extrinsic = new THREE.Matrix4().set(
        rot[0][0], rot[0][1], rot[0][2], trans[0],
        rot[1][0], rot[1][1], rot[1][2], trans[1],
        rot[2][0], rot[2][1], rot[2][2], trans[2],
        0, 0, 0, 1
    );

    const camToWorld = extrinsic.clone().invert();
    const positionWorld = new THREE.Vector3().setFromMatrixPosition(camToWorld);
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(camToWorld);
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();

    let imageName = entry.image || `camera_${index}`;
    const numericId = parseInt(imageName.split("_").pop() ?? "", 10);
    if (Number.isFinite(numericId)) {
        imageName = `IMG_${numericId + 1}.jpg`;
    }

    return {
        label: entry.image || `camera_${index}`,
        worldPosition: positionWorld,
        position: positionWorld.clone(),
        quaternion,
        forward,
        imagePath: `converted_jpg/${imageName}`
    };
}

async function loadCameraPoses() {
    try {
        const response = await fetch("cameras.json");
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const converted = Array.isArray(data.cameras)
            ? data.cameras
                  .map(convertCameraEntry)
                  .filter((cam) => cam !== null)
            : [];
        viewerState.cameraData = converted;
        updateCameraAlignment();
        updateSceneBounds();
        rebuildCameraGroup();
        if (viewerState.pointCloudBounds) {
            focusOnPointCloud();
        }
        console.log(`Loaded ${converted.length} camera poses`);
    } catch (error) {
        console.error("Unable to load camera positions", error);
    }
}

function loadPointCloud() {
    return new Promise((resolve, reject) => {
        loader.load(
            "cleaned_final_cloud.ply",
            (geometry) => {
                geometry.computeBoundingBox();
                const bbox = geometry.boundingBox;
                if (bbox) {
                    viewerState.pointCloudBounds = {
                        min: bbox.min.clone(),
                        max: bbox.max.clone()
                    };
                    viewerState.groundY = bbox.min.y;
                } else {
                    viewerState.pointCloudBounds = null;
                    viewerState.groundY = 0;
                }
                geometry.computeBoundingSphere();
                updateCameraAlignment();
                updateSceneBounds();

                const size = Math.max(viewerState.cloudRadius * 0.003, 0.01);
                const cloudMat = new THREE.PointsMaterial({
                    size,
                    vertexColors: false,
                    color: 0xffffff,
                    sizeAttenuation: true
                });

                if (viewerState.pointsObject) {
                    scene.remove(viewerState.pointsObject);
                    viewerState.pointsObject.geometry.dispose();
                    if (viewerState.pointsObject.material) {
                        viewerState.pointsObject.material.dispose();
                    }
                }

                const points = new THREE.Points(geometry, cloudMat);
                viewerState.pointsObject = points;
                scene.add(points);

                focusOnPointCloud();
                rebuildCameraGroup();
                console.log("PLY Loaded ✓");
                resolve();
            },
            undefined,
            (err) => {
                console.error("PLY load failed", err);
                reject(err);
            }
        );
    });
}

function moveViewerToCamera(cam) {
    if (!cam) {
        return;
    }
    if (!viewerState.savedView) {
        storeCurrentView();
    }
    setCameraMode(CameraModes.STABILIZED);
    camera.position.copy(cam.position);
    camera.quaternion.copy(cam.quaternion);
    const lookTarget = cam.position.clone().add(cam.forward);
    controls.target.copy(lookTarget);
    controls.update();
    if (imagePopup && popupImg) {
        const showPopup = () => {
            popupImg.style.objectFit = "cover";
            popupImg.style.width = "100%";
            popupImg.style.height = "100%";
            requestAnimationFrame(() => imagePopup.classList.add("visible"));
        };
        popupImg.src = cam.imagePath;
        if (popupImg.complete) {
            showPopup();
        } else {
            popupImg.addEventListener("load", showPopup, { once: true });
        }
    }
}

function onSceneClick(event) {
    const target = event.target;
    if (target instanceof Element && target.closest("#imagePopup")) {
        return;
    }
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(viewerState.cameraSpheres, false);
    if (!hits.length) {
        return;
    }
    const cam = hits[0].object.userData;
    moveViewerToCamera(cam);
}
window.addEventListener("click", onSceneClick);

function hidePopupAndRestore() {
    if (imagePopup) {
        imagePopup.classList.remove("visible");
    }
    restoreSavedView();
}

if (closePopupBtn && imagePopup) {
    closePopupBtn.addEventListener("click", hidePopupAndRestore);
}

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    updateStabilizedMotion(delta);
    controls.update();
    renderer.render(scene, camera);
}

async function init() {
    updateHud();
    animate();
    await Promise.allSettled([loadPointCloud(), loadCameraPoses()]);
    updateHud();
}

init();