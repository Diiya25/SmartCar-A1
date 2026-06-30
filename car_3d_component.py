"""
3D Rotating Car Component for SmartCar AI
-------------------------------------------
Drop this file into your repo as `car_3d_component.py` (same folder as app.py),
put the .glb file inside an `assets/` folder, and call render_3d_car() from app.py.
"""

import base64
import os
import streamlit as st
import streamlit.components.v1 as components


def render_3d_car(glb_path: str = "assets/chevrolet_corvette_c5_blue.glb", height: int = 480):
    """Renders a real 3D, auto-rotating, mouse-draggable GLB car model using Three.js."""

    if not os.path.exists(glb_path):
        st.error(f"3D model not found at: {glb_path}")
        return

    with open(glb_path, "rb") as f:
        glb_base64 = base64.b64encode(f.read()).decode("utf-8")

    html_code = f"""
    <div id="car-canvas-container" style="width:100%; height:{height}px; position:relative; overflow:hidden;
         border-radius:24px; background:linear-gradient(135deg, #eaf2ff 0%, #dceeff 100%);">
        <div id="loader" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
             color:#1d4ed8; font-family:Inter, sans-serif; font-size:14px; letter-spacing:1px;">
            LOADING 3D MODEL...
        </div>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from "three";
        import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";
        import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

        const container = document.getElementById('car-canvas-container');
        const width = container.clientWidth;
        const height = {height};

        const scene = new THREE.Scene();
        scene.background = null;

        const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
        camera.position.set(4.5, 1.8, 4.5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        container.appendChild(renderer.domElement);

        // Lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0xb8d4ff, 1.1);
        scene.add(hemiLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
        keyLight.position.set(5, 8, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.set(1024, 1024);
        scene.add(keyLight);

        const rimLight = new THREE.DirectionalLight(0x4fa8ff, 1.0);
        rimLight.position.set(-5, 3, -5);
        scene.add(rimLight);

        // Soft ground shadow catcher
        const groundGeo = new THREE.CircleGeometry(6, 64);
        const groundMat = new THREE.ShadowMaterial({{ opacity: 0.18 }});
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.01;
        ground.receiveShadow = true;
        scene.add(ground);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.enableZoom = true;
        controls.minDistance = 3;
        controls.maxDistance = 9;
        controls.maxPolarAngle = Math.PI / 2.1;
        controls.target.set(0, 0.4, 0);

        let autoRotate = true;
        let carModel = null;

        // Decode base64 GLB
        const b64 = "{glb_base64}";
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{ bytes[i] = binary.charCodeAt(i); }}

        const loader = new GLTFLoader();
        loader.parse(bytes.buffer, '', (gltf) => {{
            carModel = gltf.scene;

            carModel.traverse((node) => {{
                if (node.isMesh) {{
                    node.castShadow = true;
                    node.receiveShadow = false;
                }}
            }});

            // Auto-fit and center the model
            const box = new THREE.Box3().setFromObject(carModel);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2.6 / maxDim;
            carModel.scale.setScalar(scale);

            const scaledBox = new THREE.Box3().setFromObject(carModel);
            const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
            carModel.position.x -= scaledCenter.x;
            carModel.position.z -= scaledCenter.z;
            carModel.position.y -= scaledBox.min.y;

            scene.add(carModel);
            document.getElementById('loader').style.display = 'none';
        }}, (err) => {{
            document.getElementById('loader').innerText = 'Could not load model';
            console.error(err);
        }});

        // Pause auto-rotate while user drags
        renderer.domElement.addEventListener('pointerdown', () => {{ autoRotate = false; }});
        renderer.domElement.addEventListener('pointerup', () => {{
            setTimeout(() => {{ autoRotate = true; }}, 2500);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            if (carModel && autoRotate) {{
                carModel.rotation.y += 0.006;
            }}
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            camera.aspect = w / height;
            camera.updateProjectionMatrix();
            renderer.setSize(w, height);
        }});
    </script>
    """

    components.html(html_code, height=height + 10, scrolling=False)
