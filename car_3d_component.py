"""
3D Rotating Car Component for SmartCar AI (v3 - static file serving, no base64 embedding)
---------------------------------------------------------------------------------------------
SETUP (one-time):
1. In your repo root, create a folder named exactly: static
2. Put chevrolet_corvette_c5_blue.glb inside that `static` folder
   (so the path is: static/chevrolet_corvette_c5_blue.glb)
3. Create a file named .streamlit/config.toml in your repo with this content:

   [server]
   enableStaticServing = true

4. Put this car_3d_component.py file next to app.py (repo root)
5. In app.py, import and call:
   from car_3d_component import render_3d_car
   render_3d_car("chevrolet_corvette_c5_blue.glb", height=460)

   (Note: pass just the filename, not "assets/..." this time — it's served from /app/static/)
"""

import streamlit as st
import streamlit.components.v1 as components


def render_3d_car(glb_filename: str = "chevrolet_corvette_c5_blue.glb", height: int = 480):
    """Renders a real 3D, auto-rotating, mouse-draggable GLB car model using Three.js.

    Loads the model via Streamlit's static file serving (fetch), avoiding the
    base64-inline-HTML approach which can get truncated by hosting proxies on large files.
    """

    model_path = f"/app/static/{glb_filename}"

    html_code = f"""
    <div id="car-canvas-container" style="width:100%; height:{height}px; position:relative; overflow:hidden;
         border-radius:24px; background:linear-gradient(135deg, #eaf2ff 0%, #dceeff 100%);">
        <div id="loader" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
             color:#1d4ed8; font-family:Inter, sans-serif; font-size:13px; letter-spacing:1px; text-align:center; width:80%;">
            LOADING 3D MODEL...
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
    (function() {{
        function init() {{
            if (typeof THREE === 'undefined' || typeof THREE.GLTFLoader === 'undefined') {{
                document.getElementById('loader').innerText = 'Failed to load 3D library (check internet/CDN access)';
                return;
            }}

            var parentOrigin = '';
            try {{
                parentOrigin = window.parent.location.origin;
            }} catch (e) {{
                parentOrigin = window.location.origin;
            }}
            var modelUrl = parentOrigin + "{model_path}";

            var container = document.getElementById('car-canvas-container');
            var width = container.clientWidth;
            var height = {height};

            var scene = new THREE.Scene();

            var camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
            camera.position.set(4.5, 1.8, 4.5);

            var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(width, height);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            var hemiLight = new THREE.HemisphereLight(0xffffff, 0xb8d4ff, 1.1);
            scene.add(hemiLight);

            var keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
            keyLight.position.set(5, 8, 5);
            keyLight.castShadow = true;
            scene.add(keyLight);

            var rimLight = new THREE.DirectionalLight(0x4fa8ff, 1.0);
            rimLight.position.set(-5, 3, -5);
            scene.add(rimLight);

            var groundGeo = new THREE.CircleGeometry(6, 64);
            var groundMat = new THREE.ShadowMaterial({{ opacity: 0.18 }});
            var ground = new THREE.Mesh(groundGeo, groundMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.y = -0.01;
            ground.receiveShadow = true;
            scene.add(ground);

            var controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.06;
            controls.enableZoom = true;
            controls.minDistance = 3;
            controls.maxDistance = 9;
            controls.maxPolarAngle = Math.PI / 2.1;
            controls.target.set(0, 0.4, 0);

            var autoRotate = true;
            var carModel = null;

            var loader = new THREE.GLTFLoader();
            loader.load(
                modelUrl,
                function(gltf) {{
                    carModel = gltf.scene;

                    carModel.traverse(function(node) {{
                        if (node.isMesh) {{
                            node.castShadow = true;
                        }}
                    }});

                    var box = new THREE.Box3().setFromObject(carModel);
                    var size = box.getSize(new THREE.Vector3());
                    var maxDim = Math.max(size.x, size.y, size.z);
                    var scale = 2.6 / maxDim;
                    carModel.scale.setScalar(scale);

                    var scaledBox = new THREE.Box3().setFromObject(carModel);
                    var scaledCenter = scaledBox.getCenter(new THREE.Vector3());
                    carModel.position.x -= scaledCenter.x;
                    carModel.position.z -= scaledCenter.z;
                    carModel.position.y -= scaledBox.min.y;

                    scene.add(carModel);
                    document.getElementById('loader').style.display = 'none';
                }},
                function(xhr) {{
                    if (xhr.lengthComputable) {{
                        var pct = Math.round((xhr.loaded / xhr.total) * 100);
                        document.getElementById('loader').innerText = 'LOADING 3D MODEL... ' + pct + '%';
                    }}
                }},
                function(err) {{
                    document.getElementById('loader').innerText = 'Could not fetch model file. Check URL: ' + modelUrl;
                    console.error(err);
                }}
            );

            renderer.domElement.addEventListener('pointerdown', function() {{ autoRotate = false; }});
            renderer.domElement.addEventListener('pointerup', function() {{
                setTimeout(function() {{ autoRotate = true; }}, 2500);
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

            window.addEventListener('resize', function() {{
                var w = container.clientWidth;
                camera.aspect = w / height;
                camera.updateProjectionMatrix();
                renderer.setSize(w, height);
            }});
        }}

        var checkInterval = setInterval(function() {{
            if (typeof THREE !== 'undefined' && typeof THREE.GLTFLoader !== 'undefined' && typeof THREE.OrbitControls !== 'undefined') {{
                clearInterval(checkInterval);
                init();
            }}
        }}, 100);

        setTimeout(function() {{
            clearInterval(checkInterval);
            if (typeof THREE === 'undefined' || typeof THREE.GLTFLoader === 'undefined') {{
                document.getElementById('loader').innerText = 'Failed to load 3D library (timeout)';
            }}
        }}, 10000);
    }})();
    </script>
    """

    components.html(html_code, height=height + 10, scrolling=False)
