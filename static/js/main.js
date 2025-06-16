//import * as THREE from "three";
//import '/static/css/style.css';
/*import gsap from "/static/3js cnv/gsap"
import { OrbitControls } from "/static/3js cnv/three/examples/jsm/controls/OrbitControls";
*/
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x211c21);
//eye
const eyetexture = new THREE.TextureLoader().load('static/retina.png');
const normaltexture = new THREE.TextureLoader().load('static/retina1.png');
const eye = new THREE.Mesh(
  new THREE.SphereGeometry(2.6, 64, 64),
  new THREE.MeshStandardMaterial({
    map: eyetexture,
    normalMap: normaltexture
  })
);
scene.add(eye);


//sizes
const sizes ={
  width : window.innerWidth,
  height : window.innerHeight,
}
//light
const light = new THREE.PointLight(0xffffff, 2.5, 3.5)
scene.add(light)

//camera
const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 100)
camera.position.z = 20
scene.add(camera)

//helper
const lighthelper = new THREE.PointLightHelper(light)
//const gridhelper = new THREE.GridHelper(200, 50);
scene.add(lighthelper)

//renderer
const canvas = document.querySelector('.webgl')
const renderer = new THREE.WebGLRenderer({ canvas })
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(2)
renderer.render(scene, camera)


//resize
window.addEventListener('resize', ()=>{
  sizes.width = window.innerWidth
  sizes.height = window.innerHeight
  //update camera
  
  camera.aspect = sizes.width / sizes.height
  camera.updateProjectionMatrix()
  renderer.setSize(sizes.width, sizes.height)
})


//Timeline magic
const t1 = gsap.timeline({defaults: {duration: 1} })
t1.fromTo(eye.scale, {z:0, x:0, y:0}, {z:1, x:1, y:1})
t1.fromTo("nav", { y: "-100%"}, { y: "0%"})
t1.fromTo(".title", {opacity :0}, {opacity:1})
t1.fromTo(".form", {opacity :0}, {opacity:1})


//controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.enablePan = false
controls.enableZoom = false
controls.autoRotate = true
controls.autoRotateSpeed = 6
const loop = () =>{
  controls.update();
  renderer.render(scene,camera)
  window.requestAnimationFrame(loop)
}
loop()

