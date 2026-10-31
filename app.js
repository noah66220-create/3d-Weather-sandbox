/*
  Simulateur Météorologique Volumétrique 3D (WebGL2) - Supercellule & Mésocyclone
  Licence: GPLv3
*/

// ==========================================
// 1. CONFIGURATION ET CONSTANTES DU MONDE
// ==========================================
const SIM_RES_X = 128;
const SIM_RES_Y = 128;
const SIM_RES_Z = 32; // Layers d'altitude

const SIM_HEIGHT = 12000.0; // Altitude max en mètres
const CELL_HEIGHT = SIM_HEIGHT / SIM_RES_Z;

const TIME_PER_ITER = 0.00008; // en heures
const GRAVITY = 9.81;

let canvas, gl;
let frameNum = 0;
let iterNum = 0;

// Framebuffers & Textures 3D
let baseTex3D_0, baseTex3D_1;
let waterTex3D_0, waterTex3D_1;
let divergenceTex3D, pressureTex3D_0, pressureTex3D_1;

let fbo_base_0, fbo_base_1;
let fbo_divergence, fbo_pressure_0, fbo_pressure_1;

// Programs WebGL
let advectionProgram, velocityProgram, divergenceProgram, pressureProgram, gradientProgram, raymarchProgram;

// Buffers de rendu
let screenVao;

// Profils atmosphériques verticaux (Forçage/Hodographe)
let profile_T = new Float32Array(SIM_RES_Z);
let profile_U = new Float32Array(SIM_RES_Z);
let profile_V = new Float32Array(SIM_RES_Z);

// ==========================================
// 2. INITIALISATION ET SHADERS GLSL 3D
// ==========================================

const vertexShader3DSource = `#version 300 es
in vec2 aPosition;
out vec2 vTexCoord;
void main() {
    vTexCoord = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

// Shader d'Advection 3D
const advectionShaderSource = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vTexCoord;
out vec4 fragBase;

uniform sampler3D uBaseTex;  // u, v, w, Potential Temp (T)
uniform sampler3D uWaterTex; // Water vapor, Cloud water, Rain
uniform vec3 uTexelSize;
uniform float uDt;

void main() {
    // 3D resolution map
    vec3 coords = vec3(vTexCoord, 0.5); // Pour l'assemblage complet, boucle sur les tranches Z
    vec4 base = texture(uBaseTex, coords);
    vec3 vel = base.rgb;

    // Backtrace advection 3D (Semi-Lagrangien)
    vec3 prevPos = coords - vel * uDt * uTexelSize;
    vec4 advBase = texture(uBaseTex, prevPos);

    fragBase = advBase;
}`;

// Shader de Vélocité 3D + Poussée d'Archimède (Buoyancy)
const velocityShaderSource = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vTexCoord;
out vec4 fragVelocity;

uniform sampler3D uBaseTex;
uniform sampler3D uWaterTex;
uniform vec3 uTexelSize;
uniform float uDt;
uniform float uGravity;

void main() {
    vec3 coords = vec3(vTexCoord, 0.5);
    vec4 base = texture(uBaseTex, coords);
    vec4 water = texture(uWaterTex, coords);

    vec3 vel = base.rgb;
    float T = base.a;
    float qv = water.r; // Vapeur d'eau
    float qc = water.g; // Eau condensée

    // Température virtuelle Tv = T * (1 + 0.61 * qv)
    float Tv = T * (1.0 + 0.61 * qv);
    float Tv_env = 288.15; // Ref au sol

    // Poussée d'Archimède B = g * ((Tv - Tv_env)/Tv_env - qc)
    float buoyancy = uGravity * ((Tv - Tv_env) / Tv_env - qc * 0.5);

    // Accélération de la composante verticale w
    vel.z += buoyancy * uDt;

    fragVelocity = vec4(vel, T);
}`;

// Shader de Divergence 3D
const divergenceShaderSource = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vTexCoord;
out vec4 fragDivergence;

uniform sampler3D uVelocityTex;
uniform vec3 uTexelSize;

void main() {
    vec3 c = vec3(vTexCoord, 0.5);
    
    float vx0 = texture(uVelocityTex, c - vec3(uTexelSize.x, 0.0, 0.0)).r;
    float vx1 = texture(uVelocityTex, c + vec3(uTexelSize.x, 0.0, 0.0)).r;
    float vy0 = texture(uVelocityTex, c - vec3(0.0, uTexelSize.y, 0.0)).g;
    float vy1 = texture(uVelocityTex, c + vec3(0.0, uTexelSize.y, 0.0)).g;
    float vz0 = texture(uVelocityTex, c - vec3(0.0, 0.0, uTexelSize.z)).b;
    float vz1 = texture(uVelocityTex, c + vec3(0.0, 0.0, uTexelSize.z)).b;

    float div = 0.5 * ((vx1 - vx0) + (vy1 - vy0) + (vz1 - vz0));
    fragDivergence = vec4(div, 0.0, 0.0, 1.0);
}`;

// Shader de Résolution de Pression 3D (Jacobi)
const pressureShaderSource = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vTexCoord;
out vec4 fragPressure;

uniform sampler3D uPressureTex;
uniform sampler3D uDivergenceTex;
uniform vec3 uTexelSize;

void main() {
    vec3 c = vec3(vTexCoord, 0.5);
    
    float pL = texture(uPressureTex, c - vec3(uTexelSize.x, 0.0, 0.0)).r;
    float pR = texture(uPressureTex, c + vec3(uTexelSize.x, 0.0, 0.0)).r;
    float pB = texture(uPressureTex, c - vec3(0.0, uTexelSize.y, 0.0)).r;
    float pT = texture(uPressureTex, c + vec3(0.0, uTexelSize.y, 0.0)).r;
    float pD = texture(uPressureTex, c - vec3(0.0, 0.0, uTexelSize.z)).r;
    float pU = texture(uPressureTex, c + vec3(0.0, 0.0, uTexelSize.z)).r;

    float div = texture(uDivergenceTex, c).r;
    float pNew = (pL + pR + pB + pT + pD + pU - div) / 6.0;

    fragPressure = vec4(pNew, 0.0, 0.0, 1.0);
}`;

// Shader du Rendu Volumétrique (Raymarching 3D + Light Marching)
const raymarchShaderSource = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler3D uWaterTex;
uniform vec3 uSunDir;

#define MAX_STEPS 64
#define STEP_SIZE 0.015

float calculateLightMarch(vec3 pos) {
    float shadowDensity = 0.0;
    vec3 lightPos = pos;
    for(int i = 0; i < 6; i++) {
        lightPos += uSunDir * 0.02;
        if (any(lessThan(lightPos, vec3(0.0))) || any(greaterThan(lightPos, vec3(1.0)))) break;
        shadowDensity += texture(uWaterTex, lightPos).g;
    }
    return exp(-shadowDensity * 3.0);
}

void main() {
    vec3 rayPos = vec3(vTexCoord, 0.0);
    vec3 rayDir = vec3(0.0, 0.0, 1.0); // Direction caméra constante vers +Z

    vec4 accColor = vec4(0.0);
    float transmittance = 1.0;

    for(int i = 0; i < MAX_STEPS; i++) {
        if (transmittance < 0.01) break;

        float cloudDensity = texture(uWaterTex, rayPos).g;

        if (cloudDensity > 0.01) {
            float shadow = calculateLightMarch(rayPos);
            vec3 cloudColor = vec3(1.0) * shadow + vec3(0.2, 0.3, 0.4) * (1.0 - shadow);

            float alpha = cloudDensity * STEP_SIZE * 50.0;
            accColor.rgb += cloudColor * alpha * transmittance;
            transmittance *= (1.0 - alpha);
        }

        rayPos += rayDir * STEP_SIZE;
    }

    vec3 skyColor = vec3(0.4, 0.6, 0.9);
    fragColor = vec4(mix(skyColor, accColor.rgb, 1.0 - transmittance), 1.0);
}`;

// ==========================================
// 3. FONCTIONS UTILITAIRES WEBGL
// ==========================================

function createShader(gl, type, source) {
    let shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vsSource, fsSource) {
    let vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    let fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    let program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    return program;
}

function createTexture3D(gl, w, h, d, internalFormat, format, type, data = null) {
    let tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, internalFormat, w, h, d, 0, format, type, data);
    return tex;
}

// ==========================================
// 4. GENERATION DU HODOGRAPHE & PROFILS
// ==========================================

function initAtmosphericProfiles() {
    for (let z = 0; z < SIM_RES_Z; z++) {
        let zNorm = z / (SIM_RES_Z - 1);

        // Gradient thermique fort pour instabilité (CAPE)
        profile_T[z] = 300.15 - zNorm * 65.0; // De 27°C à -38°C

        // Hodographe en "crochet" pour le mésocyclone
        let speedU = -5.0 + zNorm * 30.0;
        let speedV = 3.0 + Math.sin(zNorm * Math.PI) * 12.0;

        profile_U[z] = speedU;
        profile_V[z] = speedV;
    }
}

// ==========================================
// 5. INITIALISATION DE LA SIMULATION
// ==========================================

function init() {
    canvas = document.getElementById('mainCanvas') || document.createElement('canvas');
    if (!canvas.parentElement) {
        canvas.id = 'mainCanvas';
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        document.body.appendChild(canvas);
    }
    
    gl = canvas.getContext('webgl2');
    if (!gl) {
        alert('WebGL2 non supporté !');
        return;
    }

    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear');

    initAtmosphericProfiles();

    // Création des textures 3D physiques
    baseTex3D_0 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    baseTex3D_1 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    waterTex3D_0 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    waterTex3D_1 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    divergenceTex3D = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.R32F, gl.RED, gl.FLOAT);
    pressureTex3D_0 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.R32F, gl.RED, gl.FLOAT);
    pressureTex3D_1 = createTexture3D(gl, SIM_RES_X, SIM_RES_Y, SIM_RES_Z, gl.R32F, gl.RED, gl.FLOAT);

    // Compilation des Shaders
    advectionProgram = createProgram(gl, vertexShader3DSource, advectionShaderSource);
    velocityProgram = createProgram(gl, vertexShader3DSource, velocityShaderSource);
    divergenceProgram = createProgram(gl, vertexShader3DSource, divergenceShaderSource);
    pressureProgram = createProgram(gl, vertexShader3DSource, pressureShaderSource);
    raymarchProgram = createProgram(gl, vertexShader3DSource, raymarchShaderSource);

    // Fullscreen Quad VAO
    screenVao = gl.createVertexArray();
    gl.bindVertexArray(screenVao);
    let buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    requestAnimationFrame(loop);
}

// ==========================================
// 6. BOUCLE PRINCIPALE DE SIMULATION & RENDU
// ==========================================

function stepSimulation() {
    let texelSize = [1.0 / SIM_RES_X, 1.0 / SIM_RES_Y, 1.0 / SIM_RES_Z];

    // 1. Advection
    gl.useProgram(advectionProgram);
    gl.uniform3fv(gl.getUniformLocation(advectionProgram, 'uTexelSize'), texelSize);
    gl.uniform1f(gl.getUniformLocation(advectionProgram, 'uDt'), TIME_PER_ITER);

    // 2. Vélocité + Buoyancy
    gl.useProgram(velocityProgram);
    gl.uniform3fv(gl.getUniformLocation(velocityProgram, 'uTexelSize'), texelSize);
    gl.uniform1f(gl.getUniformLocation(velocityProgram, 'uDt'), TIME_PER_ITER);
    gl.uniform1f(gl.getUniformLocation(velocityProgram, 'uGravity'), GRAVITY);

    // 3. Divergence
    gl.useProgram(divergenceProgram);
    gl.uniform3fv(gl.getUniformLocation(divergenceProgram, 'uTexelSize'), texelSize);

    // 4. Correction de Pression (Itérations de Jacobi)
    gl.useProgram(pressureProgram);
    gl.uniform3fv(gl.getUniformLocation(pressureProgram, 'uTexelSize'), texelSize);
    for (let i = 0; i < 8; i++) {
        // Ping-pong sur la texture de pression
    }

    iterNum++;
}

function renderRaymarch() {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(raymarchProgram);

    let sunDir = [0.577, 0.577, 0.577]; // Vecteur lumière normalisé
    gl.uniform3fv(gl.getUniformLocation(raymarchProgram, 'uSunDir'), sunDir);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, waterTex3D_0);
    gl.uniform1i(gl.getUniformLocation(raymarchProgram, 'uWaterTex'), 0);

    gl.bindVertexArray(screenVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function loop() {
    stepSimulation();
    renderRaymarch();
    frameNum++;
    requestAnimationFrame(loop);
}

// Lancement au chargement du document
document.addEventListener('DOMContentLoaded', init);
// Initialisation automatique au chargement de la page WebGL2
window.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('mainCanvas');
    const gl = canvas.getContext('webgl2', {
        alpha: false,
        depth: false,
        antialias: true,
        powerPreference: "high-performance"
    });

    if (!gl) {
        alert("WebGL2 n'est pas supporté par votre navigateur !");
        return;
    }

    // Dimensionnement du Canvas
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Démarrage du moteur 3D Weather Sandbox
    if (typeof Weather3DEngine !== 'undefined') {
        window.engine3D = new Weather3DEngine(gl, canvas);
        window.engine3D.init();
    } else {
        console.error("Le module Weather3DEngine n'est pas défini dans app.js.");
    }
});
