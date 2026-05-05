// wind_gl.js — GPU wind particle layer for MapLibre GL JS
//
// Technique: based on mapbox/webgl-wind (Vladimir Agafonkin).
// Wind u/v encoded in a PNG texture (R=u, G=v, normalized to 0-255).
// Particles are advected on the GPU via ping-pong framebuffers.
// Trail effect via blending the previous frame at reduced opacity.
//
// This is a MapLibre CustomLayerInterface so it plugs into the existing map
// and gets the view-projection matrix each frame.

// ── Shader sources ──────────────────────────────────────────────────────────

const QUAD_VS = `
attribute vec2 a_pos;
varying vec2 v_tex;
void main() {
  v_tex = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0, 1);
}`;

// Draw the particle trail framebuffer onto the map (with fade).
const SCREEN_FS = `
precision mediump float;
uniform sampler2D u_screen;
uniform float u_opacity;
varying vec2 v_tex;
void main() {
  vec4 c = texture2D(u_screen, v_tex);
  gl_FragColor = vec4(floor(c.rgb * u_opacity * 255.0) / 255.0, 1.0);
}`;

// Update particle positions by sampling the wind texture.
const UPDATE_VS = `
precision highp float;
attribute vec2 a_index;              // particle index (x in [0,texSize), y in [0,texSize))
uniform sampler2D u_particles;       // current particle positions (r=x, g=y) in [0,1]
uniform sampler2D u_wind;            // wind (r=u_norm, g=v_norm)
uniform vec2 u_wind_min;             // [u_min, v_min] m/s
uniform vec2 u_wind_max;             // [u_max, v_max] m/s
uniform float u_speed_factor;        // advection multiplier
uniform float u_rand_seed;
varying vec2 v_particle_pos;

// Pseudo-random from position
float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec2 texCoord = (a_index + 0.5) / float(textureSize(u_particles, 0).x);
  vec4 particle = texture2D(u_particles, texCoord);
  vec2 pos = particle.rg;

  // Sample wind at current position
  vec2 windSample = texture2D(u_wind, pos).rg;
  vec2 velocity = mix(u_wind_min, u_wind_max, windSample);

  // Advect: wind is in m/s, we need fraction of domain per frame.
  // Domain is ~4km, assume 60fps → dt ≈ 1/60 s
  // fraction = velocity * dt / domain_m
  float dt = u_speed_factor / 60.0;
  vec2 offset = velocity * dt / 4000.0;  // 4km domain
  vec2 newPos = pos + offset;

  // Drop particles that exit [0,1]
  float dropRate = 0.003;
  float keep = step(0.0, newPos.x) * step(newPos.x, 1.0)
             * step(0.0, newPos.y) * step(newPos.y, 1.0)
             * step(dropRate, rand(pos + u_rand_seed));
  newPos = mix(vec2(rand(pos + 1.3 + u_rand_seed), rand(pos + 2.1 + u_rand_seed)), newPos, keep);

  v_particle_pos = newPos;
  gl_Position = vec4(texCoord * 2.0 - 1.0, 0, 1);
  gl_PointSize = 1.0;
}`;

const UPDATE_FS = `
precision highp float;
varying vec2 v_particle_pos;
void main() {
  gl_FragColor = vec4(v_particle_pos, 0.0, 1.0);
}`;

// Draw particles as points at their wind-texture positions, mapped to screen.
const DRAW_VS = `
precision highp float;
attribute vec2 a_index;
uniform sampler2D u_particles;
uniform sampler2D u_wind;
uniform vec2 u_wind_min;
uniform vec2 u_wind_max;
uniform mat4 u_matrix;
uniform vec4 u_bbox;            // [west, south, east, north] in Mercator coords
varying float v_speed_t;

void main() {
  vec2 texCoord = (a_index + 0.5) / float(textureSize(u_particles, 0).x);
  vec2 pos = texture2D(u_particles, texCoord).rg;

  // Map [0,1] → Mercator coordinates
  float mx = mix(u_bbox.x, u_bbox.z, pos.x);
  float my = mix(u_bbox.y, u_bbox.w, 1.0 - pos.y);   // flip y: pos.y=0 is south

  gl_Position = u_matrix * vec4(mx, my, 0.0, 1.0);
  gl_PointSize = 2.0;

  // Speed for coloring
  vec2 windSample = texture2D(u_wind, pos).rg;
  vec2 velocity = mix(u_wind_min, u_wind_max, windSample);
  float speed = length(velocity);
  v_speed_t = clamp(speed / 20.0, 0.0, 1.0);   // normalize by vmax
}`;

const DRAW_FS = `
precision mediump float;
varying float v_speed_t;
void main() {
  // Viridis-ish colormap
  vec3 c0 = vec3(0.267, 0.005, 0.329);
  vec3 c1 = vec3(0.127, 0.567, 0.551);
  vec3 c2 = vec3(0.993, 0.906, 0.144);
  vec3 color = mix(mix(c0, c1, smoothstep(0.0, 0.5, v_speed_t)),
                   c2, smoothstep(0.5, 1.0, v_speed_t));
  gl_FragColor = vec4(color, 1.0);
}`;

// ── Helpers ──────────────────────────────────────────────────────────────────

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("shader: " + log);
  }
  return sh;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error("link: " + gl.getProgramInfoLog(p));
  }
  return p;
}

function createTexture(gl, filter, data, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  if (data instanceof Uint8Array) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }
  return tex;
}

// ── Encode u/v Float32Arrays into an RGBA PNG-compatible Uint8Array ───────

export function encodeWindTexture(u, v, nx, ny) {
  // Find data range
  let uMin = Infinity, uMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;
  for (let i = 0; i < nx * ny; i++) {
    if (u[i] < uMin) uMin = u[i];
    if (u[i] > uMax) uMax = u[i];
    if (v[i] < vMin) vMin = v[i];
    if (v[i] > vMax) vMax = v[i];
  }
  // Add small margin to avoid division by zero
  const uRange = Math.max(uMax - uMin, 0.01);
  const vRange = Math.max(vMax - vMin, 0.01);

  const pixels = new Uint8Array(nx * ny * 4);
  for (let i = 0; i < nx * ny; i++) {
    pixels[i * 4]     = Math.round(((u[i] - uMin) / uRange) * 255);
    pixels[i * 4 + 1] = Math.round(((v[i] - vMin) / vRange) * 255);
    pixels[i * 4 + 2] = 0;
    pixels[i * 4 + 3] = 255;
  }
  return { pixels, nx, ny, uMin, uMax: uMin + uRange, vMin, vMax: vMin + vRange };
}

// ── MapLibre Custom Layer ────────────────────────────────────────────────────

export class WindGLLayer {
  constructor(id, options = {}) {
    this.id = id;
    this.type = "custom";
    this.renderingMode = "2d";

    this.numParticles = options.numParticles || 8000;
    this.fadeOpacity = options.fadeOpacity || 0.985;
    this.speedFactor = options.speedFactor || 1.0;
    this.vmax = options.vmax || 20.0;

    this._windData = null;    // { pixels, nx, ny, uMin, uMax, vMin, vMax }
    this._bbox = null;        // [west, south, east, north] in Mercator coords
    this._initialized = false;
  }

  setWind(windData, bounds) {
    // windData from encodeWindTexture(); bounds = { west, south, east, north } in lng/lat
    this._windData = windData;
    // Convert bounds to Mercator
    const nw = maplibregl.MercatorCoordinate.fromLngLat([bounds.west, bounds.north]);
    const se = maplibregl.MercatorCoordinate.fromLngLat([bounds.east, bounds.south]);
    this._bbox = [nw.x, se.y, se.x, nw.y]; // [west_merc, south_merc, east_merc, north_merc]
    this._needsWindUpload = true;
  }

  onAdd(map, gl) {
    this.map = map;
    this.gl = gl;

    // Programs
    this.drawProgram = createProgram(gl, DRAW_VS, DRAW_FS);
    this.updateProgram = createProgram(gl, UPDATE_VS, UPDATE_FS);
    this.screenProgram = createProgram(gl, QUAD_VS, SCREEN_FS);

    // Full-screen quad
    this.quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

    // Particle index buffer
    this._rebuildParticles(gl);

    // Screen framebuffers (ping-pong for trail)
    this.screenTextures = [null, null];
    this.screenFBOs = [null, null];
    this._resizeScreen(gl);

    this._initialized = true;
  }

  _rebuildParticles(gl) {
    const texSize = Math.ceil(Math.sqrt(this.numParticles));
    this.particleTexSize = texSize;
    const n = texSize * texSize;

    // Initial random positions
    const particleData = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      particleData[i * 4]     = Math.floor(Math.random() * 256);
      particleData[i * 4 + 1] = Math.floor(Math.random() * 256);
      particleData[i * 4 + 2] = 0;
      particleData[i * 4 + 3] = 255;
    }

    // Two particle state textures (ping-pong)
    this.particleTextures = [
      createTexture(gl, gl.NEAREST, particleData, texSize, texSize),
      createTexture(gl, gl.NEAREST, particleData, texSize, texSize),
    ];
    this.particleFBOs = [gl.createFramebuffer(), gl.createFramebuffer()];
    for (let i = 0; i < 2; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.particleFBOs[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.particleTextures[i], 0);
    }

    // Particle index VBO
    const indices = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      indices[i * 2]     = i % texSize;
      indices[i * 2 + 1] = Math.floor(i / texSize);
    }
    if (this.indexBuf) gl.deleteBuffer(this.indexBuf);
    this.indexBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    this.particleSwap = 0;
  }

  _resizeScreen(gl) {
    const w = gl.canvas.width, h = gl.canvas.height;
    for (let i = 0; i < 2; i++) {
      if (this.screenTextures[i]) gl.deleteTexture(this.screenTextures[i]);
      if (this.screenFBOs[i]) gl.deleteFramebuffer(this.screenFBOs[i]);
      const emptyPixels = new Uint8Array(w * h * 4);
      this.screenTextures[i] = createTexture(gl, gl.NEAREST, emptyPixels, w, h);
      this.screenFBOs[i] = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.screenFBOs[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.screenTextures[i], 0);
    }
    this.screenSwap = 0;
    this.screenW = w;
    this.screenH = h;
  }

  render(gl, matrix) {
    if (!this._windData || !this._initialized) return;

    // Resize screen FBOs if canvas changed
    if (gl.canvas.width !== this.screenW || gl.canvas.height !== this.screenH) {
      this._resizeScreen(gl);
    }

    // Upload wind texture if new data
    if (this._needsWindUpload) {
      if (this.windTex) gl.deleteTexture(this.windTex);
      this.windTex = createTexture(gl, gl.LINEAR, this._windData.pixels,
                                    this._windData.nx, this._windData.ny);
      this._needsWindUpload = false;
    }

    // 1. Draw fade of previous frame into screenFBO[next]
    const curScreen = this.screenSwap;
    const nextScreen = 1 - curScreen;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.screenFBOs[nextScreen]);
    gl.viewport(0, 0, this.screenW, this.screenH);

    gl.useProgram(this.screenProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.screenTextures[curScreen]);
    gl.uniform1i(gl.getUniformLocation(this.screenProgram, "u_screen"), 0);
    gl.uniform1f(gl.getUniformLocation(this.screenProgram, "u_opacity"), this.fadeOpacity);
    this._drawQuad(gl, this.screenProgram);

    // 2. Draw new particles into same FBO
    gl.useProgram(this.drawProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.particleTextures[this.particleSwap]);
    gl.uniform1i(gl.getUniformLocation(this.drawProgram, "u_particles"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.windTex);
    gl.uniform1i(gl.getUniformLocation(this.drawProgram, "u_wind"), 1);
    gl.uniform2f(gl.getUniformLocation(this.drawProgram, "u_wind_min"), this._windData.uMin, this._windData.vMin);
    gl.uniform2f(gl.getUniformLocation(this.drawProgram, "u_wind_max"), this._windData.uMax, this._windData.vMax);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.drawProgram, "u_matrix"), false, matrix);
    gl.uniform4fv(gl.getUniformLocation(this.drawProgram, "u_bbox"), this._bbox);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    const aIdx = gl.getAttribLocation(this.drawProgram, "a_index");
    gl.enableVertexAttribArray(aIdx);
    gl.vertexAttribPointer(aIdx, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.particleTexSize * this.particleTexSize);

    // 3. Render screen texture to the actual map canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.screenW, this.screenH);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.screenProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.screenTextures[nextScreen]);
    gl.uniform1i(gl.getUniformLocation(this.screenProgram, "u_screen"), 0);
    gl.uniform1f(gl.getUniformLocation(this.screenProgram, "u_opacity"), 1.0);
    this._drawQuad(gl, this.screenProgram);

    this.screenSwap = nextScreen;

    // 4. Update particle positions (ping-pong)
    const curPart = this.particleSwap;
    const nextPart = 1 - curPart;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.particleFBOs[nextPart]);
    gl.viewport(0, 0, this.particleTexSize, this.particleTexSize);

    gl.useProgram(this.updateProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.particleTextures[curPart]);
    gl.uniform1i(gl.getUniformLocation(this.updateProgram, "u_particles"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.windTex);
    gl.uniform1i(gl.getUniformLocation(this.updateProgram, "u_wind"), 1);
    gl.uniform2f(gl.getUniformLocation(this.updateProgram, "u_wind_min"), this._windData.uMin, this._windData.vMin);
    gl.uniform2f(gl.getUniformLocation(this.updateProgram, "u_wind_max"), this._windData.uMax, this._windData.vMax);
    gl.uniform1f(gl.getUniformLocation(this.updateProgram, "u_speed_factor"), this.speedFactor);
    gl.uniform1f(gl.getUniformLocation(this.updateProgram, "u_rand_seed"), Math.random());

    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    const aIdx2 = gl.getAttribLocation(this.updateProgram, "a_index");
    gl.enableVertexAttribArray(aIdx2);
    gl.vertexAttribPointer(aIdx2, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.particleTexSize * this.particleTexSize);

    this.particleSwap = nextPart;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.map.triggerRepaint();
  }

  _drawQuad(gl, program) {
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    const aPos = gl.getAttribLocation(program, "a_pos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  onRemove(map, gl) {
    // Cleanup all GL resources
    [this.drawProgram, this.updateProgram, this.screenProgram].forEach(p => p && gl.deleteProgram(p));
    [this.quadBuf, this.indexBuf].forEach(b => b && gl.deleteBuffer(b));
    [this.windTex, ...this.particleTextures, ...this.screenTextures]
      .forEach(t => t && gl.deleteTexture(t));
    [...this.particleFBOs, ...this.screenFBOs].forEach(f => f && gl.deleteFramebuffer(f));
  }
}
