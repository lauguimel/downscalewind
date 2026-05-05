// particles_layer.js — MapLibre custom WebGL layer for animated 3D wind particles.
//
// Implements the MapLibre CustomLayerInterface so it shares the map's GL context
// and receives the view-projection matrix each frame. Particles live inside the
// case box in normalized (fx, fy) ∈ [0,1]² + absolute altitude (meters). Advection
// runs on the CPU by trilinear sampling of the u/v/w volumes (fast enough for a
// few thousand particles, trivial to debug, no transform feedback boilerplate).
//
// Usage:
//   const layer = new ParticlesLayer("plume-particles");
//   map.addLayer(layer);
//   layer.setCase(caseData);          // whenever a new .bin is loaded
//   layer.setCount(5000);             // GUI
//   layer.setSpeedScale(1.0);         // advection dt multiplier

const VERT_SRC = `#version 300 es
precision highp float;

in vec3 a_pos;       // mercator coordinates (x, y, z)
in float a_speed;    // wind speed at this particle, m/s

uniform mat4 u_matrix;
uniform float u_vmax;
uniform float u_pointSize;

out float v_t;       // [0,1] color ramp coordinate

void main() {
  gl_Position = u_matrix * vec4(a_pos, 1.0);
  gl_PointSize = u_pointSize;
  v_t = clamp(a_speed / u_vmax, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in float v_t;
out vec4 fragColor;

// Crude viridis ramp
vec3 viridis(float t) {
  vec3 c0 = vec3(0.267, 0.005, 0.329);
  vec3 c1 = vec3(0.127, 0.567, 0.551);
  vec3 c2 = vec3(0.993, 0.906, 0.144);
  return mix(mix(c0, c1, smoothstep(0.0, 0.5, t)),
             c2, smoothstep(0.5, 1.0, t));
}

void main() {
  // Circular sprite
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if (r > 0.5) discard;
  float alpha = smoothstep(0.5, 0.2, r);
  fragColor = vec4(viridis(v_t), alpha * 0.85);
}
`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("particle shader compile failed: " + log);
  }
  return sh;
}

function link(gl, vsSrc, fsSrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("particle program link failed: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

// Trilinear sample of a scalar (ny, nx, nz) Float32Array at normalized (fy, fx, fz).
function sampleField(field, nx, ny, nz, fx, fy, fz) {
  const xf = fx * (nx - 1);
  const yf = fy * (ny - 1);
  const zf = fz * (nz - 1);
  const i0 = Math.max(0, Math.min(nx - 2, Math.floor(xf)));
  const j0 = Math.max(0, Math.min(ny - 2, Math.floor(yf)));
  const k0 = Math.max(0, Math.min(nz - 2, Math.floor(zf)));
  const wx = xf - i0, wy = yf - j0, wz = zf - k0;

  const idx = (j, i, k) => (j * nx + i) * nz + k;

  const c000 = field[idx(j0,     i0,     k0)];
  const c100 = field[idx(j0,     i0 + 1, k0)];
  const c010 = field[idx(j0 + 1, i0,     k0)];
  const c110 = field[idx(j0 + 1, i0 + 1, k0)];
  const c001 = field[idx(j0,     i0,     k0 + 1)];
  const c101 = field[idx(j0,     i0 + 1, k0 + 1)];
  const c011 = field[idx(j0 + 1, i0,     k0 + 1)];
  const c111 = field[idx(j0 + 1, i0 + 1, k0 + 1)];

  const c00 = c000 * (1 - wx) + c100 * wx;
  const c10 = c010 * (1 - wx) + c110 * wx;
  const c01 = c001 * (1 - wx) + c101 * wx;
  const c11 = c011 * (1 - wx) + c111 * wx;

  const c0 = c00 * (1 - wy) + c10 * wy;
  const c1 = c01 * (1 - wy) + c11 * wy;

  return c0 * (1 - wz) + c1 * wz;
}

// Trilinear for 2D terrain (ny, nx) indexed as terrain[j*nx + i]
function sampleTerrain(terrain, nx, ny, fx, fy) {
  const xf = fx * (nx - 1);
  const yf = fy * (ny - 1);
  const i0 = Math.max(0, Math.min(nx - 2, Math.floor(xf)));
  const j0 = Math.max(0, Math.min(ny - 2, Math.floor(yf)));
  const wx = xf - i0, wy = yf - j0;
  const t00 = terrain[j0 * nx + i0];
  const t10 = terrain[j0 * nx + i0 + 1];
  const t01 = terrain[(j0 + 1) * nx + i0];
  const t11 = terrain[(j0 + 1) * nx + i0 + 1];
  return (t00 * (1 - wx) + t10 * wx) * (1 - wy) +
         (t01 * (1 - wx) + t11 * wx) * wy;
}

export class ParticlesLayer {
  constructor(id = "plume-particles") {
    this.id = id;
    this.type = "custom";
    this.renderingMode = "3d";

    this.gl = null;
    this.program = null;
    this.posBuf = null;
    this.speedBuf = null;
    this.vao = null;

    this.caseData = null;
    this.count = 4000;
    this.speedScale = 1.0;
    this.vmax = 20.0;
    this.pointSize = 3.0;
    this.altExaggeration = 2.5;   // visual boost so particles rise above terrain
    this.maxAgeFrames = 200;

    // Particle state (CPU side). Allocated on setCount().
    this._fx = null;   // Float32Array[count]
    this._fy = null;
    this._z = null;    // absolute altitude in meters
    this._age = null;  // frame count, reset when reborn

    // GPU upload buffers (reused each frame)
    this._mercPos = null;   // Float32Array[count * 3]
    this._speeds = null;    // Float32Array[count]

    this._mercBounds = null;
    this._meterToMerc = 0;
  }

  setCase(caseData) {
    this.caseData = caseData;
    this._recomputeMercatorBounds();
    this._allocate();
    for (let i = 0; i < this.count; i++) this._respawn(i);
  }

  setCount(n) {
    this.count = Math.max(100, Math.min(20000, n | 0));
    if (this.caseData) {
      this._allocate();
      for (let i = 0; i < this.count; i++) this._respawn(i);
    }
  }

  setSpeedScale(s) { this.speedScale = s; }
  setVmax(v) { this.vmax = v; }
  setPointSize(s) { this.pointSize = s; }
  setAltExaggeration(e) { this.altExaggeration = e; }

  _allocate() {
    this._fx = new Float32Array(this.count);
    this._fy = new Float32Array(this.count);
    this._z = new Float32Array(this.count);
    this._age = new Uint16Array(this.count);
    this._mercPos = new Float32Array(this.count * 3);
    this._speeds = new Float32Array(this.count);
  }

  _recomputeMercatorBounds() {
    if (!this.caseData) return;
    const { bounds } = this.caseData;
    const nw = maplibregl.MercatorCoordinate.fromLngLat([bounds.west, bounds.north], 0);
    const se = maplibregl.MercatorCoordinate.fromLngLat([bounds.east, bounds.south], 0);
    this._mercBounds = {
      xMin: nw.x, xMax: se.x,
      yNorth: nw.y, ySouth: se.y,
    };
    this._meterToMerc = nw.meterInMercatorCoordinateUnits();
  }

  _respawn(i) {
    const { nx, ny, terrain, zLevels } = this.caseData;
    const fx = Math.random();
    const fy = Math.random();
    this._fx[i] = fx;
    this._fy[i] = fy;
    const elev = sampleTerrain(terrain, nx, ny, fx, fy);
    // Random AGL between z_min and ~half the box top
    const zMax = zLevels[zLevels.length - 1];
    const agl = 20.0 + Math.random() * (zMax * 0.4 - 20.0);
    this._z[i] = elev + agl;
    this._age[i] = (Math.random() * this.maxAgeFrames) | 0;
  }

  // --- MapLibre CustomLayerInterface hooks ---

  onAdd(map, gl) {
    this.map = map;
    this.gl = gl;
    this.program = link(gl, VERT_SRC, FRAG_SRC);

    this.u_matrix = gl.getUniformLocation(this.program, "u_matrix");
    this.u_vmax = gl.getUniformLocation(this.program, "u_vmax");
    this.u_pointSize = gl.getUniformLocation(this.program, "u_pointSize");
    this.a_pos = gl.getAttribLocation(this.program, "a_pos");
    this.a_speed = gl.getAttribLocation(this.program, "a_speed");

    this.posBuf = gl.createBuffer();
    this.speedBuf = gl.createBuffer();
  }

  onRemove(map, gl) {
    if (this.program) gl.deleteProgram(this.program);
    if (this.posBuf) gl.deleteBuffer(this.posBuf);
    if (this.speedBuf) gl.deleteBuffer(this.speedBuf);
  }

  render(gl, matrix) {
    if (!this.caseData || !this._mercBounds) return;

    this._step();

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.u_matrix, false, matrix);
    gl.uniform1f(this.u_vmax, this.vmax);
    gl.uniform1f(this.u_pointSize, this.pointSize);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this._mercPos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.a_pos);
    gl.vertexAttribPointer(this.a_pos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.speedBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this._speeds, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.a_speed);
    gl.vertexAttribPointer(this.a_speed, 1, gl.FLOAT, false, 0, 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    gl.drawArrays(gl.POINTS, 0, this.count);

    // Ask MapLibre to redraw continuously so animation runs
    this.map.triggerRepaint();
  }

  _step() {
    const c = this.caseData;
    const { nx, ny, nz, u, v, w, zLevels, terrain } = c;
    const zMin = zLevels[0];
    const zMax = zLevels[zLevels.length - 1];
    const dt = 0.25 * this.speedScale;  // seconds per frame (ad-hoc)

    const dLonDeg = c.bounds.east - c.bounds.west;
    const dLatDeg = c.bounds.north - c.bounds.south;
    const dx_m = (dLonDeg * 111000.0 * Math.cos(c.latCenter * Math.PI / 180));
    const dy_m = (dLatDeg * 111000.0);

    const mb = this._mercBounds;
    const mercXRange = mb.xMax - mb.xMin;
    const mercYRange = mb.ySouth - mb.yNorth;
    const meterToMerc = this._meterToMerc;

    for (let i = 0; i < this.count; i++) {
      const fx = this._fx[i];
      const fy = this._fy[i];
      const zAbs = this._z[i];

      const elev = sampleTerrain(terrain, nx, ny, fx, fy);
      const agl = Math.max(zMin, zAbs - elev);
      const fz = Math.min(1.0, Math.max(0.0, (Math.log(agl) - Math.log(zMin)) / (Math.log(zMax) - Math.log(zMin))));

      const uu = sampleField(u, nx, ny, nz, fx, fy, fz);
      const vv = sampleField(v, nx, ny, nz, fx, fy, fz);
      const ww = sampleField(w, nx, ny, nz, fx, fy, fz);
      const spd = Math.hypot(uu, vv, ww);

      // Advect in meters → update fractional coords
      const newFx = fx + (uu * dt) / dx_m;
      const newFy = fy + (vv * dt) / dy_m;
      const newZ = zAbs + ww * dt;

      this._age[i]++;
      const outOfBox = newFx < 0 || newFx > 1 || newFy < 0 || newFy > 1 ||
                       (newZ - sampleTerrain(terrain, nx, ny,
                         Math.min(1, Math.max(0, newFx)),
                         Math.min(1, Math.max(0, newFy)))) > zMax;
      if (outOfBox || this._age[i] > this.maxAgeFrames) {
        this._respawn(i);
      } else {
        this._fx[i] = newFx;
        this._fy[i] = newFy;
        this._z[i] = newZ;
      }

      // Compute mercator coordinates for render
      const mx = mb.xMin + this._fx[i] * mercXRange;
      // fy=0 is south (large mercY) → yMax; fy=1 is north → yNorth
      const my = mb.ySouth - this._fy[i] * mercYRange;
      const altVisual = (this._z[i] - elev) * this.altExaggeration + elev;
      const mz = altVisual * meterToMerc;

      this._mercPos[3 * i] = mx;
      this._mercPos[3 * i + 1] = my;
      this._mercPos[3 * i + 2] = mz;
      this._speeds[i] = spd;
    }
  }
}
