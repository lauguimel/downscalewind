// loader.js — parse the PLM2 binary format produced by generate_demo_case.py
//
// Layout (little-endian):
//   magic[4] "PLM2"
//   nx, ny, nz  : u32
//   dx_m, dy_m  : f32
//   lat_c, lon_c: f32
//   pad[8]
//   z_levels    : f32[nz]
//   terrain     : f32[ny*nx]
//   u, v, w     : f32[ny*nx*nz]

export function loadCaseFromBuffer(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "PLM2") throw new Error(`bad magic: ${magic}`);

  const nx = dv.getUint32(4, true);
  const ny = dv.getUint32(8, true);
  const nz = dv.getUint32(12, true);
  const dx = dv.getFloat32(16, true);
  const dy = dv.getFloat32(20, true);
  const latC = dv.getFloat32(24, true);
  const lonC = dv.getFloat32(28, true);
  // 8 bytes pad (32..40)

  let off = 40;
  const z = new Float32Array(buf, off, nz);                     off += nz * 4;
  const terrain = new Float32Array(buf, off, ny * nx);          off += ny * nx * 4;
  const u = new Float32Array(buf, off, ny * nx * nz);           off += ny * nx * nz * 4;
  const v = new Float32Array(buf, off, ny * nx * nz);           off += ny * nx * nz * 4;
  const w = new Float32Array(buf, off, ny * nx * nz);           off += ny * nx * nz * 4;

  // Domain bounds in degrees (small-angle approximation)
  const domainKm = (nx * dx) / 1000.0;
  const dLat = (domainKm * 0.5) / 111.0;
  const dLon = (domainKm * 0.5) / (111.0 * Math.cos(latC * Math.PI / 180));

  return {
    nx, ny, nz, dx, dy,
    latCenter: latC, lonCenter: lonC,
    bounds: {
      west: lonC - dLon,
      east: lonC + dLon,
      south: latC - dLat,
      north: latC + dLat,
    },
    zLevels: z,
    terrain,
    u, v, w,
  };
}

export async function loadCase(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  return loadCaseFromBuffer(await res.arrayBuffer());
}

// Compute wind speed at a target AGL height on the full (ny,nx) grid,
// by linear interpolation in z.
export function speedAtHeight(caseData, targetZ) {
  const { nx, ny, nz, zLevels, u, v } = caseData;
  // Find bracketing indices
  let iHi = 0;
  while (iHi < nz && zLevels[iHi] < targetZ) iHi++;
  if (iHi === 0) iHi = 1;
  if (iHi >= nz) iHi = nz - 1;
  const iLo = iHi - 1;
  const wHi = (targetZ - zLevels[iLo]) / (zLevels[iHi] - zLevels[iLo]);
  const wLo = 1 - wHi;

  const out = new Float32Array(ny * nx);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const base = (j * nx + i) * nz;
      const uZ = wLo * u[base + iLo] + wHi * u[base + iHi];
      const vZ = wLo * v[base + iLo] + wHi * v[base + iHi];
      out[j * nx + i] = Math.hypot(uZ, vZ);
    }
  }
  return out;
}

// Build an RGBA image texture colored by wind speed (viridis-ish).
export function speedToRGBA(speedField, nx, ny, vmin = 0, vmax = 20) {
  const rgba = new Uint8ClampedArray(nx * ny * 4);
  for (let k = 0; k < nx * ny; k++) {
    const t = Math.max(0, Math.min(1, (speedField[k] - vmin) / (vmax - vmin)));
    // crude viridis approximation
    const r = Math.round(255 * (0.267 + t * (0.993 - 0.267) * (1 - t) * 2));
    const g = Math.round(255 * (0.005 + t * 0.9));
    const b = Math.round(255 * (0.329 + (1 - t) * 0.5));
    const idx = k * 4;
    rgba[idx] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = 180;  // semi-transparent
  }
  return rgba;
}
