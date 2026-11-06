// The 2-D/3-D view: how far the picture is warped from the orbiting 3-D camera
// onto a flat, orthographic plot of two state axes, and where the slab slice
// sits along the third.
//
// Everything here animates, because the point is that the two views are the
// same object: going flat is a continuous blend of every point's screen
// position, and changing plane while flat goes up through 3-D and back down,
// so the eye can follow which axis went where.

const AXES = { x: 0, y: 1, z: 2 };
const NAMES = ['x', 'y', 'z'];

const smooth = t => t * t * (3 - 2 * t);

export class View {
  constructor({ duration = 1.1 } = {}) {
    this.duration = duration;
    this.t = 0;              // linear progress toward flat, 0..1
    this.target = 0;
    this.axes = [0, 1, 2];   // plot horizontal, plot vertical, depth/slice
    this.pending = null;     // axes to switch to once back in 3-D
    this.slice = { on: false, at: 0.5, thickness: 0.08 };  // fractions of the depth axis's range
    this.bounds = { center: [0, 0, 0], half: [1, 1, 1] };
    this.margin = 0.86;
  }

  get flat() { return smooth(this.t); }

  // Fit the plot to where the trajectories actually are: 2-98th percentiles
  // per state axis, from the same readback the director frames the camera with.
  measure(positions) {
    const count = positions.length / 4;
    const stride = Math.max(1, Math.ceil(count / 20000));
    const cols = [[], [], []];
    for (let i = 0; i < count; i += stride) {
      const x = positions[i * 4], y = positions[i * 4 + 1], z = positions[i * 4 + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      cols[0].push(x); cols[1].push(y); cols[2].push(z);
    }
    if (!cols[0].length) return;
    const at = (a, q) => a[Math.min(a.length - 1, Math.floor(a.length * q))];
    for (let k = 0; k < 3; k++) {
      const a = cols[k].sort((p, q) => p - q);
      const lo = at(a, 0.02), hi = at(a, 0.98);
      this.bounds.center[k] = (lo + hi) / 2;
      this.bounds.half[k] = Math.max((hi - lo) / 2, 1e-6);
    }
  }

  setFlat(on) { this.target = on ? 1 : 0; }

  toggle() { this.setFlat(this.target < 0.5); }

  // `plane` is two axis letters, e.g. 'xz'; the remaining axis is depth.
  setPlane(plane) {
    const a = AXES[plane[0]], b = AXES[plane[1]];
    if (a === undefined || b === undefined || a === b) return;
    const next = [a, b, 3 - a - b];
    if (next.every((v, i) => v === this.axes[i])) return;
    if (this.t <= 0) { this.axes = next; return; }
    // Flat on the old plane: rise to 3-D, swap, come back down.
    this.pending = next;
    this.resume = this.target;
    this.target = 0;
  }

  get plane() { return NAMES[this.axes[0]] + NAMES[this.axes[1]]; }
  get depthAxis() { return NAMES[this.axes[2]]; }

  setSlice({ on, at, thickness } = {}) {
    if (on !== undefined) this.slice.on = on;
    if (at !== undefined) this.slice.at = Math.max(0, Math.min(1, at));
    if (thickness !== undefined) this.slice.thickness = Math.max(0.005, Math.min(1, thickness));
  }

  // The slice centre in state units, for a readout.
  get sliceValue() {
    const c = this.axes[2];
    return this.bounds.center[c] + (2 * this.slice.at - 1) * this.bounds.half[c];
  }

  advance(dt) {
    const step = dt / this.duration;
    if (this.t < this.target) this.t = Math.min(this.target, this.t + step);
    else if (this.t > this.target) this.t = Math.max(this.target, this.t - step);
    if (this.pending && this.t === 0) {
      this.axes = this.pending;
      this.pending = null;
      this.target = this.resume;
    }
  }

  uniforms(aspect) {
    const [a, b, c] = this.axes;
    const { center, half } = this.bounds;
    // Equal units on both plot axes, so the shape is not stretched to fill.
    const k = this.margin * Math.min(aspect / half[a], 1 / half[b]);
    return {
      flat: this.flat,
      axes: this.axes,
      center: [center[a], center[b]],
      scale: [k / aspect, k],
      depth: [center[c], half[c]],
      slice: [
        this.slice.on ? 1 : 0,
        this.sliceValue,
        Math.max(1e-6, this.slice.thickness * half[c])
      ]
    };
  }
}
