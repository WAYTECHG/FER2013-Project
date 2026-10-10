export const LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
export const COLORS = ['#f89c88', '#b9b181', '#bdabed', '#b7f76b', '#a5b7ce', '#74b4ef', '#74d8ed'];

export function clampBox(box, width, height) {
  const x = Math.max(0, Math.floor(box.originX));
  const y = Math.max(0, Math.floor(box.originY));
  const right = Math.min(width, Math.ceil(box.originX + box.width));
  const bottom = Math.min(height, Math.ceil(box.originY + box.height));
  return { x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y) };
}

// Area-weighted RGB resampling; preserves the repo's [0,1] NHWC convention.
// The alpha channel is ignored after the source has been composited on white.
export function areaTensor(rgba, width, height, size = 48) {
  const output = new Float32Array(size * size * 3);
  const sx = width / size, sy = height / size;
  for (let oy = 0; oy < size; oy++) {
    const y0 = oy * sy, y1 = (oy + 1) * sy;
    for (let ox = 0; ox < size; ox++) {
      const x0 = ox * sx, x1 = (ox + 1) * sx;
      const sums = [0, 0, 0];
      for (let iy = Math.floor(y0); iy < Math.ceil(y1) && iy < height; iy++) {
        const wy = Math.max(0, Math.min(iy + 1, y1) - Math.max(iy, y0));
        for (let ix = Math.floor(x0); ix < Math.ceil(x1) && ix < width; ix++) {
          const wx = Math.max(0, Math.min(ix + 1, x1) - Math.max(ix, x0));
          const pos = (iy * width + ix) * 4, weight = wx * wy;
          for (let c = 0; c < 3; c++) sums[c] += rgba[pos + c] * weight;
        }
      }
      const pos = (oy * size + ox) * 3;
      // OpenCV resizes uint8 pixels before converting them to float32.
      for (let c = 0; c < 3; c++) output[pos + c] = Math.round(sums[c] / (sx * sy)) / 255;
    }
  }
  return output;
}

export function probabilityVector(raw) {
  const values = Array.from(raw, Number);
  if (values.length !== 7 || values.some(x => !Number.isFinite(x) || x < 0 || x > 1.0001)) {
    throw new Error('The model must return seven softmax probabilities.');
  }
  const sum = values.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 0.02) throw new Error('Model probabilities do not sum to one.');
  return values.map(x => x / sum);
}

export function validateManifest(value) {
  if (!value || value.format !== 'onnx' || value.inputLayout !== 'NHWC' ||
      JSON.stringify(value.inputShape) !== '[1,48,48,3]' || value.normalization !== 'rgb/255' ||
      JSON.stringify(value.labels) !== JSON.stringify(LABELS) || !value.modelFile ||
      typeof value.inputName !== 'string' || typeof value.outputName !== 'string') {
    throw new Error('Model metadata does not match the project: 48×48 RGB, NHWC, RGB/255, seven labels.');
  }
  return value;
}

export function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const intersection = w * h;
  return intersection / (a.width * a.height + b.width * b.height - intersection || 1);
}

export function smoothFaces(faces, previous, live) {
  const available = [...previous];
  return faces.map(face => {
    if (!live || !face.probabilities) return face;
    let match = -1, best = 0.35;
    available.forEach((old, i) => { const score = iou(face.box, old.box); if (score > best) { best = score; match = i; } });
    if (match !== -1) {
      const old = available.splice(match, 1)[0];
      if (old.probabilities) face.probabilities = face.probabilities.map((p, i) => 0.55 * p + 0.45 * old.probabilities[i]);
    }
    return face;
  });
}
