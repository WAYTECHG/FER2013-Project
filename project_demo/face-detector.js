// YuNet decoding follows OpenCV's FaceDetectorYN implementation.
// This detector uses ONNX Runtime WASM and requires no WebGL context.
import { iou } from './core.js';

export const DETECTOR_SIZE = 640;
export const DETECTOR_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4';

export function bgrTensor(rgba, size = DETECTOR_SIZE) {
  const pixels = size * size, input = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    input[i] = rgba[i * 4 + 2];
    input[pixels + i] = rgba[i * 4 + 1];
    input[2 * pixels + i] = rgba[i * 4];
  }
  return input;
}

export function decodeFaces(outputs, width, height, resizedWidth, resizedHeight, threshold = 0.65) {
  const candidates = [];
  for (const stride of [8, 16, 32]) {
    const cls = outputs[`cls_${stride}`]?.data;
    const obj = outputs[`obj_${stride}`]?.data;
    const bbox = outputs[`bbox_${stride}`]?.data;
    const columns = DETECTOR_SIZE / stride, count = columns * columns;
    if (!cls || !obj || !bbox || cls.length !== count || obj.length !== count || bbox.length !== count * 4) {
      throw new Error('The face detector output is incompatible. Restore models/face_detector.onnx from the repair package.');
    }
    for (let i = 0; i < count; i++) {
      const score = Math.sqrt(Math.max(0, Math.min(1, cls[i])) * Math.max(0, Math.min(1, obj[i])));
      if (!Number.isFinite(score) || score < threshold) continue;
      const cx = (i % columns + bbox[i * 4]) * stride;
      const cy = (Math.floor(i / columns) + bbox[i * 4 + 1]) * stride;
      if (cx < 0 || cy < 0 || cx >= resizedWidth || cy >= resizedHeight) continue;
      const boxWidth = Math.exp(bbox[i * 4 + 2]) * stride;
      const boxHeight = Math.exp(bbox[i * 4 + 3]) * stride;
      if (!Number.isFinite(boxWidth) || !Number.isFinite(boxHeight)) continue;
      const x = Math.max(0, (cx - boxWidth / 2) * width / resizedWidth);
      const y = Math.max(0, (cy - boxHeight / 2) * height / resizedHeight);
      const right = Math.min(width, (cx + boxWidth / 2) * width / resizedWidth);
      const bottom = Math.min(height, (cy + boxHeight / 2) * height / resizedHeight);
      if (right - x > 3 && bottom - y > 3) candidates.push({ x, y, width: right - x, height: bottom - y, score });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of candidates.slice(0, 5000)) {
    if (kept.every(other => iou(box, other) <= 0.3)) kept.push(box);
  }
  return kept.sort((a, b) => b.width * b.height - a.width * a.height);
}
