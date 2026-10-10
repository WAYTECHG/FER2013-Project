// Classic worker: MediaPipe's WASM loader uses importScripts.
importScripts('./vendor/ort/ort.wasm.min.js');
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', self.location.href).href;
ort.env.wasm.numThreads = 1; // Works on static hosts and iframes without COOP/COEP.
ort.env.wasm.proxy = false; // Already running inside our own worker.
let detector, session, manifest, loadedManifestURL, previous = [];
let core, busyChain = Promise.resolve();
const frameCanvas = new OffscreenCanvas(1, 1);
const frameContext = frameCanvas.getContext('2d', { willReadFrequently: true });
const cropCanvas = new OffscreenCanvas(48, 48);
const cropContext = cropCanvas.getContext('2d', { willReadFrequently: true });

async function initialize(manifestURL) {
  core ||= await import('./core.js');
  if (!detector) {
    const { FaceDetector, FilesetResolver } = await import('./vendor/mediapipe/vision_bundle.mjs');
    const vision = await FilesetResolver.forVisionTasks(new URL('./vendor/mediapipe/wasm', self.location.href).href);
    detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: new URL('./models/blaze_face_short_range.tflite', self.location.href).href, delegate: 'CPU' },
      runningMode: 'IMAGE', minDetectionConfidence: 0.55, minSuppressionThreshold: 0.3
    });
  }
  previous = [];
  if (session && loadedManifestURL === manifestURL) return { classifier: true };
  if (session) { await session.release(); session = undefined; }
  manifest = undefined; loadedManifestURL = undefined;
  const response = await fetch(manifestURL);
  if (response.status === 404) return { classifier: false, reason: 'Your face is detected, but the selected trained emotion model is not installed. Add its ONNX model and matching metadata to enable the seven emotion labels and scores.' };
  if (!response.ok) throw new Error(`Could not load model metadata (HTTP ${response.status}).`);
  manifest = core.validateManifest(await response.json());
  const modelURL = new URL(manifest.modelFile, manifestURL);
  const modelResponse = await fetch(modelURL);
  if (!modelResponse.ok) throw new Error(`Could not download the trained emotion model (HTTP ${modelResponse.status}).`);
  const bytes = await modelResponse.arrayBuffer();
  if (manifest.sha256 && self.crypto?.subtle) {
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== manifest.sha256) throw new Error('Model integrity check failed. Upload the matching model and metadata again.');
  }
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  if (!session.inputNames.includes(manifest.inputName) || !session.outputNames.includes(manifest.outputName)) {
    await session.release(); session = undefined;
    throw new Error('Model input or output names do not match its metadata.');
  }
  // Warm up and check the actual output before claiming that the engine is ready.
  const output = await session.run({ [manifest.inputName]: new ort.Tensor('float32', new Float32Array(48 * 48 * 3), [1, 48, 48, 3]) });
  core.probabilityVector(output[manifest.outputName].data);
  loadedManifestURL = manifestURL;
  return { classifier: true };
}

async function analyze({ bitmap, live }) {
  if (!detector) throw new Error('The face detection engine is not ready.');
  const started = performance.now();
  frameCanvas.width = bitmap.width; frameCanvas.height = bitmap.height;
  frameContext.fillStyle = '#fff'; frameContext.fillRect(0, 0, bitmap.width, bitmap.height);
  frameContext.drawImage(bitmap, 0, 0); bitmap.close();
  const detected = detector.detect(frameCanvas).detections;
  const boxes = detected.map(d => core.clampBox(d.boundingBox, frameCanvas.width, frameCanvas.height))
    .filter(box => box.width > 3 && box.height > 3).sort((a, b) => b.width * b.height - a.width * a.height);
  const faces = [];
  for (const box of boxes.slice(0, 6)) {
    let probabilities = null;
    if (session) {
      cropCanvas.width = box.width; cropCanvas.height = box.height;
      cropContext.drawImage(frameCanvas, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
      const rgba = cropContext.getImageData(0, 0, box.width, box.height).data;
      const input = core.areaTensor(rgba, box.width, box.height);
      const output = await session.run({ [manifest.inputName]: new ort.Tensor('float32', input, [1, 48, 48, 3]) });
      probabilities = core.probabilityVector(output[manifest.outputName].data);
    }
    faces.push({ box, probabilities });
  }
  const smoothed = core.smoothFaces(faces, previous, live);
  previous = live ? smoothed : [];
  return { faces: smoothed, totalFaces: boxes.length, width: frameCanvas.width, height: frameCanvas.height, ms: performance.now() - started, classifier: !!session };
}

self.onmessage = ({ data }) => {
  busyChain = busyChain.then(async () => {
    try {
      let result;
      if (data.type === 'init') result = await initialize(data.manifestURL);
      else if (data.type === 'frame') result = await analyze(data);
      else if (data.type === 'reset') { previous = []; result = {}; }
      else throw new Error('Unknown engine request.');
      self.postMessage({ id: data.id, ok: true, result });
    } catch (error) {
      data.bitmap?.close();
      self.postMessage({ id: data.id, ok: false, error: error.message || String(error) });
    }
  });
};
