import { LABELS, COLORS } from './core.js';
const $ = id => document.getElementById(id);
const config = window.FER_CONFIG;
let mode = 'image', imageSource = null, stream = null, worker = null, nextID = 0;
let engineModel = null, engineInfo = null, enginePromise = null, generation = 0;
let frameBusy = false, cameraStarting = false, lastSent = 0, animation = 0, lastFrameTime = -1;
let currentResult = null, selectedFace = 0, lastResultAt = 0, uploadVersion = 0;
const requests = new Map();
const canvas = $('image-canvas'), overlay = $('overlay'), video = $('camera');

function message(text, kind = '') { $('demo-message').textContent = text; $('demo-message').className = `message ${kind}`; }
function engineStatus(text, ready = false) { $('engine-status').textContent = text; $('engine-status').classList.toggle('ready', ready); }
function clearResults(label = 'Awaiting input') {
  currentResult = null; selectedFace = 0; updateScores(null, label);
  $('face-count').textContent = '0 faces'; $('face-picker').hidden = true;
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
}

LABELS.forEach((label, i) => {
  const row = document.createElement('div'); row.className = 'score-row';
  row.innerHTML = `<div class="score-top"><span>${label}</span><span>—</span></div><div class="score-track"><div class="score-fill" style="--score-color:${COLORS[i]}"></div></div>`;
  $('scores').append(row);
});
config.models.forEach(model => { const option = new Option(model.name, model.id); $('model-select').add(option); });
$('model-select').value = config.defaultModel;

function updateScores(probabilities, emptyLabel = 'Awaiting input') {
  const top = probabilities ? probabilities.indexOf(Math.max(...probabilities)) : -1;
  [...$('scores').children].forEach((row, i) => {
    const value = probabilities?.[i];
    row.querySelector('.score-top span:last-child').textContent = value == null ? '—' : `${(100 * value).toFixed(1)}%`;
    row.querySelector('.score-fill').style.width = value == null ? '0%' : `${100 * value}%`;
    row.classList.toggle('winner', i === top);
  });
  $('prediction-label').textContent = top === -1 ? emptyLabel : LABELS[top];
  $('prediction-confidence').textContent = top === -1 ? 'No expression scores available.' : `${(100 * probabilities[top]).toFixed(1)}% model confidence${mode === 'camera' ? ' · smoothed' : ''}`;
}

function request(type, data = {}, transfer = []) {
  if (!worker) {
    worker = new Worker(new URL('./inference-worker.js?v=20261010-wasm-face', import.meta.url));
    worker.onmessage = ({ data: response }) => {
      const pending = requests.get(response.id); if (!pending) return;
      clearTimeout(pending.timeout); requests.delete(response.id);
      response.ok ? pending.resolve(response.result) : pending.reject(new Error(response.error));
    };
    worker.onerror = event => failWorker(new Error('The inference worker could not start. Restore inference-worker.js, face-detector.js and vendor/ort/ from the complete package. ' + (event.message || '')));
  }
  const id = ++nextID;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => failWorker(new Error('The inference engine timed out. Retry or choose a smaller model.')), type === 'init' ? 120000 : 30000);
    requests.set(id, { resolve, reject, timeout });
    try { worker.postMessage({ id, type, ...data }, transfer); }
    catch (error) { clearTimeout(timeout); requests.delete(id); reject(error); }
  });
}
function failWorker(error) {
  worker?.terminate(); worker = null; engineModel = null; engineInfo = null; enginePromise = null;
  for (const pending of requests.values()) { clearTimeout(pending.timeout); pending.reject(error); }
  requests.clear();
}
async function ensureEngine() {
  const model = config.models.find(x => x.id === $('model-select').value);
  if (engineModel === model.id && engineInfo) {
    engineStatus(engineInfo.classifier ? 'Emotion model ready' : 'Emotion model missing', engineInfo.classifier);
    return engineInfo;
  }
  if (enginePromise) { await enginePromise; return ensureEngine(); }
  engineStatus('Loading engine…');
  message('Loading the face detector and selected model. The first load can take a moment.');
  enginePromise = request('init', { manifestURL: new URL(model.manifest, location.href).href });
  try {
    engineInfo = await enginePromise; engineModel = model.id;
    engineStatus(engineInfo.classifier ? 'Emotion model ready' : 'Emotion model missing', engineInfo.classifier);
    if (!engineInfo.classifier) message(engineInfo.reason);
    return engineInfo;
  } catch (error) { engineStatus('Engine unavailable'); throw error; }
  finally { enginePromise = null; }
}

function resizeOverlay() {
  const rect = $('viewport').getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * ratio), h = Math.round(rect.height * ratio);
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  return { width: rect.width, height: rect.height, ratio };
}
function drawBoxes() {
  const { width, height, ratio } = resizeOverlay(), ctx = overlay.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  if (!currentResult) return;
  const scale = Math.min(width / currentResult.width, height / currentResult.height);
  const left = (width - currentResult.width * scale) / 2, top = (height - currentResult.height * scale) / 2;
  const mirrored = mode === 'camera' && $('mirror').checked;
  currentResult.faces.forEach((face, i) => {
    const b = face.box, w = b.width * scale, h = b.height * scale;
    let x = left + b.x * scale; const y = top + b.y * scale;
    if (mirrored) x = width - x - w;
    const p = face.probabilities, best = p ? p.indexOf(Math.max(...p)) : -1;
    const color = best === -1 ? '#74d8ed' : COLORS[best];
    ctx.strokeStyle = color; ctx.lineWidth = i === selectedFace ? 2.5 : 1.5;
    ctx.strokeRect(x, y, w, h);
    const label = p ? `${i + 1} · ${LABELS[best]} ${(100 * p[best]).toFixed(0)}%` : `${i + 1} · Emotion model missing`;
    ctx.font = '600 12px Arial';
    const labelWidth = Math.min(width - 8, ctx.measureText(label).width + 16);
    const lx = Math.max(4, Math.min(x, width - labelWidth - 4)), ly = y > 26 ? y - 24 : Math.min(height - 24, y + h + 2);
    ctx.fillStyle = color; ctx.fillRect(lx, ly, labelWidth, 22); ctx.fillStyle = '#0b1220'; ctx.fillText(label, lx + 8, ly + 15);
  });
}
function renderResult(result) {
  currentResult = result;
  selectedFace = Math.min(selectedFace, Math.max(0, result.faces.length - 1));
  $('face-count').textContent = `${result.totalFaces} ${result.totalFaces === 1 ? 'face' : 'faces'}${result.totalFaces > 6 ? ' · analyzing largest 6' : ''}`;
  $('face-picker').hidden = result.faces.length < 2;
  $('face-select').replaceChildren(...result.faces.map((_, i) => new Option(`Face ${i + 1}${i === 0 ? ' · largest' : ''}`, i)));
  $('face-select').value = String(selectedFace);
  updateScores(result.faces[selectedFace]?.probabilities, result.faces.length ? 'Model not installed' : 'No face found');
  drawBoxes();
  if (mode === 'image') {
    $('frame-stats').textContent = `${canvas.width} × ${canvas.height} · analysis ${Math.round(result.ms)} ms`;
    if (!result.faces.length) message('No face detected. Try a closer, front-facing photo with good lighting.');
    else if (result.classifier) message(`Analysis complete. ${result.faces.length} face${result.faces.length === 1 ? '' : 's'} analyzed.`, 'success');
    else message(engineInfo.reason);
  }
}

async function makeBitmap(source, maxWidth) {
  const width = source.videoWidth || source.width, height = source.videoHeight || source.height;
  if (!width || !height) throw new Error('The input has no image data yet.');
  const scale = Math.min(1, maxWidth / Math.max(width, height));
  return createImageBitmap(source, { resizeWidth: Math.max(1, Math.round(width * scale)), resizeHeight: Math.max(1, Math.round(height * scale)), resizeQuality: 'high' });
}
async function upload(file) {
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['jpg', 'jpeg', 'webp', 'png'].includes(ext) || (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type))) {
    message('Choose a JPG, JPEG, WEBP, or PNG image.', 'error'); return;
  }
  if (file.size > config.maxUploadMB * 1024 * 1024) { message(`Please choose an image smaller than ${config.maxUploadMB} MB.`, 'error'); return; }
  setMode('image'); const version = ++uploadVersion; ++generation;
  imageSource = null; $('analyze').disabled = true; clearResults('Loading image');
  message('Decoding your image…');
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (version !== uploadVersion) return;
    if (image.naturalWidth * image.naturalHeight > 40_000_000) throw new Error('This image is too large. Resize it below 40 megapixels and try again.');
    imageSource = image;
    const scale = Math.min(1, 2400 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale); canvas.height = Math.round(image.naturalHeight * scale);
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    $('file-name').textContent = file.name; $('viewport-empty').hidden = true; canvas.hidden = false; overlay.hidden = false;
    clearResults(); $('analyze').disabled = false;
    $('frame-stats').textContent = `${image.naturalWidth} × ${image.naturalHeight} · image ready`;
    message('Image ready. Select Analyze image to run detection.');
  } catch (error) {
    if (version === uploadVersion) {
      canvas.hidden = true; overlay.hidden = true; $('viewport-empty').hidden = false;
      $('file-name').textContent = 'or drop your image here';
      message(error.message || 'This file could not be decoded. Try another image.', 'error');
    }
  }
  finally { URL.revokeObjectURL(url); }
}
$('image-file').addEventListener('change', e => { upload(e.target.files[0]); e.target.value = ''; });
$('viewport').addEventListener('dragover', e => { e.preventDefault(); $('viewport').classList.add('dragging'); });
$('viewport').addEventListener('dragleave', () => $('viewport').classList.remove('dragging'));
$('viewport').addEventListener('drop', e => { e.preventDefault(); $('viewport').classList.remove('dragging'); upload(e.dataTransfer.files[0]); });
$('analyze').addEventListener('click', async () => {
  if (!imageSource || frameBusy) return;
  const token = generation; frameBusy = true; $('analyze').disabled = true;
  try {
    await ensureEngine(); if (token !== generation) return;
    const bitmap = await makeBitmap(canvas, 1600);
    const result = await request('frame', { bitmap, live: false }, [bitmap]);
    if (token === generation) renderResult(result);
  } catch (error) { if (token === generation) message(error.message, 'error'); }
  finally { frameBusy = false; $('analyze').disabled = !imageSource || mode !== 'image'; }
});

function setMode(next) {
  if (next !== mode) { stopCamera(); ++generation; clearResults(); }
  mode = next;
  ['image', 'camera'].forEach(type => { $(type + '-tab').classList.toggle('selected', mode === type); $(type + '-tab').setAttribute('aria-selected', String(mode === type)); $(type + '-tab').tabIndex = mode === type ? 0 : -1; $(type + '-controls').hidden = mode !== type; });
  canvas.hidden = mode !== 'image' || !imageSource; video.hidden = true;
  overlay.hidden = mode !== 'image' || !imageSource;
  $('viewport-empty').hidden = mode === 'image' && !!imageSource;
  $('empty-title').textContent = mode === 'camera' ? 'Make room for your expression.' : 'Put a face in the frame.';
  $('empty-description').textContent = mode === 'camera' ? 'Start the camera and allow access in your browser.' : 'Choose a photo to explore its expression.';
  $('analyze').disabled = !imageSource || frameBusy;
  $('frame-stats').textContent = mode === 'image' && imageSource ? `${canvas.width} × ${canvas.height} · image ready` : 'No input yet';
}
$('image-tab').onclick = () => setMode('image'); $('camera-tab').onclick = () => setMode('camera');
document.querySelector('.tabs').addEventListener('keydown', e => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) { e.preventDefault(); const type = e.key === 'Home' ? 'image' : e.key === 'End' ? 'camera' : mode === 'image' ? 'camera' : 'image'; setMode(type); $(type + '-tab').focus(); }
});
function stopCamera() {
  ++generation; cancelAnimationFrame(animation); animation = 0;
  stream?.getTracks().forEach(track => track.stop()); stream = null; video.srcObject = null;
  $('camera-badge').hidden = true; video.hidden = true; $('stop-camera').disabled = true;
  $('start-camera').disabled = cameraStarting;
  if (mode === 'camera') { $('viewport-empty').hidden = false; overlay.hidden = true; clearResults(); $('frame-stats').textContent = 'Camera stopped'; }
}
async function cameraLoop(timestamp) {
  if (!stream) return;
  animation = requestAnimationFrame(cameraLoop);
  const interval = 1000 / Number($('inference-rate').value);
  if (frameBusy || timestamp - lastSent < interval || video.currentTime === lastFrameTime || video.readyState < 2) return;
  frameBusy = true; lastSent = timestamp; lastFrameTime = video.currentTime; const token = generation;
  try {
    const bitmap = await makeBitmap(video, 640);
    if (token !== generation || !stream) { bitmap.close(); return; }
    const result = await request('frame', { bitmap, live: true }, [bitmap]);
    if (token !== generation || !stream) return;
    const now = performance.now(), fps = lastResultAt ? 1000 / (now - lastResultAt) : null; lastResultAt = now;
    renderResult(result);
    $('frame-stats').textContent = `${fps ? fps.toFixed(1) : '—'} analyses/sec · ${Math.round(result.ms)} ms · ${video.videoWidth} × ${video.videoHeight}`;
  } catch (error) { if (token === generation) { stopCamera(); message(error.message, 'error'); } }
  finally { frameBusy = false; }
}
$('start-camera').onclick = async () => {
  if (cameraStarting || stream) return;
  setMode('camera'); cameraStarting = true; $('start-camera').disabled = true;
  const token = generation;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs HTTPS or localhost. Open the deployed site in a full browser tab.');
    // Permission is requested directly from the user's click, before model downloads.
    const incoming = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 }, facingMode: 'user' }, audio: false });
    if (token !== generation) { incoming.getTracks().forEach(t => t.stop()); return; }
    stream = incoming; video.srcObject = stream;
    video.hidden = false; overlay.hidden = false; $('viewport-empty').hidden = true;
    $('camera-badge').hidden = false; $('stop-camera').disabled = false;
    await video.play();
    const info = await ensureEngine(); if (token !== generation || !stream) return;
    await request('reset'); if (token !== generation || !stream) return;
    message(info.classifier ? 'Camera running. Expression scores update automatically; your video stays on this device.' : info.reason, info.classifier ? 'success' : '');
    lastSent = 0; lastFrameTime = -1; lastResultAt = 0;
    animation = requestAnimationFrame(cameraLoop);
    stream.getVideoTracks()[0].onended = () => { stopCamera(); message('Camera access ended. Start it again to continue.'); };
  } catch (error) {
    if (token === generation) {
      stopCamera();
      const explanation = error.name === 'NotAllowedError' ? 'Camera access was blocked. Allow camera access in your browser, or open the full page outside the embedded Space.' : error.name === 'NotFoundError' ? 'No camera was found on this device. You can upload an image instead.' : error.name === 'NotReadableError' ? 'The camera is in use by another application. Close it there and retry.' : error.message;
      message(explanation, 'error');
    }
  } finally { cameraStarting = false; $('start-camera').disabled = !!stream; }
};
$('stop-camera').onclick = () => { stopCamera(); message('Camera stopped. All camera tracks have been released.'); };
$('mirror').addEventListener('change', () => { video.classList.toggle('mirrored', $('mirror').checked); drawBoxes(); });
video.classList.toggle('mirrored', $('mirror').checked);
$('inference-rate').oninput = () => $('rate-label').textContent = `${$('inference-rate').value} / sec target`;
$('face-select').onchange = () => { selectedFace = Number($('face-select').value); updateScores(currentResult?.faces[selectedFace]?.probabilities, 'Model not installed'); drawBoxes(); };
$('model-select').onchange = async () => {
  const wasLive = !!stream; stopCamera(); clearResults(); engineStatus('Engine idle');
  message(wasLive ? 'Model changed. Start the camera again to use it.' : 'Model changed. Analyze an image or start the camera to load it.');
};
new ResizeObserver(drawBoxes).observe($('viewport'));
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) { stopCamera(); message('Camera stopped while the page was hidden. Start it again to continue.'); } });
window.addEventListener('pagehide', () => { stopCamera(); failWorker(new Error('Page closed.')); });
if (window.self !== window.top) { $('open-direct').hidden = false; $('open-direct').href = location.href; }

const menu = $('navigation');
$('menu-toggle').onclick = () => { const open = menu.classList.toggle('open'); $('menu-toggle').setAttribute('aria-expanded', String(open)); };
menu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => { menu.classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded', 'false'); }));
const sectionObserver = new IntersectionObserver(entries => { for (const entry of entries) if (entry.isIntersecting) { menu.querySelectorAll('a').forEach(a => a.classList.toggle('active', a.hash === `#${entry.target.id}`)); } }, { rootMargin: '-20% 0px -55% 0px', threshold: 0 });
document.querySelectorAll('main section[id]').forEach(s => sectionObserver.observe(s));

let benchmarks = [];
async function loadBenchmarks() {
  try {
    const response = await fetch('./benchmarks.json'); if (!response.ok) throw new Error('Benchmark data unavailable.');
    benchmarks = await response.json(); renderBenchmarks();
  } catch (error) { $('benchmark-note').textContent = error.message; }
}
function renderBenchmarks() {
  $('benchmark-cards').replaceChildren(); $('accuracy-bars').replaceChildren();
  benchmarks.forEach((model, i) => {
    const accuracy = model.reportAccuracy, text = (100 * accuracy).toFixed(2);
    const card = document.createElement('article'); card.className = `benchmark-card ${i === 0 ? 'best' : ''}`;
    card.innerHTML = `${i === 0 ? '<span class="best-badge">BEST ACCURACY</span>' : ''}<h3>${model.name}</h3><div class="metric">${text}<small>%</small></div><p>Recorded test accuracy</p><span class="tiny">${model.kind}</span>`;
    $('benchmark-cards').append(card);
    const bar = document.createElement('div'); bar.className = 'benchmark-bar';
    bar.innerHTML = `<div class="bar-label"><span>${model.name}</span><span>${text}%</span></div><div class="bar-track"><div class="bar-fill" style="width:${accuracy * 100}%"></div></div>`;
    $('accuracy-bars').append(bar);
  });
  $('benchmark-note').textContent = 'Source: submitted report, Table 4.2.1 (page 60). Original evaluation figures are from the project repository.';
}
loadBenchmarks();
function plotPath() {
  const model = $('plot-model').value, type = $('plot-type').value;
  if (type === 'roc') return `./assets/results/${model === 'resnet' ? 'roc_curce_resnet' : 'roc_curve_' + model}.png`;
  return `./assets/results/${type}_${model}.png`;
}
function updatePlot() {
  const label = `${$('plot-model').selectedOptions[0].text} · ${$('plot-type').selectedOptions[0].text}`;
  $('evaluation-image').src = plotPath(); $('evaluation-image').alt = label;
  $('plot-dialog-title').textContent = label;
}
$('plot-model').onchange = updatePlot; $('plot-type').onchange = updatePlot;
$('open-plot').onclick = () => { updatePlot(); $('large-plot').src = plotPath(); $('large-plot').alt = $('evaluation-image').alt; $('plot-dialog').showModal(); };
$('close-plot').onclick = () => $('plot-dialog').close();
$('plot-dialog').addEventListener('click', e => { if (e.target === $('plot-dialog')) { const r = $('plot-dialog').getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) $('plot-dialog').close(); } });

const formEndpoint = config.formspreeEndpoint;
const validEndpoint = /^https:\/\/formspree\.io\/f\/[a-zA-Z0-9]+$/.test(formEndpoint);
if (!validEndpoint) { $('send-message').disabled = true; $('contact-status').textContent = 'The contact form is not connected yet. Please check back soon.'; }
$('contact-form').addEventListener('submit', async e => {
  e.preventDefault(); const form = e.target;
  if (!validEndpoint || !form.reportValidity()) return;
  const button = $('send-message'); button.disabled = true; button.textContent = 'Sending…';
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(formEndpoint, { method: 'POST', body: new FormData(form), headers: { Accept: 'application/json' }, signal: controller.signal });
    if (!response.ok) throw new Error(response.status === 429 ? 'Too many messages. Please try again later.' : 'Your message could not be sent. Please try again.');
    form.reset(); $('contact-status').textContent = 'Message sent. Thank you for getting in touch.';
  } catch (error) { $('contact-status').textContent = error.name === 'AbortError' ? 'The request timed out. Please try again.' : error.message; }
  finally { clearTimeout(timeout); button.disabled = false; button.textContent = 'Send message'; }
});
