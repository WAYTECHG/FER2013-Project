---
title: FER Expression Lab
emoji: 🙂
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# FER Expression Lab

**[Open the live demo on Hugging Face Spaces →](https://huggingface.co/spaces/Wizzas/Face-Expression-Recognition)**

A static interface for [WAYTECHG/FER2013-Project](https://github.com/WAYTECHG/FER2013-Project). It includes Home, About, Live Demo, Model Benchmarks, and Contact Us, with image uploads and webcam face detection. Your trained emotion models run locally in the visitor's browser through ONNX Runtime Web.

**Ready to run:** this complete package includes browser exports of your supplied `CNN_best.keras`, `mobilenetv2_best.keras`, `resnet50_best.keras`, and `vgg16_best.keras` checkpoints. All four provide the seven emotion probabilities. Custom CNN is the default, matching the classifier used by your original laptop demo. No model conversion or Python ML libraries are needed to run or deploy it. The contact form is connected to your public Formspree endpoint in `config.js`.

## Run locally on Windows

Extract the complete ZIP first and copy the entire `project_demo` folder into your project. Keep `core.js`, `face-detector.js`, `config.js`, `vendor/`, `models/`, and `assets/` together with the other files; a patch archive by itself is not a complete app. Open PowerShell in `project_demo` and run:

```powershell
py -m http.server 8000
```

Open **http://localhost:8000**. Keep that terminal running. Do not open `index.html` by double-clicking it: browser workers and WASM require an HTTP server.

## Re-export models after changing checkpoints (optional)

All four models are already exported and connected in this package. Skip this section for normal use. If you retrain a checkpoint, repeat the export to update the corresponding ONNX model and metadata together. The source `main.py` uses `models/CNN_best.keras`.

Use a separate Python 3.11 environment for export. These commands assume `project_demo` is inside the original project, alongside its `models` folder:

```powershell
py -3.11 -m venv .venv-export
.\.venv-export\Scripts\python.exe -m pip install -r tools\export-requirements.txt
.\.venv-export\Scripts\python.exe tools\export_model.py --checkpoint "..\models\CNN_best.keras" --id cnn
```

This creates **`models/cnn.onnx` and `models/cnn.json`**. Upload both with the interface. The export checks Keras-versus-ONNX probability parity on eight normalized inputs before publishing the pair. Optional verification on actual cropped test faces:

```powershell
.\.venv-export\Scripts\python.exe tools\export_model.py --checkpoint "..\models\CNN_best.keras" --id cnn --verify-images "..\fer2013\test"
```

The verification uses at most 32 images and checks conversion agreement; it does not rerun or assert FER2013 test accuracy. Verify the deployed model on representative real faces too. For other trained models:

```powershell
.\.venv-export\Scripts\python.exe tools\export_model.py --checkpoint "..\models\resnet50_best.keras" --id resnet
.\.venv-export\Scripts\python.exe tools\export_model.py --checkpoint "..\models\mobilenetv2_best.keras" --id mobilenet
.\.venv-export\Scripts\python.exe tools\export_model.py --checkpoint "..\models\vgg16_best.keras" --id vgg16
```

Adjust checkpoint paths to where the files actually exist. The notebooks also refer to Google Drive paths for some checkpoints.

All classifiers must use **48×48 RGB**, values divided by 255, NHWC layout, a seven-class softmax, and this exact class order:

| Index | Expression |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

Do not add ImageNet-specific preprocessing: the source training generators use RGB / 255 for all four models. Browser inference uses area-weighted uint8 face resizing to follow the source `main.py`. The browser uses OpenCV YuNet through ONNX Runtime WASM to locate faces; the original desktop script uses an OpenCV Haar cascade. This changes face localization, so live results are not a new measurement of the recorded FER2013 benchmark.

## Connect Formspree without exposing your email

1. Create a form at https://formspree.io and configure your recipient email inside its dashboard.
2. Copy its public endpoint, such as `https://formspree.io/f/abcdefgh`.
3. In **`config.js`**, set:

```javascript
formspreeEndpoint: 'https://formspree.io/f/abcdefgh',
```

No recipient email or secret key belongs in the frontend. Visitors fill their own name, email, subject, and message. The form sends these fields to Formspree and handles success, timeout, and failure states. Until a valid endpoint is supplied, Send message is disabled. Complete Formspree's email verification and send one test message yourself after deployment.

## Deploy to Render

Commit the entire **`project_demo` folder** to your existing GitHub repository. The large ResNet50 and VGG16 files exceed GitHub's browser upload limit; use Git from your local repository folder:

```powershell
git add project_demo
git commit -m "Connect trained FER2013 models to static demo"
git push
```

The largest exported file is about 96 MB, so the individual ONNX files fit within GitHub's normal Git file limit. Do not commit the original `.keras` files or the ZIP as frontend assets. In Render choose **New → Static Site**, connect the repo, and use:

| Setting | Value |
|---|---|
| Root directory | `project_demo` |
| Build command | `echo Static FER demo ready` |
| Publish directory | `.` |

If the interface files are at the repository root, leave Root directory blank. No Python backend, Torch, TensorFlow server, Docker, or runtime environment variables are required. For the existing `WAYTECHG/FER2013-Project` repository, deploy branch `main` with the settings above.

## Deploy to Hugging Face Spaces

Create a Space with the **Static** SDK. Upload the **contents** of `project_demo` directly to the Space repository root, including this `README.md`, `vendor/`, `models/`, and `assets/`. Its metadata already specifies:

```yaml
sdk: static
app_file: index.html
```

Do not leave the files nested under `project_demo` unless you also set `app_file: project_demo/index.html`. Keep the README metadata at the Space repository root. From PowerShell inside `project_demo`, upload the complete folder using the Hugging Face CLI, which handles the large model files:

```powershell
py -m pip install --upgrade huggingface_hub
hf auth login
hf upload Wizzas/Face-Expression-Recognition . . --repo-type space
```

The command above updates this project's existing Static Space. Run it from inside `project_demo` so `index.html`, this README, and the bundled folders are uploaded directly to the Space root.

If camera permission is blocked in an embedded Space, use **Open full page for camera access** in the demo. Open the direct `https://<your-space-subdomain>.hf.space` page and allow camera access. Render and direct HF Space URLs must use HTTPS; localhost works for local testing.

## Live performance and behavior

- Camera capture requests up to 640×480 at 30 FPS, subject to device/browser capabilities.
- Camera playback is independent of inference. Detection and classification run in one background worker.
- Only one frame is processed at a time; frames are skipped rather than queued. Live analysis defaults to a target of 8 analyses/sec, adjustable from 2 to 12. The UI reports the actual analysis rate.
- Up to six largest detected faces are analyzed. The side panel can inspect any of those faces. Face boxes use matching image coordinates, including a mirrored camera preview.
- Live probabilities use an exponential average matched by face overlap. Uploaded images show unsmoothed probabilities.
- Video stops and camera tracks are released on Stop, input-source changes, model changes, page hide, and leaving the page.
- JPG, JPEG, WEBP, and PNG are decoded by the browser. Maximum upload: 20 MB, 40 megapixels. Preview is capped at 2400 pixels on its long edge; image analysis uses up to 1600 pixels. Camera analysis uses up to 640 pixels. Transparency is composited on white.
- WASM uses one thread to work without cross-origin isolation headers on both hosts. Large models, especially VGG16/ResNet50, may reduce the analysis rate on slower devices. Start with the custom CNN.
- Inference libraries, WASM files, and face detector are bundled locally. Google Fonts is optional, with local font fallbacks. Images/video are never submitted to an inference server. Only contact messages are sent to Formspree when submitted.

## Benchmark results

All benchmark cards and bars use the submitted report, Table 4.2.1 (page 60). There is no source selector.

| Model | Report test accuracy |
|---|---:|
| Custom CNN | 66.00% |
| ResNet50 | 61.23% |
| MobileNetV2 | 60.34% |
| VGG16 | 60.13% |

Original confusion matrices, class reports, ROC curves, and Grad-CAM figures are copied from `result_images/`. They are saved evaluation figures, not live image or webcam results.

## Troubleshooting

- **Emotion model missing / Model not installed:** the deployed copy is incomplete. Restore all four `.onnx` files and their matching `.json` files from this complete package into `models/`.
- **Upload and camera tabs do nothing:** confirm `core.js` and `config.js` exist next to `index.html`; copy the whole package and run it over HTTP/HTTPS, then refresh with Ctrl + F5.
- **Model integrity error:** deploy matching metadata and ONNX bytes from the same export.
- **Engine unavailable:** confirm `vendor/ort/`, `face-detector.js`, and `models/face_detector.onnx` were uploaded. Serve via HTTP/HTTPS. Use a modern Chrome, Edge, Firefox, or Safari with WebAssembly, workers, `createImageBitmap`, and OffscreenCanvas support.
- **Contact form unavailable:** configure the Formspree endpoint in `config.js`.
- **Camera blocked:** allow camera permission or open the full page outside the embedded Space. Close applications already using the camera.
- **Old files after redeployment:** refresh without cache. Do not rename bundled WASM files or change library versions independently.

## License

The original MIT license is preserved in `LICENSE`. See `THIRD_PARTY_NOTICES.md` for runtime libraries and model assets.
