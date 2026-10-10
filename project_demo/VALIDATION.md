# Verification status

Original model integration was verified in Chromium 151 on 9 October 2026. The deployment repair below was validated on 10 October 2026.

## Trained model integration

All four original Keras checkpoints were loaded without retraining and exported to float32 ONNX. Each export checks numerical agreement on nine inputs: zero, one, six seeded normalized RGB tensors, and a real face crop. All comparisons passed with absolute tolerance 0.0001 and relative tolerance 0.001. Matching metadata records input/output names, class order, normalization, source checkpoint, and SHA-256 integrity.

| Model | Source checkpoint | ONNX size | Parity inputs | Maximum absolute probability difference |
|---|---|---:|---:|---:|
| Custom CNN | `CNN_best.keras` | 5.25 MB | 9 | 0.000000507 |
| ResNet50 | `resnet50_best.keras` | 96.31 MB | 9 | 0.000000238 |
| MobileNetV2 | `mobilenetv2_best.keras` | 10.36 MB | 9 | 0.000003159 |
| VGG16 | `vgg16_best.keras` | 60.46 MB | 9 | 0.000000894 |

The class order is Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. Input is NHWC 1×48×48×3 RGB divided by 255, matching the original laptop preprocessing. Custom CNN is the default, matching `main.py`. Original `.keras` files and optimizer state are not needed on the static site.

## Original browser checks (before the detector repair)

- All four trained models loaded through local ONNX Runtime Web WASM and returned seven valid probabilities for an uploaded face. Displayed probabilities summed to 100% within rounding tolerance. The test image was classified as Happy by all four models; this is a functional check, not a benchmark.
- JPG, JPEG, PNG, and WEBP uploads displayed and produced trained CNN emotion predictions. Upload opens the native chooser; inference begins when Analyze image is pressed.
- The live camera fixture produced a face box, Happy label, and seven scores using the trained CNN. Observed rate: approximately 8 analyses/sec with 21 ms analysis in this test environment. The fixture is prerecorded; laptop-specific performance and behavior on a physical camera can differ.
- Camera preview and worker inference run independently, with only one analysis in flight. Stop clears `video.srcObject` and releases camera tracks.
- Restored the ready status when a previously loaded classifier is reused after changing the selection.
- No uncaught browser JavaScript errors or failed HTTP responses occurred during trained-model checks.
- Report cards and bars show only 66.00%, 61.23%, 60.34%, and 60.13%; there is no benchmark source picker.
- 390px mobile layout has no horizontal overflow.

Earlier interface checks also verified multiple nearby faces, mirroring, all 16 evaluation figures, plot enlargement, mobile navigation, stale-upload protection, and contact success handling through an intercepted request. No contact message was sent externally.

## Deployment repair — 10 October 2026

- Reproduced Render's worker-start failure. `vendor/ort/ort.wasm.min.js`, ONNX Runtime WASM, and MediaPipe WASM returned HTTP 404.
- Reproduced Hugging Face's separate failure using a public test portrait: the emotion model loaded, but face detection failed with `Cannot read properties of undefined (reading activeTexture)`.
- Replaced WebGL-dependent MediaPipe detection with OpenCV YuNet executed through the bundled ONNX Runtime Web 1.20.1 WASM backend. Inference now uses only WASM and 2D canvases.
- Verified the bundled YuNet binary against its official Git LFS SHA-256 and ran ONNX structural validation.
- Tested the actual bundled browser WASM runtime under Node: one portrait yielded one face; a two-portrait collage yielded two faces; a flat image yielded no faces. Decoding includes letterbox mapping and non-maximum suppression.
- All four original emotion ONNX files and metadata remain byte-identical to the previous complete package. All four returned seven valid probabilities from the detected portrait crop. Predictions were Neutral for CNN/MobileNetV2/ResNet50 and Angry for VGG16; this is a functional check, not a claim of emotion ground truth.
- Ten repeated portrait frames passed detector-plus-CNN inference. Mean runtime was approximately 65 ms per frame in this test environment, excluding image capture, canvas resizing, rendering, and network downloads. This does not establish a laptop camera frame rate.
- The new detector implementation has not been deployed to either live account. A physical webcam and the revised UI on the user's deployments still require verification after uploading.
- New worker/app URL versions avoid stale JavaScript after redeployment. Missing runtime files now produce a specific error.

## Deployment and remaining configuration

- This is a complete static package; keep `config.js`, `core.js`, `vendor/`, `models/`, and `assets/` together. The seven-file update patch alone cannot run.
- The public Formspree endpoint is configured in `config.js`. Recipient settings remain in Formspree. Simulated submission passed; no real contact message was sent.
- The package is ready for local serving, Render Static Sites, and Hugging Face Static Spaces. It has not been pushed or published to your accounts.
- Large models require larger downloads and may lower the analysis rate. Custom CNN is recommended for live use.
- The browser face detector is OpenCV YuNet on WASM; the original laptop script uses an OpenCV Haar cascade. Face crops may differ, so identical labels on every scene are not guaranteed.
- Conversion agreement does not establish new FER2013 test accuracy. Benchmarks remain the submitted report values; no accuracy evaluation was rerun.

## Documentation references

- Render Static Sites: https://render.com/docs/static-sites
- Hugging Face Static HTML Spaces: https://huggingface.co/docs/hub/en/spaces-sdks-static
- Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- GitHub file limits: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- ONNX Runtime Web deployment: https://onnxruntime.ai/docs/tutorials/web/deploy.html
- OpenCV YuNet face detector: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Formspree HTML forms: https://formspree.io/html/
