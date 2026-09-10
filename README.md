# Cone & Rice Counter

An Android app (React Native + Expo) that photographs cones or rice grains
and returns a numbered, labeled count of each ("Rice #1", "Rice #2", "Cone
#1"...) plus a summary ("Total Cones: 3 | Total Rice Grains: 142").

## 1. Project Architecture

**Frontend: React Native + Expo.** Chosen over native Kotlin/Compose because
`expo-camera` + `expo-image-manipulator` give production-quality capture,
permission handling, and image preprocessing with a fraction of the
boilerplate of a CameraX pipeline, and `react-native-svg` renders crisp,
resolution-independent bounding boxes/labels over the photo. It still
compiles to a real installable Android app (and iOS, for free, if you ever
want it).

**Computer vision: dual pipeline, processed on a backend, not on-device.**
Rice grains and cones need fundamentally different techniques:

| | Rice grains | Cones |
|---|---|---|
| Object scale | Tiny, many per photo | Large, few per photo |
| Technique | Classical OpenCV: Otsu threshold + per-blob distance-transform/watershed splitting | YOLOv8-nano (custom-trained) with a classical HSV/shape fallback |
| Hard part | Touching/overlapping grains merging into one blob | No off-the-shelf model class exists for arbitrary cones -- needs a small fine-tuned model |

Both pipelines live in `backend/app/` and share the same output contract
(`x, y, width, height, confidence` per detected object), so the mobile app
and the API contract don't care which pipeline produced a box.

**Why a backend (FastAPI) instead of on-device TFLite/OpenCV:**

- **Accuracy over speed** was the explicit priority for this project. A
  server can run full OpenCV (watershed, connected components) and a real
  Ultralytics YOLOv8 model without the size/precision compromises of
  mobile-quantized TFLite models or hand-rolled OpenCV-Android JNI bindings.
- **Iteration speed.** Tuning threshold ratios, retraining the cone model,
  or swapping in a better algorithm is a server deploy, not an app store
  release.
- **Trade-off you're accepting:** requires network connectivity and adds
  network latency. If you need fully offline operation later, the rice
  pipeline (pure OpenCV) ports to OpenCV4Android with minimal changes, and
  the cone model can be exported to TFLite (`yolo export format=tflite`) --
  the mobile app's contract (upload photo -> get back a list of boxes)
  wouldn't need to change, only where `analyzeImage()` in
  `mobile/src/services/api.ts` sends the image.

**Why the app asks "Rice or Cones?" before opening the camera:** the two
pipelines assume very different shot framing (macro close-up for grains vs.
standing back for cones-sized objects). A single unscoped photo that tries
to contain both at counting-usable resolution isn't a realistic capture, so
the user picks a mode first and the backend only runs the relevant pipeline
(a `mode=auto` option exists on the API for scenes where you know both are
at a compatible scale).

### Request flow

```
Capture (CameraScreen)
   -> confirm/retake preview
   -> resize/compress (expo-image-manipulator)
   -> POST /detect (multipart image + mode)   [ProcessingScreen: loading spinner]
   -> FastAPI: blur check -> rice and/or cone detector -> numbered detections
   -> JSON { rice_count, cone_count, detections[] }
   -> ResultsScreen: SVG overlay drawn from detections, scaled to displayed image size
```

## 2. Setup Instructions

### Backend (FastAPI + OpenCV)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the API (reachable on your LAN at http://<your-machine-ip>:8000)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run the test suite (generates a synthetic test image with known
# rice/cone counts, so tests don't depend on a checked-in photo)
pip install pytest httpx
python -m pytest tests/ -v
```

Interactive API docs: `http://localhost:8000/docs`.

**Optional -- enabling the real YOLOv8-nano cone detector:** without a
trained model, cone detection falls back to classical HSV color + shape
matching (tuned for a bright orange cone by default -- see
`backend/app/cone_detector.py`'s `HSV_LOWER`/`HSV_UPPER`, or the "Calibrating
the cone color range" section below). For production-grade accuracy across
lighting/backgrounds, train a small custom model:

```bash
pip install ultralytics
# Label ~150-300 photos of your cone in Roboflow or CVAT, export as YOLOv8 format
yolo detect train model=yolov8n.pt data=cones.yaml epochs=100 imgsz=640
cp runs/detect/train/weights/best.pt backend/models/cone_yolov8n.pt
```

Once `backend/models/cone_yolov8n.pt` exists, `cone_detector.py` uses it
automatically -- no code changes needed.

**Calibrating the cone color range (classical fallback):** sample a few
pixels from a real photo of your cone:

```python
import cv2
img = cv2.imread("your_cone_photo.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[y, x])  # pick a pixel clearly inside the cone
```

Set `HSV_LOWER`/`HSV_UPPER` in `cone_detector.py` to bracket the hue/sat/val
you observe, with some margin.

### Mobile app (Expo / React Native)

```bash
cd mobile
npm install

# Point the app at your backend (see src/config.ts for details):
# - Android emulator: http://10.0.2.2:8000 (already the default)
# - Physical device via Expo Go: http://<your-machine-LAN-IP>:8000
# - Tunnel: use `expo start --tunnel` and an https URL (e.g. ngrok) for the backend too

npx expo start
```

Then either:
- Press `a` to launch in a connected Android emulator, or
- Scan the QR code with **Expo Go** (Play Store) on a physical Android phone.

**Building an installable `.apk`** (no Mac/Android Studio required):

```bash
npm install -g eas-cli
eas login
eas build -p android --profile preview
```

This builds in Expo's cloud and gives you a downloadable `.apk` link.
Alternatively, `npx expo prebuild` generates a native `android/` project you
can open in Android Studio and build/run locally.

## 3. Computer Vision Code

- `backend/app/rice_detector.py` -- Otsu threshold -> morphological opening
  -> **per-connected-component** distance-transform + watershed splitting.
  Working blob-by-blob (rather than one global threshold) is what makes this
  robust: a single oversized object in frame can't swamp the relative
  threshold and hide every real grain peak, and isolated single grains skip
  watershed entirely so they don't get spuriously split by their own
  curvature. Includes a foreground-coverage sanity gate so a blank/noisy
  photo returns "no detections" instead of miscounting image noise as
  grains.
- `backend/app/cone_detector.py` -- YOLOv8n custom-model path (when
  `backend/models/cone_yolov8n.pt` exists) with an automatic classical-CV
  fallback (HSV color threshold + convexity/aspect-ratio shape gate) so the
  endpoint works before you've trained anything.
- `backend/app/blur_check.py` -- Laplacian-variance sharpness gate, run
  before either detector so a shaky/out-of-focus photo gets a clear
  "retake" message instead of a garbage count.
- `backend/app/main.py` -- FastAPI endpoints wiring the above together and
  assigning the 1-based numeric labels ("Rice #1", "Cone #2", ...).
- `backend/app/draw_annotations.py` + `POST /detect/annotated` -- optional
  server-rendered debug image with numbered boxes drawn on it (useful for
  eyeballing detector quality without the mobile app).
- `backend/tests/generate_test_image.py` + `backend/tests/test_detectors.py`
  -- synthetic image with known counts, used to verify both detectors,
  the blur gate, and the full `/detect` endpoint without needing a real photo.

## 4. Frontend Code

All under `mobile/`, TypeScript throughout:

- `App.tsx` -- entry point, mounts the navigator.
- `src/navigation/index.tsx` -- 4-screen stack: Home -> Camera -> Processing -> Results.
- `src/screens/HomeScreen.tsx` -- mode picker ("Count Rice Grains" / "Count Cones").
- `src/screens/CameraScreen.tsx` -- camera permission states (request /
  denied-with-settings-link), live preview via `expo-camera`'s `CameraView`,
  capture -> confirm/retake, then downscale+compress via
  `expo-image-manipulator` before upload.
- `src/screens/ProcessingScreen.tsx` -- loading spinner while
  `services/api.ts` uploads the photo; renders an inline error + "Retake
  Photo" action on network failure, server error, or a `blurry` API status.
- `src/screens/ResultsScreen.tsx` -- renders the photo at screen width,
  overlays `DetectionOverlay` (SVG boxes + numeric labels scaled from the
  original image's pixel coordinates to however the photo is actually being
  displayed), shows a friendly empty state when `status === "no_detections"`,
  and a `SummaryBar` with the final counts.
- `src/components/DetectionOverlay.tsx` -- the scaling math that keeps boxes
  aligned to the photo regardless of screen size.
- `src/components/SummaryBar.tsx` -- the "Total Cones: X | Total Rice
  Grains: Y" bar.
- `src/services/api.ts` -- multipart upload + typed response, normalizing
  network/timeout/server errors into one `ApiError` type.
- `src/types.ts` -- shared types, including the navigator's typed param list.

### Error handling summary

| Scenario | Where handled | User sees |
|---|---|---|
| Camera permission denied | `CameraScreen` | "Enable it in Settings" button (or re-prompt if still askable) |
| Blurry photo | `blur_check.py` -> `ProcessingScreen` | "Couldn't analyze photo" + Retake |
| No objects found | `main.py` (`no_detections`) -> `ResultsScreen` | Friendly empty state with lighting/framing tips, counts shown as 0 |
| Network/server error | `api.ts` -> `ProcessingScreen` | Error message + Retake |
| Decoded image invalid | `main.py` (`error` status) | Error message + Retake |

## Known limitations & tuning notes

- The rice watershed splitter recovers most, but not all, of heavily
  **overlapping** grains (classical CV's known limitation vs. a trained
  segmentation model) -- for best accuracy, spread grains in a single layer
  with minimal overlap. The in-app hint text on the camera screen reflects this.
- The classical cone fallback assumes a single, fairly uniform cone color;
  it will need its HSV range recalibrated for a different color/material, or
  replaced by the trained YOLOv8n path for robustness across lighting and cone types.
- All area-based thresholds in `rice_detector.py`/`cone_detector.py` are
  expressed as *ratios of image area* so they scale across camera
  resolutions; they still assume a roughly consistent shooting distance
  (see in-app framing hints per mode).
