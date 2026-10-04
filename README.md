# Body Swap for Windows

Touch-friendly **Windows x64** app for the **ASUS ROG Ally X** (Windows 11, Ryzen Z1 Extreme, Radeon 780M, 24 GB). It replaces the **whole visible person** in a video — body, clothes, hair, and face together — with a person from a photo or a short clip.

This is **not** a face-only swap, and it is **not** a cloud / diffusion body generator. There is no full neural body-swap network in this build (those models are tens of gigabytes and do not fit an honest offline Ally X install). What ships is an offline ONNX pipeline you can actually run:

1. **Robust Video Matting** cuts the person out of the frame.
2. **MoveNet Lightning** finds the body joints.
3. A **pose bone warp** repaints the source person’s colours along those joints so the body follows the target pose, then feathers it back over the background.
4. The MP4 is encoded at the original resolution up to 1080p (never upscaled) with a high bitrate, or CRF 16 where libx264 exists, so it does not come out soft and pixelated on purpose.

| | This app |
|---|---|
| Package | Body Swap **1.0.0** (portable zip) |
| Acceleration | **ONNX Runtime DirectML** on the Radeon 780M, automatic **CPU fallback** |
| GPU safety | DirectML is probed in a **child process**. A 780M driver crash does not kill the window. The UI then says GPU failed and stays on CPU. |
| Input | Target **video** + source **photo or short clip** |
| Default resolution | Up to **1080p** short side |
| Default length cap | **90 seconds** per run |
| Output | `%USERPROFILE%\Videos\BodySwap` |
| Models | **~24 MB** total, first launch only |

New repo. It does not replace or modify gif-face-swap-windows, face-fusion-windows, or face-swap-video-windows.

## Download

Grab **`BodySwap-1.0.0-win64.zip`** from the [Releases](https://github.com/vanukrishnans-source/body-swap-windows/releases) page. No Python install.

### SmartScreen

The build is not code-signed. Windows will say “Windows protected your PC”:

1. Click **More info**
2. Click **Run anyway**

```powershell
Get-FileHash .\BodySwap-1.0.0-win64.zip -Algorithm SHA256
```

### Run

1. Unzip anywhere (for example `C:\Games\BodySwap\`).
2. Double-click **`BodySwap.exe`**.
3. First launch downloads **~24 MB**:
   - `movenet_singlepose_lightning.onnx` — 9.4 MB
   - `rvm_mobilenetv3_fp32.onnx` — 15.0 MB
4. Pick the **target video** (person to replace) and the **source body** (photo or short clip).
5. Tap **Swap body**. The MP4 lands in `Videos\BodySwap`.

Models stay in `%LOCALAPPDATA%\BodySwap\models`. After that download the app is offline.

Large tap targets, dark UI, meant for the 7″ 1080p screen at 150% scaling. Mouse and touch both work.

## What “body swap” means here

The source person’s **visible surface** (shirt, hair, skin, jeans) is moved onto the target person’s **pose and place in the frame**. The background of the target video stays. One main person per clip.

It is a geometric transfer, not a generated person. Expect:

- A readable clothing / hair / body change when both people are clearly visible and the camera angle is similar (both from behind, or both facing the camera).
- Stretching, smeared edges, or a cut-out look when the pose or the angle is very different (front onto side profile is a bad pair).
- **Limbs that are not in the source stay as the target.** A thigh-up photo will not invent feet. The original shoes and lower legs remain instead of a white hole.
- Occlusion (bags, crossed arms, a second person) is imperfect.
- The face follows the head warp. It is not an InsightFace identity swap. Do not expect Face Fusion quality on the face.
- Tiny people in a wide shot go soft, because MoveNet only has a few dozen pixels to work with.
- Short clips as the source pick the closest pose per frame (up to 24 frames). A still photo is more stable and less flickery.

Speed on the Ally X was **not measured on that device**. On a CPU dev box a 720p frame is well under a second. DirectML on the 780M should be faster than CPU; if the GPU probe fails you still get a video, just slower. A 10 second 720p clip is a reasonable first test. The in-app cap is 90 seconds.

## Licence

- App code: MIT (`LICENSE`).
- MoveNet: Apache-2.0.
- RVM weights: **GPL-3.0** (downloaded, not bundled). See `THIRD_PARTY.md`.
- FFmpeg in the zip: LGPL build.
- Only swap people who agreed to it. This is not a tool for impersonation.

## Build from source

Windows x64, or the GitHub Actions workflow.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt
powershell -File packaging\download_ffmpeg.ps1
$env:BSWAP_MODELS = "$PWD\.models-cache"
python scripts\prefetch_models.py
python scripts\fetch_test_clips.py clips
python -m bswap --selftest --video clips\garden.mp4 --source clips\braids.mp4 --models $env:BSWAP_MODELS --device cpu --start 4 --length 1.2 --fps 8 --max-short 480 --out selftest-out
pyinstaller BodySwap.spec --noconfirm
powershell -File packaging\make_zip.ps1
```
