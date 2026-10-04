# Third-party notices

## App code
MIT. See LICENSE.

## Models (downloaded on first launch, not inside the zip)
- **MoveNet SinglePose Lightning** ONNX (~9.4 MB). Google MoveNet, Apache-2.0. ONNX conversion published by Xenova on Hugging Face.
- **Robust Video Matting, MobileNetV3** ONNX (~15.0 MB). Peter Lin, GPL-3.0. https://github.com/PeterL1n/RobustVideoMatting  
  The weights are GPL-3.0. This app's own source is MIT and does not bundle the weights. If you combine the app with those weights, the matting piece follows GPL-3.0.

## Runtime
- ONNX Runtime / ONNX Runtime DirectML — MIT
- OpenCV — Apache-2.0
- NumPy — BSD
- PySide6 / Qt — LGPL-3.0
- FFmpeg LGPL shared build from BtbN/FFmpeg-Builds (bundled in the Windows zip). Encoders: h264_amf, h264_mf, libopenh264 as available. No GPL libx264 in the Windows build.

## Test clips
Mixkit stock videos (people walking), used only for development and CI. Not shipped in the zip.
