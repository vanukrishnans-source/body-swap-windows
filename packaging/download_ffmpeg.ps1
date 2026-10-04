# Download the BtbN LGPL FFmpeg win64 shared build and place ffmpeg.exe / ffprobe.exe under ffmpeg/.
$ErrorActionPreference = "Stop"
$tag = "latest"
$asset = "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$tag/$asset"
$dest = Join-Path $PSScriptRoot "..\ffmpeg"
$zip = Join-Path $env:TEMP $asset
Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
$extract = Join-Path $env:TEMP "ffmpeg-lgpl"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
Expand-Archive -Path $zip -DestinationPath $extract
$bin = Get-ChildItem -Path $extract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$root = $bin.Directory.Parent.FullName
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item (Join-Path $bin.Directory.FullName "ffmpeg.exe") (Join-Path $dest "ffmpeg.exe") -Force
Copy-Item (Join-Path $bin.Directory.FullName "ffprobe.exe") (Join-Path $dest "ffprobe.exe") -Force
# Shared build needs its DLLs next to the exe.
Get-ChildItem (Join-Path $root "bin") -Filter *.dll | Copy-Item -Destination $dest -Force
# License files from the build.
Get-ChildItem $root -File | Where-Object { $_.Name -match "LICENSE|COPYING|README" } | Copy-Item -Destination $dest -Force
Write-Host "FFmpeg ready at $dest"
& (Join-Path $dest "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
& (Join-Path $dest "ffmpeg.exe") -hide_banner -encoders 2>&1 | Select-String "h264_amf|h264_mf|libopenh264|h264_nvenc"
