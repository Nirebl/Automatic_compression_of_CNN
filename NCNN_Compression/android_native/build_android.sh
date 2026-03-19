#!/usr/bin/env bash
set -euo pipefail

# Требования:
#  - Android NDK (export ANDROID_NDK=/path/to/ndk)
#  - cmake, ninja, git
# Собираем:
#  android_native/ncnn-install  (include/lib для Android arm64)
#  android_native/build-android/bin/xtrim_yolo_detect

ABI=arm64-v8a
API=24

if [[ -z "${ANDROID_NDK:-}" ]]; then
  echo "ERROR: ANDROID_NDK env var is not set"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
NCNN_DIR="$ROOT/ncnn"
NCNN_BUILD="$ROOT/ncnn-build"
NCNN_INSTALL="$ROOT/ncnn-install"
APP_BUILD="$ROOT/build-android"

TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"

if [[ ! -d "$NCNN_DIR" ]]; then
  git clone --depth=1 https://github.com/Tencent/ncnn.git "$NCNN_DIR"
fi

# 1) build ncnn
rm -rf "$NCNN_BUILD"
cmake -S "$NCNN_DIR" -B "$NCNN_BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM=android-"$API" \
  -DNCNN_VULKAN=OFF -DNCNN_OPENMP=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$NCNN_INSTALL"
cmake --build "$NCNN_BUILD" -j
cmake --install "$NCNN_BUILD"

# 2) build detector binary (links against ncnn-install)
rm -rf "$APP_BUILD"
cmake -S "$ROOT" -B "$APP_BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM=android-"$API" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$APP_BUILD" -j

echo "OK: built $APP_BUILD/xtrim_yolo_detect"
mkdir -p "$APP_BUILD/bin"
cp -f "$APP_BUILD/xtrim_yolo_detect" "$APP_BUILD/bin/xtrim_yolo_detect"
echo "BIN: $APP_BUILD/bin/xtrim_yolo_detect"