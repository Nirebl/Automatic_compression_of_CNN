#pragma once
#include <android/asset_manager_jni.h>
#include <vector>
#include "ncnn/net.h"

struct Det { float x1,y1,x2,y2,score; int cls; };

class YoloV8 {
public:
    bool load(AAssetManager* mgr, const char* param, const char* bin);

    // Load model for specific input size (assets: yolov8n_320.param etc.)
    bool loadForSize(AAssetManager* mgr, int inputSize);

    // NEW: Load model from filesystem paths (adb push)
    bool loadFromFile(const char* paramPath, const char* binPath, int inputSize, int numThreads);

    std::vector<Det> detect_rgba(const uint8_t* rgba,
                                 int srcW, int srcH, int rowStride,
                                 int rotationDeg,
                                 float conf_thr, float iou_thr, int dst=640);

    void clear() { net.clear(); }

    int getLoadedSize() const { return loadedInputSize; }

    void setOptimized(bool enabled) { useOptimizations = enabled; }
    bool isOptimized() const { return useOptimizations; }

private:
    ncnn::Net net;
    int loadedInputSize = 640;
    bool useOptimizations = true;
};