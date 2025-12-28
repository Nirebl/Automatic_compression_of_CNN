#pragma once
#include <android/asset_manager_jni.h>
#include <vector>
#include "ncnn/net.h"

struct Det { float x1,y1,x2,y2,score; int cls; };

class YoloV8 {
public:
    bool load(AAssetManager* mgr, const char* param, const char* bin);
    
    // Load model for specific input size (e.g., yolov8n_320.param)
    bool loadForSize(AAssetManager* mgr, int inputSize);

    // Реалтайм-камерный путь (RGBA + stride + поворот)
    std::vector<Det> detect_rgba(const uint8_t* rgba,
                                 int srcW, int srcH, int rowStride,
                                 int rotationDeg,
                                 float conf_thr, float iou_thr, int dst=640);

    void clear() {net.clear();}
    
    int getLoadedSize() const { return loadedInputSize; }

private:
    ncnn::Net net;
    int loadedInputSize = 640;
};
