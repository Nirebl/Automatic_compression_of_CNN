#pragma once
#include <android/asset_manager_jni.h>
#include <vector>
#include "ncnn/net.h"

struct SegDet {
    float x1, y1, x2, y2;
    float score;
    int cls;
    std::vector<uint8_t> mask;  // Binary mask for the bounding box region
    int mask_w, mask_h;          // Dimensions of the mask (matches bbox size)
};

class YoloV11Seg {
public:
    bool load(AAssetManager* mgr, const char* param, const char* bin);

    // Real-time camera path (RGBA + stride + rotation)
    // Returns detections with segmentation masks
    std::vector<SegDet> detect_rgba(const uint8_t* rgba,
                                    int srcW, int srcH, int rowStride,
                                    int rotationDeg,
                                    float conf_thr, float iou_thr, int dst = 640);

    void clear() { net.clear(); }

private:
    ncnn::Net net;
    int num_class = 80;
    int mask_proto_dim = 32;   // Number of mask prototype channels
    int mask_proto_h = 160;    // Prototype mask height (for 640 input)
    int mask_proto_w = 160;    // Prototype mask width
};

