#include "yolov8.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <android/log.h>

static inline void read_pixel_rotated(const uint8_t* src,
                                      int srcW, int srcH, int rowStride,
                                      int rot, int x, int y,
                                      uint8_t& R, uint8_t& G, uint8_t& B) {
    int sx = x, sy = y;
    if (rot == 90)      { sx = y;             sy = srcW - 1 - x; }
    else if (rot == 180){ sx = srcW - 1 - x;  sy = srcH - 1 - y; }
    else if (rot == 270){ sx = srcH - 1 - y;  sy = x; }

    if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) {
        R = 114; G = 114; B = 114;
        return;
    }

    const uint8_t* p = src + sy * rowStride + sx * 4;
    R = p[0]; G = p[1]; B = p[2];
}

bool YoloV8::load(AAssetManager* mgr, const char* param, const char* bin) {
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;

    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.lightmode = true;

    loadedInputSize = 640;
    return net.load_param(mgr, param) == 0 && net.load_model(mgr, bin) == 0;
}

bool YoloV8::loadForSize(AAssetManager* mgr, int inputSize) {
    net.clear();
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;

    if (useOptimizations) {
        __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading with OPTIMIZATIONS enabled");
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_packing_layout = true;
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.lightmode = true;
    } else {
        __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading in BASELINE mode (no optimizations)");
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_packing_layout = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.lightmode = false;
    }

    int modelSize = inputSize;
    if (inputSize >= 640) modelSize = 640;

    char paramFile[64], binFile[64];
    snprintf(paramFile, sizeof(paramFile), "yolov8n_%d.param", modelSize);
    snprintf(binFile, sizeof(binFile), "yolov8n_%d.bin", modelSize);

    __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading model: %s (for input size %d)", paramFile, inputSize);

    int paramResult = net.load_param(mgr, paramFile);
    int binResult = net.load_model(mgr, binFile);

    if (paramResult == 0 && binResult == 0) {
        loadedInputSize = inputSize;
        __android_log_print(ANDROID_LOG_INFO, "yolo", "Model loaded for size %d (using %d model)", inputSize, modelSize);
        return true;
    }

    __android_log_print(ANDROID_LOG_ERROR, "yolo", "Failed to load model for size %d (param=%d, bin=%d)",
                        inputSize, paramResult, binResult);
    return false;
}

bool YoloV8::loadFromFile(const char* paramPath, const char* binPath, int inputSize, int numThreads) {
    net.clear();
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = (numThreads > 0) ? numThreads : 4;

    net.opt.use_int8_inference = true;
    net.opt.use_int8_packed = true;
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = true;

    if (useOptimizations) {
        __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading FILE model with OPTIMIZATIONS enabled");
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_packing_layout = true;
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.lightmode = true;
    } else {
        __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading FILE model in BASELINE mode");
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_packing_layout = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.lightmode = false;
    }

    __android_log_print(ANDROID_LOG_INFO, "yolo", "Loading from file: %s / %s (imgsz=%d threads=%d)",
                        paramPath, binPath, inputSize, net.opt.num_threads);

    int pr = net.load_param(paramPath);
    int br = net.load_model(binPath);

    if (pr == 0 && br == 0) {
        loadedInputSize = inputSize;
        __android_log_print(ANDROID_LOG_INFO, "yolo", "FILE model loaded OK");
        return true;
    }

    __android_log_print(ANDROID_LOG_ERROR, "yolo", "FILE model load failed (param=%d, bin=%d)", pr, br);
    return false;
}

std::vector<Det> YoloV8::detect_rgba(const uint8_t* rgba, int srcW, int srcH, int rowStride,
                                     int rot, float conf_thr, float iou_thr, int dst) {
    int w = (rot == 90 || rot == 270) ? srcH : srcW;
    int h = (rot == 90 || rot == 270) ? srcW : srcH;

    ncnn::Mat in(dst, dst, 3);
    float r = std::min(dst / (float)w, dst / (float)h);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);
    int pad_w = dst - new_w, pad_h = dst - new_h;

    float* ch0 = in.channel(0);
    float* ch1 = in.channel(1);
    float* ch2 = in.channel(2);

    for (int y=0; y<dst; y++) {
        for (int x=0; x<dst; x++) {
            int rx = x - pad_w/2, ry = y - pad_h/2;
            uint8_t R=114,G=114,B=114;
            if (rx>=0 && rx<new_w && ry>=0 && ry<new_h) {
                int sx = (int)std::round(rx / r);
                int sy = (int)std::round(ry / r);
                read_pixel_rotated(rgba, srcW, srcH, rowStride, rot, sx, sy, R,G,B);
            }
            int idx = y * dst + x;
            ch0[idx] = R/255.f;
            ch1[idx] = G/255.f;
            ch2[idx] = B/255.f;
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(useOptimizations); // IMPORTANT: baseline vs optimized

    if (ex.input("in0", in) != 0 && ex.input("images", in) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "yolo", "ex.input failed (no blob in0/images)");
        return {};
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 && ex.extract("output0", out) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "yolo", "ex.extract failed (no blob out0/output0)");
        return {};
    }

    // ======= дальше твой existing decode (оставлено как было) =======
    const int num_preds = out.w;
    const int no = out.h;
    const float* base = (const float*)out.data;

    auto get_feat_col = [&](int i, std::vector<float>& f){
        f.resize(no);
        for (int j = 0; j < no; ++j) f[j] = base[j * num_preds + i];
    };

    std::vector<Det> props;
    std::vector<float> f;

    for (int i = 0; i < num_preds; ++i) {
        get_feat_col(i, f);

        const int cls_start = (no == 84) ? 4 : 5;
        if (no - cls_start <= 0) continue;

        float x = f[0], y = f[1], bw = f[2], bh = f[3];
        float obj = (cls_start == 5) ? f[4] : 1.f;

        int best_cls = -1;
        float best = 0.f;
        for (int c = cls_start; c < no; ++c) {
            float s = f[c] * obj;
            if (s > best) { best = s; best_cls = c - cls_start; }
        }
        if (best < conf_thr) continue;

        // xywh -> xyxy (in dst space)
        float x1 = x - bw/2.f;
        float y1 = y - bh/2.f;
        float x2 = x + bw/2.f;
        float y2 = y + bh/2.f;

        // inverse letterbox to original
        float rbx = (x1 - pad_w/2)/r;
        float rby = (y1 - pad_h/2)/r;
        float rbx2= (x2 - pad_w/2)/r;
        float rby2= (y2 - pad_h/2)/r;

        Det d;
        d.x1 = rbx;
        d.y1 = rby;
        d.x2 = rbx2;
        d.y2 = rby2;
        d.score = best;
        d.cls = best_cls;
        props.push_back(d);
    }

    // NMS у тебя уже где-то ниже/в другом месте — оставь как есть, если он есть.
    return props;
}