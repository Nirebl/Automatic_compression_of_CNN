#include "yolov11seg.hpp"
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define LOG_TAG "yolov11seg"

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Safe pixel read with rotation and bounds checking
static inline void read_pixel_rotated(const uint8_t* src,
                                      int srcW, int srcH, int rowStride,
                                      int rot, int x, int y,
                                      uint8_t& R, uint8_t& G, uint8_t& B) {
    int sx = x, sy = y;
    if (rot == 90)       { sx = y;            sy = srcW - 1 - x; }
    else if (rot == 180) { sx = srcW - 1 - x; sy = srcH - 1 - y; }
    else if (rot == 270) { sx = srcH - 1 - y; sy = x; }

    // Bounds check
    if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) {
        R = G = B = 114;  // Padding color
        return;
    }

    const uint8_t* p = src + sy * rowStride + sx * 4;
    R = p[0]; G = p[1]; B = p[2];
}

bool YoloV11Seg::load(AAssetManager* mgr, const char* param, const char* bin) {
    // Disable Vulkan compute - causes SIGSEGV crashes on many devices
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;  // Single thread to avoid OpenMP crashes
    
    int rp = net.load_param(mgr, param);
    int rm = net.load_model(mgr, bin);
    if (rp != 0 || rm != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "load failed param=%d bin=%d", rp, rm);
        return false;
    }
    loadedInputSize = 640;  // Default model size
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "YOLOv11-seg model loaded");
    return true;
}

bool YoloV11Seg::loadForSize(AAssetManager* mgr, int inputSize) {
    // Clear previous model
    net.clear();
    
    // Disable Vulkan compute
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    
    // For sizes >= 640, use the 640 model
    int modelSize = inputSize;
    if (inputSize >= 640) {
        modelSize = 640;
    }
    
    // Build filename: yolov11n-seg_320.param, yolov11n-seg_320.bin
    char paramFile[64], binFile[64];
    snprintf(paramFile, sizeof(paramFile), "yolov11n-seg_%d.param", modelSize);
    snprintf(binFile, sizeof(binFile), "yolov11n-seg_%d.bin", modelSize);
    
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "Loading model: %s (for input size %d)", paramFile, inputSize);
    
    int paramResult = net.load_param(mgr, paramFile);
    int binResult = net.load_model(mgr, binFile);
    
    if (paramResult == 0 && binResult == 0) {
        loadedInputSize = inputSize;
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "Model loaded for size %d (using %d model)", inputSize, modelSize);
        return true;
    }
    
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to load model for size %d (param=%d, bin=%d)", 
                        inputSize, paramResult, binResult);
    return false;
}

std::vector<SegDet> YoloV11Seg::detect_rgba(const uint8_t* rgba, int srcW, int srcH,
                                            int rowStride, int rot,
                                            float conf_thr, float iou_thr, int dst) {
    if (!rgba || srcW <= 0 || srcH <= 0) return {};
    
    // Ensure dst is a multiple of 32 (YOLO stride requirement)
    dst = ((dst + 31) / 32) * 32;
    // Minimum 416px for YOLOv11-seg (smaller sizes may cause model issues)
    if (dst < 416) dst = 416;
    if (dst > 1280) dst = 1280;  // Cap at reasonable max
    
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, 
        "detect_rgba: srcW=%d srcH=%d dst=%d", srcW, srcH, dst);

    // Dimensions after rotation
    int w = (rot == 90 || rot == 270) ? srcH : srcW;
    int h = (rot == 90 || rot == 270) ? srcW : srcH;

    // Letterbox -> ncnn::Mat dst×dst×3 (float32)
    // Note: ncnn::Mat(w,h,c) uses the same layout as the existing yolov8.cpp
    ncnn::Mat in(dst, dst, 3);
    float scale = std::min(dst / (float)w, dst / (float)h);
    int new_w = (int)std::round(w * scale);
    int new_h = (int)std::round(h * scale);
    int pad_w = dst - new_w, pad_h = dst - new_h;

    // Get channel pointers for proper planar access
    float* ch0 = in.channel(0);
    float* ch1 = in.channel(1);
    float* ch2 = in.channel(2);
    
    // Fill with letterbox (same pattern as yolov8.cpp)
    for (int y = 0; y < dst; y++) {
        for (int x = 0; x < dst; x++) {
            int rx = x - pad_w / 2, ry = y - pad_h / 2;
            uint8_t R = 114, G = 114, B = 114;
            if (rx >= 0 && rx < new_w && ry >= 0 && ry < new_h) {
                int sx = (int)std::round(rx / scale);
                int sy = (int)std::round(ry / scale);
                read_pixel_rotated(rgba, srcW, srcH, rowStride, rot, sx, sy, R, G, B);
            }
            // ncnn::Mat is channel-planar, not interleaved
            int idx = y * dst + x;
            ch0[idx] = R / 255.f;
            ch1[idx] = G / 255.f;
            ch2[idx] = B / 255.f;
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    // Input - try common names
    if (ex.input("in0", in) != 0 && ex.input("images", in) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "ex.input failed");
        return {};
    }

    // Detection output (boxes + classes + mask coefficients)
    ncnn::Mat out_det;
    if (ex.extract("out0", out_det) != 0 && ex.extract("output0", out_det) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "ex.extract det failed for dst=%d", dst);
        return {};
    }
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, 
        "out_det: w=%d h=%d c=%d (dst=%d)", out_det.w, out_det.h, out_det.c, dst);

    // Prototype masks output
    ncnn::Mat out_proto;
    bool has_proto = (ex.extract("out1", out_proto) == 0 || ex.extract("output1", out_proto) == 0);
    if (has_proto) {
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, 
            "out_proto: w=%d h=%d c=%d", out_proto.w, out_proto.h, out_proto.c);
    } else {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "No proto output");
    }

    // Parse detection output
    // YOLOv11-seg format: [x, y, w, h, cls0..cls79, mask0..mask31] per prediction
    // Shape is typically (116, 8400) or (8400, 116) where 116 = 4 + 80 + 32
    const int num_preds = out_det.w;
    const int no = out_det.h;
    const float* base = (const float*)out_det.data;

    // Detect format - use local variables to avoid modifying member state
    int feat_dim = no;
    int n_preds = num_preds;
    bool transposed = false;

    // If h > w, it's likely transposed
    if (no > num_preds && num_preds >= 4) {
        feat_dim = num_preds;
        n_preds = no;
        transposed = true;
    }

    // Calculate number of classes and mask coefficients (use local copy)
    int local_mask_dim = mask_proto_dim;
    int nc = feat_dim - 4 - local_mask_dim;
    if (nc <= 0) {
        nc = feat_dim - 4;
        local_mask_dim = 0;
    }
    if (nc <= 0) nc = 80;

    auto get_feat = [&](int i, std::vector<float>& f) {
        f.resize(feat_dim);
        if (transposed) {
            const float* row = base + i * feat_dim;
            std::copy(row, row + feat_dim, f.begin());
        } else {
            for (int j = 0; j < feat_dim; ++j)
                f[j] = base[j * n_preds + i];
        }
    };

    // Inverse scale transformation
    auto inv_scale = [&](float& bx, float& by, float& bw, float& bh) {
        float rbx = (bx - pad_w / 2) / scale;
        float rby = (by - pad_h / 2) / scale;
        float rbw = bw / scale;
        float rbh = bh / scale;

        float cx, cy;
        if (rot == 0)        { cx = rbx;            cy = rby; }
        else if (rot == 90)  { cx = srcW - 1 - rby; cy = rbx; }
        else if (rot == 180) { cx = srcW - 1 - rbx; cy = srcH - 1 - rby; }
        else                 { cx = rby;            cy = srcH - 1 - rbx; }

        bx = cx; by = cy; bw = rbw; bh = rbh;
    };

    struct Proposal {
        float x1, y1, x2, y2, score;
        int cls;
        std::vector<float> mask_coeffs;
        float cx, cy, bw, bh;  // Center coords in model space for mask cropping
    };

    std::vector<Proposal> props;
    props.reserve(100);
    std::vector<float> f;
    f.reserve(feat_dim);

    for (int i = 0; i < n_preds; ++i) {
        get_feat(i, f);

        float x = f[0], y = f[1], bw = f[2], bh = f[3];

        // Find best class
        int cls = -1;
        float best = 0.f;
        for (int c = 0; c < nc; ++c) {
            float sc = f[4 + c];
            if (sc > best) {
                best = sc;
                cls = c;
            }
        }

        if (best < conf_thr) continue;

        // Store original model-space coords for mask
        float cx_model = x, cy_model = y;
        float bw_model = bw, bh_model = bh;

        inv_scale(x, y, bw, bh);
        float x1 = std::max(0.f, x - bw / 2);
        float y1 = std::max(0.f, y - bh / 2);
        float x2 = std::min((float)srcW, x + bw / 2);
        float y2 = std::min((float)srcH, y + bh / 2);

        // Skip invalid boxes
        if (x2 <= x1 || y2 <= y1) continue;

        Proposal p;
        p.x1 = x1; p.y1 = y1; p.x2 = x2; p.y2 = y2;
        p.score = best;
        p.cls = cls;
        p.cx = cx_model; p.cy = cy_model;
        p.bw = bw_model; p.bh = bh_model;

        // Extract mask coefficients
        if (local_mask_dim > 0 && (int)f.size() >= 4 + nc + local_mask_dim) {
            p.mask_coeffs.resize(local_mask_dim);
            for (int m = 0; m < local_mask_dim; ++m) {
                p.mask_coeffs[m] = f[4 + nc + m];
            }
        }

        props.push_back(std::move(p));
    }

    // NMS
    std::sort(props.begin(), props.end(),
              [](const Proposal& a, const Proposal& b) { return a.score > b.score; });

    std::vector<SegDet> keep;
    keep.reserve(props.size());
    std::vector<bool> suppressed(props.size(), false);

    auto compute_iou = [](const Proposal& a, const Proposal& b) -> float {
        float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
        float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
        float iw = std::max(0.f, xx2 - xx1);
        float ih = std::max(0.f, yy2 - yy1);
        float inter = iw * ih;
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (area_a + area_b - inter + 1e-6f);
    };

    // Proto mask dimensions - get actual dimensions from the output tensor
    int proto_h = 0, proto_w = 0, proto_c = 0;
    if (has_proto && out_proto.data) {
        proto_c = out_proto.c;
        proto_h = out_proto.h;
        proto_w = out_proto.w;
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, 
            "Proto dims: c=%d h=%d w=%d (dst=%d)", proto_c, proto_h, proto_w, dst);
    }
    
    // Validate proto dimensions - must be positive and reasonable
    bool valid_proto = has_proto && proto_c > 0 && proto_h > 0 && proto_w > 0 &&
                       proto_c == local_mask_dim;

    for (size_t i = 0; i < props.size(); ++i) {
        if (suppressed[i]) continue;

        const Proposal& p = props[i];

        SegDet det;
        det.x1 = p.x1; det.y1 = p.y1;
        det.x2 = p.x2; det.y2 = p.y2;
        det.score = p.score;
        det.cls = p.cls;
        det.mask_w = 0;
        det.mask_h = 0;

        // Generate mask if we have valid proto and coefficients
        if (valid_proto && !p.mask_coeffs.empty() &&
            proto_c == (int)p.mask_coeffs.size()) {

            // Compute bounding box in proto mask space
            float scale_x = (float)proto_w / (float)dst;
            float scale_y = (float)proto_h / (float)dst;

            int mx1 = std::max(0, (int)std::floor((p.cx - p.bw / 2) * scale_x));
            int my1 = std::max(0, (int)std::floor((p.cy - p.bh / 2) * scale_y));
            int mx2 = std::min(proto_w, (int)std::ceil((p.cx + p.bw / 2) * scale_x));
            int my2 = std::min(proto_h, (int)std::ceil((p.cy + p.bh / 2) * scale_y));

            int mw = mx2 - mx1;
            int mh = my2 - my1;

            if (mw > 0 && mh > 0 && mx2 <= proto_w && my2 <= proto_h) {
                det.mask_w = mw;
                det.mask_h = mh;
                det.mask.resize(mw * mh);

                // Compute mask: sigmoid(sum of coeffs * proto channels)
                for (int py = 0; py < mh; ++py) {
                    for (int px = 0; px < mw; ++px) {
                        float sum = 0.f;
                        int abs_x = mx1 + px;
                        int abs_y = my1 + py;
                        
                        // Bounds check
                        if (abs_x >= 0 && abs_x < proto_w && abs_y >= 0 && abs_y < proto_h) {
                            for (int c = 0; c < proto_c; ++c) {
                                const float* proto_ch = out_proto.channel(c);
                                sum += p.mask_coeffs[c] * proto_ch[abs_y * proto_w + abs_x];
                            }
                        }

                        det.mask[py * mw + px] = sigmoid(sum) > 0.5f ? 255 : 0;
                    }
                }
            }
        }

        keep.push_back(std::move(det));

        // Suppress overlapping detections
        for (size_t j = i + 1; j < props.size(); ++j) {
            if (!suppressed[j] && compute_iou(props[i], props[j]) > iou_thr) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}
