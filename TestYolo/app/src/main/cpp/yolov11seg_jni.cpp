#include <jni.h>
#include <android/asset_manager_jni.h>
#include "yolov11seg.hpp"

static YoloV11Seg* g_seg = nullptr;
static YoloV11Seg* g_seg_benchmark = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_MainActivity_00024YoloSegBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr,
        jstring paramPath, jstring binPath) {

    // Clean up existing instance
    if (g_seg) {
        g_seg->clear();
        delete g_seg;
    }
    g_seg = new YoloV11Seg();

    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);

    bool ok = g_seg->load(mgr, param, bin);

    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);

    return ok;
}

// ===== Benchmark-specific JNI methods =====

// Store AssetManager reference for model reloading
static AAssetManager* g_seg_assetMgr = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_YoloSegBenchmarkActivity_00024YoloSegBridge_initSeg(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr) {

    // Clean up existing instance
    if (g_seg_benchmark) {
        g_seg_benchmark->clear();
        delete g_seg_benchmark;
    }
    g_seg_benchmark = new YoloV11Seg();
    g_seg_assetMgr = AAssetManager_fromJava(env, assetMgr);

    // Don't load any model yet - will load per resolution
    return true;
}

// Load model for specific input size
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_YoloSegBenchmarkActivity_00024YoloSegBridge_loadForSize(
        JNIEnv* env, jobject /*thiz*/, jint inputSize) {
    if (!g_seg_benchmark || !g_seg_assetMgr) {
        return false;
    }
    return g_seg_benchmark->loadForSize(g_seg_assetMgr, inputSize);
}

// Get currently loaded model size
extern "C" JNIEXPORT jint JNICALL
Java_com_example_testyolo_YoloSegBenchmarkActivity_00024YoloSegBridge_getLoadedSize(
        JNIEnv* env, jobject /*thiz*/) {
    if (!g_seg_benchmark) {
        return 0;
    }
    return g_seg_benchmark->getLoadedSize();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_YoloSegBenchmarkActivity_00024YoloSegBridge_releaseSeg(
        JNIEnv*, jobject) {
    if (g_seg_benchmark) {
        g_seg_benchmark->clear();
        delete g_seg_benchmark;
        g_seg_benchmark = nullptr;
    }
    g_seg_assetMgr = nullptr;
}

// Detect with configurable input size for benchmarking
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_YoloSegBenchmarkActivity_00024YoloSegBridge_detectSegRgbaWithSize(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou, jint inputSize) {

    if (!g_seg_benchmark) {
        jclass floatArrCls = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    uint8_t* ptr = (uint8_t*)env->GetDirectBufferAddress(rgbaBuffer);
    if (!ptr) {
        jclass floatArrCls = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    std::vector<SegDet> dets = g_seg_benchmark->detect_rgba(ptr, width, height, rowStride,
                                                            rotationDeg, conf, iou, inputSize);

    jclass floatArrCls = env->FindClass("[F");
    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);

    for (jsize i = 0; i < (jsize)dets.size(); ++i) {
        const SegDet& d = dets[i];
        jfloat tmp[8] = {
            d.x1, d.y1, d.x2, d.y2,
            d.score, (float)d.cls,
            (float)d.mask_w, (float)d.mask_h
        };
        jfloatArray row = env->NewFloatArray(8);
        env->SetFloatArrayRegion(row, 0, 8, tmp);
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_MainActivity_00024YoloSegBridge_release(
        JNIEnv*, jobject) {
    if (g_seg) {
        g_seg->clear();
        delete g_seg;
        g_seg = nullptr;
    }
}

// Returns detection boxes: Array of FloatArray [x1, y1, x2, y2, score, cls, mask_w, mask_h]
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_MainActivity_00024YoloSegBridge_detectRgbaBoxesOnly(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou) {

    if (!g_seg) {
        // Return empty array instead of null
        jclass floatArrCls = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    uint8_t* ptr = (uint8_t*)env->GetDirectBufferAddress(rgbaBuffer);
    if (!ptr) {
        jclass floatArrCls = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    std::vector<SegDet> dets = g_seg->detect_rgba(ptr, width, height, rowStride,
                                                   rotationDeg, conf, iou);

    jclass floatArrCls = env->FindClass("[F");
    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);

    for (jsize i = 0; i < (jsize)dets.size(); ++i) {
        const SegDet& d = dets[i];
        jfloat tmp[8] = {
            d.x1, d.y1, d.x2, d.y2,
            d.score, (float)d.cls,
            (float)d.mask_w, (float)d.mask_h
        };
        jfloatArray row = env->NewFloatArray(8);
        env->SetFloatArrayRegion(row, 0, 8, tmp);
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }
    return out;
}

// Returns detections with contour points: Array of FloatArray
// Format: [x1, y1, x2, y2, score, cls, numContourPoints, px0, py0, px1, py1, ...]
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_MainActivity_00024YoloSegBridge_detectRgbaWithContours(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou) {

    jclass floatArrCls = env->FindClass("[F");
    
    if (!g_seg) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    uint8_t* ptr = (uint8_t*)env->GetDirectBufferAddress(rgbaBuffer);
    if (!ptr) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }

    std::vector<SegDet> dets = g_seg->detect_rgba(ptr, width, height, rowStride,
                                                   rotationDeg, conf, iou);

    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);

    for (jsize i = 0; i < (jsize)dets.size(); ++i) {
        const SegDet& d = dets[i];
        
        // Base fields: x1, y1, x2, y2, score, cls, numContourPoints (7 values)
        // Then contour points as x,y pairs
        int numContourPoints = (int)(d.contour.size() / 2);
        int totalSize = 7 + (int)d.contour.size();
        
        std::vector<jfloat> tmp(totalSize);
        tmp[0] = d.x1;
        tmp[1] = d.y1;
        tmp[2] = d.x2;
        tmp[3] = d.y2;
        tmp[4] = d.score;
        tmp[5] = (float)d.cls;
        tmp[6] = (float)numContourPoints;
        
        // Copy contour points
        for (size_t j = 0; j < d.contour.size(); ++j) {
            tmp[7 + j] = d.contour[j];
        }
        
        jfloatArray row = env->NewFloatArray(totalSize);
        env->SetFloatArrayRegion(row, 0, totalSize, tmp.data());
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }
    return out;
}