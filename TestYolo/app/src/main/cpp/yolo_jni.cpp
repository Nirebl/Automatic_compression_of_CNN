#include <jni.h>
#include <android/asset_manager_jni.h>
#include "yolov8.hpp"

static YoloV8* g = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr) {
    // Clean up existing instance
    if (g) {
        g->clear();
        delete g;
    }
    g = new YoloV8();
    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    return g->load(mgr, "yolov8n.param", "yolov8n.bin");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_release(
        JNIEnv*, jobject) {
    if (g) {
        g->clear();
        delete g;
        g = nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_detectRgba(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou) {

    jclass floatArrCls = env->FindClass("[F");
    
    // Null checks to prevent crashes
    if (!g) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    uint8_t* ptr = (uint8_t*) env->GetDirectBufferAddress(rgbaBuffer);
    if (!ptr) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    // Validate dimensions
    if (width <= 0 || height <= 0 || rowStride <= 0) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    std::vector<Det> dets = g->detect_rgba(ptr, width, height, rowStride, rotationDeg, conf, iou);

    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);
    for (jsize i=0;i<(jsize)dets.size();++i){
        jfloat tmp[6] = {dets[i].x1,dets[i].y1,dets[i].x2,dets[i].y2,dets[i].score,(float)dets[i].cls};
        jfloatArray row = env->NewFloatArray(6);
        env->SetFloatArrayRegion(row,0,6,tmp);
        env->SetObjectArrayElement(out,i,row);
        env->DeleteLocalRef(row);
    }
    return out;
}

// Benchmark version with configurable input resolution
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_detectRgbaWithSize(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou, jint inputSize) {

    jclass floatArrCls = env->FindClass("[F");
    
    if (!g) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    uint8_t* ptr = (uint8_t*) env->GetDirectBufferAddress(rgbaBuffer);
    if (!ptr) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    if (width <= 0 || height <= 0 || rowStride <= 0) {
        return env->NewObjectArray(0, floatArrCls, nullptr);
    }
    
    std::vector<Det> dets = g->detect_rgba(ptr, width, height, rowStride, rotationDeg, conf, iou, inputSize);

    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);
    for (jsize i=0;i<(jsize)dets.size();++i){
        jfloat tmp[6] = {dets[i].x1,dets[i].y1,dets[i].x2,dets[i].y2,dets[i].score,(float)dets[i].cls};
        jfloatArray row = env->NewFloatArray(6);
        env->SetFloatArrayRegion(row,0,6,tmp);
        env->SetObjectArrayElement(out,i,row);
        env->DeleteLocalRef(row);
    }
    return out;
}

// Store AssetManager reference for model reloading
static AAssetManager* g_assetMgr = nullptr;

// Init for YoloBenchmarkActivity
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr) {
    if (g) {
        g->clear();
        delete g;
    }
    g = new YoloV8();
    g_assetMgr = AAssetManager_fromJava(env, assetMgr);
    // Don't load any model yet - will load per resolution
    return true;
}

// Load model for specific input size
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_loadForSize(
        JNIEnv* env, jobject /*thiz*/, jint inputSize) {
    if (!g || !g_assetMgr) {
        return false;
    }
    return g->loadForSize(g_assetMgr, inputSize);
}

// Get currently loaded model size
extern "C" JNIEXPORT jint JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_getLoadedSize(
        JNIEnv* env, jobject /*thiz*/) {
    if (!g) {
        return 0;
    }
    return g->getLoadedSize();
}

// Release for YoloBenchmarkActivity
extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_release(
        JNIEnv*, jobject) {
    if (g) {
        g->clear();
        delete g;
        g = nullptr;
    }
    g_assetMgr = nullptr;
}

// Set optimization mode (true = optimized, false = baseline)
extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_setOptimized(
        JNIEnv*, jobject, jboolean enabled) {
    if (g) {
        g->setOptimized(enabled);
    }
}

// Get current optimization mode
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_YoloBenchmarkActivity_00024YoloBridge_isOptimized(
        JNIEnv*, jobject) {
    if (g) {
        return g->isOptimized();
    }
    return true;
}