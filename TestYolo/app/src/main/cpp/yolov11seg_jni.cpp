#include <jni.h>
#include <android/asset_manager_jni.h>
#include "yolov11seg.hpp"

static YoloV11Seg* g_seg = nullptr;

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
