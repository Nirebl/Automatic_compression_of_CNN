#include <jni.h>
#include <android/asset_manager_jni.h>
#include "yolov8.hpp"

static YoloV8* g = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr) {
    if (!g) g = new YoloV8();
    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    return g->load(mgr, "yolov8n.param", "yolov8n.bin");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_release(
        JNIEnv*, jobject) { g->clear(); }

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_MainActivity_00024YoloBridge_detectRgba(
        JNIEnv* env, jobject /*thiz*/,
        jobject rgbaBuffer,
        jint width, jint height, jint rowStride, jint rotationDeg,
        jfloat conf, jfloat iou) {

    uint8_t* ptr = (uint8_t*) env->GetDirectBufferAddress(rgbaBuffer);
    std::vector<Det> dets = g->detect_rgba(ptr, width, height, rowStride, rotationDeg, conf, iou);

    jclass floatArrCls = env->FindClass("[F");
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
