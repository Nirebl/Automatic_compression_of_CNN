#include <jni.h>
#include <string>
#include <android/asset_manager_jni.h>
#include <fastercnn.cpp>

// forward из fasterrcnn.cpp
extern FasterRCNN g_frcnn;

static std::string J2S(JNIEnv* env, jstring js) {
    const char* c = env->GetStringUTFChars(js, nullptr);
    std::string s = c ? c : "";
    env->ReleaseStringUTFChars(js, c);
    return s;
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_example_testyolo_MainActivity_00024FrcnnBridge_release(
        JNIEnv*, jobject) { g_frcnn.clear(); }

// boolean init(AssetManager, String param, String bin)
JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_MainActivity_00024FrcnnBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr, jstring jparam, jstring jbin) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    auto param = J2S(env, jparam);
    auto bin   = J2S(env, jbin);
    return g_frcnn.load(mgr, param.c_str(), bin.c_str());
}

// FloatArray[] detectRgba(...)
JNIEXPORT jobjectArray JNICALL
Java_com_example_testyolo_MainActivity_00024FrcnnBridge_detectRgba(
        JNIEnv* env, jobject /*thiz*/, jobject buf, jint w, jint h, jint rowStride,
        jint rotDeg, jfloat conf) {
    const unsigned char* rgba = (const unsigned char*)env->GetDirectBufferAddress(buf);
    if (!rgba) {
        return env->NewObjectArray(0, env->FindClass("[F"), nullptr);
    }
    auto dets = g_frcnn.detect_rgba(rgba, w, h, rowStride, rotDeg, conf);

    jclass floatArrCls = env->FindClass("[F");
    jobjectArray out = env->NewObjectArray((jsize)dets.size(), floatArrCls, nullptr);

    for (int i = 0; i < (int)dets.size(); ++i) {
        const auto &d = dets[i];
        jfloat tmp[6] = { d.x1, d.y1, d.x2, d.y2, d.score, (float)d.cls };
        jfloatArray row = env->NewFloatArray(6);
        env->SetFloatArrayRegion(row, 0, 6, tmp);
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }
    return out;
}

} // extern "C"
