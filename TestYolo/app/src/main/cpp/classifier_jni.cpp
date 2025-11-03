#include <jni.h>
#include <string>
#include <android/asset_manager_jni.h>


#include <classifier.cpp>
// forward из classifier.cpp
extern ResNet50 g_resnet;

static std::string J2S(JNIEnv* env, jstring js) {
    const char* c = env->GetStringUTFChars(js, nullptr);
    std::string s = c ? c : "";
    env->ReleaseStringUTFChars(js, c);
    return s;
}

extern "C" {

// boolean init(AssetManager, String param, String bin)
JNIEXPORT jboolean JNICALL
Java_com_example_testyolo_MainActivity_00024ResNetBridge_init(
        JNIEnv* env, jobject /*thiz*/, jobject assetMgr, jstring jparam, jstring jbin) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    auto param = J2S(env, jparam);
    auto bin   = J2S(env, jbin);
    return g_resnet.load(mgr, param.c_str(), bin.c_str());
}

// float[] classifyRgba(ByteBuffer rgba, int w, int h, int rowStride, int rotDeg, int topK)
// возвращаем массив длины 2*topK: [cls0, prob0, cls1, prob1, ...]
JNIEXPORT jfloatArray JNICALL
Java_com_example_testyolo_MainActivity_00024ResNetBridge_classifyRgba(
        JNIEnv* env, jobject /*thiz*/, jobject buf, jint w, jint h, jint rowStride, jint rotDeg, jint topK) {
    const unsigned char* rgba = (const unsigned char*)env->GetDirectBufferAddress(buf);
    if (!rgba) {
        jfloatArray ret = env->NewFloatArray(0);
        return ret;
    }
    auto top = g_resnet.classify_rgba(rgba, w, h, rowStride, rotDeg, topK);

    const int N = (int)top.size();
    jfloatArray out = env->NewFloatArray(N*2);
    std::vector<float> tmp(N*2);
    for (int i = 0; i < N; ++i) { tmp[i*2+0] = (float)top[i].first; tmp[i*2+1] = top[i].second; }
    env->SetFloatArrayRegion(out, 0, N*2, tmp.data());
    return out;
}

} // extern "C"
