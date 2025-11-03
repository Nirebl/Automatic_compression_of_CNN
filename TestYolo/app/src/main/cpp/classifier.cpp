// resnet50.cpp
// Классификация RGBA-кадра (ResNet-50, ImageNet) через ncnn.
// В твоём resnet50.param: input = "in0", output = "out0".
// Есть fallback-имена на случай другого экспорта.

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "ncnn/net.h"

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>

#ifndef LOG_TAG
#define LOG_TAG "ncnn-resnet50"
#endif

class ResNet50 {
public:
    bool load(AAssetManager* mgr, const char* param, const char* bin) {
        // Если на устройстве есть Vulkan — ncnn сам использует его для ускорения
        net.opt.use_vulkan_compute = true;

        if (net.load_param(mgr, param) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "load_param(%s) failed", param);
            return false;
        }
        if (net.load_model(mgr, bin) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "load_model(%s) failed", bin);
            return false;
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "ResNet50 loaded (param=%s, bin=%s)", param, bin);
        return true;
    }

    // RGBA8888 -> topK пар (class_id, prob in [0..1])
    std::vector<std::pair<int,float>> classify_rgba(const uint8_t* rgba,
                                                    int w, int h,
                                                    int /*rowStride*/,
                                                    int /*rotation_deg*/,
                                                    int topK = 5)
    {
        // Preprocess: resize к 224x224, нормализация как в ImageNet
        const int iw = 224, ih = 224;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, iw, ih);

        // mean/std заданы в диапазоне 0..255 (эквивалент 0.485/0.456/0.406 и 0.229/0.224/0.225)
        const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
        const float norm_vals[3] = {1.f/58.395f, 1.f/57.12f, 1.f/57.375f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);

        // ------- Подаём вход -------
        // Главные имена для твоего графа: "in0"
        // Добавлены популярные fallback-варианты для других экспортов.
        static const char* INPUT_CANDIDATES[] = {
                "in0", "0", "input", "data", "images", "pnnx_input_0", "input.1", "x", "image"
        };
        bool fed = false;
        for (const char* nm : INPUT_CANDIDATES) {
            if (ex.input(nm, in) == 0) {
                fed = true;
                // __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "fed input via '%s'", nm);
                break;
            }
        }
        if (!fed) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "failed to feed input (none of candidates matched)");
            return {};
        }

        // ------- Достаём выход -------
        // Для твоего графа главный выход: "out0"
        static const char* OUTPUT_CANDIDATES[] = {
                "out0", "prob", "softmax", "logits", "output", "output0", "out", "pnnx_output_0"
        };
        ncnn::Mat out;
        bool got = false;
        for (const char* nm : OUTPUT_CANDIDATES) {
            if (ex.extract(nm, out) == 0) {
                got = true;
                // __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "extracted output via '%s' (w=%d h=%d c=%d)", nm, out.w, out.h, out.c);
                break;
            }
        }
        if (!got) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "failed to extract output");
            return {};
        }

        // Копируем в вектор и делаем softmax
        std::vector<float> logits;
        logits.reserve((size_t)out.total());
        for (int i = 0; i < (int)out.total(); ++i) logits.push_back(out[i]);
        if (logits.empty()) return {};

        float maxv = *std::max_element(logits.begin(), logits.end());
        float sum = 0.f;
        for (auto &v : logits) { v = std::exp(v - maxv); sum += v; }
        if (sum <= 0.f) sum = 1.f;
        for (auto &v : logits) v /= sum;

        // topK индексы
        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        const int K = std::min(topK, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
                          [&](int a, int b){ return logits[a] > logits[b]; });

        std::vector<std::pair<int,float>> top;
        top.reserve(K);
        for (int i = 0; i < K; ++i) {
            int k = idx[i];
            top.emplace_back(k, logits[k]);
        }
        return top;
    }

private:
    ncnn::Net net;
};

// Глобальный экземпляр — если твой JNI-блок к нему обращается
static ResNet50 g_resnet;
