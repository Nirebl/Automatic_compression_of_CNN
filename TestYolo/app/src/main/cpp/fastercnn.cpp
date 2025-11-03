#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "ncnn/net.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>

#ifndef LOG_TAG
#define LOG_TAG "ncnn-fasterrcnn"
#endif

struct Det { float x1,y1,x2,y2,score; int cls; };

static void parse_param_io(AAssetManager* mgr,
                           const char* param_asset,
                           std::vector<std::string>& inputs,
                           std::vector<std::string>& outputs)
{
    inputs.clear(); outputs.clear();

    if (!mgr || !param_asset) return;
    AAsset* a = AAssetManager_open(mgr, param_asset, AASSET_MODE_BUFFER);
    if (!a) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "parse_param_io: cannot open %s in assets", param_asset);
        return;
    }
    const size_t len = AAsset_getLength(a);
    std::string text; text.resize(len);
    AAsset_read(a, text.data(), len);
    AAsset_close(a);

    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        // пропускаем пустые/комменты
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;
        if (line[p] == '#') continue;

        // В ncnn .param входы выглядят как "Input 0 1 <blob>"
        // Выходы — как "Output 0 1 <blob>"
        // Иногда после имени идут "0=..." параметры — их игнорируем.
        if (line.compare(p, 5, "Input") == 0 || line.compare(p, 6, "Output") == 0) {
            std::istringstream ls(line);
            std::vector<std::string> tok; std::string t;
            while (ls >> t) tok.push_back(t);
            if (!tok.empty()) {
                // имя блоба — последний токен, у которого нет "0=..." стиля
                for (int i = (int)tok.size() - 1; i >= 0; --i) {
                    if (tok[i].find('=') == std::string::npos) {
                        if (line.compare(p, 5, "Input") == 0)
                            inputs.push_back(tok[i]);
                        else
                            outputs.push_back(tok[i]);
                        break;
                    }
                }
            }
        }
    }

    auto uniq = [](std::vector<std::string>& v){
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    uniq(inputs); uniq(outputs);

    // лог для отладки
    std::string inl, outl;
    for (auto& s : inputs)  { inl  += (inl.empty()  ? "" : ","); inl  += s; }
    for (auto& s : outputs) { outl += (outl.empty() ? "" : ","); outl += s; }
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "parse_param_io: inputs=[%s] outputs=[%s]",
                        inl.c_str(), outl.c_str());
}

class FasterRCNN {
public:
    void clear() { net.clear(); }

    bool load(AAssetManager* mgr, const char* param, const char* bin) {
        mgr_ = mgr;
        param_asset_ = param ? param : "";

        net.opt.use_vulkan_compute = true;

        if (net.load_param(mgr, param) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "load_param(%s) failed", param ? param : "(null)");
            return false;
        }
        if (net.load_model(mgr, bin) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "load_model(%s) failed", bin ? bin : "(null)");
            return false;
        }

        parse_param_io(mgr_, param_asset_.c_str(), input_names_, output_names_);
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "FasterRCNN loaded (param=%s, bin=%s)",
                            param ? param : "(null)", bin ? bin : "(null)");
        return true;
    }

    std::vector<Det> detect_rgba(const uint8_t* rgba, int w, int h,
                                 int /*rowStride*/, int /*rotation_deg*/,
                                 float conf_thr = 0.25f)
    {
        // torchvision-style resize: короткая сторона -> 800, длинная <= 1333
        int short_side = std::min(w, h);
        float scale = 800.f / (float)short_side;
        int nw = (int)std::lround(w * scale);
        int nh = (int)std::lround(h * scale);
        if (std::max(nw, nh) > 1333) {
            float s2 = 1333.f / (float)std::max(nw, nh);
            nw = (int)std::lround(nw * s2);
            nh = (int)std::lround(nh * s2);
            scale *= s2;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, nw, nh);

        // mean/std ImageNet в 0..255
        const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
        const float norm_vals[3] = {1.f/58.395f, 1.f/57.12f, 1.f/57.375f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);

        // ---------- INPUT ----------
        // 1) пробуем имена, распарсенные из .param
        bool fed = false;
        for (const auto& nm : input_names_) {
            if (ex.input(nm.c_str(), in) == 0) { fed = true; __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "fed via parsed input '%s'", nm.c_str()); break; }
        }
        // 2) fallback — популярные варианты
        if (!fed) {
            static const char* INPUT_CANDIDATES[] = {
                    "0", "in0", "images", "input", "data", "pnnx_input_0", "input.1"
            };
            for (const char* nm : INPUT_CANDIDATES) {
                if (ex.input(nm, in) == 0) { fed = true; __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "fed via fallback input '%s'", nm); break; }
            }
        }
        if (!fed) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "failed to feed input (no candidate matched)");
            return {};
        }

        // ---------- OUTPUTS ----------
        // Сначала прямые pnnx-идентификаторы (если известны)
        ncnn::Mat m_boxes, m_scores, m_labels, dets;
        bool got_triplet = false, got_nx6 = false, got_ssd = false;

        // 1) Попробуем распарсенные имена как кандидатов на выход.
        //    Вытащим всё, что удаётся, а затем попробуем распознать, где boxes/scores/labels.
        std::vector<std::pair<std::string,ncnn::Mat>> parsed_outs;
        for (const auto& nm : output_names_) {
            ncnn::Mat m;
            if (ex.extract(nm.c_str(), m) == 0 && m.total() > 0) {
                parsed_outs.emplace_back(nm, m);
            }
        }
        if (!parsed_outs.empty()) {
            // ищем мат с 4-канальным измерением как boxes
            int idx_boxes = -1;
            for (int i=0; i<(int)parsed_outs.size(); ++i) {
                const ncnn::Mat& m = parsed_outs[i].second;
                if (m.w == 4 || m.h == 4 || m.c == 4 || (m.total() % 4 == 0 && (m.w==0 || m.h==0))) {
                    idx_boxes = i; break;
                }
            }
            if (idx_boxes != -1) {
                m_boxes = parsed_outs[idx_boxes].second;
                // остальные два возьмём как scores/labels (если по размеру подходят)
                for (int i=0; i<(int)parsed_outs.size(); ++i) {
                    if (i == idx_boxes) continue;
                    const ncnn::Mat& m = parsed_outs[i].second;
                    // Выбираем вначале scores, потом labels
                    if (m_scores.empty()) m_scores = m;
                    else if (m_labels.empty()) m_labels = m;
                }
                if (!m_scores.empty() && !m_labels.empty()) got_triplet = true;
            }
        }

        // 2) Жёсткие pnnx-имена, которые часто встречались: boxes="570", scores="560", labels="571"
        if (!got_triplet) {
            bool ok = (ex.extract("570", m_boxes)  == 0) &&
                      (ex.extract("560", m_scores) == 0) &&
                      (ex.extract("571", m_labels) == 0);
            if (ok && m_boxes.total() && m_scores.total() && m_labels.total())
                got_triplet = true;
        }

        // 3) Классические имена
        if (!got_triplet) {
            bool ok = (ex.extract("boxes",  m_boxes)  == 0) &&
                      (ex.extract("scores", m_scores) == 0) &&
                      (ex.extract("labels", m_labels) == 0);
            if (ok && m_boxes.total() && m_scores.total() && m_labels.total())
                got_triplet = true;
        }

        // 4) Единый Nx6: [x1 y1 x2 y2 score cls]
        if (!got_triplet) {
            if ((ex.extract("dets", dets) == 0) ||
                (ex.extract("out0", dets) == 0) ||
                (ex.extract("pnnx_output_0", dets) == 0))
            {
                if (dets.total() > 0) got_nx6 = true;
            }
        }

        // 5) SSD-стиль: [label score x1 y1 x2 y2]
        if (!got_triplet && !got_nx6) {
            if (ex.extract("detection_out", dets) == 0 && dets.total() > 0) {
                got_ssd = true;
            }
        }

        if (!got_triplet && !got_nx6 && !got_ssd) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "failed to extract outputs (no known pattern matched). Parsed outputs tried: %d", (int)output_names_.size());
            return {};
        }

        std::vector<Det> out_dets;

        if (got_triplet) {
            // Определяем N по boxes
            int N = 0;
            bool boxes_w4 = (m_boxes.w == 4 && m_boxes.h > 0);
            bool boxes_h4 = (m_boxes.h == 4 && m_boxes.w > 0);
            if (boxes_w4)      N = m_boxes.h;
            else if (boxes_h4) N = m_boxes.w;
            else               N = (int)std::min(m_scores.total(), m_labels.total());

            __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                                "triplet resolved: boxes(w=%d h=%d c=%d) scores(w=%d h=%d c=%d) labels(w=%d h=%d c=%d) N=%d",
                                m_boxes.w, m_boxes.h, m_boxes.c,
                                m_scores.w, m_scores.h, m_scores.c,
                                m_labels.w, m_labels.h, m_labels.c, N);

            for (int i = 0; i < N; ++i) {
                float x1,y1,x2,y2;
                if (boxes_w4) {
                    const float* b = m_boxes.row(i); // [4]
                    x1=b[0]; y1=b[1]; x2=b[2]; y2=b[3];
                } else if (boxes_h4) {
                    const float* r0 = m_boxes.row(0);
                    const float* r1 = m_boxes.row(1);
                    const float* r2 = m_boxes.row(2);
                    const float* r3 = m_boxes.row(3);
                    x1=r0[i]; y1=r1[i]; x2=r2[i]; y2=r3[i];
                } else {
                    x1 = m_boxes[i*4+0];
                    y1 = m_boxes[i*4+1];
                    x2 = m_boxes[i*4+2];
                    y2 = m_boxes[i*4+3];
                }

                float score = (m_scores.total() > i) ? m_scores[i] : 0.f;
                if (score < conf_thr) continue;

                int cls = 0;
                if (m_labels.total() > i) cls = (int)std::round(m_labels[i]);

                out_dets.push_back({x1,y1,x2,y2,score,cls});
            }
        }
        else if (got_nx6) {
            int N = dets.h ? dets.h : dets.w;
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "Nx6 dets: N=%d (w=%d h=%d c=%d)", N, dets.w, dets.h, dets.c);
            for (int i=0; i<N; ++i) {
                const float* p = dets.row(i);
                float score = p[4];
                if (score < conf_thr) continue;
                out_dets.push_back({p[0],p[1],p[2],p[3],score,(int)p[5]});
            }
        }
        else if (got_ssd) {
            int N = dets.h ? dets.h : dets.w;
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "SSD dets: N=%d (w=%d h=%d c=%d)", N, dets.w, dets.h, dets.c);
            for (int i=0; i<N; ++i) {
                const float* p = dets.row(i);
                float score = p[1];
                if (score < conf_thr) continue;
                out_dets.push_back({p[2],p[3],p[4],p[5],score,(int)p[0]});
            }
        }

        // Масштаб обратно к исходному кадру + клип
        float inv = 1.f / scale;
        for (auto &d : out_dets) {
            d.x1 = std::max(0.f, std::min((float)w, d.x1 * inv));
            d.y1 = std::max(0.f, std::min((float)h, d.y1 * inv));
            d.x2 = std::max(0.f, std::min((float)w, d.x2 * inv));
            d.y2 = std::max(0.f, std::min((float)h, d.y2 * inv));
        }

        std::sort(out_dets.begin(), out_dets.end(),
                  [](const Det& a, const Det& b){ return a.score > b.score; });

        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "detections: %d", (int)out_dets.size());
        return out_dets;
    }

private:
    ncnn::Net net;
    AAssetManager* mgr_ = nullptr;
    std::string param_asset_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

// Глобальный инстанс, если JNI-бридж к нему обращается
static FasterRCNN g_frcnn;
