    #include "yolov8.hpp"
#include <algorithm>
#include <cmath>

static inline void read_pixel_rotated(const uint8_t* src,
                                      int srcW, int srcH, int rowStride,
                                      int rot, int x, int y,
                                      uint8_t& R, uint8_t& G, uint8_t& B) {
    int sx = x, sy = y;
    if (rot == 90)      { sx = y;          sy = srcW - 1 - x; }
    else if (rot == 180){ sx = srcW - 1 - x; sy = srcH - 1 - y; }
    else if (rot == 270){ sx = srcH - 1 - y; sy = x; }
    
    // Bounds check to prevent out-of-bounds memory access
    if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) {
        R = 114; G = 114; B = 114;  // padding color
        return;
    }
    
    const uint8_t* p = src + sy * rowStride + sx * 4;
    R = p[0]; G = p[1]; B = p[2];
}

bool YoloV8::load(AAssetManager* mgr, const char* param, const char* bin) {
    net.opt.use_vulkan_compute = true; // если доступно
    return net.load_param(mgr, param) == 0 && net.load_model(mgr, bin) == 0;
}

std::vector<Det> YoloV8::detect_rgba(const uint8_t* rgba, int srcW, int srcH, int rowStride,
                                     int rot, float conf_thr, float iou_thr, int dst) {
    // размеры после учёта поворота
    int w = (rot == 90 || rot == 270) ? srcH : srcW;
    int h = (rot == 90 || rot == 270) ? srcW : srcH;

    // letterbox → ncnn::Mat dst×dst×3 (float32)
    ncnn::Mat in(dst, dst, 3);
    float r = std::min(dst / (float)w, dst / (float)h);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);
    int pad_w = dst - new_w, pad_h = dst - new_h;

    // Get channel pointers for proper planar access
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
            // ncnn::Mat is channel-planar, not interleaved
            int idx = y * dst + x;
            ch0[idx] = R/255.f;
            ch1[idx] = G/255.f;
            ch2[idx] = B/255.f;
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    // ВАЖНО: используй ТОЧНЫЕ имена из твоего .param
    // (если уже подставил их ранее — оставь как есть)
    if (ex.input("in0", in) != 0 && ex.input("images", in) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "yolo", "ex.input failed (нет блоба in0/images)");
        return {};
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 && ex.extract("output0", out) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "yolo", "ex.extract failed (нет блоба out0/output0)");
        return {};
    }

    __android_log_print(ANDROID_LOG_INFO, "yolo", "out shape: w=%d h=%d c=%d", out.w, out.h, out.c);

    // ------ Универсальный доступ к предсказанию i как к вектору признаков ------
    // Мы приведем любую раскладку к массиву feat[no] = [x,y,w,h,(obj?), class...]
    auto decode_one = [&](int i, std::vector<float>& feat)->bool {
        feat.clear();

        if (out.c == 1) {
            const int W = out.w, H = out.h;       // если H==num, W==no — идеально
            const float* base = (const float*)out.data;
            // Проверим два варианта раскладки
            // Вариант A: строки — предсказания (H==num, W==no)
            if (i < H && W >= 6) {
                feat.resize(W);
                const float* row = out.row(i);      // безопасно: i < H
                std::copy(row, row + W, feat.begin());
                return true;
            }
            // Вариант B: столбцы — предсказания (W==num, H==no)
            if (i < W && H >= 6) {
                feat.resize(H);
                // feat[j] = M[j][i]
                for (int j = 0; j < H; ++j) feat[j] = base[j*W + i];
                return true;
            }
            return false;
        } else {
            // Часто встречается: c==num, h==1, w==no
            if (out.h == 1 && out.w >= 6 && i < out.c) {
                const ncnn::Mat ch = out.channel(i);
                feat.resize(out.w);
                // ch.row(0) корректен при h==1
                const float* row = ch.row(0);
                std::copy(row, row + out.w, feat.begin());
                return true;
            }
            // Иной экзотический случай — можно добавить еще ветки по необходимости
            return false;
        }
    };

    // ------ Преобразование боксов обратно к исходному кадру ------
    auto inv_scale = [&](float& bx, float& by, float& bw, float& bh){
        float rbx = (bx - pad_w/2)/r, rby = (by - pad_h/2)/r;
        float rbw = bw / r, rbh = bh / r;

        float cx, cy;
        if      (rot == 0)   { cx = rbx;            cy = rby; }
        else if (rot == 90)  { cx = srcW - 1 - rby; cy = rbx; }
        else if (rot == 180) { cx = srcW - 1 - rbx; cy = srcH - 1 - rby; }
        else                 { cx = rby;            cy = srcH - 1 - rbx; }

        bx = cx; by = cy; bw = rbw; bh = rbh;
    };

    // out: w = 8400 (num_preds), h = 84 (no), c = 1
    const int num_preds = out.w;
    const int no = out.h;           // 84
    const float* base = (const float*)out.data;

    auto get_feat_col = [&](int i, std::vector<float>& f){
        f.resize(no);
        // берем столбец i: f[j] = out[j][i]
        for (int j = 0; j < no; ++j) f[j] = base[j * num_preds + i];
    };

    std::vector<Det> props;
    std::vector<float> f;

    for (int i = 0; i < num_preds; ++i) {
        get_feat_col(i, f);
        // У YOLOv8 export часто НЕТ objness: 4 + nc
        const int cls_start = (no == 84) ? 4 : 5;  // если у тебя другой датасет, подстрой
        if (no - cls_start <= 0) continue;

        float x = f[0], y = f[1], bw = f[2], bh = f[3];
        float obj = (cls_start == 5) ? f[4] : 1.f;

        int cls=-1; float best=0.f;
        for (int c = cls_start; c < no; ++c) if (f[c] > best){ best=f[c]; cls=c-cls_start; }
        float score = obj * best;
        if (score < conf_thr) continue;

        inv_scale(x,y,bw,bh);
        float x1 = std::max(0.f, x - bw/2), y1 = std::max(0.f, y - bh/2);
        float x2 = std::min((float)srcW, x + bw/2), y2 = std::min((float)srcH, y + bh/2);
        props.push_back({x1,y1,x2,y2,score,cls});
    }


    // ------ NMS ------
    std::sort(props.begin(), props.end(), [](const Det& a,const Det& b){ return a.score>b.score; });
    std::vector<Det> keep; std::vector<char> rm(props.size(), 0);
    auto iou = [](const Det& a, const Det& b){
        float xx1=std::max(a.x1,b.x1), yy1=std::max(a.y1,b.y1);
        float xx2=std::min(a.x2,b.x2), yy2=std::min(a.y2,b.y2);
        float w=std::max(0.f,xx2-xx1), h=std::max(0.f,yy2-yy1);
        float inter=w*h, ua=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
        return inter/(ua+1e-6f);
    };
    for (int i=0;i<(int)props.size();++i){
        if (rm[i]) continue; keep.push_back(props[i]);
        for (int j=i+1;j<(int)props.size();++j)
            if (!rm[j] && iou(props[i],props[j])>iou_thr) rm[j]=1;
    }
    return keep;

}
