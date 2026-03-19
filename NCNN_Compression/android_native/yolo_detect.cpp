#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "net.h"

struct Args {
    std::string param_path;
    std::string bin_path;
    std::string image_ppm;
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.6f;
    int max_det = 100;
};

static bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) return nullptr;
            i++;
            return argv[i];
        };

        if (k == "--param") {
            const char* v = need("--param");
            if (!v) return false;
            a.param_path = v;
        } else if (k == "--bin") {
            const char* v = need("--bin");
            if (!v) return false;
            a.bin_path = v;
        } else if (k == "--image") {
            const char* v = need("--image");
            if (!v) return false;
            a.image_ppm = v;
        } else if (k == "--imgsz") {
            const char* v = need("--imgsz");
            if (!v) return false;
            a.imgsz = std::atoi(v);
        } else if (k == "--conf") {
            const char* v = need("--conf");
            if (!v) return false;
            a.conf = std::atof(v);
        } else if (k == "--iou") {
            const char* v = need("--iou");
            if (!v) return false;
            a.iou = std::atof(v);
        } else if (k == "--max_det") {
            const char* v = need("--max_det");
            if (!v) return false;
            a.max_det = std::atoi(v);
        } else {
            std::cerr << "Unknown arg: " << k << "\n";
            return false;
        }
    }
    if (a.param_path.empty() || a.bin_path.empty() || a.image_ppm.empty()) return false;
    return true;
}

struct PPMImage {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb; // size w*h*3
};

static bool read_ppm_p6(const std::string& path, PPMImage& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    std::string magic;
    f >> magic;
    if (magic != "P6") return false;

    int w = 0, h = 0, maxv = 0;
    f >> w >> h >> maxv;
    f.get(); // consume single whitespace/newline after maxv

    if (w <= 0 || h <= 0 || maxv != 255) return false;

    out.w = w;
    out.h = h;
    out.rgb.resize((size_t)w * (size_t)h * 3);

    f.read((char*)out.rgb.data(), out.rgb.size());
    return f.good();
}

struct LetterboxInfo {
    float r = 1.f;
    float dw = 0.f;
    float dh = 0.f;
};

static ncnn::Mat letterbox_to_square(const PPMImage& im, int imgsz, LetterboxInfo& info) {
    int w = im.w, h = im.h;
    float r = std::min((float)imgsz / (float)h, (float)imgsz / (float)w);
    int nw = (int)std::round(w * r);
    int nh = (int)std::round(h * r);
    info.r = r;
    info.dw = (imgsz - nw) / 2.0f;
    info.dh = (imgsz - nh) / 2.0f;

    // resize using ncnn from_pixels_resize
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(im.rgb.data(), ncnn::Mat::PIXEL_RGB, w, h, nw, nh);

    // pad to imgsz with 114
    ncnn::Mat out(imgsz, imgsz, 3);
    out.fill(114.f);

    for (int c = 0; c < 3; c++) {
        float* outp = out.channel(c);
        const float* inp = resized.channel(c);
        for (int y = 0; y < nh; y++) {
            int oy = (int)info.dh + y;
            if (oy < 0 || oy >= imgsz) continue;
            for (int x = 0; x < nw; x++) {
                int ox = (int)info.dw + x;
                if (ox < 0 || ox >= imgsz) continue;
                outp[oy * imgsz + ox] = inp[y * nw + x];
            }
        }
    }

    // normalize to 0..1
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    out.substract_mean_normalize(mean_vals, norm_vals);
    return out;
}

static bool parse_ncnn_param_io(const std::string& param_path, std::string& in_blob, std::string& out_blob) {
    std::ifstream f(param_path);
    if (!f) return false;

    std::string line;
    bool got_in = false;
    out_blob.clear();

    // skip first two lines (magic + counts) but we can just read all
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        // tokenization
        std::vector<std::string> tok;
        {
            std::string cur;
            for (char ch : line) {
                if (ch == ' ' || ch == '\t') {
                    if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
                } else {
                    cur.push_back(ch);
                }
            }
            if (!cur.empty()) tok.push_back(cur);
        }
        if (tok.size() < 4) continue;

        std::string type = tok[0];
        int bottom = std::atoi(tok[2].c_str());
        int top = std::atoi(tok[3].c_str());

        // blobs start at idx=4
        if (type == "Input" && bottom == 0 && top >= 1) {
            // tops begin at idx 4+bottom
            int tops_idx = 4 + bottom;
            if ((int)tok.size() > tops_idx) {
                in_blob = tok[tops_idx];
                got_in = true;
            }
        }

        // last top blob is a reasonable output guess
        if (top >= 1) {
            int tops_idx = 4 + bottom;
            if ((int)tok.size() >= tops_idx + top) {
                out_blob = tok[tops_idx + top - 1];
            }
        }
    }
    return got_in && !out_blob.empty();
}

struct Det {
    int cls = -1;
    float score = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
};

static float iou(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float areaA = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float areaB = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    float uni = areaA + areaB - inter + 1e-9f;
    return inter / uni;
}

static std::vector<Det> nms_classwise(std::vector<Det> dets, float iou_thr, int max_det) {
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.score > b.score; });

    std::vector<Det> keep;
    keep.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); i++) {
        const Det& d = dets[i];
        bool suppressed = false;
        for (const auto& k : keep) {
            if (k.cls == d.cls) {
                if (iou(k, d) > iou_thr) {
                    suppressed = true;
                    break;
                }
            }
        }
        if (!suppressed) {
            keep.push_back(d);
            if ((int)keep.size() >= max_det) break;
        }
    }
    return keep;
}

static void decode_yolov8_like(
    const ncnn::Mat& out,
    const LetterboxInfo& lb,
    int orig_w,
    int orig_h,
    float conf_thr,
    std::vector<Det>& dets
) {
    // Attempt to interpret out as [attr, n] or [n, attr]
    int dims = out.dims;
    int w = out.w, h = out.h, c = out.c;

    // Flatten into 2D view via pointer access.
    // For most YOLO exports: out is 2D: w=n, h=attr (or swapped), c=1.
    int dim0 = h;
    int dim1 = w;
    if (dims == 3) {
        // Sometimes (w, h, c). If c==1, treat as 2D.
        if (c == 1) {
            dim0 = h;
            dim1 = w;
        } else {
            // fallback: collapse c into dim0
            dim0 = h * c;
            dim1 = w;
        }
    }

    int attr = std::min(dim0, dim1);
    int n = std::max(dim0, dim1);
    bool attr_by_row = (dim0 <= dim1); // rows=attr, cols=n

    if (attr < 6) {
        // too small, can't decode
        return;
    }

    int nc = attr - 4;

    auto get_val = [&](int row, int col) -> float {
        // out is stored row-major for 2D: access via out.row(i)
        // If attr_by_row: row=attr, col=n => value at [row, col]
        // else: row=n, col=attr => value at [row, col]
        if (attr_by_row) {
            const float* rptr = out.row(row);
            return rptr[col];
        } else {
            const float* rptr = out.row(row);
            return rptr[col];
        }
    };

    // If not attr_by_row, swap interpretation
    if (!attr_by_row) {
        // then dim0=n, dim1=attr
        // get_val(row=n_i, col=attr_j) ok
    }

    for (int i = 0; i < n; i++) {
        float x, y, ww, hh;
        if (attr_by_row) {
            x = get_val(0, i);
            y = get_val(1, i);
            ww = get_val(2, i);
            hh = get_val(3, i);
        } else {
            // row=i, col=j
            const float* rptr = out.row(i);
            x = rptr[0];
            y = rptr[1];
            ww = rptr[2];
            hh = rptr[3];
        }

        int best_cls = -1;
        float best_score = 0.f;

        if (attr_by_row) {
            for (int cls = 0; cls < nc; cls++) {
                float s = get_val(4 + cls, i);
                if (s > best_score) { best_score = s; best_cls = cls; }
            }
        } else {
            const float* rptr = out.row(i);
            for (int cls = 0; cls < nc; cls++) {
                float s = rptr[4 + cls];
                if (s > best_score) { best_score = s; best_cls = cls; }
            }
        }

        if (best_score < conf_thr) continue;

        // map back from padded/imgsz space to original image space
        float px = (x - lb.dw) / lb.r;
        float py = (y - lb.dh) / lb.r;
        float pw = ww / lb.r;
        float ph = hh / lb.r;

        float x1 = px - pw / 2.f;
        float y1 = py - ph / 2.f;
        float x2 = px + pw / 2.f;
        float y2 = py + ph / 2.f;

        x1 = std::max(0.f, std::min(x1, (float)orig_w));
        y1 = std::max(0.f, std::min(y1, (float)orig_h));
        x2 = std::max(0.f, std::min(x2, (float)orig_w));
        y2 = std::max(0.f, std::min(y2, (float)orig_h));

        Det d;
        d.cls = best_cls;
        d.score = best_score;
        d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
        dets.push_back(d);
    }
}

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        std::cerr << "Usage:\n"
                  << "  xtrim_yolo_detect --param model.param --bin model.bin --image input.ppm "
                  << "--imgsz 640 --conf 0.25 --iou 0.6 --max_det 100\n";
        return 2;
    }

    std::string in_blob, out_blob;
    if (!parse_ncnn_param_io(a.param_path, in_blob, out_blob)) {
        std::cerr << "Failed to parse NCNN param for IO blobs: " << a.param_path << "\n";
        return 3;
    }
    std::cout << "# io: input=" << in_blob << " output=" << out_blob << "\n";

    PPMImage im;
    if (!read_ppm_p6(a.image_ppm, im)) {
        std::cerr << "Failed to read PPM P6 image: " << a.image_ppm << "\n";
        return 4;
    }
    std::cout << "# image_wh=" << im.w << "x" << im.h << "\n";

    LetterboxInfo lb;
    ncnn::Mat in = letterbox_to_square(im, a.imgsz, lb);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;

    if (net.load_param(a.param_path.c_str()) != 0) {
        std::cerr << "load_param failed\n";
        return 5;
    }
    if (net.load_model(a.bin_path.c_str()) != 0) {
        std::cerr << "load_model failed\n";
        return 6;
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input(in_blob.c_str(), in);

    ncnn::Mat out;
    if (ex.extract(out_blob.c_str(), out) != 0) {
        std::cerr << "extract failed\n";
        return 7;
    }

    std::vector<Det> dets;
    dets.reserve(1024);
    decode_yolov8_like(out, lb, im.w, im.h, a.conf, dets);

    auto keep = nms_classwise(dets, a.iou, a.max_det);

    std::cout << "# det_count=" << keep.size() << "\n";
    for (const auto& d : keep) {
        std::cout << d.cls << " " << d.score << " "
                  << d.x1 << " " << d.y1 << " " << d.x2 << " " << d.y2 << "\n";
    }
    return 0;
}