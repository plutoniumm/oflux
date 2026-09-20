//go:build sam3

#include "bridge.h"

#include "sam3.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct oflux_sam3 {
    std::shared_ptr<sam3_model> model;
    sam3_params                 params;
};

static void set_err(char *err, size_t errlen, const char *msg) {
    if (!err || errlen == 0) return;
    std::strncpy(err, msg, errlen - 1);
    err[errlen - 1] = '\0';
}

// Every entry point is wrapped: an exception escaping into Go is an immediate
// crash, and libsam3 allocates std::vectors sized from model hyperparameters.
#define OFLUX_CATCH(err, errlen, failval)                          \
    catch (const std::exception &e) {                              \
        set_err((err), (errlen), e.what());                        \
        return failval;                                            \
    } catch (...) {                                                \
        set_err((err), (errlen), "unknown C++ exception");         \
        return failval;                                            \
    }

oflux_sam3 *oflux_sam3_open(const char *model_path, int n_threads, int use_gpu,
                            char *err, size_t errlen) {
    if (!model_path || !*model_path) {
        set_err(err, errlen, "empty model path");
        return nullptr;
    }
    try {
        sam3_params params;
        params.model_path = model_path;
        params.n_threads  = n_threads > 0 ? n_threads : 4;
        params.use_gpu    = use_gpu != 0;

        std::shared_ptr<sam3_model> model = sam3_load_model(params);
        if (!model) {
            set_err(err, errlen, "sam3_load_model failed (see stderr)");
            return nullptr;
        }
        // Text prompts go through sam3_segment_pcs, which bails out on
        // visual-only and SAM2 checkpoints. Catch that here, at load, rather
        // than as an empty result on every request.
        if (sam3_is_visual_only(*model) || sam3_get_model_type(*model) != SAM3_MODEL_SAM3) {
            sam3_free_model(*model);
            set_err(err, errlen,
                    "checkpoint has no text encoder (needs a full sam3-*.ggml, not sam3-visual/sam2/edgetam)");
            return nullptr;
        }
        return new oflux_sam3{std::move(model), params};
    }
    OFLUX_CATCH(err, errlen, nullptr)
}

void oflux_sam3_close(oflux_sam3 *h) {
    if (!h) return;
    // ~sam3_model does not free the ggml context, backend or buffer — those
    // are raw pointers, and sam3_free_model is what releases them. It nulls
    // each one as it goes, so dropping the shared_ptr afterwards is safe.
    if (h->model) sam3_free_model(*h->model);
    delete h;
}

void oflux_sam3_free_masks(oflux_sam3_mask *masks, int count) {
    if (!masks) return;
    for (int i = 0; i < count; ++i) std::free(masks[i].pixels);
    std::free(masks);
}

static oflux_sam3_box to_c_box(const sam3_box &b) {
    oflux_sam3_box o;
    o.x0 = b.x0;
    o.y0 = b.y0;
    o.x1 = b.x1;
    o.y1 = b.y1;
    return o;
}

int oflux_sam3_segment(oflux_sam3 *h,
                       const unsigned char *rgb, int width, int height,
                       const char *prompt,
                       const oflux_sam3_point *points, int n_points,
                       const oflux_sam3_box *box,
                       float score_threshold, float nms_threshold,
                       int separate,
                       oflux_sam3_mask **out_masks, int *out_count,
                       char *err, size_t errlen) {
    if (!h || !h->model || !rgb || !out_masks || !out_count ||
        width <= 0 || height <= 0 || n_points < 0 || (n_points > 0 && !points)) {
        set_err(err, errlen, "invalid argument");
        return -1;
    }
    *out_masks = nullptr;
    *out_count = 0;

    const bool pcs = prompt && *prompt;
    if (!pcs && n_points == 0 && !box) {
        set_err(err, errlen, "no prompt: need a text prompt, a point or a box");
        return -1;
    }

    try {
        const size_t npix = (size_t)width * (size_t)height;

        sam3_image image;
        image.width    = width;
        image.height   = height;
        image.channels = 3;
        image.data.assign(rgb, rgb + npix * 3);

        // A fresh state per call. sam3_create_state copies the model's ggml
        // backend into the state, so states are not independent and must not
        // overlap; the Go side serialises calls for the same reason.
        sam3_state_ptr state = sam3_create_state(*h->model, h->params);
        if (!state) {
            set_err(err, errlen, "sam3_create_state failed");
            return -1;
        }
        if (!sam3_encode_image(*state, *h->model, image)) {
            set_err(err, errlen, "sam3_encode_image failed (see stderr)");
            return -1;
        }

        sam3_result result;
        if (pcs) {
            sam3_pcs_params p;
            p.text_prompt     = prompt;
            p.score_threshold = score_threshold > 0.0f ? score_threshold : 0.5f;
            p.nms_threshold   = nms_threshold > 0.0f ? nms_threshold : 0.1f;
            result = sam3_segment_pcs(*state, *h->model, p);
        } else {
            sam3_pvs_params p;
            for (int i = 0; i < n_points; ++i) {
                sam3_point pt;
                pt.x = points[i].x;
                pt.y = points[i].y;
                if (points[i].negative) p.neg_points.push_back(pt);
                else                    p.pos_points.push_back(pt);
            }
            if (box) {
                p.box.x0  = box->x0;
                p.box.y0  = box->y0;
                p.box.x1  = box->x1;
                p.box.y1  = box->y1;
                p.use_box = true;
            }
            result = sam3_segment_pvs(*state, *h->model, p);
        }

        // result owns its detections and their mask vectors; both die with
        // this scope, which is why everything kept is copied out to calloc'd
        // memory the caller frees.
        std::vector<const sam3_detection *> dets;
        for (const sam3_detection &d : result.detections) {
            if (d.mask.width != width || d.mask.height != height) continue;
            if (d.mask.data.size() != npix) continue;
            dets.push_back(&d);
        }
        if (dets.empty()) return 0;

        std::stable_sort(dets.begin(), dets.end(),
                         [](const sam3_detection *a, const sam3_detection *b) {
                             return a->score > b->score;
                         });

        const int count = separate ? (int)dets.size() : 1;
        oflux_sam3_mask *out =
            (oflux_sam3_mask *)std::calloc((size_t)count, sizeof(oflux_sam3_mask));
        if (!out) {
            set_err(err, errlen, "out of memory allocating masks");
            return -1;
        }

        if (separate) {
            for (int i = 0; i < count; ++i) {
                out[i].pixels = (unsigned char *)std::calloc(npix, 1);
                if (!out[i].pixels) {
                    oflux_sam3_free_masks(out, count); // calloc zeroed the rest
                    set_err(err, errlen, "out of memory allocating mask");
                    return -1;
                }
                const uint8_t *src = dets[i]->mask.data.data();
                for (size_t p = 0; p < npix; ++p)
                    if (src[p]) out[i].pixels[p] = 255;
                out[i].score = dets[i]->score;
                out[i].box   = to_c_box(dets[i]->box);
            }
        } else {
            out[0].pixels = (unsigned char *)std::calloc(npix, 1);
            if (!out[0].pixels) {
                oflux_sam3_free_masks(out, count);
                set_err(err, errlen, "out of memory allocating mask");
                return -1;
            }
            sam3_box u = dets[0]->box;
            for (const sam3_detection *d : dets) {
                const uint8_t *src = d->mask.data.data();
                for (size_t p = 0; p < npix; ++p)
                    if (src[p]) out[0].pixels[p] = 255;
                u.x0 = std::min(u.x0, d->box.x0);
                u.y0 = std::min(u.y0, d->box.y0);
                u.x1 = std::max(u.x1, d->box.x1);
                u.y1 = std::max(u.y1, d->box.y1);
            }
            out[0].score = dets[0]->score; // sorted: the best of the merge
            out[0].box   = to_c_box(u);
        }

        *out_masks = out;
        *out_count = count;
        return 0;
    }
    OFLUX_CATCH(err, errlen, -1)
}
