//go:build sam3

// C surface over sam3.cpp's C++ API (std::string / std::vector / shared_ptr
// in the signatures, which cgo cannot call). Build-tagged so the default
// build never compiles it: third_party/ is gitignored.
//
// Ownership: oflux_sam3_open's handle must be released with oflux_sam3_close;
// *out_masks from oflux_sam3_segment is a calloc'd array whose entries each
// own a calloc'd pixel buffer, all released by oflux_sam3_free_masks. err is a
// caller-owned buffer, always NUL-terminated.

#ifndef OFLUX_SAM3_BRIDGE_H
#define OFLUX_SAM3_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oflux_sam3 oflux_sam3;

// Original-image pixel coordinates: libsam3 scales points and boxes by the
// encoded image's orig_width/orig_height itself.
typedef struct {
    float x, y;
    int   negative;
} oflux_sam3_point;

typedef struct {
    float x0, y0, x1, y1;
} oflux_sam3_box;

typedef struct {
    unsigned char *pixels; // width*height bytes of 0 or 255
    float          score;
    oflux_sam3_box box;
} oflux_sam3_mask;

oflux_sam3 *oflux_sam3_open(const char *model_path, int n_threads, int use_gpu,
                            char *err, size_t errlen);

void oflux_sam3_close(oflux_sam3 *h);

// A non-empty prompt selects the text detector (PCS) and ignores points/box; an
// empty one selects the interactive decoder (PVS), which needs at least one
// positive point or a box. separate=0 merges every detection into one mask.
// Returns 0 on success, leaving *out_count at 0 when nothing was found.
int oflux_sam3_segment(oflux_sam3 *h,
                       const unsigned char *rgb, int width, int height,
                       const char *prompt,
                       const oflux_sam3_point *points, int n_points,
                       const oflux_sam3_box *box,
                       float score_threshold, float nms_threshold,
                       int separate,
                       oflux_sam3_mask **out_masks, int *out_count,
                       char *err, size_t errlen);

void oflux_sam3_free_masks(oflux_sam3_mask *masks, int count);

#ifdef __cplusplus
}
#endif

#endif
