macOS (Apple Silicon) menu-bar build with the bundled Metal sd-server engine. Signed with a Developer ID and notarized by Apple, so it opens normally. Install: open the .dmg and drag oflux.app to Applications; the fox appears in your menu bar and models download on demand. Verify downloads against SHA256SUMS.

## The web UI is now a board

`http://127.0.0.1:11534` opens on an infinite canvas instead of a form. Images
are objects you drop, paste, move, scale, rotate, flip and stack.

- **Smart select** — press `S` and click part of an image; SAM3 finds it. Click
  again to add, Alt-click to remove, drag a box, or find it by words. Brush and
  rubber refine the selection by hand.
- **Cut out / Remove background / Erase** — lift the selected part out as its
  own object, keep only it, or remove it and have the model fill the gap.
- **Prompt bar** — with something selected it edits that; with nothing selected
  it generates a new image onto the board. Progress shows on the board and the
  job can be cancelled.
- **Text objects**, **undo/redo**, **export or copy as PNG**, and the board is
  autosaved in the browser so a reload does not lose it.

Without SAM3 (a `SAM3=0` build) the smart tools hide and the brush still works.
Manage is unchanged.

## `oflux edit` and `oflux generate`

The CLI can finally make an image instead of printing a curl line:

```
oflux edit qwe-2511 photo.png "make it night time" -o night.png
oflux generate zim-1-turbo "a fox in a spacesuit"
```

`edit` takes several images (subject first, references after) with the prompt
last. Both stream progress, save a PNG and print the seed. `oflux help` lists
the flags.

## Fixes

- An edit that named only `width` or only `height` got the engine's 512 for the
  other side. The missing side now follows the image's aspect ratio.
- `oflux run` suggested `/v1/edit` even for generate-only models; it now prints
  the command the model actually supports.
- `oflux rm` accepts a preset name, since `oflux ls` lists presets beside models.
