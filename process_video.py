#!/usr/bin/env python3
"""
process_video.py
Usage:
    python process_video.py input.mp4 output.mp4

This script applies a set of visual effects to approximate
the reference video's "look": color grading, contrast,
film grain, chromatic aberration (slight RGB shift),
scanlines and vignette.
Requires: moviepy, numpy, pillow, imageio-ffmpeg
"""
import sys
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def make_scanline_image(w, h, intensity=0.08, line_height=1, gap=2):
    """
    Create a semi-transparent scanline image (RGBA) with horizontal dark lines.
    intensity: darkness of lines (0-1)
    line_height: pixel rows of each dark line
    gap: pixel rows between dark lines
    """
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    alpha_line = int(255 * intensity)
    y = 0
    while y < h:
        draw.rectangle([0, y, w, min(h, y + line_height - 1)], fill=(0,0,0,alpha_line))
        y += line_height + gap
    # slight blur for softness
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img

def make_vignette_mask(w, h, strength=0.6):
    """
    Return an RGB image (W,H,3) with multiplicative vignette (0..1)
    center is 1.0, edges down to (1-strength)
    """
    cx, cy = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx*cx + cy*cy)
    Y, X = np.ogrid[0:h, 0:w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    vign = 1.0 - strength * (dist / max_dist)
    vign = np.clip(vign, 0.0, 1.0)
    # repeat to 3 channels
    vign3 = np.dstack([vign, vign, vign]).astype(np.float32)
    return vign3

def frame_effect_factory(width, height):
    """
    Returns a function that will be applied to each frame (numpy uint8 HxWx3).
    The function performs:
      - Slight RGB channel offset (chromatic aberration)
      - Contrast and brightness adjustment
      - Film grain (random noise)
      - Slight blur/sharpen via PIL (optional)
    """
    scanline_img = make_scanline_image(width, height, intensity=0.08, line_height=1, gap=2)
    vignette = make_vignette_mask(width, height, strength=0.45)

    def effect(frame):
        # frame: H x W x 3, uint8
        img = frame.astype(np.int16)

        # chromatic aberration: shift R left, B right a few pixels
        shift = max(1, width // 640 * 2)  # scale small shift with width
        r = np.roll(img[:, :, 0], -shift, axis=1)
        g = img[:, :, 1]
        b = np.roll(img[:, :, 2], shift, axis=1)
        combined = np.stack([r, g, b], axis=2).astype(np.int16)

        # contrast and brightness tweak (simple linear transform around 128)
        combined = 128 + 1.06 * (combined - 128) + 4  # gain + bias
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # film grain (gaussian noise)
        grain_std = 6  # adjust grain amount
        noise = np.random.normal(loc=0, scale=grain_std, size=combined.shape).astype(np.int16)
        combined = np.clip(combined.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # convert to PIL for slight unsharp mask for crispiness
        pil = Image.fromarray(combined)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        # overlay scanlines (multiply-ish)
        scan = scanline_img.convert("RGBA")
        base_rgba = pil.convert("RGBA")
        base_rgba.alpha_composite(scan)  # blend scanlines on top (alpha composite)
        pil2 = base_rgba.convert("RGB")

        # apply vignette (multiply)
        arr = np.asarray(pil2).astype(np.float32) / 255.0
        arr = arr * vignette
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        return arr

    return effect

def process(in_path, out_path):
    print("Loading clip:", in_path)
    clip = VideoFileClip(in_path)
    # target width (keep aspect ratio)
    target_w = 1280
    if clip.w > target_w:
        clip = clip.resize(width=target_w)

    w, h = clip.w, clip.h
    print(f"Processing at {w}x{h}, fps={clip.fps:.2f}, duration={clip.duration:.1f}s")

    # color boost via moviepy colorx (saturation-like)
    clip = clip.fx(lambda c: c)  # no-op placeholder

    # Apply frame-by-frame effect
    effect = frame_effect_factory(w, h)
    processed = clip.fl_image(effect)

    # Write file
    # Use x264 with moderate quality; keep audio
    print("Writing output:", out_path)
    processed.write_videofile(out_path,
                              codec="libx264",
                              audio_codec="aac",
                              temp_audiofile="temp-audio.m4a",
                              remove_temp=True,
                              bitrate="4M",
                              threads=4,
                              preset="medium",
                              ffmpeg_params=["-pix_fmt", "yuv420p"])
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_video.py input.mp4 output.mp4")
        sys.exit(1)
    in_p = sys.argv[1]
    out_p = sys.argv[2]
    process(in_p, out_p)
