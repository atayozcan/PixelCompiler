//===- pixel_runtime.h - Pixel Runtime Library --------------------------===//
// Runtime support functions for image loading, saving, and manipulation
//===----------------------------------------------------------------------===//
#pragma once
#include <cstdint>

extern "C" {
//===----------------------------------------------------------------------===//
// Image Structure
//===----------------------------------------------------------------------===//
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels; // 1=grayscale, 3=RGB, 4=RGBA
    uint8_t *data;
} PixelImage;

//===----------------------------------------------------------------------===//
// Image I/O Functions
//===----------------------------------------------------------------------===//
// Load an image from file (supports BMP, PNG, JPG)
PixelImage *pixel_load_image(const char *filepath);

// Save an image to file
// format: "bmp", "png", "jpg"
int pixel_save_image(const PixelImage *img, const char *filepath,
                     const char *format);

// Free image memory
void pixel_free_image(PixelImage *img);

//===----------------------------------------------------------------------===//
// Image Processing Functions
//===----------------------------------------------------------------------===//

// Invert image colors (255 - value)
PixelImage *pixel_invert(const PixelImage *img);

// Convert to grayscale
PixelImage *pixel_grayscale(const PixelImage *img);

// Apply blur filter
PixelImage *pixel_blur(const PixelImage *img, int kernel_size);

// Rotate image by angle (degrees)
PixelImage *pixel_rotate(const PixelImage *img, float angle);

// Adjust brightness (-255 to 255)
PixelImage *pixel_brightness(const PixelImage *img, int adjustment);

// Adjust contrast (0.0 to 2.0, 1.0 = no change)
PixelImage *pixel_contrast(const PixelImage *img, float factor);

// Crop image
PixelImage *pixel_crop(const PixelImage *img, uint32_t x, uint32_t y,
                       uint32_t width, uint32_t height);

// Resize image
PixelImage *pixel_resize(const PixelImage *img, uint32_t new_width,
                         uint32_t new_height);

// Flip horizontally
PixelImage *pixel_flip_horizontal(const PixelImage *img);

// Flip vertically
PixelImage *pixel_flip_vertical(const PixelImage *img);

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// Create a blank image
PixelImage *pixel_create_image(uint32_t width, uint32_t height,
                               uint32_t channels);

// Clone an image
PixelImage *pixel_clone_image(const PixelImage *img);

// Get pixel value at (x, y, channel)
uint8_t pixel_get_pixel(const PixelImage *img, uint32_t x, uint32_t y,
                        uint32_t channel);

// Set pixel value at (x, y, channel)
void pixel_set_pixel(PixelImage *img, uint32_t x, uint32_t y, uint32_t channel,
                     uint8_t value);
}
