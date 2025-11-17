//===- pixel_runtime.cpp - Pixel Runtime Implementation -----------------===//
//
// Implementation of runtime support for BMP image processing
//
//===----------------------------------------------------------------------===//
#include "pixel_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
//===----------------------------------------------------------------------===//
// BMP File Format Structures
//===----------------------------------------------------------------------===//
#pragma pack(push, 1)
typedef struct {
  uint16_t type;      // Magic identifier: 0x4d42
  uint32_t size;      // File size in bytes
  uint16_t reserved1; // Not used
  uint16_t reserved2; // Not used
  uint32_t offset;    // Offset to image data in bytes
} BMPHeader;

typedef struct {
  uint32_t size;            // Header size in bytes
  int32_t width;            // Width of the image
  int32_t height;           // Height of the image
  uint16_t planes;          // Number of color planes
  uint16_t bits;            // Bits per pixel
  uint32_t compression;     // Compression type
  uint32_t imagesize;       // Image size in bytes
  int32_t xresolution;      // Pixels per meter
  int32_t yresolution;      // Pixels per meter
  uint32_t ncolors;         // Number of colors
  uint32_t importantcolors; // Important colors
} BMPInfoHeader;
#pragma pack(pop)

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//
PixelImage *pixel_create_image(const uint32_t width, const uint32_t height,
                               const uint32_t channels) {
  const auto img = static_cast<PixelImage *>(malloc(sizeof(PixelImage)));
  if (!img)
    return nullptr;

  img->width = width;
  img->height = height;
  img->channels = channels;
  img->data = static_cast<uint8_t *>(
      calloc(width * height * channels, sizeof(uint8_t)));

  if (!img->data) {
    free(img);
    return nullptr;
  }
  return img;
}

void pixel_free_image(PixelImage *img) {
  if (!img)
    return;
  if (img->data)
    free(img->data);
  free(img);
}

PixelImage *pixel_clone_image(const PixelImage *img) {
  if (!img)
    return nullptr;

  PixelImage *clone =
      pixel_create_image(img->width, img->height, img->channels);
  if (!clone)
    return nullptr;

  memcpy(clone->data, img->data, img->width * img->height * img->channels);
  return clone;
}

//===----------------------------------------------------------------------===//
// BMP I/O Functions
//===----------------------------------------------------------------------===//
PixelImage *pixel_load_image(const char *filepath) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file '%s'\n", filepath);
    return nullptr;
  }

  // Read BMP header
  BMPHeader header;
  if (fread(&header, sizeof(BMPHeader), 1, file) != 1) {
    fprintf(stderr, "Error: Cannot read BMP header\n");
    fclose(file);
    return nullptr;
  }

  // Verify BMP magic number
  if (header.type != 0x4D42) {
    fprintf(stderr, "Error: Not a valid BMP file\n");
    fclose(file);
    return nullptr;
  }

  // Read BMP info header
  BMPInfoHeader infoHeader;
  if (fread(&infoHeader, sizeof(BMPInfoHeader), 1, file) != 1) {
    fprintf(stderr, "Error: Cannot read BMP info header\n");
    fclose(file);
    return nullptr;
  }

  // Only support 24-bit RGB BMPs
  if (infoHeader.bits != 24) {
    fprintf(stderr, "Error: Only 24-bit BMP files are supported\n");
    fclose(file);
    return nullptr;
  }

  // Create image structure
  const uint32_t width = infoHeader.width;
  const uint32_t height = abs(infoHeader.height);
  PixelImage *img = pixel_create_image(width, height, 3);
  if (!img) {
    fclose(file);
    return nullptr;
  }

  // Seek to pixel data
  fseek(file, header.offset, SEEK_SET);

  // BMP rows are padded to 4-byte boundaries
  const uint32_t row_size = (width * 3 + 3) / 4 * 4;
  const auto row = static_cast<uint8_t *>(malloc(row_size));

  // Read pixel data (BMP stores bottom-to-top, BGR format)
  for (int32_t y = height - 1; y >= 0; y--) {
    if (fread(row, 1, row_size, file) != row_size) {
      fprintf(stderr, "Error: Failed to read pixel data\n");
      free(row);
      pixel_free_image(img);
      fclose(file);
      return nullptr;
    }

    // Convert BGR to RGB
    for (uint32_t x = 0; x < width; x++) {
      const uint32_t idx = (y * width + x) * 3;
      img->data[idx + 0] = row[x * 3 + 2]; // R
      img->data[idx + 1] = row[x * 3 + 1]; // G
      img->data[idx + 2] = row[x * 3 + 0]; // B
    }
  }

  free(row);
  fclose(file);
  return img;
}

int pixel_save_image(const PixelImage *img, const char *filepath,
                     const char *format) {
  if (!img || !filepath)
    return -1;

  // Only support BMP for now
  if (strcmp(format, "bmp") != 0) {
    fprintf(stderr, "Error: Only BMP format is currently supported\n");
    return -1;
  }

  FILE *file = fopen(filepath, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot create file '%s'\n", filepath);
    return -1;
  }

  // Calculate padding
  const uint32_t row_size = (img->width * 3 + 3) / 4 * 4;
  const uint32_t image_size = row_size * img->height;

  // Prepare headers
  BMPHeader header;
  header.type = 0x4D42;
  header.size = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + image_size;
  header.reserved1 = 0;
  header.reserved2 = 0;
  header.offset = sizeof(BMPHeader) + sizeof(BMPInfoHeader);

  BMPInfoHeader infoHeader;
  infoHeader.size = sizeof(BMPInfoHeader);
  infoHeader.width = img->width;
  infoHeader.height = img->height;
  infoHeader.planes = 1;
  infoHeader.bits = 24;
  infoHeader.compression = 0;
  infoHeader.imagesize = image_size;
  infoHeader.xresolution = 2835; // 72 DPI
  infoHeader.yresolution = 2835;
  infoHeader.ncolors = 0;
  infoHeader.importantcolors = 0;

  // Write headers
  fwrite(&header, sizeof(BMPHeader), 1, file);
  fwrite(&infoHeader, sizeof(BMPInfoHeader), 1, file);

  // Write pixel data (bottom-to-top, BGR format)
  const auto row = static_cast<uint8_t *>(calloc(row_size, 1));

  for (int32_t y = img->height - 1; y >= 0; y--) {
    // Convert RGB to BGR
    for (uint32_t x = 0; x < img->width; x++) {
      const uint32_t idx = (y * img->width + x) * img->channels;
      row[x * 3 + 0] =
          img->channels >= 3 ? img->data[idx + 2] : img->data[idx]; // B
      row[x * 3 + 1] =
          img->channels >= 2 ? img->data[idx + 1] : img->data[idx]; // G
      row[x * 3 + 2] = img->data[idx + 0];                          // R
    }
    fwrite(row, 1, row_size, file);
  }

  free(row);
  fclose(file);
  return 0;
}

//===----------------------------------------------------------------------===//
// Image Processing Functions
//===----------------------------------------------------------------------===//
PixelImage *pixel_invert(const PixelImage *img) {
  if (!img)
    return nullptr;

  PixelImage *result = pixel_clone_image(img);
  if (!result)
    return nullptr;

  const uint32_t total_pixels = img->width * img->height * img->channels;
  for (uint32_t i = 0; i < total_pixels; i++)
    result->data[i] = 255 - img->data[i];
  return result;
}

PixelImage *pixel_grayscale(const PixelImage *img) {
  if (!img)
    return nullptr;

  // If already grayscale, just clone
  if (img->channels == 1)
    return pixel_clone_image(img);

  PixelImage *result = pixel_create_image(img->width, img->height, 3);
  if (!result)
    return nullptr;

  for (uint32_t y = 0; y < img->height; y++) {
    for (uint32_t x = 0; x < img->width; x++) {
      const uint32_t idx = (y * img->width + x) * img->channels;
      const uint8_t r = img->data[idx + 0];
      const uint8_t g = img->data[idx + 1];
      const uint8_t b = img->data[idx + 2];

      // Luminosity method: 0.299*R + 0.587*G + 0.114*B
      const uint8_t gray =
          static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);

      const uint32_t out_idx = (y * result->width + x) * 3;
      result->data[out_idx + 0] = gray;
      result->data[out_idx + 1] = gray;
      result->data[out_idx + 2] = gray;
    }
  }
  return result;
}

PixelImage *pixel_rotate(const PixelImage *img, float angle) {
  if (!img)
    return nullptr;

  // Normalize angle to 0-360
  while (angle < 0)
    angle += 360.0f;
  while (angle >= 360.0f)
    angle -= 360.0f;

  // Handle 90-degree rotations specially for better quality
  if (fabs(angle - 0.0f) < 0.1f)
    return pixel_clone_image(img);
  if (fabs(angle - 90.0f) < 0.1f) {
    // Rotate 90 degrees clockwise
    PixelImage *result =
        pixel_create_image(img->height, img->width, img->channels);
    if (!result)
      return nullptr;
    for (uint32_t y = 0; y < img->height; y++)
      for (uint32_t x = 0; x < img->width; x++) {
        const uint32_t src_idx = (y * img->width + x) * img->channels;
        const uint32_t dst_idx =
            (x * result->width + (img->height - 1 - y)) * img->channels;
        for (uint32_t c = 0; c < img->channels; c++)
          result->data[dst_idx + c] = img->data[src_idx + c];
      }
    return result;
  }
  if (fabs(angle - 180.0f) < 0.1f) {
    // Rotate 180 degrees
    PixelImage *result =
        pixel_create_image(img->width, img->height, img->channels);
    if (!result)
      return nullptr;

    for (uint32_t y = 0; y < img->height; y++)
      for (uint32_t x = 0; x < img->width; x++) {
        const uint32_t src_idx = (y * img->width + x) * img->channels;
        const uint32_t dst_idx =
            ((img->height - 1 - y) * img->width + (img->width - 1 - x)) *
            img->channels;
        for (uint32_t c = 0; c < img->channels; c++)
          result->data[dst_idx + c] = img->data[src_idx + c];
      }
    return result;
  }
  if (fabs(angle - 270.0f) < 0.1f) {
    // Rotate 270 degrees clockwise (90 counter-clockwise)
    PixelImage *result =
        pixel_create_image(img->height, img->width, img->channels);
    if (!result)
      return nullptr;
    for (uint32_t y = 0; y < img->height; y++)
      for (uint32_t x = 0; x < img->width; x++) {
        const uint32_t src_idx = (y * img->width + x) * img->channels;
        const uint32_t dst_idx =
            ((img->width - 1 - x) * result->width + y) * img->channels;
        for (uint32_t c = 0; c < img->channels; c++)
          result->data[dst_idx + c] = img->data[src_idx + c];
      }
    return result;
  }

  // For arbitrary angles, use rotation matrix
  // This is a simplified implementation - just return a copy for now
  fprintf(
      stderr,
      "Warning: Arbitrary angle rotation not fully implemented, angle=%.2f\n",
      angle);
  return pixel_clone_image(img);
}

//===----------------------------------------------------------------------===//
// Runtime C Interface for MLIR (using void* for PixelImage*)
//===----------------------------------------------------------------------===//
void *_pixel_load(const char *filepath) { return pixel_load_image(filepath); }

void _pixel_save(void *img, const char *filepath) {
  pixel_save_image(static_cast<PixelImage *>(img), filepath, "bmp");
}

void *_pixel_invert(void *img) {
  return pixel_invert(static_cast<PixelImage *>(img));
}

void *_pixel_grayscale(void *img) {
  return pixel_grayscale(static_cast<PixelImage *>(img));
}

void *_pixel_rotate(void *img, const float angle) {
  return pixel_rotate(static_cast<PixelImage *>(img), angle);
}

void _pixel_free(void *img) {
  pixel_free_image(static_cast<PixelImage *>(img));
}

} // extern "C"
