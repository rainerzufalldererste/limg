#include "limg.h"

#include <stdio.h>
#include <inttypes.h>

#define STB_IMAGE_IMPLEMENTATION (1)
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION (1)
#include "stb_image_write.h"

#define FAIL(result, msg, ...) do { printf(msg, __VA_ARGS__); return result; } while (false)

inline void Write(const char *filename, void *pData, const size_t size)
{
  FILE *pFile = fopen(filename, "wb");

  fwrite(pData, 1, size, pFile);

  fclose(pFile);
}

int32_t main(const int32_t argc, const char **pArgv)
{
  if (argc == 1)
    FAIL(EXIT_SUCCESS, "Usage: limg <InputFile>");

  const char *sourceImagePath = pArgv[1];

  size_t sizeX = 0, sizeY = 0;
  bool hasAlpha = false;
  const uint32_t *pSourceImage = nullptr;
  uint32_t *pTargetImage = nullptr;

  // Read image.
  {
    int32_t width, height, channels;
    pSourceImage = reinterpret_cast<const uint32_t *>(stbi_load(sourceImagePath, &width, &height, &channels, 4));

    if (pSourceImage == nullptr)
      FAIL(EXIT_FAILURE, "Failed to read source image from '%s'.\n", sourceImagePath);

    sizeX = width;
    sizeY = height;
    hasAlpha = (channels == 4);
  }

  // Allocate space for decoded image.
  {
    pTargetImage = reinterpret_cast<uint32_t *>(malloc(sizeX * sizeY * sizeof(uint32_t)));

    if (pTargetImage == nullptr)
      FAIL(EXIT_FAILURE, "Failed to allocate target buffer.\n");
  }

  uint32_t *pA, *pB;
  uint8_t *pBlockError, *pFactors;

  // Allocate space for a, b, factors, blockError.
  {
    pA = reinterpret_cast<uint32_t *>(malloc(sizeX / 8 * sizeY / 8 * sizeof(uint32_t)));
    pB = reinterpret_cast<uint32_t *>(malloc(sizeX / 8 * sizeY / 8 * sizeof(uint32_t)));
    pBlockError = reinterpret_cast<uint8_t *>(malloc(sizeX / 8 * sizeY / 8 * sizeof(uint8_t)));
    pFactors = reinterpret_cast<uint8_t *>(malloc(sizeX * sizeY * sizeof(uint8_t)));
  }

  limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pA, uint32_t *pB, uint32_t *pDecoded, uint8_t *pFactors, uint8_t *pBlockError, const bool hasAlpha);

  const limg_result result = limg_encode_test(pSourceImage, sizeX, sizeY, pA, pB, pTargetImage, pFactors, pBlockError, hasAlpha);

  printf("limg_encode_test completed with exit code 0x%" PRIX32 ".\n", result);
  printf("%" PRIu64 " x %" PRIu64 " pixels.\n", (sizeX / 8) * 8, (sizeY / 8) * 8);
  printf("%" PRIu64 " x %" PRIu64 " blocks.\n", sizeX / 8, sizeY / 8);

  // Write everything.
  {
    Write("C:\\data\\limg_a.raw", pA, sizeX / 8 * sizeY / 8 * sizeof(uint32_t));
    Write("C:\\data\\limg_b.raw", pB, sizeX / 8 * sizeY / 8 * sizeof(uint32_t));
    Write("C:\\data\\limg_blk_err.raw", pBlockError, sizeX / 8 * sizeY / 8 * sizeof(uint8_t));
    Write("C:\\data\\limg_fac.raw", pFactors, sizeX * sizeY * sizeof(uint8_t));
    Write("C:\\data\\limg_out.raw", pTargetImage, sizeX * sizeY * sizeof(uint32_t));

    stbi_write_png("C:\\data\\limg_out.png", (int32_t)(sizeX / 8) * 8, (int32_t)(sizeY / 8) * 8, 4, pTargetImage, (int32_t)sizeX * sizeof(uint32_t));
    stbi_write_png("C:\\data\\limg_blk_err.png", (int32_t)(sizeX / 8), (int32_t)(sizeY / 8), 1, pBlockError, (int32_t)(sizeX / 8));
    stbi_write_png("C:\\data\\limg_fac.png", (int32_t)(sizeX / 8) * 8, (int32_t)(sizeY / 8) * 8, 1, pFactors, (int32_t)(sizeX));
  }

  return EXIT_SUCCESS;
}
