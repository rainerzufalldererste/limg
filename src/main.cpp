#include "limg.h"

#include <stdio.h>
#include <inttypes.h>

#include <chrono>

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

inline int64_t CurrentTimeNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline int32_t Hash(const int32_t value)
{
  const uint64_t oldstate = value * 6364136223846793005ULL + (value | 1);

  const uint32_t xorshifted = (uint32_t)(((oldstate >> 18) ^ oldstate) >> 27);
  const uint32_t rot = (uint32_t)(oldstate >> 59);

  return (xorshifted >> rot) | (xorshifted << (uint32_t)((-(int32_t)rot) & 31));
}

uint64_t ParseUInt(const char *text)
{
  uint64_t ret = 0;

  while (true)
  {
    const uint8_t d = (uint8_t)(*text - '0');
    text++;

    if (d > 9)
      break;

    ret = (ret << 1) + (ret << 3) + d;
  }

  return ret;
}

static const char Arg_NoWrite[] = "--no-output";
static const char Arg_ErrorFactor[] = "--error-factor";

int32_t main(const int32_t argc, const char **pArgv)
{
  if (argc == 1)
    FAIL(EXIT_SUCCESS, "Usage: limg <InputFile> [%s | %s <Factor>]", Arg_NoWrite, Arg_ErrorFactor);

  const char *sourceImagePath = pArgv[1];
  bool writeEncodedImages = true;
  uint32_t errorFactor = 4;

  size_t sizeX = 0, sizeY = 0;
  bool hasAlpha = false;
  const uint32_t *pSourceImage = nullptr;
  uint32_t *pTargetImage = nullptr;

  // Parse Args.
  {
    int32_t argIndex = 2;

    while (true)
    {
      const int32_t argsRemaining = argc - argIndex;

      if (argsRemaining <= 0)
        break;

      if (argsRemaining >= 1 && strncmp(Arg_NoWrite, pArgv[argIndex], sizeof(Arg_NoWrite)) == 0)
      {
        argIndex++;
        writeEncodedImages = false;
      }
      else if (argsRemaining >= 2 && strncmp(Arg_ErrorFactor, pArgv[argIndex], sizeof(Arg_ErrorFactor)) == 0)
      {
        errorFactor = (uint32_t)ParseUInt(pArgv[argIndex + 1]);
        argIndex += 2;
      }
      else
      {
        FAIL(EXIT_FAILURE, "Invalid Parameter: '%s'. Aborting.\n", pArgv[argIndex]);
      }
    }
  }

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
    pTargetImage = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));

    if (pTargetImage == nullptr)
      FAIL(EXIT_FAILURE, "Failed to allocate target buffer.\n");
  }

  uint32_t *pA, *pB, *pBlockIndex;
  uint8_t *pBlockError, *pFactors, *pShift;

  // Allocate space for a, b, factors, blockError.
  {
    pA = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
    pB = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
    pBlockIndex = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
    pBlockError = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
    pFactors = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
    pShift = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
  }

  // Print Image Info.
  {
    printf("%" PRIu64 " x %" PRIu64 " pixels.\n", sizeX, sizeY);
  }

  size_t totalBlockArea = 0;

  // Encode.
  {
    const int64_t before = CurrentTimeNs();

    const limg_result result = limg_encode_test(pSourceImage, sizeX, sizeY, pTargetImage, pA, pB, pBlockIndex, pFactors, pBlockError, pShift, hasAlpha, &totalBlockArea, errorFactor);

    const int64_t after = CurrentTimeNs();

    printf("limg_encode_test completed with exit code 0x%" PRIX32 ".\n", result);
    printf("Elapsed Time: %f ms\n", (after - before) * 1e-6f);
    printf("Throughput: %f Mpx/S\n", (sizeX * sizeY * 1e-6) / ((after - before) * 1e-9f));
  }

  // Compare.
  {
    double mean, max;
    const double psnr = limg_compare(pSourceImage, pTargetImage, sizeX, sizeY, hasAlpha, &mean, &max);

    printf("\nImage Perceptual RGB(A) PSNR: %4.2f dB (mean: %5.3f => %7.5f%% | sqrt: %5.3f%%)\n", psnr, mean, (mean / max) * 100.0, (sqrt(mean) / sqrt(max)) * 100.0);

    const double meanOverBlocks = mean * (double)(sizeX * sizeY) / (double)totalBlockArea;
    const double block_psnr = 10.0 * log10((double)max / meanOverBlocks);

    printf("Block Perceptual RGB(A) PSNR: %4.2f dB (mean: %5.3f => %7.5f%% | sqrt: %5.3f%%)\n", block_psnr, meanOverBlocks, (meanOverBlocks / max) * 100.0, (sqrt(meanOverBlocks) / sqrt(max)) * 100.0);
  }

  // Write everything.
  if (writeEncodedImages)
  {
    stbi_write_bmp("C:\\data\\limg_out.bmp", (int32_t)sizeX, (int32_t)sizeY, 4, pTargetImage);
    stbi_write_bmp("C:\\data\\limg_blk_err.bmp", (int32_t)sizeX, (int32_t)sizeY, 1, pBlockError);
    stbi_write_bmp("C:\\data\\limg_shift.bmp", (int32_t)sizeX, (int32_t)sizeY, 1, pShift);
    stbi_write_bmp("C:\\data\\limg_fac.bmp", (int32_t)sizeX, (int32_t)sizeY, 1, pFactors);
    stbi_write_bmp("C:\\data\\limg_a.bmp", (int32_t)sizeX, (int32_t)sizeY, 4, pA);
    stbi_write_bmp("C:\\data\\limg_b.bmp", (int32_t)sizeX, (int32_t)sizeY, 4, pB);

    for (size_t i = 0; i < sizeX * sizeY; i++)
      pBlockIndex[i] = Hash(pBlockIndex[i]) | 0xFF000000;

    stbi_write_bmp("C:\\data\\limg_index.bmp", (int32_t)sizeX, (int32_t)sizeY, 4, pBlockIndex);
  }

  return EXIT_SUCCESS;
}
