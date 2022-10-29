#include "limg.h"

#include <stdio.h>
#include <inttypes.h>

#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#define STB_IMAGE_IMPLEMENTATION (1)
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION (1)
#include "stb_image_write.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef __GNUC__
#define FAIL(result, msg, ...) do { printf(msg, ##__VA_ARGS__); return result; } while (false)
#else
#define FAIL(result, msg, ...) do { printf(msg, __VA_ARGS__); return result; } while (false)
#endif

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
static const char Arg_AccurateBitCrushing[] = "--accurate-bit-crushing";
static const char Arg_SingleThreaded[] = "--single-thread";
static const char Arg_ListCount[] = "--count";
static const char Arg_List[] = "--";

static bool _WriteEncodedImages = true;
static uint32_t _ErrorFactor = 4;
static bool _FastBitCrushing = true;
static bool _UseThreadPool = true;
static size_t _ListCount = 1;

int32_t main(const int32_t argc, const char **pArgv)
{
  if (argc == 1)
    FAIL(EXIT_SUCCESS, "Usage:\nlimg [<InputFile> | --] [%s | %s <Factor> | %s | %s] \n  if input file is --:\n    [%s <Count>] -- <list of files>)\n", Arg_NoWrite, Arg_ErrorFactor, Arg_AccurateBitCrushing, Arg_SingleThreaded, Arg_ListCount);

  const char *sourceImagePath = pArgv[1];

  size_t sizeX = 0, sizeY = 0;
  bool hasAlpha = false;
  uint32_t *pSourceImage = nullptr;
  uint32_t *pTargetImage = nullptr;

  int32_t argIndex = 2;

  // Parse Args.
  {
    while (true)
    {
      const int32_t argsRemaining = argc - argIndex;

      if (argsRemaining <= 0)
        break;

      if (argsRemaining >= 1 && strncmp(Arg_NoWrite, pArgv[argIndex], sizeof(Arg_NoWrite)) == 0)
      {
        argIndex++;
        _WriteEncodedImages = false;
      }
      else if (argsRemaining >= 1 && strncmp(Arg_AccurateBitCrushing, pArgv[argIndex], sizeof(Arg_AccurateBitCrushing)) == 0)
      {
        argIndex++;
        _FastBitCrushing = false;
      }
      else if (argsRemaining >= 1 && strncmp(Arg_SingleThreaded, pArgv[argIndex], sizeof(Arg_SingleThreaded)) == 0)
      {
        argIndex++;
        _UseThreadPool = false;

#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        SetThreadIdealProcessor(GetCurrentThread(), SetThreadIdealProcessor(GetCurrentThread(), MAXIMUM_PROCESSORS)); // Set to current ideal processor.
#endif
      }
      else if (argsRemaining >= 2 && strncmp(Arg_ErrorFactor, pArgv[argIndex], sizeof(Arg_ErrorFactor)) == 0)
      {
        _ErrorFactor = (uint32_t)ParseUInt(pArgv[argIndex + 1]);
        argIndex += 2;
      }
      else if (argsRemaining > 1 && strncmp(Arg_List, pArgv[argIndex], sizeof(Arg_List)) == 0)
      {
        if (strncmp(sourceImagePath, Arg_List, sizeof(Arg_List)) != 0)
          FAIL(EXIT_FAILURE, "'%s' is only supported with input file '%s', found '%s'.\n", pArgv[argIndex], Arg_List, sourceImagePath);

        _WriteEncodedImages = false;
        sourceImagePath = nullptr;
        argIndex++;

        break;
      }
      else if (argsRemaining > 1 && strncmp(Arg_ListCount, pArgv[argIndex], sizeof(Arg_ListCount)) == 0)
      {
        if (strncmp(sourceImagePath, Arg_List, sizeof(Arg_List)) != 0)
          FAIL(EXIT_FAILURE, "'%s' is only supported with input file '%s', found '%s'.\n", pArgv[argIndex], Arg_List, sourceImagePath);

        _ListCount = ParseUInt(pArgv[argIndex + 1]);
        argIndex += 2;
      }
      else
      {
        FAIL(EXIT_FAILURE, "Invalid Parameter: '%s'. Aborting.\n", pArgv[argIndex]);
      }
    }
  }

  limg_thread_pool *pThreadPool = nullptr;

  if (_UseThreadPool)
    pThreadPool = limg_thread_pool_new(limg_threading_max_threads());

  size_t pixels = 0;
  size_t nanosecs = 0;
  const bool singlePerfEval = sourceImagePath == nullptr && argc == argIndex + 1 && _ListCount > 1;

  do
  {
    const char *filename = sourceImagePath;

    if (filename == nullptr)
    {
      filename = pArgv[argIndex];
      argIndex++;

      if (!singlePerfEval)
        printf("\r'%s' (%" PRIi32 " remaining) (~ %8.4f Mpx/s) ...", filename, argc - argIndex, (pixels * 1e-6) / (nanosecs * 1e-9f));
    }

    // Read image.
    {
      int32_t width, height, channels;
      pSourceImage = reinterpret_cast<uint32_t *>(stbi_load(filename, &width, &height, &channels, 4));

      if (pSourceImage == nullptr)
        FAIL(EXIT_FAILURE, "Failed to read source image from '%s'.\n", sourceImagePath);

      sizeX = width;
      sizeY = height;
      hasAlpha = (channels == 4);
    }

    // Allocate space for decoded image.
    if (sourceImagePath != nullptr)
    {
      pTargetImage = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));

      if (pTargetImage == nullptr)
        FAIL(EXIT_FAILURE, "Failed to allocate target buffer.\n");
    }

    uint32_t *pShift = nullptr;
    uint8_t *pFactorsA = nullptr, *pFactorsB = nullptr, *pFactorsC = nullptr;
    uint32_t *pColAMin = nullptr, *pColAMax = nullptr, *pColBMin = nullptr, *pColBMax = nullptr, *pColCMin = nullptr, *pColCMax = nullptr;

    // Allocate space for a, b, factors, blockError.
    if (sourceImagePath != nullptr)
    {
      pFactorsA = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
      pFactorsB = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
      pFactorsC = reinterpret_cast<uint8_t *>(calloc(sizeX * sizeY, sizeof(uint8_t)));
      pShift = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColAMin = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColAMax = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColBMin = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColBMax = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColCMin = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
      pColCMax = reinterpret_cast<uint32_t *>(calloc(sizeX * sizeY, sizeof(uint32_t)));
    }

    // Print Image Info.
    if (sourceImagePath != nullptr)
    {
      printf("%" PRIu64 " x %" PRIu64 " pixels.\n", sizeX, sizeY);
    }

    // Encode.
    if (sourceImagePath != nullptr)
    {
      const int64_t before = CurrentTimeNs();

      const limg_result result = limg_encode3d_test(pSourceImage, sizeX, sizeY, pTargetImage, pFactorsA, pFactorsB, pFactorsC, pShift, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, hasAlpha, _ErrorFactor, pThreadPool, _FastBitCrushing);

      const int64_t after = CurrentTimeNs();

      printf("limg_encode_test completed with exit code 0x%" PRIX32 ".\n", result);
      printf("Elapsed Time: %f ms\n", (after - before) * 1e-6);
      printf("Throughput: %f Mpx/s\n", (sizeX * sizeY * 1e-6) / ((after - before) * 1e-9));
    }
    else if (singlePerfEval)
    {
      uint64_t timeSum = 0, min = UINT64_MAX, max = 0;
      uint64_t *pTimeNs = reinterpret_cast<uint64_t *>(malloc(sizeof(uint64_t) * _ListCount));

      if (pTimeNs == nullptr)
        FAIL(EXIT_FAILURE, "Failed to allocate memory.\n");

      const double megapixels = (sizeX * sizeY * 1e-6);

      for (size_t i = 0; i < _ListCount; i++)
      {
        const int64_t before = CurrentTimeNs();      
        const limg_result result = limg_encode3d_test_perf(pSourceImage, sizeX, sizeY, hasAlpha, _ErrorFactor, pThreadPool, _FastBitCrushing);
        const int64_t after = CurrentTimeNs();

        if (limg_success != result)
          FAIL(EXIT_FAILURE, "Encode failed with exit code 0x%" PRIX32 ".\n", result);

        const uint64_t timeNs = after - before;
        timeSum += timeNs;
        pTimeNs[i] = timeNs;

        if (timeNs > max)
          max = timeNs;

        if (timeNs < min)
          min = timeNs;

        printf("\rThroughput: ~%5.3f Mpx/s", megapixels / (timeNs * 1e-9));
      }

      const double mean = timeSum / (double)_ListCount;

      double std_dev = 0;

      for (size_t i = 0; i < _ListCount; i++)
      {
        const double diff = pTimeNs[i] - mean;
        std_dev += diff * diff;
      }

      std_dev = sqrt(std_dev / (double)(_ListCount - 1));

      printf("\rMean Elapsed Time: %8.4f ms (%8.4f - %8.4f ms | %8.4f - %8.4f ms std dev)\n", mean * 1e-6, min * 1e-6, max * 1e-6, (mean - std_dev) * 1e-6, (mean + std_dev) * 1e-6);
      printf("Throughput: %5.3f Mpx/s (%5.3f - %5.3f Mpx/s | %5.3f - %5.3f Mpx/s std dev)\n", megapixels / (mean * 1e-9), megapixels / (max * 1e-9), megapixels / (min * 1e-9), megapixels / ((mean + std_dev) * 1e-9), megapixels / ((mean - std_dev) * 1e-9));
    }
    else
    {
      const int64_t before = CurrentTimeNs();

      limg_result result;

      for (size_t i = 0; i < _ListCount; i++)
        if (limg_success != (result = limg_encode3d_test_perf(pSourceImage, sizeX, sizeY, hasAlpha, _ErrorFactor, pThreadPool, _FastBitCrushing)))
          FAIL(EXIT_FAILURE, "Encode failed with exit code 0x%" PRIX32 ".\n", result);

      const int64_t after = CurrentTimeNs();

      pixels += sizeX * sizeY * _ListCount;
      nanosecs += after - before;
    }

    // Compare.
    if (sourceImagePath != nullptr)
    {
      double mean, max;
      const double psnr = limg_compare(pSourceImage, pTargetImage, sizeX, sizeY, hasAlpha, &mean, &max);

      printf("\nImage Perceptual RGB(A) PSNR: %4.2f dB (mean: %5.3f => %7.5f%% | sqrt: %5.3f%%)\n\n", psnr, mean, (mean / max) * 100.0, (sqrt(mean) / sqrt(max)) * 100.0);
    }

    // Write everything.
    if (_WriteEncodedImages)
    {
      if (stbi_write_tga("limg_out.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pTargetImage))
        puts("Wrote decoded file.");
      else
        puts("Failed to write decoded file.");

      stbi_write_tga("limg_fac_a.tga", (int32_t)sizeX, (int32_t)sizeY, 1, pFactorsA);
      stbi_write_tga("limg_fac_b.tga", (int32_t)sizeX, (int32_t)sizeY, 1, pFactorsB);
      stbi_write_tga("limg_fac_c.tga", (int32_t)sizeX, (int32_t)sizeY, 1, pFactorsC);
      stbi_write_tga("limg_bits.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pShift);
      stbi_write_tga("limg_col_a_min.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColAMin);
      stbi_write_tga("limg_col_a_max.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColAMax);
      stbi_write_tga("limg_col_b_min.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColBMin);
      stbi_write_tga("limg_col_b_max.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColBMax);
      stbi_write_tga("limg_col_c_min.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColCMin);
      stbi_write_tga("limg_col_c_max.tga", (int32_t)sizeX, (int32_t)sizeY, 4, pColCMax);
    }

    free(pSourceImage);
    pSourceImage = nullptr;

    free(pTargetImage);
    pTargetImage = nullptr;

    free(pShift);
    pShift = nullptr;

    free(pFactorsA);
    pFactorsA = nullptr;

    free(pFactorsB);
    pFactorsB = nullptr;

    free(pFactorsC);
    pFactorsC = nullptr;

    free(pColAMin);
    pColAMin = nullptr;

    free(pColAMax);
    pColAMax = nullptr;

    free(pColBMin);
    pColBMin = nullptr;

    free(pColBMax);
    pColBMax = nullptr;

    free(pColCMin);
    pColCMin = nullptr;

    free(pColCMax);
    pColCMax = nullptr;

  } while (sourceImagePath == nullptr && argIndex < argc);

  if (sourceImagePath == nullptr && !singlePerfEval)
    printf("\rComplete.   \nProcessed %5.3f Mpx in %5.3f sec / %5.3f mins \nThroughput: %8.5f MPx/s\n\n\n", pixels * 1e-6, nanosecs * 1e-9, (nanosecs * 1e-9) / 60.0, (pixels * 1e-6) / (nanosecs * 1e-9));

  return EXIT_SUCCESS;
}
