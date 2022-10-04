#include "limg.h"

#include <malloc.h>
#include <memory.h>

#define PRINT_TEST_OUTPUT

#ifdef PRINT_TEST_OUTPUT
#include <stdio.h>
#include <inttypes.h>
#endif

#include <type_traits>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>

#define __debugbreak() __builtin_trap()
#endif

#define LIMG_SUCCESS(errorCode) (errorCode == fpR_Success)
#define LIMG_FAILED(errorCode) (!(LIMG_SUCCESS(errorCode)))

#define LIMG_ERROR_SET(errorCode) \
  do \
  { result = errorCode; \
    goto epilogue; \
  } while (0) 

#define LIMG_ERROR_IF(booleanExpression, errorCode) \
  do \
  { if (booleanExpression) \
      LIMG_ERROR_SET(errorCode); \
  } while (0)

#define LIMG_ERROR_CHECK(functionCall) \
  do \
  { result = (functionCall); \
    if (LIMG_FAILED(result)) \
      LIMG_ERROR_SET(result); \
  } while (0)

template <typename T>
inline void limgFreePtr(T **ppData)
{
  if (*ppData != nullptr)
    free(*ppData);

  *ppData = nullptr;
}

template <typename T, typename U>
constexpr inline auto limgMax(const T & a, const U & b) -> decltype(a > b ? a : b)
{
  return a > b ? a : b;
}

template <typename T, typename U>
constexpr inline auto limgMin(const T & a, const U & b) -> decltype(a < b ? a : b)
{
  return a < b ? a : b;
}

template <typename T>
constexpr inline T limgClamp(const T & a, const T & min, const T & max)
{
  if (a < min)
    return min;

  if (a > max)
    return max;

  return a;
}

template <typename T, typename U>
constexpr inline auto limgLerp(const T a, const T b, const U ratio) -> decltype(a + (b - a) * ratio) { return a + (b - a) * ratio; }

template <typename T, typename U = typename std::conditional_t<std::is_integral<T>::value, float_t, T>>
constexpr inline U limgInverseLerp(const T value, const T min, const T max) { return (U)(value - min) / (U)(max - min); }

#define LIMG_ARRAYSIZE_C_STYLE(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))

#ifdef LIMG_FORCE_ARRAYSIZE_C_STYLE
#define LIMG_ARRAYSIZE(arrayName) LIMG_ARRAYSIZE_C_STYLE(arrayName)
#else
template <typename T, size_t TCount>
inline constexpr size_t LIMG_ARRAYSIZE(const T(&)[TCount]) { return TCount; }
#endif

//////////////////////////////////////////////////////////////////////////

//struct Buffer
//{
//  uint8_t *pData = nullptr;
//  size_t capacity = 0;
//  size_t size = 0;
//};
//
//static void Buffer_Init(Buffer *pBuffer);
//static limg_result Buffer_AddData(Buffer *pBuffer, const void *pAppendData, const size_t dataSize, size_t *pIndex);
//static void Buffer_Destroy(Buffer *pBuffer);
//
////////////////////////////////////////////////////////////////////////////
//
//static void Buffer_Destroy(Buffer *pBuffer)
//{
//  if (pBuffer)
//    limgFreePtr(&pBuffer->pData);
//}
//
//static void Buffer_Init(Buffer *pBuffer)
//{
//  pBuffer->pData = nullptr;
//  pBuffer->capacity = 0;
//  pBuffer->size = 0;
//}
//
//static limg_result Buffer_AddData(Buffer *pBuffer, const void *pAppendData, const size_t dataSize, size_t *pIndex)
//{
//  limg_result result = limg_success;
//
//  if (pBuffer->pData == nullptr || pBuffer->size + dataSize > pBuffer->capacity)
//  {
//    const size_t newCapacity = limgMax(limgMax((pBuffer->capacity + 1) * 2, pBuffer->capacity + dataSize), 1024ULL);
//
//    pBuffer->pData = (uint8_t *)realloc(pBuffer->pData, newCapacity);
//    LIMG_ERROR_IF(!pBuffer->pData, limg_error_MemoryAllocationFailure);
//    
//    pBuffer->capacity = newCapacity;
//  }
//
//  *pIndex = pBuffer->size;
//  memcpy(pBuffer->pData + pBuffer->size, pAppendData, dataSize);
//  pBuffer->size += dataSize;
//
//  goto epilogue;
//
//epilogue:
//  return result;
//}

//////////////////////////////////////////////////////////////////////////

struct limg_encode_context
{
  const uint32_t *pSourceImage;
  uint32_t *pBlockInfo;
  size_t sizeX, sizeY;
  size_t maxPixelBlockError, // maximum error of a single pixel when trying to fit pixels into blocks.
    maxBlockPixelError, // maximum average error of pixels per block when trying to fit them into blocks. (accum_err * 0xFF / (rangeX * rangeY))
    maxPixelChannelBlockError, // maximum error of a single pixel on a single channel when trying to fit pixels into blocks.
    maxBlockExpandError, // maximum error of linear factor deviation when trying to expand the factors of a block in order to expand the block.
    maxPixelBitCrushError, // maximum error of a single pixel when trying to bit crush blocks.
    maxBlockBitCrushError; // maximum average error of pixels per block when trying to bit crush blocks. (accum_err * 0xFF / (rangeX * rangeY))
  bool hasAlpha, ditheringEnabled;
};

constexpr uint32_t BlockInfo_InUse = (uint32_t)((uint32_t)1 << 31);

constexpr size_t limg_BlockExpandStep = 2; // must be a power of two.
constexpr size_t limg_MinBlockSize = limg_BlockExpandStep * 4;
constexpr bool limg_ColorDependentBlockError = true;
constexpr bool limg_LuminanceDependentPixelError = true;
constexpr bool limg_ColorDependentABError = true;
constexpr bool limg_RetrievePreciseDecomposition = true;

//#define LIMG_DO_NOT_INLINE

#ifndef LIMG_DO_NOT_INLINE
#define LIMG_INLINE inline
#define LIMG_DEBUG_NO_INLINE
#else
#define LIMG_INLINE __declspec(noinline)
#define LIMG_DEBUG_NO_INLINE __declspec(noinline)
#endif

struct limg_ui8_4
{
  uint8_t v[4];

  LIMG_INLINE uint8_t &operator[](const size_t index) { return v[index]; }
  LIMG_INLINE const uint8_t &operator[](const size_t index) const { return v[index]; }

  LIMG_INLINE bool equals_w_alpha(const limg_ui8_4 &other)
  {
    return v[0] == other[0] && v[1] == other[1] && v[2] == other[2] && v[3] == other[3];
  }

  LIMG_INLINE bool equals_wo_alpha(const limg_ui8_4 &other)
  {
    return v[0] == other[0] && v[1] == other[1] && v[2] == other[2];
  }

  LIMG_INLINE bool equals(const limg_ui8_4 &other, const bool hasAlpha)
  {
    return v[0] == other[0] && v[1] == other[1] && v[2] == other[2] && (hasAlpha || v[3] == other[3]);
  }
};

// This appears to be either slower or make no significant difference.
//struct limg_ui8_4
//{
//  union
//  {
//    uint8_t v[4];
//    uint32_t n;
//  } u;
//
//  LIMG_INLINE uint8_t &operator[](const size_t index) { return u.v[index]; }
//  LIMG_INLINE const uint8_t &operator[](const size_t index) const { return u.v[index]; }
//
//  LIMG_INLINE bool equals_w_alpha(const limg_ui8_4 &other)
//  {
//    return (u.n ^ other.u.n) == 0;
//  }
//
//  LIMG_INLINE bool equals_wo_alpha(const limg_ui8_4 &other)
//  {
//    return ((u.n ^ other.u.n) & 0x00FFFFFF) == 0;
//  }
//
//  LIMG_INLINE bool equals(const limg_ui8_4 &other, const bool hasAlpha)
//  {
//    const uint32_t xored = (u.n ^ other.u.n);
//
//    return u.v[0] == other[0] && u.v[1] == other[1] && u.v[2] == other[2] && (hasAlpha || u.v[3] == other[3]);
//  }
//};

// This appears to be slower.
//struct limg_ui8_4
//{
//  uint32_t v;
//
//  LIMG_INLINE uint8_t &operator[](const size_t index) { return *(reinterpret_cast<uint8_t *>(&v) + index); }
//  LIMG_INLINE const uint8_t operator[](const size_t index) const { return (uint8_t)(v >> (index * 8)); }
//
//  LIMG_INLINE bool equals_w_alpha(const limg_ui8_4 &other)
//  {
//    return (v ^ other.v) == 0;
//  }
//
//  LIMG_INLINE bool equals_wo_alpha(const limg_ui8_4 &other)
//  {
//    return ((v ^ other.v) & 0x00FFFFFF) == 0;
//  }
//
//  LIMG_INLINE bool equals(const limg_ui8_4 &other, const bool hasAlpha)
//  {
//    const uint32_t xored = (v ^ other.v);
//
//    return xored == 0 || (hasAlpha && (xored & 0x00FFFFFF) == 0);
//  }
//};

static_assert(sizeof(limg_ui8_4) == 4, "Invalid Configuration");

//////////////////////////////////////////////////////////////////////////

template <size_t channels>
LIMG_INLINE static size_t limg_color_error(const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  size_t error;

  int_fast16_t redError = (int_fast16_t)a[0] - (int_fast16_t)b[0];
  redError *= redError;

  if (redError < 0x4000 /* = 0x80 * 0x80 */)
  {
    const uint8_t factors[4] = { 2, 4, 3, 3 };

    error = redError * factors[0];

    for (size_t i = 1; i < channels; i++)
    {
      const int_fast16_t e = (int_fast16_t)a[i] - (int_fast16_t)b[i];
      error += (size_t)(e * e) * (size_t)factors[i];
    }
  }
  else
  {
    const uint8_t factors[4] = { 3, 4, 2, 3 };

    error = redError * factors[0];

    for (size_t i = 1; i < channels; i++)
    {
      const int_fast16_t e = (int16_t)a[i] - (int16_t)b[i];
      error += (size_t)(e * e) * (size_t)factors[i];
    }
  }

  return error;
}

template <size_t channels>
LIMG_INLINE static size_t limg_vector_error(const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  size_t error = 0;

  for (size_t i = 0; i < channels; i++)
  {
    const int_fast16_t e = (int_fast16_t)a[i] - (int_fast16_t)b[i];
    error += (size_t)(e * e);
  }

  return error;
}

template <size_t channels>
static void limg_decode_block_from_factors(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[channels];

  for (size_t i = 0; i < channels; i++)
    diff[i] = b[i] - a[i];

  constexpr uint32_t bias = 1 << 7;

  for (size_t y = 0; y < rangeY; y++)
  {
    limg_ui8_4 *pOutLine = reinterpret_cast<limg_ui8_4 *>(pOut + y * sizeX);

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint8_t fac = *pFactors;
      pFactors++;

      limg_ui8_4 px;

      for (size_t i = 0; i < channels; i++)
        px[i] = (uint8_t)(a[i] + (((fac << shift) * diff[i] + bias) >> 8));

      *pOutLine = px;
      pOutLine++;
    }
  }
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError, bool ReadWriteRangeSize, size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_check_area_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError, const size_t startBlockError, size_t *pAdditionalRangeForInitialBlockError)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  size_t blockError = startBlockError;

  float dist[channels];
  float dist_or_one[channels];
  float inverse_dist_complete_or_one = 0;

  for (size_t i = 0; i < channels; i++)
  {
    const uint8_t diff = (b[i] - a[i]);
    dist[i] = (float)diff;
    dist_or_one[i] = limgMax(1, dist[i]);

    if (diff != 0)
      inverse_dist_complete_or_one += (float)dist[i];
  }

  inverse_dist_complete_or_one = 1.f / limgMax(1.f, inverse_dist_complete_or_one);;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      if constexpr (CheckBounds)
        for (size_t i = 0; i < channels; i++)
          if ((int64_t)px[i] < a[i] - (int64_t)pCtx->maxPixelChannelBlockError || (int64_t)px[i] > b[i] + (int64_t)pCtx->maxPixelChannelBlockError)
            return false;

      float offset[channels];

      for (size_t i = 0; i < channels; i++)
        offset[i] = (float)(px[i] - a[i]);

      float avg = 0;

      for (size_t i = 0; i < channels; i++)
        avg += offset[i];

      avg *= inverse_dist_complete_or_one;

      if constexpr (WriteToFactors)
      {
        *pFactors = avg;
        pFactors++;
      }

      size_t error = 0;
      size_t lum = 0;

      if constexpr (limg_ColorDependentBlockError)
      {
        if (px[0] < 0x80)
        {
          const uint8_t factors[4] = { 2, 4, 3, 3 };

          for (size_t i = 0; i < channels; i++)
          {
            const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one[i] - avg) * dist[i]));
            error += e * e * factors[i];

            if constexpr (limg_LuminanceDependentPixelError)
              lum += px[i];
          }
        }
        else
        {
          const uint8_t factors[4] = { 3, 4, 2, 3 };

          for (size_t i = 0; i < channels; i++)
          {
            const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one[i] - avg) * dist[i]));
            error += e * e * factors[i];

            if constexpr (limg_LuminanceDependentPixelError)
              lum += px[i];
          }
        }
      }
      else
      {
        for (size_t i = 0; i < channels; i++)
        {
          const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one[i] - avg) * dist[i]));
          error += e * e;

          if constexpr (limg_LuminanceDependentPixelError)
            lum += px[i];
        }
      }
      
      if constexpr (limg_LuminanceDependentPixelError)
      {
        size_t ilum;
        ilum = 0xFF * 12 - lum * (12 / channels);
        ilum *= ilum;
        lum = (ilum >> 20) + 8;
        error = lum * error;
      }

      if constexpr (CheckPixelError)
        if (error > pCtx->maxPixelBlockError)
          return false;

      blockError += error;
    }
  }

  if constexpr (WriteBlockError)
    *pBlockError = blockError;

  size_t rangeSize = rangeX * rangeY;

  if constexpr (ReadWriteRangeSize)
  {
    rangeSize += *pAdditionalRangeForInitialBlockError;
    *pAdditionalRangeForInitialBlockError = rangeSize;
  }

  return (((blockError * 0x10) / rangeSize) < pCtx->maxBlockPixelError);
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError, bool ReadWriteRangeSize>
static LIMG_INLINE bool limg_encode_check_area(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError, const size_t startBlockError, size_t *pAdditionalRangeForInitialBlockError)
{
  if (pCtx->hasAlpha)
    return limg_encode_check_area_<WriteToFactors, WriteBlockError, CheckBounds, CheckPixelError, ReadWriteRangeSize, 4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, pFactors, pBlockError, startBlockError, pAdditionalRangeForInitialBlockError);
  else
    return limg_encode_check_area_<WriteToFactors, WriteBlockError, CheckBounds, CheckPixelError, ReadWriteRangeSize, 3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, pFactors, pBlockError, startBlockError, pAdditionalRangeForInitialBlockError);
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  limg_ui8_4 a = out_a;
  limg_ui8_4 b = out_b;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      int64_t low = 0;

      for (size_t i = 0; i < channels; i++)
        low += limgMax(-1LL, ((int64_t)a[i] - px[i]));

      if (low > 0)
      {
        // is `a` a linear combination of `px` and `b`?

        float offset[channels];
        float dist_px[channels];
        float dist_or_one_px[channels];
        float dist_complete_or_one_px = 0;

        for (size_t i = 0; i < channels; i++)
        {
          offset[i] = (float)(a[i] - px[i]);

          const uint8_t diff = (b[i] - px[i]);

          dist_px[i] = (float)diff;

          if (diff != 0)
          {
            dist_or_one_px[i] = dist_px[i];
            dist_complete_or_one_px += dist_px[i];
          }
          else
          {
            dist_or_one_px[i] = 0;
          }
        }

        dist_complete_or_one_px = limgMax(1.f, dist_complete_or_one_px);

        float avg_offset = 0;

        for (size_t i = 0; i < channels; i++)
          avg_offset += offset[i];

        avg_offset /= dist_complete_or_one_px;

        size_t error = 0;
        size_t lum = 0;

        if constexpr (limg_ColorDependentBlockError)
        {
          if (px[0] < 0x80)
          {
            const uint8_t factors[4] = { 2, 4, 3, 3 };

            for (size_t i = 0; i < channels; i++)
            {
              const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
              error += e * e * factors[i];

              if constexpr (limg_LuminanceDependentPixelError)
                lum += px[i];
            }
          }
          else
          {
            const uint8_t factors[4] = { 3, 4, 2, 3 };

            for (size_t i = 0; i < channels; i++)
            {
              const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
              error += e * e * factors[i];

              if constexpr (limg_LuminanceDependentPixelError)
                lum += px[i];
            }
          }
        }
        else
        {
          for (size_t i = 0; i < channels; i++)
          {
            const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
            error += e * e;

            if constexpr (limg_LuminanceDependentPixelError)
              lum += px[i];
          }
        }

        if constexpr (limg_LuminanceDependentPixelError)
        {
          size_t ilum = 0xFF * 12 - lum * (12 / channels);
          ilum *= ilum;
          lum = (ilum >> 20) + 8;
          error = lum * error;
        }

        if (error > pCtx->maxBlockExpandError)
          return false;

        a = px;
      }
      else
      {
        int64_t high = 0;

        for (size_t i = 0; i < channels; i++)
          high += limgMax(-1LL, (px[i] - (int64_t)b[i]));

        if (high > 0)
        {
          // is `b` a linear combination of `a` and `px`?

          float offset[channels];
          float dist_px[channels];
          float dist_or_one_px[channels];
          float dist_complete_or_one_px = 0;

          for (size_t i = 0; i < channels; i++)
          {
            offset[i] = (float)(b[i] - a[i]);

            const uint8_t diff = (px[i] - a[i]);

            dist_px[i] = (float)diff;

            if (diff != 0)
            {
              dist_or_one_px[i] = dist_px[i];
              dist_complete_or_one_px += dist_px[i];
            }
            else
            {
              dist_or_one_px[i] = 0;
            }
          }

          dist_complete_or_one_px = limgMax(1.f, dist_complete_or_one_px);

          float avg_offset = 0;
          
          for (size_t i = 0; i < channels; i++)
            avg_offset += offset[i];
          
          avg_offset /= dist_complete_or_one_px;

          size_t error = 0;
          size_t lum = 0;

          if constexpr (limg_ColorDependentBlockError)
          {
            if (px[0] < 0x80)
            {
              const uint8_t factors[4] = { 2, 4, 3, 3 };

              for (size_t i = 0; i < channels; i++)
              {
                const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
                error += e * e * factors[i];

                if constexpr (limg_LuminanceDependentPixelError)
                  lum += px[i];
              }
            }
            else
            {
              const uint8_t factors[4] = { 3, 4, 2, 3 };

              for (size_t i = 0; i < channels; i++)
              {
                const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
                error += e * e * factors[i];

                if constexpr (limg_LuminanceDependentPixelError)
                  lum += px[i];
              }
            }
          }
          else
          {
            for (size_t i = 0; i < channels; i++)
            {
              const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
              error += e * e;

              if constexpr (limg_LuminanceDependentPixelError)
                lum += px[i];
            }
          }

          if constexpr (limg_LuminanceDependentPixelError)
          {
            size_t ilum = 0xFF * 12 - lum * (12 / channels);
            ilum *= ilum;
            lum = (ilum >> 20) + 8;
            error = lum * error;
          }

          if (error > pCtx->maxBlockExpandError)
            return false;

          b = px;
        }
      }
    }
  }
  
  out_a = a;
  out_b = b;

  return true;
}

static LIMG_INLINE bool limg_encode_attempt_include_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  if (pCtx->hasAlpha)
    return limg_encode_attempt_include_pixels_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
  else  
    return limg_encode_attempt_include_pixels_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
}

static LIMG_INLINE bool limg_encode_check_pixel_unused(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY)
{
  const uint32_t *pStart = (pCtx->pBlockInfo + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      if (!!(*pLine & BlockInfo_InUse))
        return false;

      pLine++;
    }
  }

  return true;
}

static LIMG_INLINE bool limg_encode_attempt_include_unused_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  return limg_encode_check_pixel_unused(pCtx, offsetX, offsetY, rangeX, rangeY) && limg_encode_attempt_include_pixels(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_min_max_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  a = pStart[0];
  b = a;
  
  size_t x = 1;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      int64_t low = 0;

      for (size_t i= 0; i < channels; i++)
        low += limgMax(-1LL, ((int64_t)a[i] - px[i]));

      if (low > 0)
      {
        a = px;
      }
      else
      {
        int64_t high = 0;
        
        for (size_t i = 0; i < channels; i++)
          high += limgMax(-1LL, (px[i] - (int64_t)b[i]));

        if (high > 0)
          b = px;
      }
    }

    x = 0;
  }
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_min_max_per_channel(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  limg_ui8_4 low[channels];
  limg_ui8_4 high[channels];

  for (size_t i = 0; i < channels; i++)
    high[i] = low[i] = *pStart;

  size_t x = 1;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
      {
        if (px[i] < low[i][i])
          low[i] = px;
        else if (px[i] > high[i][i])
          high[i] = px;
      }
    }

    x = 0;
  }

  limg_ui8_4 max_l = low[0];
  limg_ui8_4 max_h = high[0];

  size_t maxDist;

  if constexpr (limg_ColorDependentABError)
    maxDist = limg_color_error<channels>(max_l, max_h);
  else
    maxDist = limg_vector_error<channels>(max_l, max_h);

  size_t h_index = 1;

  for (size_t l_index = 0; l_index < channels; l_index++)
  {
    for (; h_index < channels; h_index++)
    {
      const limg_ui8_4 l = low[l_index];
      const limg_ui8_4 h = high[h_index];

      size_t dist;

      if constexpr (limg_ColorDependentABError)
        dist = limg_color_error<channels>(l, h);
      else
        dist = limg_vector_error<channels>(l, h);

      if (dist > maxDist)
      {
        maxDist = dist;
        max_l = l;
        max_h = h;
      }
    }

    h_index = 0;
  }

  a = max_l;
  b = max_h;
}

static LIMG_INLINE void limg_encode_get_block_a_b(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  if constexpr (limg_RetrievePreciseDecomposition)
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_per_channel<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
    else
      limg_encode_get_block_min_max_per_channel<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
  }
  else
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
    else
      limg_encode_get_block_min_max_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
  }
}

bool LIMG_DEBUG_NO_INLINE limg_encode_find_block_expand(limg_encode_context *pCtx, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB, const bool up, const bool down, const bool left, const bool right)
{
  int64_t ox = *pOffsetX;
  int64_t oy = *pOffsetY;
  int64_t rx = *pRangeX;
  int64_t ry = *pRangeY;

  bool canGrowUp = up;
  bool canGrowDown = down;
  bool canGrowLeft = left;
  bool canGrowRight = right;

#ifdef DEBUG
  if (!limg_encode_check_pixel_unused(pCtx, ox, oy, rx, ry))
  {
    __debugbreak(); // This function should've only been called if this was true, so... eh...
    return false;
  }
#endif

  limg_ui8_4 a, b;
  limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b);

  size_t blockError = 0;
  size_t rangeSize = 0;

  if (!limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, rx, ry, a, b, nullptr, &blockError, 0, &rangeSize))
    return false;

  while (canGrowUp || canGrowDown || canGrowLeft || canGrowRight)
  {
    if (canGrowRight)
    {
      const int64_t newRx = limgMin(rx + limg_BlockExpandStep, pCtx->sizeX - ox);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;

      bool cantGrowFurther = newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, ox + rx, oy, newRx - rx, ry, newA, newB);

      if (!cantGrowFurther)
      {
        if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
        {
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox + rx, oy, newRx - rx, ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
        }
        else
        {
          newRangeSize = 0;
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, newRx, ry, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
        }
      }

      if (cantGrowFurther)
      {
        canGrowRight = false;
      }
      else
      {
        rx = newRx;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
      }
    }

    if (canGrowDown)
    {
      const int64_t newRy = limgMin(ry + limg_BlockExpandStep, pCtx->sizeY - oy);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;

      bool cantGrowFurther = newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, ox, oy + ry, rx, newRy - ry, newA, newB);

      if (!cantGrowFurther)
      {
        if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy + ry, rx, newRy - ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
        else
        {
          newRangeSize = 0;
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, rx, newRy, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
        }
      }

      if (cantGrowFurther)
      {
        canGrowDown = false;
      }
      else
      {
        ry = newRy;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
      }
    }

    if (canGrowUp)
    {
      const int64_t newOx = limgMax(0LL, ox - (int64_t)limg_BlockExpandStep);
      const int64_t newRx = rx + (ox - newOx);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;

      bool cantGrowFurther = newOx == ox || !limg_encode_attempt_include_unused_pixels(pCtx, newOx, oy, ox - newOx, ry, newA, newB);

      if (!cantGrowFurther)
      {
        if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, newOx, oy, ox - newOx, ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
        else
        {
          newRangeSize = 0;
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, newOx, oy, newRx, ry, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
        }
      }

      if (cantGrowFurther)
      {
        canGrowUp = false;
      }
      else
      {
        rx = newRx;
        ox = newOx;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
      }
    }

    if (canGrowLeft)
    {
      const int64_t newOy = limgMax(0LL, oy - (int64_t)limg_BlockExpandStep);
      const int64_t newRy = ry + (oy - newOy);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;

      bool cantGrowFurther = newOy == oy || !limg_encode_attempt_include_unused_pixels(pCtx, ox, newOy, rx, oy - newOy, newA, newB);

      if (!cantGrowFurther)
      {
        if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, newOy, rx, oy - newOy, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
        else
        {
          newRangeSize = 0;
          cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, newOy, rx, newRy, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
        }
      }

      if (cantGrowFurther)
      {
        canGrowLeft = false;
      }
      else
      {
        ry = newRy;
        oy = newOy;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
      }
    }
  }

  *pOffsetX = ox;
  *pOffsetY = oy;
  *pRangeX = rx;
  *pRangeY = ry;
  *pA = a;
  *pB = b;

  return true;
}

bool LIMG_DEBUG_NO_INLINE limg_encode_find_block(limg_encode_context *pCtx, size_t &staticX, size_t &staticY, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB)
{
  size_t ox = staticX;
  size_t oy = staticY;

  for (; oy < pCtx->sizeY; oy += limg_BlockExpandStep)
  {
    const uint32_t *pBlockInfoLine = &pCtx->pBlockInfo[oy * pCtx->sizeX];

    for (; ox < pCtx->sizeX; ox += limg_BlockExpandStep)
    {
      if (!!(pBlockInfoLine[ox] & BlockInfo_InUse))
        continue;

      const size_t maxRx = pCtx->sizeX - ox;
      const size_t maxRy = pCtx->sizeY - oy;

      size_t rx = limgMin(limg_MinBlockSize, maxRx);
      size_t ry = limgMin(limg_MinBlockSize, maxRy);

      if (!limg_encode_check_pixel_unused(pCtx, ox, oy, rx, ry))
        continue;

      limg_ui8_4 a, b;
      limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b);

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;

      if (!limg_encode_find_block_expand(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB, false, true, false, true))
        continue;

      rx = *pRangeX;
      ry = *pRangeY;

      *pOffsetX = (ox + rx / 2 - limg_MinBlockSize / 2) & ~(size_t)(limg_BlockExpandStep - 1);
      *pOffsetY = (oy + ry / 2 - limg_MinBlockSize / 2) & ~(size_t)(limg_BlockExpandStep - 1);
      *pRangeX = limgMin(limg_MinBlockSize, rx);
      *pRangeY = limgMin(limg_MinBlockSize, ry);

      if (rx >= limg_MinBlockSize && ry >= limg_MinBlockSize && limg_encode_find_block_expand(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB, true, true, true, true))
      {
        staticX = ox + limg_BlockExpandStep;
        staticY = oy;

        return true;
      }

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;

      staticX = ox + rx;
      staticY = oy;

      return true;
    }

    ox = 0;
  }

  staticX = ox;
  staticY = oy;

  return false;
}

template <size_t channels>
static LIMG_INLINE bool limg_encode_try_bit_crush_block(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[channels];

  for (size_t i = 0; i < channels; i++)
    diff[i] = b[i] - a[i];

  constexpr uint32_t bias = 1 << 7;

  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  size_t blockError = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      const limg_ui8_4 px = *pLine;
      pLine++;

      const uint8_t fac = (*pFactors) >> shift;
      pFactors++;

      limg_ui8_4 dec;

      for (size_t i = 0; i < channels; i++)
        dec[i] = (uint8_t)(a[i] + (((fac << shift) * diff[i] + bias) >> 8));

      const size_t error = limg_color_error<channels>(dec, px);

      if (error > pCtx->maxPixelBitCrushError)
        return false;

      blockError += error;
    }
  }

  return ((blockError * 0x10) / (rangeX * rangeY) < pCtx->maxBlockBitCrushError);
}

static LIMG_INLINE uint8_t limg_encode_find_shift_for_block(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const uint8_t *pFactors, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint8_t shift = 0;

  if (pCtx->hasAlpha)
  {
    for (uint8_t i = 1; i < 8; i++)
    {
      if (!limg_encode_try_bit_crush_block<4>(pCtx, offsetX, offsetY, rangeX, rangeY, pFactors, i, a, b))
        break;
      else
        shift = i;
    }
  }
  else
  {
    for (uint8_t i = 1; i < 8; i++)
    {
      if (!limg_encode_try_bit_crush_block<3>(pCtx, offsetX, offsetY, rangeX, rangeY, pFactors, i, a, b))
        break;
      else
        shift = i;
    }
  }

  return shift;
}

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint32_t *pA, uint32_t *pB, uint32_t *pBlockIndex, uint8_t *pFactors, uint8_t *pBlockError, uint8_t *pShift, const bool hasAlpha)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  ctx.pSourceImage = pIn;
  ctx.pBlockInfo = pBlockIndex;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelBlockError = 0x10;
  ctx.maxBlockPixelError = 0x1C;
  ctx.maxPixelChannelBlockError = 0xA;
  ctx.maxBlockExpandError = 0x80;
  ctx.maxPixelBitCrushError = 0xC0 * (size_t)(ctx.hasAlpha ? 10 : 7);
  ctx.maxBlockBitCrushError = 0x30 * (size_t)(ctx.hasAlpha ? 10 : 7);
  ctx.ditheringEnabled = true;

  if constexpr (limg_LuminanceDependentPixelError)
  {
    ctx.maxPixelBlockError *= 0x10;
    ctx.maxBlockPixelError *= 0x10;
  }

  if constexpr (limg_ColorDependentBlockError)
  {
    //ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    //ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 6 : 4);
    ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 6 : 4);
  }

  memset(ctx.pBlockInfo, 0, sizeof(uint32_t) * ctx.sizeX * ctx.sizeY);

  size_t blockFactorsCapacity = 1024 * 1024;
  float *pBlockFactors = reinterpret_cast<float *>(malloc(blockFactorsCapacity * sizeof(float)));
  LIMG_ERROR_IF(pBlockFactors == nullptr, limg_error_MemoryAllocationFailure);

  size_t blockFindStaticX = 0;
  size_t blockFindStaticY = 0;

  uint32_t blockIndex = 0;
  size_t accumBlockSize = 0;
  size_t accumBits = 0;
  size_t unweightedBits = 0;
  size_t accumBlockError = 0;
  size_t unweightedBlockError = 0;

  uint64_t ditherLast = 0xCA7F00D15BADF00D;

  while (true)
  {
    size_t ox, oy, rx, ry;
    limg_ui8_4 a, b;

    if (!limg_encode_find_block(&ctx, blockFindStaticX, blockFindStaticY, &ox, &oy, &rx, &ry, &a, &b))
      break;

    if (rx * ry > blockFactorsCapacity)
    {
      blockFactorsCapacity = (rx * ry + 1023) & ~(size_t)1023;
      pBlockFactors = reinterpret_cast<float *>(realloc(pBlockFactors, blockFactorsCapacity * sizeof(float)));
      LIMG_ERROR_IF(pBlockFactors == nullptr, limg_error_MemoryAllocationFailure);
    }

    size_t blockError = 0;
    size_t rangeSize = 0;
    limg_encode_check_area<true, true, false, false, true>(&ctx, ox, oy, rx, ry, a, b, pBlockFactors, &blockError, 0, &rangeSize);
    // Could be: `limg_encode_check_area<true, false, false, false, false>(&ctx, ox, oy, rx, ry, a, b, pBlockFactors, nullptr, 0, nullptr);` if we weren't writing block errors to a buffer.

    accumBlockError += blockError * 0x10;
    blockError = (blockError * 0x10) / rangeSize;
    unweightedBlockError += blockError;

#ifdef _DEBUG
    if (blockError > ctx.maxBlockPixelError)
      __debugbreak();
#endif

    float *pBFacs = pBlockFactors;
    uint8_t *pBFacsU8Start = reinterpret_cast<uint8_t *>(pBFacs);
    uint8_t *pBFacsU8 = pBFacsU8Start;

    for (size_t i = 0; i < rangeSize; i++)
    {
      *pBFacsU8 = (uint8_t)limgClamp((int32_t)(*pBFacs * (float_t)0xFF + 0.5f), 0, 0xFF);
      pBFacs++;
      pBFacsU8++;
    }

    const uint8_t shift = limg_encode_find_shift_for_block(&ctx, ox, oy, rx, ry, pBFacsU8Start, a, b);
    accumBits += (8 - shift) * rangeSize;
    unweightedBits += (8 - shift);

    if (shift)
    {
      if (ctx.ditheringEnabled)
      {
        const uint32_t ditherSize = (1 << shift) - 1;
        const int32_t ditherOffset = 1 << (shift - 1);

        for (size_t i = 0; i < rangeSize; i++)
        {
          const uint64_t oldstate_hi = ditherLast;
          ditherLast = oldstate_hi * 6364136223846793005ULL + 1;

          const uint32_t xorshifted_hi = (uint32_t)(((oldstate_hi >> 18) ^ oldstate_hi) >> 27);
          const uint32_t rot_hi = (uint32_t)(oldstate_hi >> 59);

          const uint32_t hi = (xorshifted_hi >> rot_hi) | (xorshifted_hi << (uint32_t)((-(int32_t)rot_hi) & 31));

          const int32_t rand = (int32_t)(hi & ditherSize) - ditherOffset;

          pBFacsU8Start[i] = (uint8_t)limgClamp((int32_t)pBFacsU8Start[i] + rand, 0, 0xFF) >> shift;
        }
      }
      else
      {
        for (size_t i = 0; i < rangeSize; i++)
          pBFacsU8Start[i] >>= shift;
      }
    }

    for (size_t y = 0; y < ry; y++)
    {
      uint32_t *pBlockIndexLine = ctx.pBlockInfo + (y + oy) * ctx.sizeX + ox;
      uint32_t *pALine = pA + (y + oy) * ctx.sizeX + ox;
      uint32_t *pBLine = pB + (y + oy) * ctx.sizeX + ox;
      uint8_t *pFactorsLine = pFactors + (y + oy) * ctx.sizeX + ox;
      uint8_t *pBlockErrorLine = pBlockError + (y + oy) * ctx.sizeX + ox;
      uint8_t *pShiftLine = pShift + (y + oy) * ctx.sizeX + ox;

      for (size_t x = 0; x < rx; x++)
      {
        *pBlockIndexLine = blockIndex | BlockInfo_InUse;
        pBlockIndexLine++;

        *pALine = *reinterpret_cast<const uint32_t *>(&a);
        pALine++;

        *pBLine = *reinterpret_cast<const uint32_t *>(&b);
        pBLine++;

        *pFactorsLine = pBFacsU8Start[x + y * rx] << shift;
        pFactorsLine++;

        *pBlockErrorLine = (uint8_t)limgMin(blockError >> 3, 0xFFULL);
        pBlockErrorLine++;

        *pShiftLine = (uint8_t)(1 << shift);
        pShiftLine++;
      }
    }

    uint32_t *pDecodedStart = pDecoded + oy * ctx.sizeX + ox;

    if (hasAlpha)
      limg_decode_block_from_factors<4>(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, shift, a, b);
    else
      limg_decode_block_from_factors<3>(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, shift, a, b);
    
    blockIndex++;
    accumBlockSize += (rx * ry);
  }

#ifdef PRINT_TEST_OUTPUT
  printf("\n%" PRIu32 " Blocks generated.\n%5.3f %% Coverage\nAverage Size: %5.3f Pixels [(%5.3f px)^2].\nMinimum Block Size: %" PRIu64 "\nBlock Size Grow Step: %" PRIu64 "\nAverage Block Error: %5.3f (Unweighted: %5.3f)\nAverage Block Bits: %5.3f (Unweighted: %5.3f)\n\n", blockIndex, (accumBlockSize / (double)(sizeX * sizeY)) * 100.0, accumBlockSize / (double)blockIndex, sqrt(accumBlockSize / (double)blockIndex), limg_MinBlockSize, limg_BlockExpandStep, accumBlockError / (double)accumBlockSize, unweightedBlockError / (double)blockIndex, accumBits / (double)accumBlockSize, unweightedBits / (double)blockIndex);
#endif

  for (size_t i = 0; i < sizeX * sizeY; i++)
    if (!(ctx.pBlockInfo[i] & BlockInfo_InUse))
      pDecoded[i] = pIn[i];

  if (!hasAlpha)
    for (size_t i = 0; i < sizeX * sizeY; i++)
      pDecoded[i] |= 0xFF000000;

  goto epilogue;

epilogue:
  limgFreePtr(&pBlockFactors);

  return result;
}
