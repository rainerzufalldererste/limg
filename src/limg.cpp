#include "limg.h"

#include <malloc.h>
#include <memory.h>

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
  int32_t maxPixelError, maxBlockError, maxChannelError, maxBlockExpandError;
  bool hasAlpha;
};

struct limg_ui8_4
{
  uint8_t v[4];

  inline uint8_t &operator[](const size_t index) { return v[index]; }
  inline const uint8_t &operator[](const size_t index) const { return v[index]; }
};

static_assert(sizeof(limg_ui8_4) == 4, "Invalid Configuration");

constexpr uint32_t BlockInfo_InUse = (uint32_t)((uint32_t)1 << 31);

constexpr size_t limg_MinBlockSize = 4;
constexpr size_t limg_BlockExpandStep = 2;

//////////////////////////////////////////////////////////////////////////

static void limg_decode_block_from_factors_alpha(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[4];

  for (size_t i = 0; i < 4; i++)
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

      for (size_t i = 0; i < 4; i++)
        px[i] = (uint8_t)(a[i] + (((fac << shift) * diff[i] + bias) >> 8));

      *pOutLine = px;
      pOutLine++;
    }
  }
}

static void limg_decode_block_from_factors_no_alpha(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[3];

  for (size_t i = 0; i < 3; i++)
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

      for (size_t i = 0; i < 3; i++)
        px[i] = (uint8_t)(a[i] + (((fac << shift) * diff[i] + bias) >> 8));

      *pOutLine = px;
      pOutLine++;
    }
  }
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError>
static bool limg_encode_check_area(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  size_t blockError = 0;

  if (pCtx->hasAlpha)
  {
    constexpr size_t channels = 4;

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
            if ((int32_t)px[i] < a[i] - pCtx->maxChannelError || (int32_t)px[i] > b[i] + pCtx->maxChannelError)
              return false;

        float fac[channels];
        
        for (size_t i = 0; i < channels; i++)
          fac[i] = (float)(px[i] - a[i]);

        const float avg = (fac[0] + fac[1] + fac[2] + fac[3]) * inverse_dist_complete_or_one; // R G B A.

        if constexpr (WriteToFactors)
        {
          *pFactors = avg;
          pFactors++;
        }

        size_t error = 0;

        for (size_t i = 0; i < channels; i++)
        {
          const size_t e = (size_t)(0.5f + fabsf((fac[i] / dist_or_one[i] - avg) * dist[i]));
          error += e * e;
        }

        if constexpr (CheckPixelError)
          if (error > pCtx->maxPixelError)
            return false;

        blockError += error;
      }
    }
  }
  else
  {
    constexpr size_t channels = 3;

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
            if ((int32_t)px[i] < a[i] - pCtx->maxChannelError || (int32_t)px[i] > b[i] + pCtx->maxChannelError)
              return false;

        float fac[channels];

        for (size_t i = 0; i < channels; i++)
          fac[i] = (float)(px[i] - a[i]);

        const float avg = (fac[0] + fac[1] + fac[2]) * inverse_dist_complete_or_one; // R G B.

        if constexpr (WriteToFactors)
        {
          *pFactors = avg;
          pFactors++;
        }

        size_t error = 0;

        for (size_t i = 0; i < channels; i++)
        {
          const size_t e = (size_t)(0.5f + fabsf((fac[i] / dist_or_one[i] - avg) * dist[i]));
          error += e * e;
        }

        if constexpr (CheckPixelError)
          if (error > pCtx->maxPixelError)
            return false;

        blockError += error;
      }
    }
  }

  if constexpr (WriteBlockError)
    *pBlockError = blockError;
  
  return (blockError < pCtx->maxBlockError);
}

static bool limg_encode_attempt_include_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  limg_ui8_4 a = out_a;
  limg_ui8_4 b = out_b;

  if (pCtx->hasAlpha)
  {
    constexpr size_t channels = 4;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        const int64_t low =
          (limgMax(-1LL, ((int64_t)a[0] - px[0]))) +
          (limgMax(-1LL, ((int64_t)a[1] - px[1]))) +
          (limgMax(-1LL, ((int64_t)a[2] - px[2]))) +
          (limgMax(-1LL, ((int64_t)a[3] - px[3]))); // R G B A.

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

          const float avg_offset = (offset[0] + offset[1] + offset[2]) / dist_complete_or_one_px;

          size_t error = 0;

          for (size_t i = 0; i < channels; i++)
          {
            const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
            error += e * e;
          }

          if (error > pCtx->maxBlockExpandError)
            return false;

          a = px;
        }
        else
        {
          const int64_t high =
            (limgMax(-1LL, (px[0] - (int64_t)b[0]))) +
            (limgMax(-1LL, (px[1] - (int64_t)b[1]))) +
            (limgMax(-1LL, (px[2] - (int64_t)b[2]))) +
            (limgMax(-1LL, (px[3] - (int64_t)b[3]))); // R G B A.

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

            const float avg_offset = (offset[0] + offset[1] + offset[2] + offset[3]) / dist_complete_or_one_px; // R G B A.

            size_t error = 0;

            for (size_t i = 0; i < channels; i++)
            {
              const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
              error += e * e;
            }

            if (error > pCtx->maxBlockExpandError)
              return false;

            b = px;
          }
        }
      }
    }
  }
  else
  {
    constexpr size_t channels = 3;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        const int64_t low =
          (limgMax(-1LL, ((int64_t)a[0] - px[0]))) +
          (limgMax(-1LL, ((int64_t)a[1] - px[1]))) +
          (limgMax(-1LL, ((int64_t)a[2] - px[2]))); // R G B.

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

          const float avg_offset = (offset[0] + offset[1] + offset[2]) / dist_complete_or_one_px;

          size_t error = 0;

          for (size_t i = 0; i < channels; i++)
          {
            const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
            error += e * e;
          }

          if (error > pCtx->maxBlockExpandError)
            return false;

          a = px;
        }
        else
        {
          const int64_t high =
            (limgMax(-1LL, (px[0] - (int64_t)b[0]))) +
            (limgMax(-1LL, (px[1] - (int64_t)b[1]))) +
            (limgMax(-1LL, (px[2] - (int64_t)b[2]))); // R G B.

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

            const float avg_offset = (offset[0] + offset[1] + offset[2]) / dist_complete_or_one_px; // R G B.

            size_t error = 0;

            for (size_t i = 0; i < channels; i++)
            {
              const size_t e = (size_t)(0.5f + fabsf((offset[i] / dist_or_one_px[i] - avg_offset) * dist_px[i]));
              error += e * e;
            }

            if (error > pCtx->maxBlockExpandError)
              return false;

            b = px;
          }
        }
      }
    }
  }

  out_a = a;
  out_b = b;

  return true;
}

static bool limg_encode_check_pixel_unused(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY)
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

static bool limg_encode_attempt_include_unused_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  return limg_encode_check_pixel_unused(pCtx, offsetX, offsetY, rangeX, rangeY) && limg_encode_attempt_include_pixels(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
}

static void limg_encode_get_block_min_max(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  a = pStart[0];
  b = a;
  
  if (pCtx->hasAlpha)
  {
    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        const int64_t low =
          (limgMax(-1LL, ((int64_t)a[0] - px[0]))) +
          (limgMax(-1LL, ((int64_t)a[1] - px[1]))) +
          (limgMax(-1LL, ((int64_t)a[2] - px[2]))) +
          (limgMax(-1LL, ((int64_t)a[3] - px[3])));

        if (low > 0)
        {
          a = px;
        }
        else
        {
          const int64_t high =
            (limgMax(-1LL, (px[0] - (int64_t)b[0]))) +
            (limgMax(-1LL, (px[1] - (int64_t)b[1]))) +
            (limgMax(-1LL, (px[2] - (int64_t)b[2]))) +
            (limgMax(-1LL, (px[3] - (int64_t)b[3])));

          if (high > 0)
            b = px;
        }
      }
    }
  }
  else
  {
    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        const int64_t low =
          (limgMax(-1LL, ((int64_t)a[0] - px[0]))) +
          (limgMax(-1LL, ((int64_t)a[1] - px[1]))) +
          (limgMax(-1LL, ((int64_t)a[2] - px[2])));

        if (low > 0)
        {
          a = px;
        }
        else
        {
          const int64_t high =
            (limgMax(-1LL, (px[0] - (int64_t)b[0]))) +
            (limgMax(-1LL, (px[1] - (int64_t)b[1]))) +
            (limgMax(-1LL, (px[2] - (int64_t)b[2])));

          if (high > 0)
            b = px;
        }
      }
    }
  }
}

bool limg_encode_find_block_from_center(limg_encode_context *pCtx, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB)
{
  int64_t ox = *pOffsetX;
  int64_t oy = *pOffsetY;
  int64_t rx = *pRangeX;
  int64_t ry = *pRangeY;

  ox += rx / 2 - limg_MinBlockSize / 2;
  oy += ry / 2 - limg_MinBlockSize / 2;
  rx = limg_MinBlockSize;
  ry = limg_MinBlockSize;

  bool canGrowUp = true;
  bool canGrowDown = true;
  bool canGrowLeft = true;
  bool canGrowRight = true;

#ifdef DEBUG
  if (!limg_encode_check_pixel_unused(pCtx, ox, oy, rx, ry))
  {
    __debugbreak(); // This function should've only been called if this was true, so... eh...
    return false;
  }
#endif

  limg_ui8_4 a, b;
  limg_encode_get_block_min_max(pCtx, ox, oy, rx, ry, a, b);

  if (!limg_encode_check_area<false, false, true, true>(pCtx, ox, oy, rx, ry, a, b, nullptr, nullptr))
  {
    //__debugbreak(); // This function should've only been called if this was true, so... eh...
    return false;
  }

  while (canGrowUp || canGrowDown || canGrowLeft || canGrowRight)
  {
    if (canGrowRight)
    {
      const int64_t newRx = limgMin(rx + limg_BlockExpandStep, pCtx->sizeX - ox);

      if (newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, ox + rx, oy, newRx - rx, ry, a, b))
      {
        canGrowRight = false;
      }
      else
      {
        rx = newRx;
      }
    }

    if (canGrowDown)
    {
      const int64_t newRy = limgMin(ry + limg_BlockExpandStep, pCtx->sizeY - oy);

      if (newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, ox, oy + ry, rx, newRy - ry, a, b))
      {
        canGrowDown = false;
      }
      else
      {
        ry = newRy;
      }
    }

    if (canGrowUp)
    {
      const int64_t newOx = limgMax(0LL, ox - (int64_t)limg_BlockExpandStep);

      if (newOx == ox || !limg_encode_attempt_include_unused_pixels(pCtx, newOx, oy, ox - newOx, ry, a, b))
      {
        canGrowUp = false;
      }
      else
      {
        rx += ox - newOx;
        ox = newOx;
      }
    }

    if (canGrowLeft)
    {
      const int64_t newOy = limgMax(0LL, oy - (int64_t)limg_BlockExpandStep);

      if (newOy == oy || !limg_encode_attempt_include_unused_pixels(pCtx, ox, newOy, rx, oy - newOy, a, b))
      {
        canGrowLeft = false;
      }
      else
      {
        ry += oy - newOy;
        oy = newOy;
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

bool limg_encode_find_block(limg_encode_context *pCtx, size_t &staticX, size_t &staticY, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB)
{
  for (; staticY < pCtx->sizeY; staticY++)
  {
    const uint32_t *pBlockInfoLine = &pCtx->pBlockInfo[staticY * pCtx->sizeX];

    for (; staticX < pCtx->sizeX; staticX++)
    {
      if (!!(pBlockInfoLine[staticX] & BlockInfo_InUse))
        continue;

      const size_t maxRx = pCtx->sizeX - staticX;
      const size_t maxRy = pCtx->sizeY - staticY;

      size_t rx = limgMin(limg_MinBlockSize, maxRx);
      size_t ry = limgMin(limg_MinBlockSize, maxRy);

      if (!limg_encode_check_pixel_unused(pCtx, staticX, staticY, rx, ry))
        continue;

      limg_ui8_4 a, b;
      limg_encode_get_block_min_max(pCtx, staticX, staticY, rx, ry, a, b);

      if (!limg_encode_check_area<false, false, true, true>(pCtx, staticX, staticY, rx, ry, a, b, nullptr, nullptr))
        continue;

      while (true)
      {
        const size_t newRx = limgMin(rx + limg_BlockExpandStep, maxRx);

        if (newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, staticX + rx, staticY, newRx - rx, ry, a, b))
          goto only_seek_y;

        rx = newRx;

        const size_t newRy = limgMin(ry + limg_BlockExpandStep, maxRy);

        if (newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, staticX, staticY + ry, rx, newRy - ry, a, b))
          goto only_seek_x;

        ry = newRy;
      }

      goto after_seek;

    only_seek_y:
      while (true)
      {
        const size_t newRy = limgMin(ry + limg_BlockExpandStep, maxRy);

        if (newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, staticX, staticY + ry, rx, newRy - ry, a, b))
          goto after_seek;

        ry = newRy;
      }

      goto after_seek;

    only_seek_x:
      while (true)
      {
        const size_t newRx = limgMin(rx + limg_BlockExpandStep, maxRx);

        if (newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, staticX + rx, staticY, newRx - rx, ry, a, b))
          goto after_seek;

        rx = newRx;
      }

      goto after_seek;

    after_seek:
      *pOffsetX = staticX;
      *pOffsetY = staticY;
      *pRangeX = rx;
      *pRangeY = ry;

      if (rx >= limg_MinBlockSize && ry >= limg_MinBlockSize && limg_encode_find_block_from_center(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB))
        return true;

      *pA = a;
      *pB = b;

      staticX += rx;

      return true;
    }

    staticX = 0;
  }

  return false;
}

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint32_t *pA, uint32_t *pB, uint32_t *pBlockIndex, uint8_t *pFactors, uint8_t *pShift, const bool hasAlpha)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  ctx.pSourceImage = pIn;
  ctx.pBlockInfo = pBlockIndex;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelError = 0x10;
  ctx.maxBlockError = 0x10000;
  ctx.maxChannelError = 0x10;
  ctx.maxBlockExpandError = 0x10;

  memset(ctx.pBlockInfo, 0, sizeof(uint32_t) * ctx.sizeX * ctx.sizeY);

  size_t blockFactorsCapacity = 1024 * 1024;
  float *pBlockFactors = reinterpret_cast<float *>(malloc(blockFactorsCapacity * sizeof(float)));
  LIMG_ERROR_IF(pBlockFactors == nullptr, limg_error_MemoryAllocationFailure);

  size_t blockFindStaticX = 0;
  size_t blockFindStaticY = 0;

  uint32_t blockIndex = 0;

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
    limg_encode_check_area<true, true, false, false>(&ctx, ox, oy, rx, ry, a, b, pBlockFactors, &blockError);

    float *pBFacs = pBlockFactors;
    uint8_t *pBFacsU8Start = reinterpret_cast<uint8_t *>(pBFacs);
    uint8_t *pBFacsU8 = pBFacsU8Start;
    const uint8_t shift = 0;

    blockError /= (rx * ry);

    for (size_t y = 0; y < ry; y++)
    {
      uint32_t *pBlockIndexLine = ctx.pBlockInfo + (y + oy) * ctx.sizeX + ox;
      uint32_t *pALine = pA + (y + oy) * ctx.sizeX + ox;
      uint32_t *pBLine = pB + (y + oy) * ctx.sizeX + ox;
      uint8_t *pFactorsLine = pFactors + (y + oy) * ctx.sizeX + ox;
      uint8_t *pShiftLine = pShift + (y + oy) * ctx.sizeX + ox;

      for (size_t x = 0; x < rx; x++)
      {
        *pBlockIndexLine = blockIndex | BlockInfo_InUse;
        pBlockIndexLine++;

        *pALine = *reinterpret_cast<const uint32_t *>(&a);
        pALine++;

        *pBLine = *reinterpret_cast<const uint32_t *>(&b);
        pBLine++;

        *pFactorsLine = *pBFacsU8 = (uint8_t)limgClamp((int32_t)(*pBFacs * (float_t)0xFF + 0.5f), 0, 0xFF) >> shift;
        pBFacs++;
        pBFacsU8++;
        pFactorsLine++;

        *pShiftLine = (uint8_t)limgMin(blockError / 8, 0xFFULL);
        pShiftLine++;
      }
    }

    uint32_t *pDecodedStart = pDecoded + oy * ctx.sizeX + ox;

    if (hasAlpha)
      limg_decode_block_from_factors_alpha(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, 0, a, b);
    else
      limg_decode_block_from_factors_no_alpha(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, 0, a, b);
    
    blockIndex++;
  }

  //for (size_t y = 0; y < ctx.blocksY; y++)
  //{
  //  uint8_t *pFactorLine = pFactors + (y * ctx.sizeX * limg_BlockSize);
  //  uint32_t *pDecodedLine = pDecoded + (y * sizeX * limg_BlockSize);
  //
  //  for (size_t x = 0; x < ctx.blocksX; x++)
  //  {
  //    limg_ui8_4 a, b;
  //
  //    limg_encode_get_block_min_max(&ctx, x, y, a, b);
  //    
  //    float factors[limg_BlockSize * limg_BlockSize];
  //    size_t blockError;
  //
  //    if (!limg_encode_check_block(&ctx, x, y, a, b, factors, &blockError))
  //      __debugbreak();
  //
  //    *pA = *reinterpret_cast<const uint32_t *>(&a);
  //    pA++;
  //
  //    *pB = *reinterpret_cast<const uint32_t *>(&b);
  //    pB++;
  //
  //    uint8_t u8factors[limg_BlockSize * limg_BlockSize];
  //    uint8_t shift = 0;
  //
  //    if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && (!ctx.hasAlpha || a[3] == b[3]))
  //    {
  //      shift = 8;
  //
  //      memset(u8factors, 0, sizeof(u8factors));
  //    }
  //    else
  //    {
  //      size_t error_rem = blockError;
  //
  //      while (error_rem < 0xFFFF)
  //      {
  //        shift++;
  //        error_rem = (error_rem * 3) + 1;
  //      }
  //
  //      size_t i = 0;
  //
  //      for (size_t yy = 0; yy < limg_BlockSize; yy++)
  //      {
  //        for (size_t xx = 0; xx < limg_BlockSize; xx++, i++)
  //        {
  //          u8factors[i] = (uint8_t)limgClamp((int32_t)(factors[xx + yy * limg_BlockSize] * (float_t)0xFF + 0.5f), 0, 0xFF) >> shift;
  //          pFactorLine[xx + yy * ctx.sizeX] = u8factors[i] << shift;
  //        }
  //      }
  //    }
  //
  //    *pShift = (uint8_t)limgMin(1 << shift, 0xFFULL);
  //    pShift++;
  //
  //    pFactorLine += limg_BlockSize;
  //
  //    if (hasAlpha)
  //      limg_decode_block_from_factors_alpha(pDecodedLine, sizeX, pFactors, *pShift, *reinterpret_cast<const limg_ui8_4 *>(pA), *reinterpret_cast<const limg_ui8_4 *>(pB));
  //    else
  //      limg_decode_block_from_factors_no_alpha(pDecodedLine, sizeX, pFactors, *pShift, *reinterpret_cast<const limg_ui8_4 *>(pA), *reinterpret_cast<const limg_ui8_4 *>(pB));
  //
  //    pDecodedLine += limg_BlockSize;
  //  }
  //}

  if (!hasAlpha)
    for (size_t i = 0; i < sizeX * sizeY; i++)
      pDecoded[i] |= 0xFF000000;

  goto epilogue;

epilogue:
  limgFreePtr(&pBlockFactors);

  return result;
}
