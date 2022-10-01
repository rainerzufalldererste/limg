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
  uint8_t *pBlockInfo;
  size_t sizeX, sizeY;
  size_t blocksX, blocksY;
  int32_t maxPixelError, maxBlockError;
  bool hasAlpha;
};

struct limg_ui8_4
{
  uint8_t v[4];

  inline uint8_t &operator[](const size_t index) { return v[index]; }
  inline const uint8_t &operator[](const size_t index) const { return v[index]; }
};

static_assert(sizeof(limg_ui8_4) == 4, "Invalid Configuration");

constexpr size_t BlockInfo_InUse = 1 << 0;

constexpr size_t limg_BlockSize = 8;

//////////////////////////////////////////////////////////////////////////

static inline const uint32_t * limg_block(const uint32_t *pImage, const size_t sizeX, const size_t blockX, const size_t blockY)
{
  return pImage + (blockX + blockY * sizeX) * limg_BlockSize;
}

static void limg_decode_block_from_factors_alpha(uint32_t *pOut, const size_t sizeX, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[4];

  for (size_t i = 0; i < 4; i++)
    diff[i] = b[i] - a[i];

  constexpr uint32_t bias = 1 << 7;

  for (size_t y = 0; y < limg_BlockSize; y++)
  {
    limg_ui8_4 *pOutLine = reinterpret_cast<limg_ui8_4 *>(pOut + y * sizeX);

    for (size_t x = 0; x < limg_BlockSize; x++)
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

static void limg_decode_block_from_factors_no_alpha(uint32_t *pOut, const size_t sizeX, const uint8_t *pFactors, const uint8_t shift, const limg_ui8_4 &a, const limg_ui8_4 &b)
{
  uint32_t diff[3];

  for (size_t i = 0; i < 3; i++)
    diff[i] = b[i] - a[i];

  constexpr uint32_t bias = 1 << 7;

  for (size_t y = 0; y < limg_BlockSize; y++)
  {
    limg_ui8_4 *pOutLine = reinterpret_cast<limg_ui8_4 *>(pOut + y * sizeX);

    for (size_t x = 0; x < limg_BlockSize; x++)
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

static bool limg_encode_check_block(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(limg_block(pCtx->pSourceImage, pCtx->sizeX, offsetX, offsetY));

  size_t blockError = 0;

  if (pCtx->hasAlpha)
  {
    float dist[4];
    float dist_or_one[4];
    float inverse_dist_complete_or_one = 0;

    for (size_t i = 0; i < 4; i++)
    {
      const uint8_t diff = (b[i] - a[i]);
      dist[i] = (float)diff;
      dist_or_one[i] = limgMax(1, dist[i]);

      if (diff != 0)
        inverse_dist_complete_or_one += (float)dist[i];
    }

    inverse_dist_complete_or_one = 1.f / limgMax(1.f, inverse_dist_complete_or_one);;

    for (size_t y = 0; y < limg_BlockSize; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < limg_BlockSize; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        for (size_t i = 0; i < 4; i++)
          if ((int32_t)px[i] < a[i] - pCtx->maxPixelError || (int32_t)px[i] > b[i] + pCtx->maxPixelError)
            return false;

        float fac[4];
        
        for (size_t i = 0; i < 4; i++)
          fac[i] = (float)(px[i] - a[i]);

        const float avg = (fac[0] + fac[1] + fac[2]) * inverse_dist_complete_or_one;

        *pFactors = avg;
        pFactors++;

        size_t error = 0;

        for (size_t i = 0; i < 4; i++)
          error += (size_t)(0.5f + fabsf((fac[i] / dist_or_one[i] - avg) * dist[i]));

        if (error > pCtx->maxPixelError)
          return false;

        blockError += error;
      }
    }
  }
  else
  {
    float dist[3];
    float dist_or_one[3];
    float inverse_dist_complete_or_one = 0;

    for (size_t i = 0; i < 3; i++)
    {
      const uint8_t diff = (b[i] - a[i]);
      dist[i] = (float)diff;
      dist_or_one[i] = limgMax(1, dist[i]);
      
      if (diff != 0)
        inverse_dist_complete_or_one += (float)dist[i];
    }

    inverse_dist_complete_or_one = 1.f / limgMax(1.f, inverse_dist_complete_or_one);;

    for (size_t y = 0; y < limg_BlockSize; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < limg_BlockSize; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        for (size_t i = 0; i < 3; i++)
          if ((int32_t)px[i] < a[i] - pCtx->maxPixelError || (int32_t)px[i] > b[i] + pCtx->maxPixelError)
            return false;

        float fac[3];

        for (size_t i = 0; i < 3; i++)
          fac[i] = (float)(px[i] - a[i]);

        const float avg = (fac[0] + fac[1] + fac[2]) * inverse_dist_complete_or_one;

        *pFactors = avg;
        pFactors++;

        size_t error = 0;

        for (size_t i = 0; i < 3; i++)
          error += (size_t)(0.5f + fabsf((fac[i] / dist_or_one[i] - avg) * dist[i]));

        if (error > pCtx->maxPixelError)
          return false;

        blockError += error;
      }
    }
  }

  *pBlockError = blockError;
  
  return (blockError < pCtx->maxBlockError);
}

static void limg_encode_get_block_min_max(limg_encode_context *pCtx, const size_t blockX, const size_t blockY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(limg_block(pCtx->pSourceImage, pCtx->sizeX, blockX, blockY));

  a = pStart[0];
  b = a;
  
  if (pCtx->hasAlpha)
  {
    for (size_t y = 0; y < limg_BlockSize; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < limg_BlockSize; x++)
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
    for (size_t y = 0; y < limg_BlockSize; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < limg_BlockSize; x++)
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

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pA, uint32_t *pB, uint32_t *pDecoded, uint8_t *pFactors, uint8_t *pBlockError, const bool hasAlpha)
{
  limg_result result = limg_success;
  
  limg_encode_context ctx;
  ctx.pSourceImage = pIn;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.blocksX = ctx.sizeX / limg_BlockSize;
  ctx.blocksY = ctx.sizeY / limg_BlockSize;
  ctx.hasAlpha = hasAlpha;
  ctx.pBlockInfo = nullptr;
  ctx.maxPixelError = 0xFFFFFF;
  ctx.maxBlockError = 0xFFFFFF;

  for (size_t y = 0; y < ctx.blocksY; y++)
  {
    uint8_t *pFactorLine = pFactors + (y * ctx.sizeX * limg_BlockSize);
    uint32_t *pDecodedLine = pDecoded + (y * ctx.sizeX * limg_BlockSize);

    for (size_t x = 0; x < ctx.blocksX; x++)
    {
      limg_ui8_4 a, b;

      limg_encode_get_block_min_max(&ctx, x, y, a, b);
      
      float factors[limg_BlockSize * limg_BlockSize];
      size_t blockError;

      if (!limg_encode_check_block(&ctx, x, y, a, b, factors, &blockError))
        __debugbreak();

      *pA = *reinterpret_cast<const uint32_t *>(&a);
      pA++;

      *pB = *reinterpret_cast<const uint32_t *>(&b);
      pB++;

      uint8_t u8factors[limg_BlockSize * limg_BlockSize];
      uint8_t shift = 0;

      if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && (!ctx.hasAlpha || a[3] == b[3]))
      {
        shift = 8;

        memset(u8factors, 0, sizeof(u8factors));
      }
      else
      {
        size_t error_rem = blockError;

        while (error_rem < 0xFFFF)
        {
          shift++;
          error_rem = (error_rem * 3) + 1;
        }

        size_t i = 0;

        for (size_t yy = 0; yy < limg_BlockSize; yy++)
        {
          for (size_t xx = 0; xx < limg_BlockSize; xx++, i++)
          {
            u8factors[i] = (uint8_t)limgClamp((int32_t)(factors[xx + yy * limg_BlockSize] * (float_t)0xFF + 0.5f), 0, 0xFF) >> shift;
            pFactorLine[xx + yy * ctx.sizeX] = u8factors[i] << shift;
          }
        }
      }

      *pBlockError = (uint8_t)limgMin(1 << shift, 0xFFULL);
      pBlockError++;

      pFactorLine += limg_BlockSize;

      if (hasAlpha)
        limg_decode_block_from_factors_alpha(pDecodedLine, ctx.sizeX, u8factors, shift, a, b);
      else
        limg_decode_block_from_factors_no_alpha(pDecodedLine, ctx.sizeX, u8factors, shift, a, b);

      pDecodedLine += limg_BlockSize;
    }
  }

  if (!ctx.hasAlpha)
    for (size_t i = 0; i < ctx.sizeX * ctx.sizeY; i++)
      pDecoded[i] |= 0xFF000000;

  goto epilogue;

epilogue:
  return result;
}
