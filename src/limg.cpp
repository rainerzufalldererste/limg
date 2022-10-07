#include "limg.h"

#include <malloc.h>
#include <memory.h>
#include <math.h>
#include <float.h>

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

constexpr uint32_t BlockInfo_InUse = (uint32_t)((uint32_t)1 << 31);

constexpr size_t limg_BlockExpandStep = 2; // must be a power of two.
constexpr size_t limg_MinBlockSize = limg_BlockExpandStep * 4;
constexpr bool limg_ColorDependentBlockError = true;
constexpr bool limg_LuminanceDependentPixelError = false;
constexpr bool limg_ColorDependentABError = true;
constexpr bool limg_DiagnoseCulprits = true;

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

  size_t culprits, 
    culpritWasPixelBlockError, 
    culpritWasBlockPixelError, 
    culpritWasPixelChannelBlockError, 
    culpritWasBlockExpandError, 
    culpritWasPixelBitCrushError, 
    culpritWasBlockBitCrushError;
};

#define LIMG_PRECISE_DECOMPOSITION 2

#ifdef LIMG_PRECISE_DECOMPOSITION
constexpr size_t limg_RetrievePreciseDecomposition = LIMG_PRECISE_DECOMPOSITION;
#else
#define LIMG_PRECISE_DECOMPOSITION 0
constexpr size_t limg_RetrievePreciseDecomposition = 0;
#endif

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
  union
  {
    uint8_t v[4];
    uint32_t n;
  } u;

  LIMG_INLINE uint8_t &operator[](const size_t index) { return u.v[index]; }
  LIMG_INLINE const uint8_t &operator[](const size_t index) const { return u.v[index]; }

  LIMG_INLINE bool equals_w_alpha(const limg_ui8_4 &other) const
  {
    return (u.n ^ other.u.n) == 0;
  }
  
  LIMG_INLINE bool equals_wo_alpha(const limg_ui8_4 &other) const 
  {
    return ((u.n ^ other.u.n) & 0x00FFFFFF) == 0;
  }

  template <size_t channels>
  LIMG_INLINE bool equals(const limg_ui8_4 &other) const
  {
    if constexpr (channels == 4)
      return equals_w_alpha(other);
    else
      return equals_wo_alpha(other);
  }

  LIMG_INLINE bool equals(const limg_ui8_4 &other, const bool hasAlpha) const
  {
    return u.v[0] == other[0] && u.v[1] == other[1] && u.v[2] == other[2] && (hasAlpha || u.v[3] == other[3]);
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

struct limg_encode_decomposition_state
{
#if LIMG_PRECISE_DECOMPOSITION == 1
  limg_ui8_4 low[4];
  limg_ui8_4 high[4];
  size_t maxDist;
#elif LIMG_PRECISE_DECOMPOSITION == 2
  size_t sum[4];
#endif
};

template <size_t channels>
struct limg_color_error_state
{
#if LIMG_PRECISE_DECOMPOSITION == 2
  float normal[channels];
  float inv_dot_normal;
#else
  float dist[channels];
  float inv_dist_or_one[channels];
  float inverse_dist_complete_or_one = 0;
#endif
};

//////////////////////////////////////////////////////////////////////////

template <typename T, size_t channels>
static LIMG_INLINE T limg_dot(const T a[channels], const T b[channels])
{
  T sum = (T)0;

  for (size_t i = 0; i < channels; i++)
    sum += a[i] * b[i];

  return sum;
}

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
static LIMG_INLINE void limg_init_color_error_state_accurate_(const limg_ui8_4 &a, const limg_ui8_4 &b, limg_color_error_state<channels> &state)
{
  for (size_t i = 0; i < channels; i++)
    state.normal[i] = (float)((int16_t)b[i] - (int16_t)a[i]);

  state.inv_dot_normal = 1.f / limg_dot<float, channels>(state.normal, state.normal);
}

template <size_t channels>
static LIMG_INLINE void limg_init_color_error_state_(const limg_ui8_4 &a, const limg_ui8_4 &b, limg_color_error_state<channels> &state)
{
  for (size_t i = 0; i < channels; i++)
  {
    const uint8_t diff = (b[i] - a[i]);
    state.dist[i] = (float)diff;
    state.inv_dist_or_one[i] = 1.f / (limgMax(1, state.dist[i]));

    if (diff != 0)
      state.inverse_dist_complete_or_one += (float)state.dist[i];
  }

  state.inverse_dist_complete_or_one = 1.f / limgMax(1.f, state.inverse_dist_complete_or_one);
}

template <size_t channels>
LIMG_INLINE static void limg_init_color_error_state(const limg_ui8_4 &a, const limg_ui8_4 &b, limg_color_error_state<channels> &state)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
    limg_init_color_error_state_accurate_<channels>(a, b, state);
  else
    limg_init_color_error_state_<channels>(a, b, state);
}

template <size_t channels>
LIMG_INLINE static size_t limg_color_error_state_get_error_(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  float offset[channels];

  for (size_t i = 0; i < channels; i++)
    offset[i] = (float)(color[i] - a[i]);

  float avg = 0;

  for (size_t i = 0; i < channels; i++)
    avg += offset[i];

  avg *= state.inverse_dist_complete_or_one;

  factor = avg;

  size_t error = 0;
  size_t lum = 0;

  if constexpr (limg_ColorDependentBlockError)
  {
    if (color[0] < 0x80)
    {
      const uint8_t factors[4] = { 2, 4, 3, 3 };

      for (size_t i = 0; i < channels; i++)
      {
        const size_t e = (size_t)(0.5f + fabsf((offset[i] * state.inv_dist_or_one[i] - avg) * state.dist[i]));
        error += e * e * factors[i];

        if constexpr (limg_LuminanceDependentPixelError)
          lum += color[i];
      }
    }
    else
    {
      const uint8_t factors[4] = { 3, 4, 2, 3 };

      for (size_t i = 0; i < channels; i++)
      {
        const size_t e = (size_t)(0.5f + fabsf((offset[i] * state.inv_dist_or_one[i] - avg) * state.dist[i]));
        error += e * e * factors[i];

        if constexpr (limg_LuminanceDependentPixelError)
          lum += color[i];
      }
    }
  }
  else
  {
    for (size_t i = 0; i < channels; i++)
    {
      const size_t e = (size_t)(0.5f + fabsf((offset[i] * state.inv_dist_or_one[i] - avg) * state.dist[i]));
      error += e * e;

      if constexpr (limg_LuminanceDependentPixelError)
        lum += color[i];
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

  return error;
}

template <size_t channels>
LIMG_INLINE static void limg_color_error_state_get_error_(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  float offset[channels];

  for (size_t i = 0; i < channels; i++)
    offset[i] = (float)(color[i] - a[i]);

  float avg = 0;

  for (size_t i = 0; i < channels; i++)
    avg += offset[i];

  factor = avg * state.inverse_dist_complete_or_one;
}

template <size_t channels>
LIMG_INLINE static size_t limg_color_error_from_error_vec_(const limg_ui8_4 &color, const float error_vec[channels])
{
  size_t lum = 0;
  float error = 0;

  if constexpr (limg_ColorDependentBlockError)
  {
    if (color[0] < 0x80)
    {
      const float factors[4] = { 2, 4, 3, 3 };

      for (size_t i = 0; i < channels; i++)
      {
        error += error_vec[i] * error_vec[i] * factors[i];

        if constexpr (limg_LuminanceDependentPixelError)
          lum += color[i];
      }
    }
    else
    {
      const float factors[4] = { 3, 4, 2, 3 };

      for (size_t i = 0; i < channels; i++)
      {
        error += error_vec[i] * error_vec[i] * factors[i];

        if constexpr (limg_LuminanceDependentPixelError)
          lum += color[i];
      }
    }
  }
  else
  {
    for (size_t i = 0; i < channels; i++)
    {
      error += error_vec[i] * error_vec[i];

      if constexpr (limg_LuminanceDependentPixelError)
        lum += color[i];
    }
  }

  if constexpr (limg_LuminanceDependentPixelError)
  {
    size_t ilum;
    ilum = 0xFF * 12 - lum * (12 / channels);
    ilum *= ilum;
    lum = (ilum >> 20) + 8;

    return (size_t)(((float)lum * error) + 0.5f);
  }
  else
  {
    return (size_t)error;
  }
}

template <size_t channels>
LIMG_INLINE static size_t limg_color_error_state_get_error_accurate_(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  float lineOriginToPx[channels];

  for (size_t i = 0; i < channels; i++)
    lineOriginToPx[i] = (float)color[i] - a[i];

  const float f = limg_dot<float, channels>(lineOriginToPx, state.normal) * state.inv_dot_normal;
  factor = f;

  float on_line[channels];

  for (size_t i = 0; i < channels; i++)
    on_line[i] = a[i] + f * state.normal[i];

  float error_vec[channels];

  for (size_t i = 0; i < channels; i++)
    error_vec[i] = (float)color[i] - on_line[i];

  return limg_color_error_from_error_vec_<channels>(color, error_vec);
}

template <size_t channels>
LIMG_INLINE static void limg_color_error_state_get_factor_accurate_(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  float lineOriginToPx[channels];

  for (size_t i = 0; i < channels; i++)
    lineOriginToPx[i] = (float)color[i] - a[i];

  factor = limg_dot<float, channels>(lineOriginToPx, state.normal) * state.inv_dot_normal;
}

template <size_t channels>
LIMG_INLINE static size_t limg_color_error_state_get_error(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
    return limg_color_error_state_get_error_accurate_<channels>(color, a, state, factor);
  else
    return limg_color_error_state_get_error_<channels>(color, a, state, factor);
}

template <size_t channels>
LIMG_INLINE static void limg_color_error_state_get_factor(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
    limg_color_error_state_get_factor_accurate_<channels>(color, a, state, factor);
  else
    limg_color_error_state_get_factor_<channels>(color, a, state, factor);
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

template <size_t channels, bool CheckPixelAndBlockError>
static LIMG_DEBUG_NO_INLINE bool limg_encode_get_block_factors_accurate_from_state_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  float avg[channels];

  const float inv_count = 1.f / (float)(rangeX * rangeY);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      float corrected[channels];
      float max_abs = 0;

      for (size_t i = 0; i < channels; i++)
      {
        corrected[i] = (float)px[i] - avg[i];

        if (fabsf(corrected[i]) > max_abs)
          max_abs = corrected[i];
      }

      if (max_abs != 0)
      {
        float vec[channels];
        float lengthSquared = 0;

        for (size_t i = 0; i < channels; i++)
        {
          vec[i] = corrected[i] / max_abs;
          lengthSquared += vec[i] * vec[i];
        }

        //const float inv_length = 1.f / sqrtf(lengthSquared);
        const float inv_length = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared)));

        for (size_t i = 0; i < channels; i++)
          diff_xi[i] += vec[i] * inv_length;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
    diff_xi[i] *= inv_count;

  // Find Edge Points that result in mimimal error and contain the values.
  float_t min;
  float_t max;

  bool all_zero = false;

  for (size_t i = 0; i < channels; i++)
  {
    if (diff_xi[i] != 0)
    {
      all_zero = false;
      break;
    }
  }

  size_t blockError = 0;

  if (!all_zero)
  {
    min = FLT_MAX;
    max = FLT_MIN;
    const float inv_dot_diff_xi = 1.f / limg_dot<float, channels>(diff_xi, diff_xi);

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = (float)px[i] - avg[i];

        const float f = limg_dot<float, channels>(lineOriginToPx, diff_xi) * inv_dot_diff_xi;

        if constexpr (CheckPixelAndBlockError)
        {
          float error_vec[channels];

          for (size_t i = 0; i < channels; i++)
            error_vec[i] = (float)px[i] - (avg[i] + f * diff_xi[i]);

          const size_t pixelError = limg_color_error_from_error_vec_<channels>(px, error_vec);

          if (pixelError > pCtx->maxPixelBlockError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasPixelBlockError++;
            }

            return false;
          }

          blockError += pixelError;
        }

        min = limgMin(min, f);
        max = limgMax(max, f);
      }
    }
  }
  else
  {
    min = 0;
    max = 0;
  }

  for (size_t i = 0; i < channels; i++)
  {
    a[i] = (uint8_t)limgClamp((int32_t)(avg[i] + min * diff_xi[i] + 0.5f), 0, 0xFF);
    b[i] = (uint8_t)limgClamp((int32_t)(avg[i] + max * diff_xi[i] + 0.5f), 0, 0xFF);
  }

  if constexpr (CheckPixelAndBlockError)
  {
    const size_t rangeSize = rangeX * rangeY;
    const bool ret = (((blockError * 0x10) / rangeSize) < pCtx->maxBlockPixelError);

    if constexpr (limg_DiagnoseCulprits)
    {
      if (!ret)
      {
        pCtx->culprits++;
        pCtx->culpritWasBlockPixelError++;
      }
    }

    return ret;
  }
  else
  {
    return true;
  }
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError, bool ReadWriteRangeSize, size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_check_area_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError, const size_t startBlockError, size_t *pAdditionalRangeForInitialBlockError)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  size_t blockError = startBlockError;

  limg_color_error_state<channels> color_error_state;
  limg_init_color_error_state<channels>(a, b, color_error_state);

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      if constexpr (CheckBounds)
      {
        for (size_t i = 0; i < channels; i++)
        {
          if ((int64_t)px[i] < a[i] - (int64_t)pCtx->maxPixelChannelBlockError || (int64_t)px[i] > b[i] + (int64_t)pCtx->maxPixelChannelBlockError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasPixelChannelBlockError++;
            }

            return false;
          }
        }
      }

      float factor;
      size_t error;

      if constexpr (!CheckPixelError && !WriteBlockError)
        error = limg_color_error_state_get_factor<channels>(px, a, color_error_state, factor);
      else
        error = limg_color_error_state_get_error<channels>(px, a, color_error_state, factor);

      if constexpr (CheckPixelError)
      {
        if (error > pCtx->maxPixelBlockError)
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasPixelBlockError++;
          }

          return false;
        }
      }

      blockError += error;

      if constexpr (WriteToFactors)
      {
        *pFactors = factor;
        pFactors++;
      }
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

  const bool ret = (((blockError * 0x10) / rangeSize) < pCtx->maxBlockPixelError);

  if constexpr (limg_DiagnoseCulprits)
  {
    if (!ret)
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockPixelError++;
    }
  }

  return ret;
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
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_min_max_per_channel_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
      {
        if (px[i] < state.low[i][i])
          state.low[i] = px;
        else if (px[i] > state.high[i][i])
          state.high[i] = px;
      }
    }
  }

  limg_ui8_4 max_l = state.low[0];
  limg_ui8_4 max_h = state.high[0];

  size_t maxDist = state.maxDist;

  if constexpr (limg_ColorDependentABError)
    maxDist = limg_color_error<channels>(max_l, max_h);
  else
    maxDist = limg_vector_error<channels>(max_l, max_h);

  size_t h_index = 1;

  for (size_t l_index = 0; l_index < channels; l_index++)
  {
    for (; h_index < channels; h_index++)
    {
      const limg_ui8_4 l = state.low[l_index];
      const limg_ui8_4 h = state.high[h_index];

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

  if (!out_a.equals<channels>(max_l) || !out_b.equals<channels>(max_h))
  {
    limg_color_error_state<channels> color_error_state;
    limg_init_color_error_state<channels>(max_l, max_h, color_error_state);

    if (!out_a.equals<channels>(max_l))
    {
      float factor;
      const size_t error = limg_color_error_state_get_error<channels>(out_a, max_l, color_error_state, factor);

      if (error > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }

    if (!out_b.equals<channels>(max_h))
    {
      float factor;
      const size_t error = limg_color_error_state_get_error<channels>(out_b, max_l, color_error_state, factor);

      if (error > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }
  }

  out_a = max_l;
  out_b = max_h;

  state.maxDist = maxDist;

  return true;
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_min_max_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
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
        limg_color_error_state<channels> color_error_state;
        limg_init_color_error_state(px, b, color_error_state);

        float factor;
        const size_t error = limg_color_error_state_get_error<channels>(a, px, color_error_state, factor);

        if (error > pCtx->maxBlockExpandError)
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasBlockExpandError++;
          }

          return false;
        }

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
          limg_color_error_state<channels> color_error_state;
          limg_init_color_error_state(a, px, color_error_state);

          float factor;
          const size_t error = limg_color_error_state_get_error<channels>(b, a, color_error_state, factor);

          if (error > pCtx->maxBlockExpandError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasBlockExpandError++;
            }

            return false;
          }

          b = px;
        }
      }
    }
  }
  
  out_a = a;
  out_b = b;

  return true;
}

template <size_t channels>
static LIMG_INLINE void limg_encode_attempt_include_pixels_to_sum_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  // Calculate Sum (for average).
  size_t sum[channels];

  for (size_t i = 0; i < channels; i++)
    sum[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
        sum[i] += px[i];
    }
  }

  // Store stuff in the decomposition state.
  for (size_t i = 0; i < channels; i++)
    state.sum[i] += sum[i];
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_accurate_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  const limg_ui8_4 old_a = out_a;
  const limg_ui8_4 old_b = out_b;

  limg_encode_attempt_include_pixels_to_sum_<channels>(pCtx, offsetX, offsetY, rangeX, rangeY, state);
  
  if (!limg_encode_get_block_factors_accurate_from_state_<channels, true>(pCtx, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY, out_a, out_b, state))
    return false;

  if (!old_a.equals_w_alpha(out_a) || !old_b.equals_w_alpha(out_b))
  {
    limg_color_error_state<channels> color_error_state;
    limg_init_color_error_state<channels>(out_a, out_b, color_error_state);

    if (!old_a.equals<channels>(out_a))
    {
      float _unused;

      if (limg_color_error_state_get_error<channels>(old_a, out_a, color_error_state, _unused) > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }

    if (!old_b.equals<channels>(out_b))
    {
      float _unused;

      if (limg_color_error_state_get_error<channels>(old_b, out_a, color_error_state, _unused) > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }
  }

  return true;
}

static LIMG_INLINE bool limg_encode_attempt_include_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_accurate_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
    else
      return limg_encode_attempt_include_pixels_accurate_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
  }
  else if constexpr (limg_RetrievePreciseDecomposition == 1)
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_min_max_per_channel_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state);
    else
      return limg_encode_attempt_include_pixels_min_max_per_channel_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state);
  }
  else
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_min_max_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
    else
      return limg_encode_attempt_include_pixels_min_max_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
  }
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

static LIMG_INLINE bool limg_encode_attempt_include_unused_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  return limg_encode_check_pixel_unused(pCtx, offsetX, offsetY, rangeX, rangeY) && limg_encode_attempt_include_pixels(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
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
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_min_max_per_channel_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
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

  for (size_t i = 0; i < channels; i++)
  {
    state.low[i] = low[i];
    state.high[i] = high[i];
  }

  state.maxDist = maxDist;
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_get_block_factors_accurate_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  // Calculate Sum (for average).
  size_t sum[channels];

  for (size_t i = 0; i < channels; i++)
    sum[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
        sum[i] += px[i];
    }
  }

  // Store stuff in the decomposition state.
  for (size_t i = 0; i < channels; i++)
    state.sum[i] = sum[i];

  return limg_encode_get_block_factors_accurate_from_state_<channels, true>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
}

static LIMG_INLINE bool limg_encode_get_block_a_b(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (pCtx->hasAlpha)
      return limg_encode_get_block_factors_accurate_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
    else
      return limg_encode_get_block_factors_accurate_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
  }
  else if constexpr (limg_RetrievePreciseDecomposition == 1)
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_per_channel_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
    else
      limg_encode_get_block_min_max_per_channel_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);

    return true;
  }
  else
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
    else
      limg_encode_get_block_min_max_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);

    return true;
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
  limg_encode_decomposition_state decomp_state;

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (!limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b, decomp_state))
      return false;
  }
  else
  {
    limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b, decomp_state);
  }

  size_t blockError = 0;
  size_t rangeSize = 0;

  if constexpr (limg_RetrievePreciseDecomposition != 2)
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
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, ox + rx, oy, newRx - rx, ry, newA, newB, new_decomp_state, ox, oy, newRx, ry);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
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
        decomp_state = new_decomp_state;
      }
    }

    if (canGrowDown)
    {
      const int64_t newRy = limgMin(ry + limg_BlockExpandStep, pCtx->sizeY - oy);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, ox, oy + ry, rx, newRy - ry, newA, newB, new_decomp_state, ox, oy, rx, newRy);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
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
        decomp_state = new_decomp_state;
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
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newOx == ox || !limg_encode_attempt_include_unused_pixels(pCtx, newOx, oy, ox - newOx, ry, newA, newB, new_decomp_state, newOx, oy, newRx, ry);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
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
        decomp_state = new_decomp_state;
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
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newOy == oy || !limg_encode_attempt_include_unused_pixels(pCtx, ox, newOy, rx, oy - newOy, newA, newB, new_decomp_state, ox, newOy, rx, newRy);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
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
        decomp_state = new_decomp_state;
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
      const limg_ui8_4 a = *pA;
      const limg_ui8_4 b = *pB;

      if (rx >= limg_MinBlockSize && ry >= limg_MinBlockSize && limg_encode_find_block_expand(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB, true, true, true, true))
      {
        staticX = ox;
        staticY = oy;

        return true;
      }

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;
      *pA = a;
      *pB = b;

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
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasPixelBitCrushError++;
        }

        return false;
      }

      blockError += error;
    }
  }

  const bool ret = ((blockError * 0x10) / (rangeX * rangeY) < pCtx->maxBlockBitCrushError);

  if constexpr (limg_DiagnoseCulprits)
  {
    if (!ret)
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockBitCrushError++;
    }
  }

  return ret;
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

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint32_t *pA, uint32_t *pB, uint32_t *pBlockIndex, uint8_t *pFactors, uint8_t *pBlockError, uint8_t *pShift, const bool hasAlpha, size_t *pTotalBlockArea, const uint32_t errorFactor)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  memset(&ctx, 0, sizeof(ctx));

  ctx.pSourceImage = pIn;
  ctx.pBlockInfo = pBlockIndex;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelBlockError = 0x12 * (errorFactor);
  ctx.maxBlockPixelError = 0x1C * (errorFactor / 3); // error is multiplied by 0x10.
  ctx.maxPixelChannelBlockError = 0x40 * (errorFactor / 2);
  ctx.maxBlockExpandError = 0x20 * (errorFactor);
  ctx.maxPixelBitCrushError = 0x5 * (errorFactor / 2);
  ctx.maxBlockBitCrushError = 0x2 * (errorFactor / 2); // error is multiplied by 0x10.
  ctx.ditheringEnabled = true;

  if constexpr (limg_LuminanceDependentPixelError)
  {
    ctx.maxPixelBlockError *= 0x10;
    ctx.maxBlockPixelError *= 0x10;
    ctx.maxPixelBitCrushError *= 0x10;
    ctx.maxBlockBitCrushError *= 0x10;
  }

  if constexpr (limg_ColorDependentBlockError)
  {
    //ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    //ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 6 : 4);
    ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 6 : 4);
    ctx.maxPixelBitCrushError *= (size_t)(ctx.hasAlpha ? 10 : 7);
    ctx.maxBlockBitCrushError *= (size_t)(ctx.hasAlpha ? 10 : 7);
  }

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    ctx.maxPixelBlockError *= 0x1;
    ctx.maxBlockPixelError *= 0x1;
    ctx.maxPixelBitCrushError *= 0x1;
    ctx.maxBlockBitCrushError *= 0x1;
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
    // Could be: `limg_encode_check_area<true, false, false, false, false>(&ctx, ox, oy, rx, ry, a, b, nullptr, 0, nullptr);` if we weren't writing block errors to a buffer.

    accumBlockError += blockError * 0x10;
    blockError = (blockError * 0x10) / rangeSize;
    unweightedBlockError += blockError;

#ifdef _DEBUG
    // If we're super accurate this may accidentally occur, since we calculate the error using the floating point average position and direction rather than the uint8_t min and max. Technically shouldn't matter since it only occurs due to a lack of precision. (at least if that's the reason why it's occuring...)
    //if (blockError > ctx.maxBlockPixelError)
    //  __debugbreak();
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

  *pTotalBlockArea = accumBlockSize;

#ifdef PRINT_TEST_OUTPUT
  printf("\n%" PRIu32 " Blocks generated.\n%5.3f %% Coverage\nAverage Size: %5.3f Pixels [(%5.3f px)^2].\nMinimum Block Size: %" PRIu64 "\nBlock Size Grow Step: %" PRIu64 "\nAverage Block Error: %5.3f (Unweighted: %5.3f)\nAverage Block Bits: %5.3f (Unweighted: %5.3f)\n\n", blockIndex, (accumBlockSize / (double)(sizeX * sizeY)) * 100.0, accumBlockSize / (double)blockIndex, sqrt(accumBlockSize / (double)blockIndex), limg_MinBlockSize, limg_BlockExpandStep, accumBlockError / (double)accumBlockSize, unweightedBlockError / (double)blockIndex, accumBits / (double)accumBlockSize, unweightedBits / (double)blockIndex);

  if constexpr (limg_DiagnoseCulprits)
  {
    printf("CULPRIT info: (%" PRIu64 " culprits)\n", ctx.culprits);
    printf("PixelBlockErrorCulprit: % 8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBlockError, (ctx.culpritWasPixelBlockError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBlockError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("BlockPixelError       : % 8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockPixelError, (ctx.culpritWasBlockPixelError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockPixelError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("PixelChannelBlockError: % 8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelChannelBlockError, (ctx.culpritWasPixelChannelBlockError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelChannelBlockError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("BlockExpandError      : % 8" PRIu64 " (%7.3f%%)\n", ctx.culpritWasBlockExpandError, (ctx.culpritWasBlockExpandError / (double)ctx.culprits) * 100.0);
    printf("PixelBitCrushError    : % 8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBitCrushError, (ctx.culpritWasPixelBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    printf("BlockBitCrushError    : % 8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockBitCrushError, (ctx.culpritWasBlockBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    puts("");
  }

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

double limg_compare(const uint32_t *pImageA, const uint32_t *pImageB, const size_t sizeX, const size_t sizeY, const bool hasAlpha, double *pMeanSquaredError, double *pMaxPossibleSquaredError)
{
  size_t error = 0;
  size_t maxError;

  const limg_ui8_4 *pA = reinterpret_cast<const limg_ui8_4 *>(pImageA);
  const limg_ui8_4 *pB = reinterpret_cast<const limg_ui8_4 *>(pImageB);

  const limg_ui8_4 min = { 0, 0, 0, 0 };
  const limg_ui8_4 max = { 0xFF, 0xFF, 0xFF, 0xFF };

  if (hasAlpha)
  {
    maxError = limg_color_error<4>(min, max);

    for (size_t i = 0; i < sizeX * sizeY; i++)
      error += limg_color_error<4>(pA[i], pB[i]);
  }
  else
  {
    maxError = limg_color_error<3>(min, max);

    for (size_t i = 0; i < sizeX * sizeY; i++)
      error += limg_color_error<3>(pA[i], pB[i]);
  }

  const double mse = error / (double)(sizeX * sizeY);
  const double psnr = 10.0 * log10((double)maxError / mse);

  if (pMeanSquaredError != nullptr)
    *pMeanSquaredError = mse;

  if (pMaxPossibleSquaredError != nullptr)
    *pMaxPossibleSquaredError = (double)maxError;

  return psnr;
}
