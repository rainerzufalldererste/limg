#ifndef limg_internal_h__
#define limg_internal_h__

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

#define LIMG_SUCCESS(errorCode) (errorCode == limg_success)
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

#ifdef _MSC_VER
#define LIMG_ALIGN(bytes) __declspec(align(bytes))
#else
#define LIMG_ALIGN(bytes) __attribute__((aligned(bytes)))
#endif

template <typename T, typename U>
constexpr inline auto limgMax(const T &a, const U &b) -> decltype(a > b ? a : b)
{
  return a > b ? a : b;
}

template <typename T, typename U>
constexpr inline auto limgMin(const T &a, const U &b) -> decltype(a < b ? a : b)
{
  return a < b ? a : b;
}

template <typename T>
constexpr inline T limgClamp(const T &a, const T &min, const T &max)
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
constexpr bool limg_DiagnoseCulprits = false;

struct limg_encode_context
{
  const uint32_t *pSourceImage;
  uint32_t *pBlockInfo;
  size_t sizeX, sizeY;
  size_t blockX, blockY;
  void *pBlockColorDecompositions;
  size_t maxPixelBlockError, // maximum error of a single pixel when trying to fit pixels into blocks.
    maxBlockPixelError, // maximum average error of pixels per block when trying to fit them into blocks. (accum_err * 0xFF / (rangeX * rangeY))
    maxPixelChannelBlockError, // maximum error of a single pixel on a single channel when trying to fit pixels into blocks.
    maxBlockExpandError, // maximum error of linear factor deviation when trying to expand the factors of a block in order to expand the block.
    maxPixelBitCrushError, // maximum error of a single pixel when trying to bit crush blocks.
    maxBlockBitCrushError; // maximum average error of pixels per block when trying to bit crush blocks. (accum_err * 0xFF / (rangeX * rangeY))
  bool hasAlpha, ditheringEnabled, fastBitCrush, guessCrush, crushBits, coarseFineBitCrush, errorPixelRetainingBitCrush;

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
  float inv_length_normal;
#else
  float dist[channels];
  float inv_dist_or_one[channels];
  float inverse_dist_complete_or_one = 0;
#endif
};

template <size_t channels>
struct limg_color_error_state_3d
{
  float normalA[channels];
  float normalB[channels];
  float normalC[channels];
  float inv_length_normalA;
  float inv_length_normalB;
  float inv_length_normalC;
};

template <size_t channels>
struct limg_encode_3d_output
{
  float avg[channels];
  int16_t dirA_min[channels];
  int16_t dirA_max[channels];
  int16_t dirB_offset[channels];
  int16_t dirB_mag[channels];
  int16_t dirC_offset[channels];
  int16_t dirC_mag[channels];
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

template <typename T>
static LIMG_INLINE void limg_cross(const T a[3], const T b[3], T out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
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
static LIMG_INLINE void limg_init_color_error_state_3d(const limg_encode_3d_output<channels> &in, limg_color_error_state_3d<channels> &state)
{
  bool nonzero[3] = { false, false, false };

  for (size_t i = 0; i < channels; i++)
  {
    state.normalA[i] = (float)((int16_t)in.dirA_max[i] - (int16_t)in.dirA_min[i]);
    state.normalB[i] = (float)((int16_t)in.dirB_mag[i] - (int16_t)in.dirB_offset[i]);
    state.normalC[i] = (float)((int16_t)in.dirC_mag[i] - (int16_t)in.dirC_offset[i]);

    nonzero[0] |= (state.normalA[i] != 0);
    nonzero[1] |= (state.normalB[i] != 0);
    nonzero[2] |= (state.normalC[i] != 0);
  }

  memset(&state.inv_length_normalA, 0, sizeof(float) * 3); // hah, always living on the edge.

  if (nonzero[0])
    state.inv_length_normalA = 1.f / limg_dot<float, channels>(state.normalA, state.normalA);
  
  if (nonzero[1])
    state.inv_length_normalB = 1.f / limg_dot<float, channels>(state.normalB, state.normalB);
  
  if (nonzero[2])
    state.inv_length_normalC = 1.f / limg_dot<float, channels>(state.normalC, state.normalC);
}

template <size_t channels>
static LIMG_INLINE void limg_init_color_error_state_accurate_(const limg_ui8_4 &a, const limg_ui8_4 &b, limg_color_error_state<channels> &state)
{
  for (size_t i = 0; i < channels; i++)
    state.normal[i] = (float)((int16_t)b[i] - (int16_t)a[i]);

  state.inv_length_normal = 1.f / limg_dot<float, channels>(state.normal, state.normal);
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
LIMG_INLINE static void limg_color_error_state_get_factor_(const limg_ui8_4 &color, const limg_ui8_4 &a, const limg_color_error_state<channels> &state, float &factor)
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

  const float f = limg_dot<float, channels>(lineOriginToPx, state.normal) * state.inv_length_normal;
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

  factor = limg_dot<float, channels>(lineOriginToPx, state.normal) * state.inv_length_normal;
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

LIMG_INLINE int16_t limg_fast_round_int16(const float in)
{
  return (int16_t)(in + 256.5f) - 256;
}

LIMG_INLINE int16_t limg_fast_ceil_int16(const float in)
{
  return (int16_t)(in + 256.99f) - 256;
}

LIMG_INLINE int16_t limg_fast_floor_int16(const float in)
{
  return (int16_t)(in + 256.f) - 256;
}

#endif // limg_internal_h__
