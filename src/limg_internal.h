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
  bool hasAlpha, ditheringEnabled, fastBitCrush, guessCrush, crushBits;

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
static LIMG_INLINE void limg_color_error_state_3d_get_factors(const limg_ui8_4 &color, const limg_encode_3d_output<channels> &in, const limg_color_error_state_3d<channels> &state, float &fac_a, float &fac_b, float &fac_c)
{
  float minAtoCol[channels];

  for (size_t i = 0; i < channels; i++)
    minAtoCol[i] = (float)(color[i] - in.dirA_min[i]);

  const float facA = fac_a = limg_dot<float, channels>(minAtoCol, state.normalA) * state.inv_length_normalA;

  float colEst[channels];
  float minBToColADiff[channels];

  for (size_t i = 0; i < channels; i++)
  {
    colEst[i] = ((float)in.dirA_min[i] + facA * state.normalA[i]);
    minBToColADiff[i] = ((float)color[i] - colEst[i]) - (float)in.dirB_offset[i];
  }

  const float facB = fac_b = limg_dot<float, channels>(minBToColADiff, state.normalB) * state.inv_length_normalB;

  float minCToColBDiff[channels];

  for (size_t i = 0; i < channels; i++)
  {
    colEst[i] = (colEst[i] + facB * state.normalB[i]);
    minCToColBDiff[i] = ((float)color[i] - colEst[i]) - (float)in.dirC_offset[i];
  }

  const float facC = fac_c = limg_dot<float, channels>(minCToColBDiff, state.normalC) * state.inv_length_normalC;

  (void)facC;
}

template <size_t channels>
LIMG_INLINE void limg_color_error_state_3d_get_all_factors_(const limg_encode_context *, const limg_encode_3d_output<channels> &decomposition, const limg_color_error_state_3d<channels> &color_error_state, const uint32_t *pPixels, const size_t size, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8)
{
  const limg_ui8_4 *pLine = reinterpret_cast<const limg_ui8_4 *>(pPixels);

  for (size_t i = 0; i < size; i++)
  {
    float a, b, c;

    limg_color_error_state_3d_get_factors<channels>(pLine[i], decomposition, color_error_state, a, b, c);

    *pAu8 = (uint8_t)limgClamp((int32_t)(a * (float_t)0xFF + 0.5f), 0, 0xFF);
    *pBu8 = (uint8_t)limgClamp((int32_t)(b * (float_t)0xFF + 0.5f), 0, 0xFF);
    *pCu8 = (uint8_t)limgClamp((int32_t)(c * (float_t)0xFF + 0.5f), 0, 0xFF);

    pAu8++;
    pBu8++;
    pCu8++;
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
LIMG_INLINE void limg_color_error_state_3d_get_all_factors_3_sse41(const limg_encode_context *, const limg_encode_3d_output<3> &decomposition, const limg_color_error_state_3d<3> &color_error_state, const uint32_t *pPixels, const size_t size, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128 dirA_min_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirA_min))));
  const __m128 dirB_offset_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirB_offset))));
  const __m128 dirC_offset_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirC_offset))));
  const __m128 normalA_ = _mm_loadu_ps(color_error_state.normalA);
  const __m128 normalB_ = _mm_loadu_ps(color_error_state.normalB);
  const __m128 normalC_ = _mm_loadu_ps(color_error_state.normalC);
  const __m128 inv_length_normalA_ = _mm_set1_ps(color_error_state.inv_length_normalA);
  const __m128 inv_length_normalB_ = _mm_set1_ps(color_error_state.inv_length_normalB);
  const __m128 inv_length_normalC_ = _mm_set1_ps(color_error_state.inv_length_normalC);
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128 hexFF_ps = _mm_set1_ps((float)0xFF);

  for (size_t i = 0; i < size; i++)
  {
    const __m128 color_ = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
    const __m128 minAtoCol = _mm_sub_ps(color_, dirA_min_);

    const __m128 dotA = _mm_dp_ps(minAtoCol, normalA_, 0x7F);
    const __m128 facA = _mm_mul_ps(_mm_shuffle_ps(dotA, dotA, _MM_SHUFFLE(0, 0, 0, 0)), inv_length_normalA_);

    // I've tried a bunch of variations, but this one appears to be the fastest way of doing this that I could come up with.
    *pAu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facA)))), 0);
    pAu8++;

    __m128 colEst = _mm_add_ps(dirA_min_, _mm_mul_ps(normalA_, facA));
    const __m128 minBtoCol = _mm_sub_ps(_mm_sub_ps(color_, colEst), dirB_offset_);

    const __m128 dotB = _mm_dp_ps(minBtoCol, normalB_, 0x7F);
    const __m128 facB = _mm_mul_ps(_mm_shuffle_ps(dotB, dotB, _MM_SHUFFLE(0, 0, 0, 0)), inv_length_normalB_);

    *pBu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facB)))), 0);
    pBu8++;

    colEst = _mm_add_ps(colEst, _mm_mul_ps(normalB_, facB));
    const __m128 minCtoCol = _mm_sub_ps(_mm_sub_ps(color_, colEst), dirC_offset_);

    const __m128 dotC = _mm_dp_ps(minCtoCol, normalC_, 0x7F);
    const __m128 facC = _mm_mul_ps(dotC, inv_length_normalC_);

    *pCu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facC)))), 0);
    pCu8++;
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
LIMG_INLINE void limg_color_error_state_3d_get_all_factors_4_sse41(const limg_encode_context *, const limg_encode_3d_output<4> &decomposition, const limg_color_error_state_3d<4> &color_error_state, const uint32_t *pPixels, const size_t size, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128 dirA_min_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirA_min))));
  const __m128 dirB_offset_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirB_offset))));
  const __m128 dirC_offset_ = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(decomposition.dirC_offset))));
  const __m128 normalA_ = _mm_loadu_ps(color_error_state.normalA);
  const __m128 normalB_ = _mm_loadu_ps(color_error_state.normalB);
  const __m128 normalC_ = _mm_loadu_ps(color_error_state.normalC);
  const __m128 inv_length_normalA_ = _mm_set1_ps(color_error_state.inv_length_normalA);
  const __m128 inv_length_normalB_ = _mm_set1_ps(color_error_state.inv_length_normalB);
  const __m128 inv_length_normalC_ = _mm_set1_ps(color_error_state.inv_length_normalC);
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128 hexFF_ps = _mm_set1_ps((float)0xFF);

  for (size_t i = 0; i < size; i++)
  {
    const __m128 color_ = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
    const __m128 minAtoCol = _mm_sub_ps(color_, dirA_min_);

    const __m128 dotA = _mm_dp_ps(minAtoCol, normalA_, 0xFF);
    const __m128 facA = _mm_mul_ps(_mm_shuffle_ps(dotA, dotA, _MM_SHUFFLE(0, 0, 0, 0)), inv_length_normalA_);

    *pAu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facA)))), 0);
    pAu8++;

    __m128 colEst = _mm_add_ps(dirA_min_, _mm_mul_ps(normalA_, facA));
    const __m128 minBtoCol = _mm_sub_ps(_mm_sub_ps(color_, colEst), dirB_offset_);

    const __m128 dotB = _mm_dp_ps(minBtoCol, normalB_, 0xFF);
    const __m128 facB = _mm_mul_ps(_mm_shuffle_ps(dotB, dotB, _MM_SHUFFLE(0, 0, 0, 0)), inv_length_normalB_);

    *pBu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facB)))), 0);
    pBu8++;

    colEst = _mm_add_ps(colEst, _mm_mul_ps(normalB_, facB));
    const __m128 minCtoCol = _mm_sub_ps(_mm_sub_ps(color_, colEst), dirC_offset_);

    const __m128 dotC = _mm_dp_ps(minCtoCol, normalC_, 0xFF);
    const __m128 facC = _mm_mul_ps(dotC, inv_length_normalC_);

    *pCu8 = (uint8_t)_mm_extract_epi8(_mm_max_epi32(_mm_setzero_si128(), _mm_min_epi32(hexFF_, _mm_cvtps_epi32(_mm_mul_ps(hexFF_ps, facC)))), 0);
    pCu8++;
  }
}

template <size_t channels>
LIMG_INLINE void limg_color_error_state_3d_get_all_factors(const limg_encode_context *pCtx, const limg_encode_3d_output<channels> &decomposition, const limg_color_error_state_3d<channels> &color_error_state, const uint32_t *pPixels, const size_t size, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8)
{
  if (sse41Supported)
  {
    if constexpr (channels == 3)
      limg_color_error_state_3d_get_all_factors_3_sse41(pCtx, decomposition, color_error_state, pPixels, size, pAu8, pBu8, pCu8);
    else
      limg_color_error_state_3d_get_all_factors_4_sse41(pCtx, decomposition, color_error_state, pPixels, size, pAu8, pBu8, pCu8);
  }
  else
  {
    limg_color_error_state_3d_get_all_factors_(pCtx, decomposition, color_error_state, pPixels, size, pAu8, pBu8, pCu8);
  }
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
