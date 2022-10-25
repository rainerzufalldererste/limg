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
  bool hasAlpha, ditheringEnabled, fastBitCrush, guessCrush;

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

template <size_t channels>
struct limg_color_error_state_3d
{
  float normalA[channels];
  float normalB[channels];
  float normalC[channels];
  float inv_dot_normalA;
  float inv_dot_normalB;
  float inv_dot_normalC;
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

#endif // limg_internal_h__
