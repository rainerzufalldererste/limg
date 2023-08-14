#ifndef limg_h__
#define limg_h__

#include <stddef.h>
#include <stdint.h>

#include "limg_threading.h"

enum limg_result
{
  limg_success = 0,

  limg_error_Generic = 100,
  limg_error_InvalidParameter,
  limg_error_ArgumentNull,
  limg_error_OutOfBounds,
  limg_error_MemoryAllocationFailure,
};

struct limg_encode_info
{
  uint32_t *pDecoded, *pA, *pB, *pBlockIndex;
  uint8_t *pFactors, *pBlockError, *pShift;
  size_t totalBlockArea;
};

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, limg_encode_info *pInfo, const uint32_t errorFactor);

struct limg_encode3d_info
{
  uint32_t *pDecoded, *pShiftABCX, *pColAMin, *pColAMax, *pColBMin, *pColBMax, *pColCMin, *pColCMax;
  uint8_t *pFactorsA, *pFactorsB, *pFactorsC;
};

limg_result limg_encode3d_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, limg_encode3d_info *pInfo, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing);

limg_result limg_encode3d_test_perf(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing);

struct limg_blocked_encode3d_info
{
  uint32_t *pDecoded;
  uint8_t *pFactorsA, *pFactorsB, *pFactorsC, *pBlockError, *pBitsPerPixel;
  uint32_t *pShiftABCX, *pColAMin, *pColAMax, *pColBMin, *pColBMax, *pColCMin, *pColCMax, *pBlockIndex;
};

limg_result limg_blocked_encode3d_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, limg_blocked_encode3d_info *pInfo, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing);

double limg_compare(const uint32_t *pImageA, const uint32_t *pImageB, const size_t sizeX, const size_t sizeY, const bool hasAlpha, double *pMeanSquaredError, double *pMaxPossibleSquaredError);

#endif // limg_h__
