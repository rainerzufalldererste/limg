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

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint32_t *pA, uint32_t *pB, uint32_t *pBlockIndex, uint8_t *pFactors, uint8_t *pBlockError, uint8_t *pShift, const bool hasAlpha, size_t *pTotalBlockArea, const uint32_t errorFactor);

limg_result limg_encode3d_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing);

limg_result limg_encode3d_test_perf(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing);

double limg_compare(const uint32_t *pImageA, const uint32_t *pImageB, const size_t sizeX, const size_t sizeY, const bool hasAlpha, double *pMeanSquaredError, double *pMaxPossibleSquaredError);

#endif // limg_h__
