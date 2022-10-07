#ifndef limg_h__
#define limg_h__

#include <stdint.h>

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

double limg_compare(const uint32_t *pImageA, const uint32_t *pImageB, const size_t sizeX, const size_t sizeY, const bool hasAlpha, double *pMeanSquaredError, double *pMaxPossibleSquaredError);

#endif // limg_h__
