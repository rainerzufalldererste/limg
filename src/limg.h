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

#endif // limg_h__
