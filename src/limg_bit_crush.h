#ifndef limg_bit_crush_h__
#define limg_bit_crush_h__

#include "limg_bit_crush_simd.h"

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

template <size_t channels>
static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d_(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, const limg_encode_3d_output<channels> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  int32_t normalA[channels];
  int32_t normalB[channels];
  int32_t normalC[channels];

  int32_t minA[channels];
  int32_t minB[channels];
  int32_t minC[channels];

  for (size_t i = 0; i < channels; i++)
  {
    normalA[i] = in.dirA_max[i] - in.dirA_min[i];
    normalB[i] = in.dirB_mag[i] - in.dirB_offset[i];
    normalC[i] = in.dirC_mag[i] - in.dirC_offset[i];

    minA[i] = in.dirA_min[i];
    minB[i] = in.dirB_offset[i];
    minC[i] = in.dirC_offset[i];
  }

  if (shift[0] > 7)
    for (size_t i = 0; i < 3; i++)
      normalA[i] = 0;

  if (shift[1] > 7)
  {
    for (size_t i = 0; i < 3; i++)
      normalB[i] = 0;

    for (size_t i = 0; i < 3; i++)
      minB[i] = 0;
  }

  if (shift[2] > 7)
  {
    for (size_t i = 0; i < 3; i++)
      normalC[i] = 0;

    for (size_t i = 0; i < 3; i++)
      minC[i] = 0;
  }

  uint8_t decode_bias[3] = { 0, 0, 0 };

  for (size_t i = 0; i < 3; i++)
    for (uint8_t j = (1 << (shift[i] - 1)) >> (7 - shift[i]); j; j >>= (8 - shift[i]))
      decode_bias[i] |= j;

  constexpr uint32_t bias = 1 << 7;
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pPixels);
  size_t blockError = 0;

  for (size_t i = 0; i < size; i++)
  {
    const limg_ui8_4 px = *pStart;
    pStart++;

    const uint8_t fA = *pA;
    const uint8_t fB = *pB;
    const uint8_t fC = *pC;

    pA++;
    pB++;
    pC++;

    limg_ui8_4 dec;

    const uint8_t encoded[3] = { (uint8_t)(fA >> shift[0]), (uint8_t)(fB >> shift[1]), (uint8_t)(fC >> shift[2]) };
    uint8_t decoded[3];

    for (size_t j = 0; j < 3; j++)
      decoded[j] = (encoded[j] << shift[j]) + (encoded[j] * decode_bias[j]);

    for (size_t j = 0; j < channels; j++)
    {
      int32_t estCol;
      estCol = (int32_t)(minA[j] + ((decoded[0] * normalA[j] + (int32_t)bias) >> 8));
      estCol += (int32_t)(minB[j] + ((decoded[1] * normalB[j] + (int32_t)bias) >> 8));
      estCol += (int32_t)(minC[j] + ((decoded[2] * normalC[j] + (int32_t)bias) >> 8));

      dec[j] = (uint8_t)limgClamp(estCol, 0, 0xFF);
    }

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

  const bool ret = ((blockError * 0x10) < pCtx->maxBlockBitCrushError * size);

  if constexpr (limg_DiagnoseCulprits)
  {
    if (!ret)
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockBitCrushError++;
    }
  }

  *pBlockError = blockError;

  return ret;
}

template <size_t channels>
static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, const limg_encode_3d_output<channels> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  if (sse41Supported)
  {
    if constexpr (channels == 4)
      return limg_encode_try_bit_crush_block_3d_4_sse41(pCtx, pPixels, size, in, pA, pB, pC, shift, pBlockError);
    else
      return limg_encode_try_bit_crush_block_3d_3_sse41(pCtx, pPixels, size, in, pA, pB, pC, shift, pBlockError);
  }
  else
  {
    return limg_encode_try_bit_crush_block_3d_<channels>(pCtx, pPixels, size, in, pA, pB, pC, shift, pBlockError);
  }
}

template <size_t channels>
static LIMG_INLINE void limg_encode_guess_shift_for_block_3d(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, const limg_encode_3d_output<channels> &decomposition, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8, uint8_t shift[3], size_t *pMinBlockError)
{
  size_t blockError;
  //size_t max_shift = 0;
  size_t min_block_error = (size_t)-1;
  uint8_t shift_try[3] = { 4, 5, 6 };

  if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
  {
    for (size_t i = 0; i < 3; i++)
      shift[i] = shift_try[i];

    //max_shift = (size_t)shift_try[0] + (size_t)shift_try[1] + (size_t)shift_try[2];
    min_block_error = blockError;

    shift_try[0] = 5;
    shift_try[1] = 8;
    shift_try[2] = 8;

    if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
    {
      for (size_t i = 0; i < 3; i++)
        shift[i] = shift_try[i];

      //max_shift = (size_t)shift_try[0] + (size_t)shift_try[1] + (size_t)shift_try[2];
      min_block_error = blockError;
    }
    else
    {
      shift_try[0] = 4;
      shift_try[1] = 6;
      shift_try[2] = 8;

      if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
      {
        for (size_t i = 0; i < 3; i++)
          shift[i] = shift_try[i];

        //max_shift = (size_t)shift_try[0] + (size_t)shift_try[1] + (size_t)shift_try[2];
        min_block_error = blockError;
      }
    }
  }
  else
  {
    shift_try[0] = 2;
    shift_try[1] = 4;
    shift_try[2] = 5;

    if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
    {
      for (size_t i = 0; i < 3; i++)
        shift[i] = shift_try[i];

      //max_shift = (size_t)shift_try[0] + (size_t)shift_try[1] + (size_t)shift_try[2];
      min_block_error = blockError;
    }
  }

  *pMinBlockError = min_block_error;
}

template <size_t channels>
static LIMG_INLINE void limg_encode_find_shift_for_block_3d(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, const limg_encode_3d_output<channels> &decomposition, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8, uint8_t shift[3], const size_t minBlockError)
{
  size_t max_shift = shift[0] + shift[1] + shift[2];
  size_t min_block_error = minBlockError;
  uint8_t shift_try[3];
  size_t blockError;

  // Only replace with *more* max shift.
  {
    uint8_t a = 0;
    uint8_t b = 0;
    uint8_t c = 1; // to get rid of the check for 0, 0, 0.

    for (; a <= 8; a++)
    {
      shift_try[0] = a;

      for (; b <= 8; b++)
      {
        shift_try[1] = b;

        for (; c <= 8; c++)
        {
          if ((size_t)a + (size_t)b + (size_t)c > max_shift && (a != shift[0] || b != shift[1] || c != shift[2]))
          {
            shift_try[2] = c;

            if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
            {
              for (size_t i = 0; i < 3; i++)
                shift[i] = shift_try[i];

              max_shift = (size_t)a + (size_t)b + (size_t)c;
              min_block_error = blockError;
            }
            else
            {
              break;
            }
          }
        }

        if (c == 0)
          break;

        c = 0;
      }

      if (b == 0)
        break;

      b = 0;
    }
  }

  // (Potentially) check other max shifts.
  if (max_shift > 0 && !pCtx->fastBitCrush)
  {
    uint8_t a = shift[0];
    uint8_t b = shift[1];
    uint8_t c = shift[2] + 1;

    for (; a <= 8; a++)
    {
      shift_try[0] = a;

      for (; b <= 8; b++)
      {
        shift_try[1] = b;

        for (; c <= 8; c++)
        {
          if ((size_t)a + (size_t)b + (size_t)c == max_shift)
          {
            shift_try[2] = c;

            if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
            {
              if (min_block_error > blockError)
              {
                for (size_t i = 0; i < 3; i++)
                  shift[i] = shift_try[i];

                min_block_error = blockError;
              }
            }
            else
            {
              break;
            }
          }
        }

        if (c == 0)
          break;

        c = 0;
      }

      if (b == 0)
        break;

      b = 0;
    }
  }
}

template <size_t channels>
static LIMG_INLINE void limg_encode_find_shift_for_block_stepwise_3d(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, const limg_encode_3d_output<channels> &decomposition, uint8_t *pAu8, uint8_t *pBu8, uint8_t *pCu8, uint8_t shift[3], const size_t minBlockError)
{
  uint8_t max_shift = shift[0] + shift[1] + shift[2];
  size_t min_block_error = minBlockError;
  uint8_t shift_try[3];
  size_t blockError;

  // Coarse Pass. Only replace with *more* max shift.
  {
    uint8_t a = shift[0] & 0b1111;
    uint8_t b = shift[1] & 0b1111;
    uint8_t c = (shift[2] & 0b1111) + 2; // to get rid of the potential check for 0, 0, 0.

    for (; a <= 8; a += 2)
    {
      shift_try[0] = a;

      for (; b <= 8; b += 2)
      {
        shift_try[1] = b;

        for (; c <= 8; c += 2)
        {
          if (a + b + c > max_shift)
          {
            shift_try[2] = c;

            if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
            {
              for (size_t i = 0; i < 3; i++)
                shift[i] = shift_try[i];

              max_shift = (size_t)a + (size_t)b + (size_t)c;
              min_block_error = blockError;
            }
            else
            {
              break;
            }
          }
        }

        if (c == b)
          break;

        c = b;
      }

      if (b == a)
        break;

      b = a;
    }
  }

  // Fine Pass. Still Only replace with *more* max shift.
  {
    const uint8_t pre_a = (uint8_t)shift[0];
    const uint8_t pre_b = (uint8_t)shift[1];
    const uint8_t pre_c = (uint8_t)shift[2];
    const size_t max_a = !(pre_a & 1) && pre_a != 8;
    const size_t max_b = !(pre_b & 1) && pre_b != 8;
    const size_t max_c = !(pre_c & 1) && pre_c != 8;
    uint8_t fine_shift = 0;

    uint8_t a = 0;
    uint8_t b = 0;
    uint8_t c = 1; // to get rid of the check for 0, 0, 0.

    for (; a <= max_a; a++)
    {
      shift_try[0] = pre_a + a;

      for (; b <= max_b; b++)
      {
        shift_try[1] = pre_b + b;

        for (; c <= max_c; c++)
        {
          if (a + b + c > fine_shift)
          {
            shift_try[2] = pre_c + c;

            if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
            {
              max_shift = 0;

              for (size_t i = 0; i < 3; i++)
                max_shift += (shift[i] = shift_try[i]);

              fine_shift = a + b + c;
              min_block_error = blockError;
            }
            else
            {
              break;
            }
          }
        }

        if (c == 0)
          break;

        c = 0;
      }

      if (b == 0)
        break;

      b = 0;
    }
  }

  // (Potentially) check other max shifts.
  if (max_shift > 0 && !pCtx->fastBitCrush)
  {
    uint8_t a = shift[0];
    uint8_t b = shift[1];
    uint8_t c = shift[2] + 1;

    for (; a <= 8; a++)
    {
      shift_try[0] = a;

      for (; b <= 8; b++)
      {
        shift_try[1] = b;

        for (; c <= 8; c++)
        {
          if (a + b + c == max_shift)
          {
            shift_try[2] = c;

            if (limg_encode_try_bit_crush_block_3d<channels>(pCtx, pPixels, size, decomposition, pAu8, pBu8, pCu8, shift_try, &blockError))
            {
              if (min_block_error > blockError)
              {
                for (size_t i = 0; i < 3; i++)
                  shift[i] = shift_try[i];

                min_block_error = blockError;
              }
            }
            else
            {
              break;
            }
          }
        }

        if (c == 0)
          break;

        c = 0;
      }

      if (b == 0)
        break;

      b = 0;
    }
  }
}

#endif // limg_bit_crush_h__
