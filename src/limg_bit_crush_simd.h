#ifndef limg_bit_crush_simd_h__
#define limg_bit_crush_simd_h__

#include "limg_internal.h"
#include "limg_simd.h"

static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d_3_sse41_floatA(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_encode_3d_output<3> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  constexpr size_t channels = 3;

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

  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128i shift_mul_ = _mm_set_epi32(0, 1 << shift[2], 1 << shift[1], 1 << shift[0]);
  const __m128i decode_bias_ = _mm_add_epi32(shift_mul_, _mm_set_epi32(0, decode_bias[2], decode_bias[1], decode_bias[0]));
  const __m128 inv_hexFFs_ = _mm_set1_ps(1.f / (float)0xFF);
  const __m128 decode_bias_s = _mm_cvtepi32_ps(decode_bias_);
  const __m128 minA_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minA[2], minA[1], minA[0]));
  const __m128 minB_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minB[2], minB[1], minB[0]));
  const __m128 minC_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minC[2], minC[1], minC[0]));
  const __m128 normalA_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalA[2], normalA[1], normalA[0])));
  const __m128 normalB_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalB[2], normalB[1], normalB[0])));
  const __m128 normalC_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalC[2], normalC[1], normalC[0])));
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128i redThreshold_ = _mm_set1_epi32(0x4000); // = 0x80 * 0x80.
  const __m128i index13_ = _mm_set_epi32(-1, 0, -1, 0);
  const __m128i low_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 3, 4, 2); // `_mm_set_epi32(3, 3, 4, 2);` with four channels.
  const __m128i high_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 2, 4, 3); // `_mm_set_epi32(3, 2, 4, 3);` with four channels. 

  __m128i block_error_ = _mm_setzero_si128();

  const uint32_t *pStart = reinterpret_cast<const uint32_t *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint32_t px = *pLine;
      pLine++;

      const __m128i px_ = _mm_cvtepu8_epi32(_mm_set1_epi32(px));

      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC >> shift[2], fB >> shift[1], fA >> shift[0]);  // this could be `_mm_srlv_epi32` with AVX2.
      const __m128 decoded_ = _mm_mul_ps(_mm_cvtepi32_ps(encoded_), decode_bias_s);
      const __m128 decoded_0 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128 decoded_1 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128 decoded_2 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128 estCol_0 = _mm_add_ps(_mm_mul_ps(decoded_0, normalA_), minA_);
      const __m128 estCol_1 = _mm_add_ps(_mm_mul_ps(decoded_1, normalB_), minB_);
      const __m128 estCol_2 = _mm_add_ps(_mm_mul_ps(decoded_2, normalC_), minC_);

      // This appears to be the bottleneck, but delaying the conversion to break up the dependency chain a bit (and contining the comparison with floating point numbers) was slower.
      const __m128i dec_ = _mm_min_epi32(hexFF_, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_add_ps(estCol_0, estCol_1), estCol_2)))); // ensure in range [0, 0xFF].

      const __m128i diff_ = _mm_sub_epi32(px_, dec_);
      const __m128i diff_sq_ = _mm_mullo_epi32(diff_, diff_);

      const __m128i diff_sq_red_ = _mm_shuffle_epi32(diff_sq_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i cmp_red_thresh_ = _mm_or_si128(index13_, _mm_cmplt_epi32(diff_sq_red_, redThreshold_));
      const __m128i col_error_mul_ = _mm_or_si128(_mm_and_si128(cmp_red_thresh_, low_red_error_cmp_flag_to_mul_), _mm_andnot_si128(cmp_red_thresh_, high_red_error_cmp_flag_to_mul_));

      const __m128i error_ = _mm_mullo_epi32(diff_sq_, col_error_mul_);
      const __m128i error_13_24_ = _mm_add_epi32(error_, _mm_srli_si128(error_, sizeof(uint32_t) * 2));
      const __m128i error_1234_ = _mm_add_epi32(error_13_24_, _mm_srli_si128(error_, sizeof(uint32_t)));

      block_error_ = _mm_add_epi32(block_error_, error_1234_);

      if (_mm_extract_epi32(error_1234_, 0) > pCtx->maxPixelBitCrushError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasPixelBitCrushError++;
        }

        return false;
      }
    }
  }

  const size_t blockError = (size_t)_mm_extract_epi32(block_error_, 0);

  const bool ret = ((blockError * 0x10) < pCtx->maxBlockBitCrushError * (rangeX * rangeY));

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

static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d_3_sse41_floatB(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_encode_3d_output<3> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  constexpr size_t channels = 3;

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

  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128i shift_mul_ = _mm_set_epi32(0, 1 << shift[2], 1 << shift[1], 1 << shift[0]);
  const __m128i decode_bias_ = _mm_add_epi32(shift_mul_, _mm_set_epi32(0, decode_bias[2], decode_bias[1], decode_bias[0]));
  const __m128 inv_hexFFs_ = _mm_set1_ps(1.f / (float)0xFF);
  const __m128 decode_bias_s = _mm_cvtepi32_ps(decode_bias_);
  const __m128 minA_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minA[2], minA[1], minA[0]));
  const __m128 minB_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minB[2], minB[1], minB[0]));
  const __m128 minC_ = _mm_cvtepi32_ps(_mm_set_epi32(0, minC[2], minC[1], minC[0]));
  const __m128 normalA_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalA[2], normalA[1], normalA[0])));
  const __m128 normalB_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalB[2], normalB[1], normalB[0])));
  const __m128 normalC_ = _mm_mul_ps(inv_hexFFs_, _mm_cvtepi32_ps(_mm_set_epi32(0, normalC[2], normalC[1], normalC[0])));
  const __m128 hexFF_ = _mm_set1_ps((float)0xFF);
  const __m128 redThreshold_ = _mm_set1_ps((float)0x4000); // = 0x80 * 0x80.
  const __m128i index13_ = _mm_set_epi32(-1, 0, -1, 0);
  const __m128i low_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 3, 4, 2); // `_mm_set_epi32(3, 3, 4, 2);` with four channels.
  const __m128i high_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 2, 4, 3); // `_mm_set_epi32(3, 2, 4, 3);` with four channels. 

  __m128i block_error_ = _mm_setzero_si128();

  const uint32_t *pStart = reinterpret_cast<const uint32_t *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint32_t px = *pLine;
      pLine++;

      const __m128 px_ = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_set1_epi32(px)));

      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC >> shift[2], fB >> shift[1], fA >> shift[0]);  // this could be `_mm_srlv_epi32` with AVX2.
      const __m128 decoded_ = _mm_mul_ps(_mm_cvtepi32_ps(encoded_), decode_bias_s);
      const __m128 decoded_0 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128 decoded_1 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128 decoded_2 = _mm_shuffle_ps(decoded_, decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128 estCol_0 = _mm_add_ps(_mm_mul_ps(decoded_0, normalA_), minA_);
      const __m128 estCol_1 = _mm_add_ps(_mm_mul_ps(decoded_1, normalB_), minB_);
      const __m128 estCol_2 = _mm_add_ps(_mm_mul_ps(decoded_2, normalC_), minC_);

      const __m128 dec_ = _mm_min_ps(hexFF_, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_add_ps(estCol_0, estCol_1), estCol_2))); // ensure in range [0, 0xFF].

      const __m128 diff_ = _mm_sub_ps(px_, dec_);
      const __m128 diff_sq_ = _mm_mul_ps(diff_, diff_);
      const __m128i diff_sq_32 = _mm_cvtps_epi32(diff_sq_);

      const __m128 diff_sq_red_ = _mm_shuffle_ps(diff_sq_, diff_sq_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i cmp_red_thresh_ = _mm_or_si128(index13_, _mm_castps_si128(_mm_cmplt_ps(diff_sq_red_, redThreshold_)));
      const __m128i col_error_mul_ = _mm_or_si128(_mm_and_si128(cmp_red_thresh_, low_red_error_cmp_flag_to_mul_), _mm_andnot_si128(cmp_red_thresh_, high_red_error_cmp_flag_to_mul_));

      const __m128i error_ = _mm_mullo_epi32(diff_sq_32, col_error_mul_);
      const __m128i error_13_24_ = _mm_add_epi32(error_, _mm_srli_si128(error_, sizeof(uint32_t) * 2));
      const __m128i error_1234_ = _mm_add_epi32(error_13_24_, _mm_srli_si128(error_, sizeof(uint32_t)));

      block_error_ = _mm_add_epi32(block_error_, error_1234_);

      if (_mm_extract_epi32(error_1234_, 0) > pCtx->maxPixelBitCrushError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasPixelBitCrushError++;
        }

        return false;
      }
    }
  }

  const size_t blockError = (size_t)_mm_extract_epi32(block_error_, 0);

  const bool ret = ((blockError * 0x10) < pCtx->maxBlockBitCrushError * (rangeX * rangeY));

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

static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d_3_sse41(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_encode_3d_output<3> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  constexpr size_t channels = 3;

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

  const __m128i shift_mul_ = _mm_set_epi32(0, 1 << shift[2], 1 << shift[1], 1 << shift[0]);
  const __m128i decode_bias_ = _mm_add_epi32(shift_mul_, _mm_set_epi32(0, decode_bias[2], decode_bias[1], decode_bias[0]));
  const __m128i bias_ = _mm_set1_epi32(bias);
  const __m128i minA_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0, minA[2], minA[1], minA[0]), 8), bias_); // reducing the addition of `bias` and `minA` into one value.
  const __m128i minB_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0, minB[2], minB[1], minB[0]), 8), bias_); // reducing the addition of `bias` and `minB` into one value.
  const __m128i minC_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0, minC[2], minC[1], minC[0]), 8), bias_); // reducing the addition of `bias` and `minC` into one value.
  const __m128i normalA_ = _mm_set_epi32(0, normalA[2], normalA[1], normalA[0]);
  const __m128i normalB_ = _mm_set_epi32(0, normalB[2], normalB[1], normalB[0]);
  const __m128i normalC_ = _mm_set_epi32(0, normalC[2], normalC[1], normalC[0]);
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128i redThreshold_ = _mm_set1_epi32(0x4000); // = 0x80 * 0x80.
  const __m128i index13_ = _mm_set_epi32(-1, 0, -1, 0);
  const __m128i low_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 3, 4, 2); // `_mm_set_epi32(3, 3, 4, 2);` with four channels.
  const __m128i high_red_error_cmp_flag_to_mul_ = _mm_set_epi32(0, 2, 4, 3); // `_mm_set_epi32(3, 2, 4, 3);` with four channels. 

  __m128i block_error_ = _mm_setzero_si128();

  const uint32_t *pStart = reinterpret_cast<const uint32_t *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint32_t px = *pLine;
      pLine++;

      const __m128i px_ = _mm_cvtepu8_epi32(_mm_set1_epi32(px));

      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC >> shift[2], fB >> shift[1], fA >> shift[0]);  // this could be `_mm_srlv_epi32` with AVX2.
      const __m128i decoded_ = _mm_mullo_epi32(encoded_, decode_bias_);
      const __m128i decoded_0 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i decoded_1 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128i decoded_2 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128i estCol_0 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_0, normalA_), minA_), 8);
      const __m128i estCol_1 = _mm_add_epi32(estCol_0, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_1, normalB_), minB_), 8));
      const __m128i estCol_2 = _mm_add_epi32(estCol_1, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_2, normalC_), minC_), 8));
      const __m128i dec_ = _mm_min_epi32(hexFF_, _mm_max_epi32(_mm_setzero_si128(), estCol_2)); // ensure in range [0, 0xFF].

      const __m128i diff_ = _mm_sub_epi32(px_, dec_);
      const __m128i diff_sq_ = _mm_mullo_epi32(diff_, diff_);

      const __m128i diff_sq_red_ = _mm_shuffle_epi32(diff_sq_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i cmp_red_thresh_ = _mm_or_si128(index13_, _mm_cmplt_epi32(diff_sq_red_, redThreshold_));
      const __m128i col_error_mul_ = _mm_or_si128(_mm_and_si128(cmp_red_thresh_, low_red_error_cmp_flag_to_mul_), _mm_andnot_si128(cmp_red_thresh_, high_red_error_cmp_flag_to_mul_));

      const __m128i error_ = _mm_mullo_epi32(diff_sq_, col_error_mul_);
      const __m128i error_13_24_ = _mm_add_epi32(error_, _mm_srli_si128(error_, sizeof(uint32_t) * 2));
      const __m128i error_1234_ = _mm_add_epi32(error_13_24_, _mm_srli_si128(error_, sizeof(uint32_t)));

      block_error_ = _mm_add_epi32(block_error_, error_1234_);

      if (_mm_extract_epi32(error_1234_, 0) > pCtx->maxPixelBitCrushError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasPixelBitCrushError++;
        }

        return false;
      }
    }
  }

  const size_t blockError = (size_t)_mm_extract_epi32(block_error_, 0);

  const bool ret = ((blockError * 0x10) < pCtx->maxBlockBitCrushError * (rangeX * rangeY));

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

static LIMG_INLINE bool limg_encode_try_bit_crush_block_3d_4_sse41(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_encode_3d_output<4> &in, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const uint8_t shift[3], size_t *pBlockError)
{
  constexpr size_t channels = 4;

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

  const __m128i shift_mul_ = _mm_set_epi32(0, 1 << shift[2], 1 << shift[1], 1 << shift[0]);
  const __m128i decode_bias_ = _mm_add_epi32(shift_mul_, _mm_set_epi32(0, decode_bias[2], decode_bias[1], decode_bias[0]));
  const __m128i bias_ = _mm_set1_epi32(bias);
  const __m128i minA_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(minA[3], minA[2], minA[1], minA[0]), 8), bias_); // reducing the addition of `bias` and `minA` into one value.
  const __m128i minB_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(minB[3], minB[2], minB[1], minB[0]), 8), bias_); // reducing the addition of `bias` and `minB` into one value.
  const __m128i minC_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(minC[3], minC[2], minC[1], minC[0]), 8), bias_); // reducing the addition of `bias` and `minC` into one value.
  const __m128i normalA_ = _mm_set_epi32(normalA[3], normalA[2], normalA[1], normalA[0]);
  const __m128i normalB_ = _mm_set_epi32(normalB[3], normalB[2], normalB[1], normalB[0]);
  const __m128i normalC_ = _mm_set_epi32(normalC[3], normalC[2], normalC[1], normalC[0]);
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128i redThreshold_ = _mm_set1_epi32(0x4000); // = 0x80 * 0x80.
  const __m128i index13_ = _mm_set_epi32(-1, 0, -1, 0);
  const __m128i low_red_error_cmp_flag_to_mul_ = _mm_set_epi32(3, 3, 4, 2);
  const __m128i high_red_error_cmp_flag_to_mul_ = _mm_set_epi32(3, 2, 4, 3);

  __m128i block_error_ = _mm_setzero_si128();

  const uint32_t *pStart = reinterpret_cast<const uint32_t *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint32_t px = *pLine;
      pLine++;

      const __m128i px_ = _mm_cvtepu8_epi32(_mm_set1_epi32(px));

      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC >> shift[2], fB >> shift[1], fA >> shift[0]);  // this could be `_mm_srlv_epi32` with AVX2.
      const __m128i decoded_ = _mm_mullo_epi32(encoded_, decode_bias_);
      const __m128i decoded_0 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i decoded_1 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128i decoded_2 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128i estCol_0 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_0, normalA_), minA_), 8);
      const __m128i estCol_1 = _mm_add_epi32(estCol_0, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_1, normalB_), minB_), 8));
      const __m128i estCol_2 = _mm_add_epi32(estCol_1, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_2, normalC_), minC_), 8));
      const __m128i dec_ = _mm_min_epi32(hexFF_, _mm_max_epi32(_mm_setzero_si128(), estCol_2)); // ensure in range [0, 0xFF].

      const __m128i diff_ = _mm_sub_epi32(px_, dec_);
      const __m128i diff_sq_ = _mm_mullo_epi32(diff_, diff_);

      const __m128i diff_sq_red_ = _mm_shuffle_epi32(diff_sq_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i cmp_red_thresh_ = _mm_or_si128(index13_, _mm_cmplt_epi32(diff_sq_red_, redThreshold_));
      const __m128i col_error_mul_ = _mm_or_si128(_mm_and_si128(cmp_red_thresh_, low_red_error_cmp_flag_to_mul_), _mm_andnot_si128(cmp_red_thresh_, high_red_error_cmp_flag_to_mul_));

      const __m128i error_ = _mm_mullo_epi32(diff_sq_, col_error_mul_);
      const __m128i error_13_24_ = _mm_add_epi32(error_, _mm_srli_si128(error_, sizeof(uint32_t) * 2));
      const __m128i error_1234_ = _mm_add_epi32(error_13_24_, _mm_srli_si128(error_, sizeof(uint32_t)));

      block_error_ = _mm_add_epi32(block_error_, error_1234_);

      if (_mm_extract_epi32(error_1234_, 0) > pCtx->maxPixelBitCrushError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasPixelBitCrushError++;
        }

        return false;
      }
    }
  }

  const size_t blockError = (size_t)_mm_extract_epi32(block_error_, 0);

  const bool ret = ((blockError * 0x10) < pCtx->maxBlockBitCrushError * (rangeX * rangeY));

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

#endif // limg_bit_crush_simd_h__
