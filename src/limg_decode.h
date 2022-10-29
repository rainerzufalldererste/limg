#ifndef limg_decode_h__
#define limg_decode_h__

#include "limg_internal.h"

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

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
static void limg_decode_block_from_factors_3d_3_sse41(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const limg_encode_3d_output<3> &in, const uint8_t shift[3])
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
  const __m128i minA_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0xFFFF, minA[2], minA[1], minA[0]), 8), bias_); // reducing the addition of `bias` and `minA` into one value, ensure that alpha will be clamped to 0xFF.
  const __m128i minB_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0xFFFF, minB[2], minB[1], minB[0]), 8), bias_); // reducing the addition of `bias` and `minB` into one value, ensure that alpha will be clamped to 0xFF.
  const __m128i minC_ = _mm_add_epi32(_mm_slli_epi32(_mm_set_epi32(0xFFFF, minC[2], minC[1], minC[0]), 8), bias_); // reducing the addition of `bias` and `minC` into one value, ensure that alpha will be clamped to 0xFF.
  const __m128i normalA_ = _mm_set_epi32(0, normalA[2], normalA[1], normalA[0]);
  const __m128i normalB_ = _mm_set_epi32(0, normalB[2], normalB[1], normalB[0]);
  const __m128i normalC_ = _mm_set_epi32(0, normalC[2], normalC[1], normalC[0]);
  const __m128i hexFF_ = _mm_set1_epi32(0xFF);
  const __m128i shuffle_pattern_ = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

  for (size_t y = 0; y < rangeY; y++)
  {
    uint32_t *pOutLine = reinterpret_cast<uint32_t *>(pOut + y * sizeX);

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC, fB, fA);
      const __m128i decoded_ = _mm_mullo_epi32(encoded_, decode_bias_);
      const __m128i decoded_0 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i decoded_1 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128i decoded_2 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128i estCol_0 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_0, normalA_), minA_), 8);
      const __m128i estCol_1 = _mm_add_epi32(estCol_0, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_1, normalB_), minB_), 8));
      const __m128i estCol_2 = _mm_add_epi32(estCol_1, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_2, normalC_), minC_), 8));
      const __m128i dec_ = _mm_min_epi32(hexFF_, _mm_max_epi32(_mm_setzero_si128(), estCol_2)); // ensure in range [0, 0xFF].

      const __m128i packed_ = _mm_shuffle_epi8(dec_, shuffle_pattern_);

      *pOutLine = _mm_extract_epi32(packed_, 0);
      pOutLine++;
    }
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
static void limg_decode_block_from_factors_3d_4_sse41(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const limg_encode_3d_output<4> &in, const uint8_t shift[3])
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
  const __m128i shuffle_pattern_ = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

  for (size_t y = 0; y < rangeY; y++)
  {
    uint32_t *pOutLine = reinterpret_cast<uint32_t *>(pOut + y * sizeX);

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      const __m128i encoded_ = _mm_set_epi32(0, fC, fB, fA);
      const __m128i decoded_ = _mm_mullo_epi32(encoded_, decode_bias_);
      const __m128i decoded_0 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(0, 0, 0, 0));
      const __m128i decoded_1 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(1, 1, 1, 1));
      const __m128i decoded_2 = _mm_shuffle_epi32(decoded_, _MM_SHUFFLE(2, 2, 2, 2));

      const __m128i estCol_0 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_0, normalA_), minA_), 8);
      const __m128i estCol_1 = _mm_add_epi32(estCol_0, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_1, normalB_), minB_), 8));
      const __m128i estCol_2 = _mm_add_epi32(estCol_1, _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(decoded_2, normalC_), minC_), 8));
      const __m128i dec_ = _mm_min_epi32(hexFF_, _mm_max_epi32(_mm_setzero_si128(), estCol_2)); // ensure in range [0, 0xFF].

      const __m128i packed_ = _mm_shuffle_epi8(dec_, shuffle_pattern_);

      *pOutLine = _mm_extract_epi32(packed_, 0);
      pOutLine++;
    }
  }
}

template <size_t channels>
static void limg_decode_block_from_factors_3d_(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const limg_encode_3d_output<channels> &in, const uint8_t shift[3])
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

  for (size_t y = 0; y < rangeY; y++)
  {
    limg_ui8_4 *pOutLine = reinterpret_cast<limg_ui8_4 *>(pOut + y * sizeX);

    for (size_t x = 0; x < rangeX; x++)
    {
      const uint8_t fA = *pA;
      const uint8_t fB = *pB;
      const uint8_t fC = *pC;

      pA++;
      pB++;
      pC++;

      limg_ui8_4 px;

      const int32_t fA_dec = (fA << shift[0]) + (fA * decode_bias[0]);
      const int32_t fB_dec = (fB << shift[1]) + (fB * decode_bias[1]);
      const int32_t fC_dec = (fC << shift[2]) + (fC * decode_bias[2]);

      for (size_t i = 0; i < channels; i++)
      {
        int32_t estCol;
        estCol = (int32_t)(minA[i] + ((fA_dec * normalA[i] + (int32_t)bias) >> 8));
        estCol += (int32_t)(minB[i] + ((fB_dec * normalB[i] + (int32_t)bias) >> 8));
        estCol += (int32_t)(minC[i] + ((fC_dec * normalC[i] + (int32_t)bias) >> 8));

        px[i] = (uint8_t)limgClamp(estCol, 0, 0xFF);
      }

      *pOutLine = px;
      pOutLine++;
    }
  }
}

template <size_t channels>
static void limg_decode_block_from_factors_3d(uint32_t *pOut, const size_t sizeX, const size_t rangeX, const size_t rangeY, const uint8_t *pA, const uint8_t *pB, const uint8_t *pC, const limg_encode_3d_output<channels> &in, const uint8_t shift[3])
{
  if (sse41Supported)
  {
    if constexpr (channels == 4)
      limg_decode_block_from_factors_3d_4_sse41(pOut, sizeX, rangeX, rangeY, pA, pB, pC, in, shift);
    else
      limg_decode_block_from_factors_3d_3_sse41(pOut, sizeX, rangeX, rangeY, pA, pB, pC, in, shift);
  }
  else
  {
    limg_decode_block_from_factors_3d_(pOut, sizeX, rangeX, rangeY, pA, pB, pC, in, shift);
  }
}

#endif // limg_decode_h__
