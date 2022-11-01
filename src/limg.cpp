#include "limg.h"

#include "limg_simd.h"
#include "limg_bit_crush.h"
#include "limg_decode.h"

//////////////////////////////////////////////////////////////////////////

template <size_t channels, bool CheckPixelAndBlockError>
static LIMG_DEBUG_NO_INLINE bool limg_encode_get_block_factors_accurate_from_state_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  float avg[channels];

  const float inv_count = 1.f / (float)(rangeX * rangeY);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      float corrected[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t i = 0; i < channels; i++)
      {
        corrected[i] = (float)px[i] - avg[i];

        const float abs_val = fabsf(corrected[i]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = i;
        }
      }

      if (max_abs != 0)
      {
        max_abs = corrected[max_idx];

        float vec[channels];
        float lengthSquared = 0;

        for (size_t i = 0; i < channels; i++)
        {
          vec[i] = corrected[i] / max_abs;
          lengthSquared += vec[i] * vec[i];
        }

        //const float inv_length = 1.f / sqrtf(lengthSquared);
        const float inv_length = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared)));

        for (size_t i = 0; i < channels; i++)
          diff_xi[i] += vec[i] * inv_length;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
    diff_xi[i] *= inv_count;

  // Find Edge Points that result in mimimal error and contain the values.
  float_t min;
  float_t max;

  bool any_nonzero = false;

  for (size_t i = 0; i < channels; i++)
    any_nonzero |= (diff_xi[i] != 0);

  size_t blockError = 0;

  if (any_nonzero)
  {
    min = FLT_MAX;
    max = -FLT_MAX;
    const float inv_length_diff_xi = 1.f / limg_dot<float, channels>(diff_xi, diff_xi);

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = (float)px[i] - avg[i];

        const float f = limg_dot<float, channels>(lineOriginToPx, diff_xi) * inv_length_diff_xi;

        if constexpr (CheckPixelAndBlockError)
        {
          float error_vec[channels];

          for (size_t i = 0; i < channels; i++)
            error_vec[i] = (float)px[i] - (avg[i] + f * diff_xi[i]);

          const size_t pixelError = limg_color_error_from_error_vec_<channels>(px, error_vec);

          if (pixelError > pCtx->maxPixelBlockError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasPixelBlockError++;
            }

            return false;
          }

          blockError += pixelError;
        }

        min = limgMin(min, f);
        max = limgMax(max, f);
      }
    }
  }
  else
  {
    min = 0;
    max = 0;
  }

  for (size_t i = 0; i < channels; i++)
  {
    a[i] = (uint8_t)limgClamp((int32_t)(avg[i] + min * diff_xi[i] + 0.5f), 0, 0xFF);
    b[i] = (uint8_t)limgClamp((int32_t)(avg[i] + max * diff_xi[i] + 0.5f), 0, 0xFF);
  }

  if constexpr (CheckPixelAndBlockError)
  {
    const size_t rangeSize = rangeX * rangeY;
    const bool ret = (((blockError * 0x10) / rangeSize) < pCtx->maxBlockPixelError);

    if constexpr (limg_DiagnoseCulprits)
    {
      if (!ret)
      {
        pCtx->culprits++;
        pCtx->culpritWasBlockPixelError++;
      }
    }

    return ret;
  }
  else
  {
    return true;
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_3_sse41(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_3d_output<3> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 3;

  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128 sign_bit = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
  const __m128 inv_sign_bit = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
  const __m128 zero_alpha = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
  const __m128 preferenceBias = _mm_set_ps(0, FLT_EPSILON * 1, FLT_EPSILON * 2, FLT_EPSILON * 3);

  const uint32_t *pStart = reinterpret_cast<const uint32_t *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  uint32_t avg[channels];

  for (size_t i = 0; i < channels; i++)
    avg[i] = (uint32_t)state.sum[i];

  const __m128 inv_count_ = _mm_set1_ps(1.f / (float)(rangeX * rangeY));
  const __m128 avg_ = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(avg))), inv_count_);

  __m128 diff_xi_dirA_ = _mm_setzero_ps();

  {
    const uint32_t *pLine = pStart;

    for (size_t y = 0; y < rangeY; y++)
    {
      for (size_t x = 0; x < rangeX; x++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[x]))));
        const __m128 corrected = _mm_and_ps(zero_alpha, _mm_sub_ps(px, avg_));

        const int32_t mask = _mm_movemask_ps(_mm_cmpeq_ps(corrected, _mm_setzero_ps()));

        if (0b1111 != mask) // can we use a different kind of `cmp` here? maybe `epi32` works fine as well?
        {
          const __m128 minBiased = _mm_sub_ps(corrected, preferenceBias);
          const __m128 maxBiased = _mm_add_ps(corrected, preferenceBias);
          const __m128 min_23 = _mm_shuffle_ps(minBiased, minBiased, _MM_SHUFFLE(0, 0, 3, 2));
          const __m128 max_23 = _mm_shuffle_ps(maxBiased, maxBiased, _MM_SHUFFLE(0, 0, 3, 2));
          const __m128 half_min = _mm_min_ps(minBiased, min_23);
          const __m128 half_max = _mm_max_ps(maxBiased, max_23);
          const __m128 abs_min = _mm_and_ps(inv_sign_bit, _mm_min_ps(half_min, _mm_shuffle_ps(half_min, half_min, _MM_SHUFFLE(0, 0, 0, 1))));
          const __m128 max = _mm_max_ps(half_max, _mm_shuffle_ps(half_max, half_max, _MM_SHUFFLE(0, 0, 0, 1)));
          const __m128 flip_sign = _mm_and_ps(sign_bit, _mm_cmpgt_ps(abs_min, max));

          const __m128 invLength = _mm_xor_ps(flip_sign, _mm_rsqrt_ps(_mm_dp_ps(corrected, corrected, 0x7F))); // should be 0xFF with 4 channels.
          const __m128 invLength4 = _mm_shuffle_ps(invLength, invLength, _MM_SHUFFLE(0, 0, 0, 0));
          
          const __m128 val = _mm_mul_ps(corrected, invLength4);
          diff_xi_dirA_ = _mm_add_ps(diff_xi_dirA_, val);

        }
      }

      pLine += pCtx->sizeX;
    }

    diff_xi_dirA_ = _mm_mul_ps(diff_xi_dirA_, inv_count_);
  }

  __m128 min_dirA = _mm_setzero_ps();
  __m128 max_dirA = _mm_setzero_ps();
  __m128 min_dirB = _mm_setzero_ps();
  __m128 max_dirB = _mm_setzero_ps();
  __m128 min_dirC = _mm_setzero_ps();
  __m128 max_dirC = _mm_setzero_ps();

  __m128 *pEstimate = reinterpret_cast<__m128 *>(pScratch);

  __m128 diff_xi_dirB_ = _mm_setzero_ps();
  __m128 diff_xi_dirC_ = _mm_setzero_ps();

  if (0b1111 != _mm_movemask_ps(_mm_cmpeq_ps(diff_xi_dirA_, _mm_setzero_ps()))) // can we use a different kind of `cmp` here? maybe `epi32` works fine as well?
  {
    // we originally set `min/max_dirA` to +/- FLT_MAX here, but that should be irrelevant, as we're counting from the average, so some should be below, some above or all zero.

    const __m128 inv_length_diff_xi_dirA = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirA_, diff_xi_dirA_, 0x7F)); // should be 0xFF with 4 channels.

    {
      const uint32_t *pLine = pStart;

      for (size_t y = 0; y < rangeY; y++)
      {
        for (size_t x = 0; x < rangeX; x++)
        {
          const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[x]))));
          const __m128 lineOriginToPx = _mm_sub_ps(px, avg_);

          const __m128 facA = _mm_mul_ps(_mm_dp_ps(lineOriginToPx, diff_xi_dirA_, 0x7F), inv_length_diff_xi_dirA); // should be 0xFF with 4 channels.
          const __m128 facA_full = _mm_shuffle_ps(facA, facA, _MM_SHUFFLE(0, 0, 0, 0));

          min_dirA = _mm_min_ps(min_dirA, facA);
          max_dirA = _mm_max_ps(max_dirA, facA);

          const __m128 estimateA = _mm_add_ps(avg_, _mm_mul_ps(facA_full, diff_xi_dirA_));
          const __m128 error_vec_dirA = _mm_and_ps(zero_alpha, _mm_sub_ps(px, estimateA));

          _mm_storeu_ps(reinterpret_cast<float *>(pEstimate), estimateA);
          pEstimate++;

          const int32_t mask = _mm_movemask_ps(_mm_cmpeq_ps(error_vec_dirA, _mm_setzero_ps()));

          if (0b1111 != mask) // can we use a different kind of `cmp` here? maybe `epi32` works fine as well?
          {
            const __m128 minBiased = _mm_sub_ps(error_vec_dirA, preferenceBias);
            const __m128 maxBiased = _mm_add_ps(error_vec_dirA, preferenceBias);
            const __m128 min_23 = _mm_shuffle_ps(minBiased, minBiased, _MM_SHUFFLE(0, 0, 3, 2));
            const __m128 max_23 = _mm_shuffle_ps(maxBiased, maxBiased, _MM_SHUFFLE(0, 0, 3, 2));
            const __m128 half_min = _mm_min_ps(minBiased, min_23);
            const __m128 half_max = _mm_max_ps(maxBiased, max_23);
            const __m128 abs_min = _mm_and_ps(inv_sign_bit, _mm_min_ps(half_min, _mm_shuffle_ps(half_min, half_min, _MM_SHUFFLE(0, 0, 0, 1))));
            const __m128 max = _mm_max_ps(half_max, _mm_shuffle_ps(half_max, half_max, _MM_SHUFFLE(0, 0, 0, 1)));
            const __m128 flip_sign = _mm_and_ps(sign_bit, _mm_cmpgt_ps(abs_min, max));

            const __m128 invLength = _mm_xor_ps(flip_sign, _mm_rsqrt_ps(_mm_dp_ps(error_vec_dirA, error_vec_dirA, 0x7F))); // should be 0xFF with 4 channels.
            const __m128 invLength4 = _mm_shuffle_ps(invLength, invLength, _MM_SHUFFLE(0, 0, 0, 0));

            diff_xi_dirB_ = _mm_add_ps(diff_xi_dirB_, _mm_mul_ps(error_vec_dirA, invLength4));
          }
        }

        pLine += pCtx->sizeX;
      }

      diff_xi_dirB_ = _mm_mul_ps(diff_xi_dirB_, inv_count_);
    }

    // diff_xi_dirC = diff_xi_dirA x diff_xi_dirB
    {
      const __m128 shufA = _mm_permute_ps(diff_xi_dirA_, _MM_SHUFFLE(3, 0, 2, 1));
      const __m128 shufB = _mm_permute_ps(diff_xi_dirB_, _MM_SHUFFLE(3, 1, 0, 2));
      const __m128 mul0 = _mm_mul_ps(shufA, shufB);
      const __m128 shufA1 = _mm_permute_ps(shufA, _MM_SHUFFLE(3, 0, 2, 1));
      const __m128 shufB1 = _mm_permute_ps(shufB, _MM_SHUFFLE(3, 1, 0, 2));

      diff_xi_dirC_ = _mm_sub_ps(mul0, _mm_mul_ps(shufA1, shufB1));
    }

    pEstimate = reinterpret_cast<__m128 *>(pScratch);

    const __m128 inv_length_diff_xi_dirB = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirB_, diff_xi_dirB_, 0x7F)); // should be 0xFF with 4 channels.
    const __m128 inv_length_diff_xi_dirC = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirC_, diff_xi_dirC_, 0x7F)); // should be 0xFF with 4 channels.

    min_dirC = min_dirB = _mm_set1_ps(FLT_MAX);
    max_dirC = max_dirB = _mm_set1_ps(-FLT_MAX);

    {
      const uint32_t *pLine = pStart;

      for (size_t y = 0; y < rangeY; y++)
      {
        for (size_t x = 0; x < rangeX; x++)
        {
          const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[x]))));

          const __m128 estimateA = _mm_loadu_ps(reinterpret_cast<float *>(pEstimate));
          pEstimate++;

          const __m128 lineOriginToPx = _mm_sub_ps(px, estimateA);

          const __m128 facB = _mm_mul_ps(_mm_dp_ps(lineOriginToPx, diff_xi_dirB_, 0x7F), inv_length_diff_xi_dirB); // should be 0xFF with 4 channels.
          const __m128 facB_full = _mm_shuffle_ps(facB, facB, _MM_SHUFFLE(0, 0, 0, 0));

          min_dirB = _mm_min_ps(min_dirB, facB);
          max_dirB = _mm_max_ps(max_dirB, facB);

          const __m128 estimateB = _mm_add_ps(estimateA, _mm_mul_ps(facB_full, diff_xi_dirB_));
          const __m128 error_vec_dirAB = _mm_sub_ps(px, estimateB);

          const __m128 facC = _mm_mul_ps(_mm_dp_ps(error_vec_dirAB, diff_xi_dirC_, 0x7F), inv_length_diff_xi_dirC); // should be 0xFF with 4 channels.

          min_dirC = _mm_min_ps(min_dirC, facC);
          max_dirC = _mm_max_ps(max_dirC, facC);
        }

        pLine += pCtx->sizeX;
      }
    }
  }

  const __m128i dirAmin = _mm_cvtps_epi32(_mm_add_ps(avg_, _mm_mul_ps(_mm_shuffle_ps(min_dirA, min_dirA, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirA_)));
  const __m128i dirAmax = _mm_cvtps_epi32(_mm_add_ps(avg_, _mm_mul_ps(_mm_shuffle_ps(max_dirA, max_dirA, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirA_)));
  const __m128i dirBmin = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(min_dirB, min_dirB, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirB_));
  const __m128i dirBmax = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(max_dirB, max_dirB, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirB_));
  const __m128i dirCmin = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(min_dirC, min_dirC, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirC_));
  const __m128i dirCmax = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(max_dirC, max_dirC, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirC_));

  _mm_storeu_ps(out.avg, avg_);

  out.dirA_min[0] = (int16_t)_mm_extract_epi32(dirAmin, 0);
  out.dirA_min[1] = (int16_t)_mm_extract_epi32(dirAmin, 1);
  out.dirA_min[2] = (int16_t)_mm_extract_epi32(dirAmin, 2);

  out.dirA_max[0] = (int16_t)_mm_extract_epi32(dirAmax, 0);
  out.dirA_max[1] = (int16_t)_mm_extract_epi32(dirAmax, 1);
  out.dirA_max[2] = (int16_t)_mm_extract_epi32(dirAmax, 2);

  out.dirB_offset[0] = (int16_t)_mm_extract_epi32(dirBmin, 0);
  out.dirB_offset[1] = (int16_t)_mm_extract_epi32(dirBmin, 1);
  out.dirB_offset[2] = (int16_t)_mm_extract_epi32(dirBmin, 2);

  out.dirB_mag[0] = (int16_t)_mm_extract_epi32(dirBmax, 0);
  out.dirB_mag[1] = (int16_t)_mm_extract_epi32(dirBmax, 1);
  out.dirB_mag[2] = (int16_t)_mm_extract_epi32(dirBmax, 2);

  out.dirC_offset[0] = (int16_t)_mm_extract_epi32(dirCmin, 0);
  out.dirC_offset[1] = (int16_t)_mm_extract_epi32(dirCmin, 1);
  out.dirC_offset[2] = (int16_t)_mm_extract_epi32(dirCmin, 2);

  out.dirC_mag[0] = (int16_t)_mm_extract_epi32(dirCmax, 0);
  out.dirC_mag[1] = (int16_t)_mm_extract_epi32(dirCmax, 1);
  out.dirC_mag[2] = (int16_t)_mm_extract_epi32(dirCmax, 2);
}

LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_3(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_3d_output<3> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 3;

  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  float avg[channels];

  const float inv_count = 1.f / (float)(rangeX * rangeY);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi_dirA[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      float corrected[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t i = 0; i < channels; i++)
      {
        corrected[i] = (float)px[i] - avg[i];

        const float abs_val = fabsf(corrected[i]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = i;
        }
      }

      if (max_abs != 0)
      {
        max_abs = corrected[max_idx];

        float lengthSquared = 0;

        for (size_t i = 0; i < channels; i++)
          lengthSquared += corrected[i] * corrected[i];

        const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

        for (size_t i = 0; i < channels; i++)
          diff_xi_dirA[i] += corrected[i] * inv_length;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] *= inv_count;

  // Find Edge Points that result in mimimal error and contain the values.
  float_t min_dirA;
  float_t max_dirA;
  float_t min_dirB;
  float_t max_dirB;
  float_t min_dirC;
  float_t max_dirC;

  bool any_nonzero = false;

  for (size_t i = 0; i < channels; i++)
    any_nonzero |= (diff_xi_dirA[i] != 0);

  float *pEstimate = pScratch;

  float diff_xi_dirB[channels];
  float diff_xi_dirC[channels];

  if (!any_nonzero)
  {
    min_dirA = 0;
    max_dirA = 0;
    min_dirB = 0;
    max_dirB = 0;
    min_dirC = 0;
    max_dirC = 0;
  }
  else
  {
    min_dirA = FLT_MAX;
    max_dirA = -FLT_MAX;

    const float inv_length_diff_xi_dirA = 1.f / limg_dot<float, channels>(diff_xi_dirA, diff_xi_dirA);

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirC[i] = diff_xi_dirB[i] = 0;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = (float)px[i] - avg[i];

        const float facA = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirA) * inv_length_diff_xi_dirA;

        min_dirA = limgMin(min_dirA, facA);
        max_dirA = limgMax(max_dirA, facA);

        float error_vec_dirA[channels];
        float max_abs = 0;
        size_t max_idx = 0;

        for (size_t i = 0; i < channels; i++)
        {
          pEstimate[i] = avg[i] + facA * diff_xi_dirA[i];
          error_vec_dirA[i] = (float)px[i] - pEstimate[i];

          const float abs_val = fabsf(error_vec_dirA[i]);

          if (abs_val > max_abs)
          {
            max_abs = abs_val;
            max_idx = i;
          }
        }

        pEstimate += channels;

        if (max_abs != 0)
        {
          max_abs = error_vec_dirA[max_idx];

          float lengthSquared = 0;

          for (size_t i = 0; i < channels; i++)
            lengthSquared += error_vec_dirA[i] * error_vec_dirA[i];

          const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

          for (size_t i = 0; i < channels; i++)
            diff_xi_dirB[i] += error_vec_dirA[i] * inv_length;
        }
      }
    }

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirB[i] *= inv_count;

    limg_cross(diff_xi_dirA, diff_xi_dirB, diff_xi_dirC);

    pEstimate = pScratch;

    const float inv_length_diff_xi_dirB = 1.f / limg_dot<float, channels>(diff_xi_dirB, diff_xi_dirB);
    const float inv_length_diff_xi_dirC = 1.f / limg_dot<float, channels>(diff_xi_dirC, diff_xi_dirC);

    min_dirB = FLT_MAX;
    max_dirB = -FLT_MAX;
    min_dirC = FLT_MAX;
    max_dirC = -FLT_MAX;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = px[i] - pEstimate[i];

        const float facB = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirB) * inv_length_diff_xi_dirB;

        min_dirB = limgMin(min_dirB, facB);
        max_dirB = limgMax(max_dirB, facB);

        float error_vec_dirAB[channels];

        for (size_t i = 0; i < channels; i++)
        {
          const float estimage = (pEstimate[i] + facB * diff_xi_dirB[i]);
          error_vec_dirAB[i] = (float)px[i] - estimage;
        }

        const float facC = limg_dot<float, channels>(error_vec_dirAB, diff_xi_dirC) * inv_length_diff_xi_dirC;

        min_dirC = limgMin(min_dirC, facC);
        max_dirC = limgMax(max_dirC, facC);

        pEstimate += channels;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
  {
    out.avg[i] = avg[i];
    out.dirA_min[i] = (int16_t)limg_fast_round_int16(avg[i] + min_dirA * diff_xi_dirA[i]);
    out.dirA_max[i] = (int16_t)limg_fast_round_int16(avg[i] + max_dirA * diff_xi_dirA[i]);
    out.dirB_offset[i] = (int16_t)limg_fast_round_int16(min_dirB * diff_xi_dirB[i]);
    out.dirB_mag[i] = (int16_t)limg_fast_round_int16(max_dirB * diff_xi_dirB[i]);
    out.dirC_offset[i] = (int16_t)limg_fast_round_int16(min_dirC * diff_xi_dirC[i]);
    out.dirC_mag[i] = (int16_t)limg_fast_round_int16(max_dirC * diff_xi_dirC[i]);
  }
}

// Since the cross product isn't defined in R4 and using three vectors we can't possibly represent all values in R4, so we'd rather get the closest possible direction we can get.
LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_4(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_3d_output<4> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 4;

  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  float avg[channels];

  const float inv_count = 1.f / (float)(rangeX * rangeY);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi_dirA[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      float corrected[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t i = 0; i < channels; i++)
      {
        corrected[i] = (float)px[i] - avg[i];

        const float abs_val = fabsf(corrected[i]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = i;
        }
      }

      if (max_abs != 0)
      {
        max_abs = corrected[max_idx];

        float lengthSquared = 0;

        for (size_t i = 0; i < channels; i++)
          lengthSquared += corrected[i] * corrected[i];

        const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

        for (size_t i = 0; i < channels; i++)
          diff_xi_dirA[i] += corrected[i] * inv_length;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] *= inv_count;

  // Find Edge Points that result in mimimal error and contain the values.
  float_t min_dirA;
  float_t max_dirA;
  float_t min_dirB;
  float_t max_dirB;
  float_t min_dirC;
  float_t max_dirC;

  bool any_nonzero = false;

  for (size_t i = 0; i < channels; i++)
    any_nonzero |= (diff_xi_dirA[i] != 0);

  float *pEstimate = pScratch;

  float diff_xi_dirB[channels];
  float diff_xi_dirC[channels];

  if (!any_nonzero)
  {
    min_dirA = 0;
    max_dirA = 0;
    min_dirB = 0;
    max_dirB = 0;
    min_dirC = 0;
    max_dirC = 0;
  }
  else
  {
    min_dirA = FLT_MAX;
    max_dirA = -FLT_MAX;

    const float inv_length_diff_xi_dirA = 1.f / limg_dot<float, channels>(diff_xi_dirA, diff_xi_dirA);

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirC[i] = diff_xi_dirB[i] = 0;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = (float)px[i] - avg[i];

        const float facA = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirA) * inv_length_diff_xi_dirA;

        min_dirA = limgMin(min_dirA, facA);
        max_dirA = limgMax(max_dirA, facA);

        float error_vec_dirA[channels];
        float max_abs = 0;
        size_t max_idx = 0;

        for (size_t i = 0; i < channels; i++)
        {
          pEstimate[i] = avg[i] + facA * diff_xi_dirA[i];
          error_vec_dirA[i] = (float)px[i] - pEstimate[i];

          const float abs_val = fabsf(error_vec_dirA[i]);

          if (abs_val > max_abs)
          {
            max_abs = abs_val;
            max_idx = i;
          }
        }

        pEstimate += channels;

        if (max_abs != 0)
        {
          max_abs = error_vec_dirA[max_idx];

          float lengthSquared = 0;

          for (size_t i = 0; i < channels; i++)
            lengthSquared += error_vec_dirA[i] * error_vec_dirA[i];

          const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

          for (size_t i = 0; i < channels; i++)
            diff_xi_dirB[i] += error_vec_dirA[i] * inv_length;
        }
      }
    }

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirB[i] *= inv_count;

    const float inv_length_diff_xi_dirB = 1.f / limg_dot<float, channels>(diff_xi_dirB, diff_xi_dirB);

    min_dirB = FLT_MAX;
    max_dirB = -FLT_MAX;

    pEstimate = pScratch;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = px[i] - pEstimate[i];

        const float facB = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirB) * inv_length_diff_xi_dirB;

        min_dirB = limgMin(min_dirB, facB);
        max_dirB = limgMax(max_dirB, facB);

        float error_vec_dirAB[channels];
        float max_abs = 0;
        size_t max_idx = 0;

        for (size_t i = 0; i < channels; i++)
        {
          pEstimate[i] = (pEstimate[i] + facB * diff_xi_dirB[i]);
          error_vec_dirAB[i] = (float)px[i] - pEstimate[i];

          const float abs_val = fabsf(error_vec_dirAB[i]);

          if (abs_val > max_abs)
          {
            max_abs = abs_val;
            max_idx = i;
          }
        }

        pEstimate += channels;

        if (max_abs != 0)
        {
          max_abs = error_vec_dirAB[max_idx];

          float lengthSquared = 0;

          for (size_t i = 0; i < channels; i++)
            lengthSquared += error_vec_dirAB[i] * error_vec_dirAB[i];

          const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

          for (size_t i = 0; i < channels; i++)
            diff_xi_dirC[i] += error_vec_dirAB[i] * inv_length;
        }
      }
    }

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirC[i] *= inv_count;

    const float inv_length_diff_xi_dirC = 1.f / limg_dot<float, channels>(diff_xi_dirC, diff_xi_dirC);

    pEstimate = pScratch;

    min_dirC = FLT_MAX;
    max_dirC = -FLT_MAX;

    for (size_t y = 0; y < rangeY; y++)
    {
      const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

      for (size_t x = 0; x < rangeX; x++)
      {
        limg_ui8_4 px = *pLine;
        pLine++;

        float lineOriginToPx[channels];

        for (size_t i = 0; i < channels; i++)
          lineOriginToPx[i] = px[i] - pEstimate[i];

        const float facC = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirC) * inv_length_diff_xi_dirC;

        min_dirC = limgMin(min_dirC, facC);
        max_dirC = limgMax(max_dirC, facC);

        pEstimate += channels;
      }
    }
  }

  for (size_t i = 0; i < channels; i++)
  {
    out.avg[i] = avg[i];
    out.dirA_min[i] = (int16_t)limg_fast_round_int16(avg[i] + min_dirA * diff_xi_dirA[i]);
    out.dirA_max[i] = (int16_t)limg_fast_round_int16(avg[i] + max_dirA * diff_xi_dirA[i]);
    out.dirB_offset[i] = (int16_t)limg_fast_round_int16(min_dirB * diff_xi_dirB[i]);
    out.dirB_mag[i] = (int16_t)limg_fast_round_int16(max_dirB * diff_xi_dirB[i]);
    out.dirC_offset[i] = (int16_t)limg_fast_round_int16(min_dirC * diff_xi_dirC[i]);
    out.dirC_mag[i] = (int16_t)limg_fast_round_int16(max_dirC * diff_xi_dirC[i]);
  }
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_3d_output<channels> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  if constexpr (channels == 3)
  {
    if (sse41Supported)
      limg_encode_get_block_factors_accurate_from_state_3d_3_sse41(pCtx, offsetX, offsetY, rangeX, rangeY, out, state, pScratch);
    else
      limg_encode_get_block_factors_accurate_from_state_3d_3(pCtx, offsetX, offsetY, rangeX, rangeY, out, state, pScratch);
  }
  else
  {
    limg_encode_get_block_factors_accurate_from_state_3d_4(pCtx, offsetX, offsetY, rangeX, rangeY, out, state, pScratch);
  }
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError, bool ReadWriteRangeSize, size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_check_area_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError, const size_t startBlockError, size_t *pAdditionalRangeForInitialBlockError)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  size_t blockError = startBlockError;

  limg_color_error_state<channels> color_error_state;
  limg_init_color_error_state<channels>(a, b, color_error_state);

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      if constexpr (CheckBounds)
      {
        for (size_t i = 0; i < channels; i++)
        {
          if ((int64_t)px[i] < a[i] - (int64_t)pCtx->maxPixelChannelBlockError || (int64_t)px[i] > b[i] + (int64_t)pCtx->maxPixelChannelBlockError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasPixelChannelBlockError++;
            }

            return false;
          }
        }
      }

      float factor;
      size_t error;

      if constexpr (!CheckPixelError && !WriteBlockError)
        error = limg_color_error_state_get_factor<channels>(px, a, color_error_state, factor);
      else
        error = limg_color_error_state_get_error<channels>(px, a, color_error_state, factor);

      if constexpr (CheckPixelError)
      {
        if (error > pCtx->maxPixelBlockError)
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasPixelBlockError++;
          }

          return false;
        }
      }

      blockError += error;

      if constexpr (WriteToFactors)
      {
        *pFactors = factor;
        pFactors++;
      }
    }
  }

  if constexpr (WriteBlockError)
    *pBlockError = blockError;

  size_t rangeSize = rangeX * rangeY;

  if constexpr (ReadWriteRangeSize)
  {
    rangeSize += *pAdditionalRangeForInitialBlockError;
    *pAdditionalRangeForInitialBlockError = rangeSize;
  }

  const bool ret = (((blockError * 0x10) / rangeSize) < pCtx->maxBlockPixelError);

  if constexpr (limg_DiagnoseCulprits)
  {
    if (!ret)
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockPixelError++;
    }
  }

  return ret;
}

template <bool WriteToFactors, bool WriteBlockError, bool CheckBounds, bool CheckPixelError, bool ReadWriteRangeSize>
static LIMG_INLINE bool limg_encode_check_area(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, const limg_ui8_4 &a, const limg_ui8_4 &b, float *pFactors, size_t *pBlockError, const size_t startBlockError, size_t *pAdditionalRangeForInitialBlockError)
{
  if (pCtx->hasAlpha)
    return limg_encode_check_area_<WriteToFactors, WriteBlockError, CheckBounds, CheckPixelError, ReadWriteRangeSize, 4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, pFactors, pBlockError, startBlockError, pAdditionalRangeForInitialBlockError);
  else
    return limg_encode_check_area_<WriteToFactors, WriteBlockError, CheckBounds, CheckPixelError, ReadWriteRangeSize, 3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, pFactors, pBlockError, startBlockError, pAdditionalRangeForInitialBlockError);
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_min_max_per_channel_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state)
{
#if LIMG_PRECISE_DECOMPOSITION != 1
  (void)pCtx;
  (void)offsetX;
  (void)offsetY;
  (void)rangeX;
  (void)rangeY;
  (void)out_a;
  (void)out_b;
  (void)state;

  return false;
#else
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
      {
        if (px[i] < state.low[i][i])
          state.low[i] = px;
        else if (px[i] > state.high[i][i])
          state.high[i] = px;
      }
    }
  }

  limg_ui8_4 max_l = state.low[0];
  limg_ui8_4 max_h = state.high[0];

  size_t maxDist = state.maxDist;

  if constexpr (limg_ColorDependentABError)
    maxDist = limg_color_error<channels>(max_l, max_h);
  else
    maxDist = limg_vector_error<channels>(max_l, max_h);

  size_t h_index = 1;

  for (size_t l_index = 0; l_index < channels; l_index++)
  {
    for (; h_index < channels; h_index++)
    {
      const limg_ui8_4 l = state.low[l_index];
      const limg_ui8_4 h = state.high[h_index];

      size_t dist;

      if constexpr (limg_ColorDependentABError)
        dist = limg_color_error<channels>(l, h);
      else
        dist = limg_vector_error<channels>(l, h);

      if (dist > maxDist)
      {
        maxDist = dist;
        max_l = l;
        max_h = h;
      }
    }

    h_index = 0;
  }

  if (!out_a.equals<channels>(max_l) || !out_b.equals<channels>(max_h))
  {
    limg_color_error_state<channels> color_error_state;
    limg_init_color_error_state<channels>(max_l, max_h, color_error_state);

    if (!out_a.equals<channels>(max_l))
    {
      float factor;
      const size_t error = limg_color_error_state_get_error<channels>(out_a, max_l, color_error_state, factor);

      if (error > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }

    if (!out_b.equals<channels>(max_h))
    {
      float factor;
      const size_t error = limg_color_error_state_get_error<channels>(out_b, max_l, color_error_state, factor);

      if (error > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }
  }

  out_a = max_l;
  out_b = max_h;

  state.maxDist = maxDist;

  return true;
#endif
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_min_max_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  limg_ui8_4 a = out_a;
  limg_ui8_4 b = out_b;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      int64_t low = 0;

      for (size_t i = 0; i < channels; i++)
        low += limgMax(-1LL, ((int64_t)a[i] - px[i]));

      if (low > 0)
      {
        // is `a` a linear combination of `px` and `b`?
        limg_color_error_state<channels> color_error_state;
        limg_init_color_error_state(px, b, color_error_state);

        float factor;
        const size_t error = limg_color_error_state_get_error<channels>(a, px, color_error_state, factor);

        if (error > pCtx->maxBlockExpandError)
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasBlockExpandError++;
          }

          return false;
        }

        a = px;
      }
      else
      {
        int64_t high = 0;

        for (size_t i = 0; i < channels; i++)
          high += limgMax(-1LL, (px[i] - (int64_t)b[i]));

        if (high > 0)
        {
          // is `b` a linear combination of `a` and `px`?
          limg_color_error_state<channels> color_error_state;
          limg_init_color_error_state(a, px, color_error_state);

          float factor;
          const size_t error = limg_color_error_state_get_error<channels>(b, a, color_error_state, factor);

          if (error > pCtx->maxBlockExpandError)
          {
            if constexpr (limg_DiagnoseCulprits)
            {
              pCtx->culprits++;
              pCtx->culpritWasBlockExpandError++;
            }

            return false;
          }

          b = px;
        }
      }
    }
  }

  out_a = a;
  out_b = b;

  return true;
}

template <size_t channels>
static LIMG_INLINE void limg_encode_attempt_include_pixels_to_sum_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  // Calculate Sum (for average).
  size_t sum[channels];

  for (size_t i = 0; i < channels; i++)
    sum[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
        sum[i] += px[i];
    }
  }

  // Store stuff in the decomposition state.
  for (size_t i = 0; i < channels; i++)
    state.sum[i] += sum[i];
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_attempt_include_pixels_accurate_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  const limg_ui8_4 old_a = out_a;
  const limg_ui8_4 old_b = out_b;

  limg_encode_attempt_include_pixels_to_sum_<channels>(pCtx, offsetX, offsetY, rangeX, rangeY, state);

  if (!limg_encode_get_block_factors_accurate_from_state_<channels, true>(pCtx, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY, out_a, out_b, state))
    return false;

  if (!old_a.equals_w_alpha(out_a) || !old_b.equals_w_alpha(out_b))
  {
    limg_color_error_state<channels> color_error_state;
    limg_init_color_error_state<channels>(out_a, out_b, color_error_state);

    if (!old_a.equals<channels>(out_a))
    {
      float _unused;

      if (limg_color_error_state_get_error<channels>(old_a, out_a, color_error_state, _unused) > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }

    if (!old_b.equals<channels>(out_b))
    {
      float _unused;

      if (limg_color_error_state_get_error<channels>(old_b, out_a, color_error_state, _unused) > pCtx->maxBlockExpandError)
      {
        if constexpr (limg_DiagnoseCulprits)
        {
          pCtx->culprits++;
          pCtx->culpritWasBlockExpandError++;
        }

        return false;
      }
    }
  }

  return true;
}

static LIMG_INLINE bool limg_encode_attempt_include_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_accurate_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
    else
      return limg_encode_attempt_include_pixels_accurate_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
  }
  else if constexpr (limg_RetrievePreciseDecomposition == 1)
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_min_max_per_channel_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state);
    else
      return limg_encode_attempt_include_pixels_min_max_per_channel_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state);
  }
  else
  {
    if (pCtx->hasAlpha)
      return limg_encode_attempt_include_pixels_min_max_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
    else
      return limg_encode_attempt_include_pixels_min_max_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b);
  }
}

static LIMG_INLINE bool limg_encode_check_pixel_unused(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY)
{
  const uint32_t *pStart = (pCtx->pBlockInfo + offsetX + offsetY * pCtx->sizeX);

  for (size_t y = 0; y < rangeY; y++)
  {
    const uint32_t *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      if (!!(*pLine & BlockInfo_InUse))
        return false;

      pLine++;
    }
  }

  return true;
}

static LIMG_INLINE bool limg_encode_attempt_include_unused_pixels(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &out_a, limg_ui8_4 &out_b, limg_encode_decomposition_state &state, const size_t fullOffsetX, const size_t fullOffsetY, const size_t fullRangeX, const size_t fullRangeY)
{
  return limg_encode_check_pixel_unused(pCtx, offsetX, offsetY, rangeX, rangeY) && limg_encode_attempt_include_pixels(pCtx, offsetX, offsetY, rangeX, rangeY, out_a, out_b, state, fullOffsetX, fullOffsetY, fullRangeX, fullRangeY);
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_min_max_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  a = pStart[0];
  b = a;

  size_t x = 1;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      int64_t low = 0;

      for (size_t i = 0; i < channels; i++)
        low += limgMax(-1LL, ((int64_t)a[i] - px[i]));

      if (low > 0)
      {
        a = px;
      }
      else
      {
        int64_t high = 0;

        for (size_t i = 0; i < channels; i++)
          high += limgMax(-1LL, (px[i] - (int64_t)b[i]));

        if (high > 0)
          b = px;
      }
    }

    x = 0;
  }
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_min_max_per_channel_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
#if LIMG_PRECISE_DECOMPOSITION != 1
  (void)pCtx;
  (void)offsetX;
  (void)offsetY;
  (void)rangeX;
  (void)rangeY;
  (void)a;
  (void)b;
  (void)state;
#else
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  limg_ui8_4 low[channels];
  limg_ui8_4 high[channels];

  for (size_t i = 0; i < channels; i++)
    high[i] = low[i] = *pStart;

  size_t x = 1;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
      {
        if (px[i] < low[i][i])
          low[i] = px;
        else if (px[i] > high[i][i])
          high[i] = px;
      }
    }

    x = 0;
  }

  limg_ui8_4 max_l = low[0];
  limg_ui8_4 max_h = high[0];

  size_t maxDist;

  if constexpr (limg_ColorDependentABError)
    maxDist = limg_color_error<channels>(max_l, max_h);
  else
    maxDist = limg_vector_error<channels>(max_l, max_h);

  size_t h_index = 1;

  for (size_t l_index = 0; l_index < channels; l_index++)
  {
    for (; h_index < channels; h_index++)
    {
      const limg_ui8_4 l = low[l_index];
      const limg_ui8_4 h = high[h_index];

      size_t dist;

      if constexpr (limg_ColorDependentABError)
        dist = limg_color_error<channels>(l, h);
      else
        dist = limg_vector_error<channels>(l, h);

      if (dist > maxDist)
      {
        maxDist = dist;
        max_l = l;
        max_h = h;
      }
    }

    h_index = 0;
  }

  a = max_l;
  b = max_h;

  for (size_t i = 0; i < channels; i++)
  {
    state.low[i] = low[i];
    state.high[i] = high[i];
  }

  state.maxDist = maxDist;
#endif
}

template <size_t channels>
static LIMG_DEBUG_NO_INLINE bool limg_encode_get_block_factors_accurate_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetX + offsetY * pCtx->sizeX);

  // Calculate Sum (for average).
  size_t sum[channels];

  for (size_t i = 0; i < channels; i++)
    sum[i] = 0;

  for (size_t y = 0; y < rangeY; y++)
  {
    const limg_ui8_4 *pLine = pStart + pCtx->sizeX * y;

    for (size_t x = 0; x < rangeX; x++)
    {
      limg_ui8_4 px = *pLine;
      pLine++;

      for (size_t i = 0; i < channels; i++)
        sum[i] += px[i];
    }
  }

  // Store stuff in the decomposition state.
  for (size_t i = 0; i < channels; i++)
    state.sum[i] = sum[i];

  return limg_encode_get_block_factors_accurate_from_state_<channels, true>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
}

static LIMG_INLINE bool limg_encode_get_block_a_b(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_ui8_4 &a, limg_ui8_4 &b, limg_encode_decomposition_state &state)
{
  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (pCtx->hasAlpha)
      return limg_encode_get_block_factors_accurate_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
    else
      return limg_encode_get_block_factors_accurate_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
  }
  else if constexpr (limg_RetrievePreciseDecomposition == 1)
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_per_channel_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);
    else
      limg_encode_get_block_min_max_per_channel_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b, state);

    return true;
  }
  else
  {
    if (pCtx->hasAlpha)
      limg_encode_get_block_min_max_<4>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);
    else
      limg_encode_get_block_min_max_<3>(pCtx, offsetX, offsetY, rangeX, rangeY, a, b);

    return true;
  }
}

template <size_t channels>
static LIMG_INLINE void limg_encode_sum_to_decomposition_state_(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_decomposition_state &state)
{
  for (size_t i = 0; i < channels; i++)
    state.sum[i] = 0;

  for (size_t oy = 0; oy < rangeY; oy++)
  {
    const limg_ui8_4 *pLine = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + (offsetY + oy) * pCtx->sizeX + offsetX);

    for (size_t ox = 0; ox < rangeX; ox++)
    {
      for (size_t i = 0; i < channels; i++)
        state.sum[i] += (*pLine)[i];

      pLine++;
    }
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
static LIMG_INLINE void limg_encode_sum_to_decomposition_state_sse41(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_decomposition_state &state) // this could be made SSSE3 compatible with very little effort.
{
  __m128i sum = _mm_setzero_si128();

  const limg_ui8_4 *pLine = reinterpret_cast<const limg_ui8_4 *>(pCtx->pSourceImage + offsetY * pCtx->sizeX + offsetX);

  const __m128i shuffle_8lo_16 = _mm_set_epi8(-1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);
  const __m128i shuffle_8hi_16 = _mm_set_epi8(-1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8);

  const size_t rangeX_si128 = (size_t)limgMax(0LL, (int64_t)(rangeX - sizeof(__m128i) / sizeof(uint32_t)));

  for (size_t oy = 0; oy < rangeY; oy++)
  {
    size_t ox = 0;

    for (; ox <= rangeX_si128; ox += sizeof(__m128i) / sizeof(uint32_t))
    {
      const __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[ox]));

      const __m128i sum16 = _mm_add_epi16(_mm_shuffle_epi8(val, shuffle_8lo_16), _mm_shuffle_epi8(val, shuffle_8hi_16));
      const __m128i sum32 = _mm_cvtepi16_epi32(_mm_hadd_epi16(sum16, sum16));

      sum = _mm_add_epi32(sum, sum32);
    }

    for (; ox < rangeX; ox++)
      sum = _mm_add_epi32(sum, _mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[ox]))));

    pLine += pCtx->sizeX;
  }

  _mm_storeu_si128(reinterpret_cast<__m128i *>(&state.sum[0]), _mm_cvtepu32_epi64(sum));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(&state.sum[2]), _mm_cvtepu32_epi64(_mm_bsrli_si128(sum, 8)));
}

template <size_t channels>
static LIMG_INLINE void limg_encode_sum_to_decomposition_state(limg_encode_context *pCtx, const size_t offsetX, const size_t offsetY, const size_t rangeX, const size_t rangeY, limg_encode_decomposition_state &state)
{
  if (sse41Supported)
    limg_encode_sum_to_decomposition_state_sse41(pCtx, offsetX, offsetY, rangeX, rangeY, state);
  else
    limg_encode_sum_to_decomposition_state_<channels>(pCtx, offsetX, offsetY, rangeX, rangeY, state);
}

bool LIMG_DEBUG_NO_INLINE limg_encode_find_block_expand(limg_encode_context *pCtx, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB, const bool up, const bool down, const bool left, const bool right)
{
  int64_t ox = *pOffsetX;
  int64_t oy = *pOffsetY;
  int64_t rx = *pRangeX;
  int64_t ry = *pRangeY;

  bool canGrowUp = up;
  bool canGrowDown = down;
  bool canGrowLeft = left;
  bool canGrowRight = right;

#ifdef DEBUG
  if (!limg_encode_check_pixel_unused(pCtx, ox, oy, rx, ry))
  {
    __debugbreak(); // This function should've only been called if this was true, so... eh...
    return false;
  }
#endif

  limg_ui8_4 a, b;
  limg_encode_decomposition_state decomp_state;

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    if (!limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b, decomp_state))
      return false;
  }
  else
  {
    limg_encode_get_block_a_b(pCtx, ox, oy, rx, ry, a, b, decomp_state);
  }

  size_t blockError = 0;
  size_t rangeSize = 0;

  if constexpr (limg_RetrievePreciseDecomposition != 2)
    if (!limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, rx, ry, a, b, nullptr, &blockError, 0, &rangeSize))
      return false;

  while (canGrowUp || canGrowDown || canGrowLeft || canGrowRight)
  {
    if (canGrowRight)
    {
      const int64_t newRx = limgMin(rx + limg_BlockExpandStep, pCtx->sizeX - ox);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newRx == rx || !limg_encode_attempt_include_unused_pixels(pCtx, ox + rx, oy, newRx - rx, ry, newA, newB, new_decomp_state, ox, oy, newRx, ry);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
        if (!cantGrowFurther)
        {
          if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
          {
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox + rx, oy, newRx - rx, ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
          }
          else
          {
            newRangeSize = 0;
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, newRx, ry, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
          }
        }
      }

      if (cantGrowFurther)
      {
        canGrowRight = false;
      }
      else
      {
        rx = newRx;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
        decomp_state = new_decomp_state;
      }
    }

    if (canGrowDown)
    {
      const int64_t newRy = limgMin(ry + limg_BlockExpandStep, pCtx->sizeY - oy);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newRy == ry || !limg_encode_attempt_include_unused_pixels(pCtx, ox, oy + ry, rx, newRy - ry, newA, newB, new_decomp_state, ox, oy, rx, newRy);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
        if (!cantGrowFurther)
        {
          if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy + ry, rx, newRy - ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
          else
          {
            newRangeSize = 0;
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, oy, rx, newRy, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
          }
        }
      }

      if (cantGrowFurther)
      {
        canGrowDown = false;
      }
      else
      {
        ry = newRy;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
        decomp_state = new_decomp_state;
      }
    }

    if (canGrowUp)
    {
      const int64_t newOx = limgMax(0LL, ox - (int64_t)limg_BlockExpandStep);
      const int64_t newRx = rx + (ox - newOx);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newOx == ox || !limg_encode_attempt_include_unused_pixels(pCtx, newOx, oy, ox - newOx, ry, newA, newB, new_decomp_state, newOx, oy, newRx, ry);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
        if (!cantGrowFurther)
        {
          if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, newOx, oy, ox - newOx, ry, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
          else
          {
            newRangeSize = 0;
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, newOx, oy, newRx, ry, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
          }
        }
      }

      if (cantGrowFurther)
      {
        canGrowUp = false;
      }
      else
      {
        rx = newRx;
        ox = newOx;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
        decomp_state = new_decomp_state;
      }
    }

    if (canGrowLeft)
    {
      const int64_t newOy = limgMax(0LL, oy - (int64_t)limg_BlockExpandStep);
      const int64_t newRy = ry + (oy - newOy);
      limg_ui8_4 newA = a;
      limg_ui8_4 newB = b;
      size_t newBlockError = 0;
      size_t newRangeSize = rangeSize;
      limg_encode_decomposition_state new_decomp_state = decomp_state;

      bool cantGrowFurther = newOy == oy || !limg_encode_attempt_include_unused_pixels(pCtx, ox, newOy, rx, oy - newOy, newA, newB, new_decomp_state, ox, newOy, rx, newRy);

      if constexpr (limg_RetrievePreciseDecomposition != 2)
      {
        if (!cantGrowFurther)
        {
          if (a.equals(newA, pCtx->hasAlpha) && b.equals(newB, pCtx->hasAlpha))
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, newOy, rx, oy - newOy, newA, newB, nullptr, &newBlockError, blockError, &newRangeSize);
          else
          {
            newRangeSize = 0;
            cantGrowFurther = !limg_encode_check_area<false, true, true, true, true>(pCtx, ox, newOy, rx, newRy, newA, newB, nullptr, &newBlockError, 0, &newRangeSize);
          }
        }
      }

      if (cantGrowFurther)
      {
        canGrowLeft = false;
      }
      else
      {
        ry = newRy;
        oy = newOy;
        a = newA;
        b = newB;
        blockError = newBlockError;
        rangeSize = newRangeSize;
        decomp_state = new_decomp_state;
      }
    }
  }

  *pOffsetX = ox;
  *pOffsetY = oy;
  *pRangeX = rx;
  *pRangeY = ry;
  *pA = a;
  *pB = b;

  return true;
}

bool LIMG_DEBUG_NO_INLINE limg_encode_find_block(limg_encode_context *pCtx, size_t &staticX, size_t &staticY, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_ui8_4 *pA, limg_ui8_4 *pB)
{
  size_t ox = staticX;
  size_t oy = staticY;

  for (; oy < pCtx->sizeY; oy += limg_BlockExpandStep)
  {
    const uint32_t *pBlockInfoLine = &pCtx->pBlockInfo[oy * pCtx->sizeX];

    for (; ox < pCtx->sizeX; ox += limg_BlockExpandStep)
    {
      if (!!(pBlockInfoLine[ox] & BlockInfo_InUse))
        continue;

      const size_t maxRx = pCtx->sizeX - ox;
      const size_t maxRy = pCtx->sizeY - oy;

      size_t rx = limgMin(limg_MinBlockSize, maxRx);
      size_t ry = limgMin(limg_MinBlockSize, maxRy);

      if (!limg_encode_check_pixel_unused(pCtx, ox, oy, rx, ry))
        continue;

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;

      if (!limg_encode_find_block_expand(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB, false, true, false, true))
        continue;

      rx = *pRangeX;
      ry = *pRangeY;

      *pOffsetX = (ox + rx / 2 - limg_MinBlockSize / 2) & ~(size_t)(limg_BlockExpandStep - 1);
      *pOffsetY = (oy + ry / 2 - limg_MinBlockSize / 2) & ~(size_t)(limg_BlockExpandStep - 1);
      *pRangeX = limgMin(limg_MinBlockSize, rx);
      *pRangeY = limgMin(limg_MinBlockSize, ry);
      const limg_ui8_4 a = *pA;
      const limg_ui8_4 b = *pB;

      if (rx >= limg_MinBlockSize && ry >= limg_MinBlockSize && limg_encode_find_block_expand(pCtx, pOffsetX, pOffsetY, pRangeX, pRangeY, pA, pB, true, true, true, true))
      {
        staticX = ox;
        staticY = oy;

        return true;
      }

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;
      *pA = a;
      *pB = b;

      staticX = ox + rx;
      staticY = oy;

      return true;
    }

    ox = 0;
  }

  staticX = ox;
  staticY = oy;

  return false;
}

// returns new `ditherHash`.
LIMG_INLINE static uint64_t limg_encode_dither(const uint8_t shift, const size_t rangeSize, uint64_t ditherHash, uint8_t *pFactorsU8)
{
  if (shift > 7)
    return ditherHash;

  const uint32_t ditherSize = (1 << shift) - 1;
  const int32_t ditherOffset = 1 << (shift - 1);

  for (size_t i = 0; i < rangeSize; i++)
  {
    ditherHash = ditherHash * 6364136223846793005ULL + 1;

    const uint32_t xorshifted_hi = (uint32_t)(((ditherHash >> 18) ^ ditherHash) >> 27);
    const uint32_t rot_hi = (uint32_t)(ditherHash >> 59);

    const uint32_t hi = (xorshifted_hi >> rot_hi) | (xorshifted_hi << (uint32_t)((-(int32_t)rot_hi) & 31));

    const int32_t rand = (int32_t)(hi & ditherSize) - ditherOffset;

    pFactorsU8[i] = (uint8_t)limgClamp((int32_t)pFactorsU8[i] + rand, 0, 0xFF) >> shift;
  }

  return ditherHash;
}

limg_result limg_encode_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint32_t *pA, uint32_t *pB, uint32_t *pBlockIndex, uint8_t *pFactors, uint8_t *pBlockError, uint8_t *pShift, const bool hasAlpha, size_t *pTotalBlockArea, const uint32_t errorFactor)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  memset(&ctx, 0, sizeof(ctx));

  ctx.pSourceImage = pIn;
  ctx.pBlockInfo = pBlockIndex;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelBlockError = 0x12 * (errorFactor);
  ctx.maxBlockPixelError = 0x1C * (errorFactor / 3); // error is multiplied by 0x10.
  ctx.maxPixelChannelBlockError = 0x40 * (errorFactor / 2);
  ctx.maxBlockExpandError = 0x20 * (errorFactor);
  ctx.maxPixelBitCrushError = 0x5 * (errorFactor / 2);
  ctx.maxBlockBitCrushError = 0x2 * (errorFactor / 2); // error is multiplied by 0x10.
  ctx.ditheringEnabled = true;

  if constexpr (limg_LuminanceDependentPixelError)
  {
    ctx.maxPixelBlockError *= 0x10;
    ctx.maxBlockPixelError *= 0x10;
    ctx.maxPixelBitCrushError *= 0x10;
    ctx.maxBlockBitCrushError *= 0x10;
  }

  if constexpr (limg_ColorDependentBlockError)
  {
    //ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    //ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 10 : 7); // technically correct but doesn't seem to produce similar results.
    ctx.maxPixelBlockError *= (size_t)(ctx.hasAlpha ? 6 : 4);
    ctx.maxBlockPixelError *= (size_t)(ctx.hasAlpha ? 6 : 4);
    ctx.maxPixelBitCrushError *= (size_t)(ctx.hasAlpha ? 10 : 7);
    ctx.maxBlockBitCrushError *= (size_t)(ctx.hasAlpha ? 10 : 7);
  }

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    ctx.maxPixelBlockError *= 0x1;
    ctx.maxBlockPixelError *= 0x1;
    ctx.maxPixelBitCrushError *= 0x1;
    ctx.maxBlockBitCrushError *= 0x1;
  }

  memset(ctx.pBlockInfo, 0, sizeof(uint32_t) * ctx.sizeX * ctx.sizeY);

  size_t blockFindStaticX = 0;
  size_t blockFindStaticY = 0;

  uint32_t blockIndex = 0;
  size_t accumBlockSize = 0;
  size_t accumBits = 0;
  size_t unweightedBits = 0;
  size_t accumBlockError = 0;
  size_t unweightedBlockError = 0;

  uint64_t ditherLast = 0xCA7F00D15BADF00D;

  size_t blockFactorsCapacity = 1024 * 1024;
  float *pBlockFactors = reinterpret_cast<float *>(malloc(blockFactorsCapacity * sizeof(float)));
  LIMG_ERROR_IF(pBlockFactors == nullptr, limg_error_MemoryAllocationFailure);

  while (true)
  {
    size_t ox, oy, rx, ry;
    limg_ui8_4 a, b;

    if (!limg_encode_find_block(&ctx, blockFindStaticX, blockFindStaticY, &ox, &oy, &rx, &ry, &a, &b))
      break;

    if (rx * ry > blockFactorsCapacity)
    {
      blockFactorsCapacity = (rx * ry + 1023) & ~(size_t)1023;
      pBlockFactors = reinterpret_cast<float *>(realloc(pBlockFactors, blockFactorsCapacity * sizeof(float)));
      LIMG_ERROR_IF(pBlockFactors == nullptr, limg_error_MemoryAllocationFailure);
    }

    size_t blockError = 0;
    size_t rangeSize = 0;
    limg_encode_check_area<true, true, false, false, true>(&ctx, ox, oy, rx, ry, a, b, pBlockFactors, &blockError, 0, &rangeSize);
    // Could be: `limg_encode_check_area<true, false, false, false, false>(&ctx, ox, oy, rx, ry, a, b, nullptr, 0, nullptr);` if we weren't writing block errors to a buffer.

    accumBlockError += blockError * 0x10;
    blockError = (blockError * 0x10) / rangeSize;
    unweightedBlockError += blockError;

#ifdef _DEBUG
    // If we're super accurate this may accidentally occur, since we calculate the error using the floating point average position and direction rather than the uint8_t min and max. Technically shouldn't matter since it only occurs due to a lack of precision. (at least if that's the reason why it's occuring...)
    //if (blockError > ctx.maxBlockPixelError)
    //  __debugbreak();
#endif

    float *pBFacs = pBlockFactors;
    uint8_t *pBFacsU8Start = reinterpret_cast<uint8_t *>(pBFacs);
    uint8_t *pBFacsU8 = pBFacsU8Start;

    for (size_t i = 0; i < rangeSize; i++)
    {
      *pBFacsU8 = (uint8_t)limgClamp((int32_t)(*pBFacs * (float_t)0xFF + 0.5f), 0, 0xFF);
      pBFacs++;
      pBFacsU8++;
    }

    const uint8_t shift = limg_encode_find_shift_for_block(&ctx, ox, oy, rx, ry, pBFacsU8Start, a, b);
    accumBits += (8 - shift) * rangeSize;
    unweightedBits += (8 - shift);

    if (shift)
    {
      if (ctx.ditheringEnabled)
      {
        ditherLast = limg_encode_dither(shift, rangeSize, ditherLast, pBFacsU8Start);
      }
      else
      {
        for (size_t i = 0; i < rangeSize; i++)
          pBFacsU8Start[i] >>= shift;
      }
    }

    for (size_t y = 0; y < ry; y++)
    {
      uint32_t *pBlockIndexLine = ctx.pBlockInfo + (y + oy) * ctx.sizeX + ox;
      uint32_t *pALine = pA + (y + oy) * ctx.sizeX + ox;
      uint32_t *pBLine = pB + (y + oy) * ctx.sizeX + ox;
      uint8_t *pFactorsLine = pFactors + (y + oy) * ctx.sizeX + ox;
      uint8_t *pBlockErrorLine = pBlockError + (y + oy) * ctx.sizeX + ox;
      uint8_t *pShiftLine = pShift + (y + oy) * ctx.sizeX + ox;

      for (size_t x = 0; x < rx; x++)
      {
        *pBlockIndexLine = blockIndex | BlockInfo_InUse;
        pBlockIndexLine++;

        *pALine = *reinterpret_cast<const uint32_t *>(&a);
        pALine++;

        *pBLine = *reinterpret_cast<const uint32_t *>(&b);
        pBLine++;

        *pFactorsLine = pBFacsU8Start[x + y * rx] << shift;
        pFactorsLine++;

        *pBlockErrorLine = (uint8_t)limgMin(blockError >> 3, 0xFFULL);
        pBlockErrorLine++;

        *pShiftLine = (uint8_t)(1 << shift);
        pShiftLine++;
      }
    }

    uint32_t *pDecodedStart = pDecoded + oy * ctx.sizeX + ox;

    if (hasAlpha)
      limg_decode_block_from_factors<4>(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, shift, a, b);
    else
      limg_decode_block_from_factors<3>(pDecodedStart, sizeX, rx, ry, pBFacsU8Start, shift, a, b);

    blockIndex++;
    accumBlockSize += (rx * ry);
  }

  *pTotalBlockArea = accumBlockSize;

#ifdef PRINT_TEST_OUTPUT
  printf("\n%" PRIu32 " Blocks generated.\n%5.3f %% Coverage\nAverage Size: %5.3f Pixels [(%5.3f px)^2].\nMinimum Block Size: %" PRIu64 "\nBlock Size Grow Step: %" PRIu64 "\nAverage Block Error: %5.3f (Unweighted: %5.3f)\nAverage Block Bits: %5.3f (Unweighted: %5.3f)\n\n", blockIndex, (accumBlockSize / (double)(sizeX * sizeY)) * 100.0, accumBlockSize / (double)blockIndex, sqrt(accumBlockSize / (double)blockIndex), limg_MinBlockSize, limg_BlockExpandStep, accumBlockError / (double)accumBlockSize, unweightedBlockError / (double)blockIndex, accumBits / (double)accumBlockSize, unweightedBits / (double)blockIndex);

  if constexpr (limg_DiagnoseCulprits)
  {
    printf("CULPRIT info: (%" PRIu64 " culprits)\n", ctx.culprits);
    printf("PixelBlockErrorCulprit: %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBlockError, (ctx.culpritWasPixelBlockError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBlockError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("BlockPixelError       : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockPixelError, (ctx.culpritWasBlockPixelError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockPixelError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("PixelChannelBlockError: %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelChannelBlockError, (ctx.culpritWasPixelChannelBlockError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelChannelBlockError / (double)(ctx.culpritWasPixelBlockError + ctx.culpritWasBlockPixelError + ctx.culpritWasPixelChannelBlockError)) * 100.0);
    printf("BlockExpandError      : %8" PRIu64 " (%7.3f%%)\n", ctx.culpritWasBlockExpandError, (ctx.culpritWasBlockExpandError / (double)ctx.culprits) * 100.0);
    printf("PixelBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBitCrushError, (ctx.culpritWasPixelBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    printf("BlockBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockBitCrushError, (ctx.culpritWasBlockBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    puts("");
  }

#endif

  for (size_t i = 0; i < sizeX * sizeY; i++)
    if (!(ctx.pBlockInfo[i] & BlockInfo_InUse))
      pDecoded[i] = pIn[i];

  if (!hasAlpha)
    for (size_t i = 0; i < sizeX * sizeY; i++)
      pDecoded[i] |= 0xFF000000;

  goto epilogue;

epilogue:
  limgFreePtr(&pBlockFactors);

  return result;
}

template <size_t channels, bool store_factors_shift, bool decode, bool store_accum_bits>
void limg_encode3d_test_y_range(limg_encode_context *pCtx, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, size_t accum_bits[3 + 3 * 8], const size_t y_start, const size_t y_end)
{
  float scratchBuffer[limg_MinBlockSize * limg_MinBlockSize * 4]; // technically `* 3` or `* 4` depending on `hasAlpha` being either `false` or `true`.
  uint8_t scratch_u8[limg_MinBlockSize * limg_MinBlockSize * 3];

  uint64_t ditherLast = 0xCA7F00D15BADF00D;

  for (size_t y = y_start; y < y_end; y += limg_MinBlockSize)
  {
    for (size_t x = 0; x < pCtx->sizeX; x += limg_MinBlockSize)
    {
      const size_t rx = limgMin(pCtx->sizeX - x, limg_MinBlockSize);
      const size_t ry = limgMin(pCtx->sizeY - y, limg_MinBlockSize);

      limg_encode_decomposition_state encode_state;
      limg_encode_sum_to_decomposition_state<channels>(pCtx, x, y, rx, ry, encode_state);

      limg_encode_3d_output<channels> decomposition;
      limg_encode_get_block_factors_accurate_from_state_3d<channels>(pCtx, x, y, rx, ry, decomposition, encode_state, scratchBuffer);

      limg_color_error_state_3d<channels> color_error_state;
      limg_init_color_error_state_3d<channels>(decomposition, color_error_state);

      uint8_t *pAu8 = scratch_u8;
      uint8_t *pBu8 = pAu8 + limg_MinBlockSize * limg_MinBlockSize;
      uint8_t *pCu8 = pBu8 + limg_MinBlockSize * limg_MinBlockSize;

      limg_color_error_state_3d_get_all_factors(pCtx, decomposition, color_error_state, x, y, rx, ry, pAu8, pBu8, pCu8);

      uint8_t shift[3] = { 0, 0, 0 };

      // Try best guesses.
      if (pCtx->guessCrush)
        limg_encode_guess_shift_for_block_3d<channels>(pCtx, x, y, rx, ry, decomposition, pAu8, pBu8, pCu8, shift);
      
      limg_encode_find_shift_for_block_3d<channels>(pCtx, x, y, rx, ry, decomposition, pAu8, pBu8, pCu8, shift);

      const size_t rangeSize = rx * ry;

      if (shift[0] || shift[1] || shift[2])
      {
        if (pCtx->ditheringEnabled)
        {
          if (shift[0] && shift[0] != 8)
            ditherLast = limg_encode_dither(shift[0], rangeSize, ditherLast, pAu8);

          if (shift[1] && shift[1] != 8)
            ditherLast = limg_encode_dither(shift[1], rangeSize, ditherLast, pBu8);

          if (shift[2] && shift[2] != 8)
            ditherLast = limg_encode_dither(shift[2], rangeSize, ditherLast, pCu8);
        }
        else
        {
          // it's probably faster to not `if` the shift values individually.
          for (size_t i = 0; i < rangeSize; i++)
          {
            pAu8[i] >>= shift[0];
            pBu8[i] >>= shift[1];
            pCu8[i] >>= shift[2];
          }
        }

        if constexpr (store_accum_bits)
        {
          for (size_t i = 0; i < 3; i++)
          {
            accum_bits[i] += (8 - shift[i]) * rangeSize;
            accum_bits[3 + i * 9 + shift[i]] += rangeSize;
          }
        }
      }
      else
      {
        if constexpr (store_accum_bits)
        {
          for (size_t i = 0; i < 3; i++)
          {
            accum_bits[i] += 8 * rangeSize;
            accum_bits[3 + i * 9] += rangeSize;
          }
        }
      }

      if constexpr (store_factors_shift)
      {
        const uint8_t bit_to_pattern[9] = { 0, 0x22, 0x44, 0x66, 0x88, 0xAA, 0xCC, 0xEE, 0xFF };

        const uint32_t shift_val = 0xFF000000 | (bit_to_pattern[shift[0]] << 16) | (bit_to_pattern[shift[1]] << 8) | bit_to_pattern[shift[2]];

        uint32_t colAMin = 0;
        uint32_t colAMax = 0;
        uint32_t colBMin = 0;
        uint32_t colBMax = 0;
        uint32_t colCMin = 0;
        uint32_t colCMax = 0;

        uint8_t channelOffset[4] = { 0, 8, 16, 24 };

        for (size_t i = 0; i < channels; i++)
        {
          colAMin |= (uint32_t)limgClamp((int32_t)decomposition.dirA_min[i], 0, 0xFF) << channelOffset[i];
          colAMax |= (uint32_t)limgClamp((int32_t)decomposition.dirA_max[i], 0, 0xFF) << channelOffset[i];
          colBMin |= (uint32_t)limgClamp((int32_t)decomposition.dirB_offset[i] + 0x80, 0, 0xFF) << channelOffset[i];
          colBMax |= (uint32_t)limgClamp((int32_t)decomposition.dirB_mag[i] + 0x80, 0, 0xFF) << channelOffset[i];
          colCMin |= (uint32_t)limgClamp((int32_t)decomposition.dirC_offset[i] + 0x80, 0, 0xFF) << channelOffset[i];
          colCMax |= (uint32_t)limgClamp((int32_t)decomposition.dirC_mag[i] + 0x80, 0, 0xFF) << channelOffset[i];
        }

        if constexpr (channels == 3)
        {
          colAMin |= 0xFF000000;
          colAMax |= 0xFF000000;
          colBMin |= 0xFF000000;
          colBMax |= 0xFF000000;
          colCMin |= 0xFF000000;
          colCMax |= 0xFF000000;
        }

        for (size_t oy = 0; oy < ry; oy++)
        {
          uint8_t *pFactorsALine = pFactorsA + (y + oy) * pCtx->sizeX + x;
          uint8_t *pFactorsBLine = pFactorsB + (y + oy) * pCtx->sizeX + x;
          uint8_t *pFactorsCLine = pFactorsC + (y + oy) * pCtx->sizeX + x;
          uint32_t *pShiftLine = pShiftABCX + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColAMinLine = pColAMin + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColAMaxLine = pColAMax + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColBMinLine = pColBMin + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColBMaxLine = pColBMax + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColCMinLine = pColCMin + (y + oy) * pCtx->sizeX + x;
          uint32_t *pColCMaxLine = pColCMax + (y + oy) * pCtx->sizeX + x;

          for (size_t ox = 0; ox < rx; ox++)
          {
            *pFactorsALine = *pAu8 << shift[0];
            pFactorsALine++;
            pAu8++;

            *pFactorsBLine = *pBu8 << shift[1];
            pFactorsBLine++;
            pBu8++;

            *pFactorsCLine = *pCu8 << shift[2];
            pFactorsCLine++;
            pCu8++;

            *pShiftLine = shift_val;
            pShiftLine++;

            *pColAMinLine = colAMin;
            pColAMinLine++;

            *pColAMaxLine = colAMax;
            pColAMaxLine++;

            *pColBMinLine = colBMin;
            pColBMinLine++;

            *pColBMaxLine = colBMax;
            pColBMaxLine++;

            *pColCMinLine = colCMin;
            pColCMinLine++;

            *pColCMaxLine = colCMax;
            pColCMaxLine++;

          }
        }

        pAu8 = scratch_u8;
        pBu8 = pAu8 + limg_MinBlockSize * limg_MinBlockSize;
        pCu8 = pBu8 + limg_MinBlockSize * limg_MinBlockSize;
      }

      if constexpr (decode)
      {
        uint32_t *pDecodedStart = pDecoded + y * pCtx->sizeX + x;

        limg_decode_block_from_factors_3d<channels>(pDecodedStart, pCtx->sizeX, rx, ry, pAu8, pBu8, pCu8, decomposition, shift);
      }
    }
  }
}

template <size_t channels>
void limg_encode3d_test_(limg_encode_context *pCtx, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, size_t accum_bits[3 + 3 * 8], limg_thread_pool *pThreadPool)
{
  if (pThreadPool == nullptr)
  {
    limg_encode3d_test_y_range<channels, true, true, true>(pCtx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, accum_bits, 0, pCtx->sizeY);
  }
  else
  {
    size_t thread_count = limg_thread_pool_thread_count(pThreadPool) * 4;
    size_t y_range = ((pCtx->sizeY / limg_MinBlockSize) / thread_count) * limg_MinBlockSize;

    if (y_range == 0)
    {
      thread_count = limg_thread_pool_thread_count(pThreadPool);
      y_range = ((pCtx->sizeY / limg_MinBlockSize) / thread_count) * limg_MinBlockSize;
    }

    size_t y_start = 0;

    for (size_t i = 1; i < thread_count; i++)
    {
      const size_t start = y_start;
      const size_t end = y_start + y_range;
      y_start += y_range;

      limg_thread_pool_add(pThreadPool, [&, start, end]() { limg_encode3d_test_y_range<channels, true, true, true>(pCtx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, accum_bits, start, end); });
    }

    limg_thread_pool_add(pThreadPool, [&]() { limg_encode3d_test_y_range<channels, true, true, true>(pCtx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, accum_bits, y_start, pCtx->sizeY); });

    limg_thread_pool_await(pThreadPool);
  }
}

template <size_t channels>
void limg_encode3d_test_perf_(limg_encode_context *pCtx, limg_thread_pool *pThreadPool)
{
  if (pThreadPool == nullptr)
  {
    limg_encode3d_test_y_range<channels, false, false, false>(pCtx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, pCtx->sizeY);
  }
  else
  {
    size_t thread_count = limg_thread_pool_thread_count(pThreadPool) * 4;
    size_t y_range = ((pCtx->sizeY / limg_MinBlockSize) / thread_count) * limg_MinBlockSize;

    if (y_range == 0)
    {
      thread_count = limg_thread_pool_thread_count(pThreadPool);
      y_range = ((pCtx->sizeY / limg_MinBlockSize) / thread_count) * limg_MinBlockSize;
    }

    size_t y_start = 0;

    for (size_t i = 1; i < thread_count; i++)
    {
      const size_t start = y_start;
      const size_t end = y_start + y_range;
      y_start += y_range;

      limg_thread_pool_add(pThreadPool, [&, start, end]() { limg_encode3d_test_y_range<channels, false, false, false>(pCtx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, start, end); });
    }

    limg_thread_pool_add(pThreadPool, [&]() { limg_encode3d_test_y_range<channels, false, false, false>(pCtx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, y_start, pCtx->sizeY); });

    limg_thread_pool_await(pThreadPool);
  }
}

limg_result limg_encode3d_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  memset(&ctx, 0, sizeof(ctx));

  ctx.pSourceImage = pIn;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelBlockError = 0x12 * (errorFactor);
  ctx.maxBlockPixelError = 0x1C * (errorFactor / 3); // error is multiplied by 0x10.
  ctx.maxPixelChannelBlockError = 0x40 * (errorFactor / 2);
  ctx.maxBlockExpandError = 0x20 * (errorFactor);
  ctx.maxPixelBitCrushError = 0x6 * (errorFactor / 2);
  ctx.maxBlockBitCrushError = 0x4 * (errorFactor / 2); // error is multiplied by 0x10.
  ctx.ditheringEnabled = true;
  ctx.fastBitCrush = fastBitCrushing;
  ctx.guessCrush = true;

  if constexpr (limg_LuminanceDependentPixelError)
  {
    ctx.maxPixelBlockError *= 0x10;
    ctx.maxBlockPixelError *= 0x10;
    ctx.maxPixelBitCrushError *= 0x10;
    ctx.maxBlockBitCrushError *= 0x10;
  }

  if constexpr (limg_ColorDependentBlockError)
  {
    ctx.maxPixelBlockError *= 4;
    ctx.maxBlockPixelError *= 4;
    ctx.maxPixelBitCrushError *= 7;
    ctx.maxBlockBitCrushError *= 7;
  }

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    ctx.maxPixelBlockError *= 0x1;
    ctx.maxBlockPixelError *= 0x1;
    ctx.maxPixelBitCrushError *= 0x1;
    ctx.maxBlockBitCrushError *= 0x1;
  }

  _DetectCPUFeatures();

  size_t accum_bits[3 + 3 * 9] = { 0 };

  if (ctx.hasAlpha)
    limg_encode3d_test_<4>(&ctx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, accum_bits, pThreadPool);
  else
    limg_encode3d_test_<3>(&ctx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, accum_bits, pThreadPool);

#ifdef PRINT_TEST_OUTPUT
  const size_t totalPixels = ctx.sizeX * ctx.sizeY;

  printf("\nAverage Block Bits: %5.3f (A: %5.3f | B: %5.3f | C: %5.3f)\n\n", (accum_bits[0] + accum_bits[1] + accum_bits[2]) / (double)totalPixels, accum_bits[0] / (double)totalPixels, accum_bits[1] / (double)totalPixels, accum_bits[2] / (double)totalPixels);

  for (size_t i = 0; i < 9; i++)
    printf(" %" PRIu64 " bit   ", 8 - i);

  for (size_t i = 0; i < 3; i++)
  {
    puts("");

    for (size_t j = 0; j < 9; j++)
      printf("%7.4f  ", accum_bits[3 + i * 9 + j] * 100.0 / (double)totalPixels);
  }

  puts("\n");

  if constexpr (limg_DiagnoseCulprits)
  {
    printf("CULPRIT info: (%" PRIu64 " culprits)\n", ctx.culprits);
    printf("PixelBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBitCrushError, (ctx.culpritWasPixelBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    printf("BlockBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockBitCrushError, (ctx.culpritWasBlockBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    puts("");
  }

#endif

  goto epilogue;

epilogue:

  return result;
}

limg_result limg_encode3d_test_perf(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing)
{
  limg_result result = limg_success;

  limg_encode_context ctx;
  memset(&ctx, 0, sizeof(ctx));

  ctx.pSourceImage = pIn;
  ctx.sizeX = sizeX;
  ctx.sizeY = sizeY;
  ctx.hasAlpha = hasAlpha;
  ctx.maxPixelBlockError = 0x12 * (errorFactor);
  ctx.maxBlockPixelError = 0x1C * (errorFactor / 3); // error is multiplied by 0x10.
  ctx.maxPixelChannelBlockError = 0x40 * (errorFactor / 2);
  ctx.maxBlockExpandError = 0x20 * (errorFactor);
  ctx.maxPixelBitCrushError = 0x6 * (errorFactor / 2);
  ctx.maxBlockBitCrushError = 0x4 * (errorFactor / 2); // error is multiplied by 0x10.
  ctx.ditheringEnabled = true;
  ctx.fastBitCrush = fastBitCrushing;
  ctx.guessCrush = true;

  if constexpr (limg_LuminanceDependentPixelError)
  {
    ctx.maxPixelBlockError *= 0x10;
    ctx.maxBlockPixelError *= 0x10;
    ctx.maxPixelBitCrushError *= 0x10;
    ctx.maxBlockBitCrushError *= 0x10;
  }

  if constexpr (limg_ColorDependentBlockError)
  {
    ctx.maxPixelBlockError *= 4;
    ctx.maxBlockPixelError *= 4;
    ctx.maxPixelBitCrushError *= 7;
    ctx.maxBlockBitCrushError *= 7;
  }

  if constexpr (limg_RetrievePreciseDecomposition == 2)
  {
    ctx.maxPixelBlockError *= 0x1;
    ctx.maxBlockPixelError *= 0x1;
    ctx.maxPixelBitCrushError *= 0x1;
    ctx.maxBlockBitCrushError *= 0x1;
  }

  _DetectCPUFeatures();

  if (ctx.hasAlpha)
    limg_encode3d_test_perf_<4>(&ctx, pThreadPool);
  else
    limg_encode3d_test_perf_<3>(&ctx, pThreadPool);

  goto epilogue;

epilogue:

  return result;
}

double limg_compare(const uint32_t *pImageA, const uint32_t *pImageB, const size_t sizeX, const size_t sizeY, const bool hasAlpha, double *pMeanSquaredError, double *pMaxPossibleSquaredError)
{
  size_t error = 0;
  size_t maxError;

  const limg_ui8_4 *pA = reinterpret_cast<const limg_ui8_4 *>(pImageA);
  const limg_ui8_4 *pB = reinterpret_cast<const limg_ui8_4 *>(pImageB);

  const limg_ui8_4 min = { { { 0, 0, 0, 0 } } };
  const limg_ui8_4 max = { { { 0xFF, 0xFF, 0xFF, 0xFF } } };

  if (hasAlpha)
  {
    maxError = limg_color_error<4>(min, max);

    for (size_t i = 0; i < sizeX * sizeY; i++)
      error += limg_color_error<4>(pA[i], pB[i]);
  }
  else
  {
    maxError = limg_color_error<3>(min, max);

    for (size_t i = 0; i < sizeX * sizeY; i++)
      error += limg_color_error<3>(pA[i], pB[i]);
  }

  const double mse = error / (double)(sizeX * sizeY);
  const double psnr = 10.0 * log10((double)maxError / mse);

  if (pMeanSquaredError != nullptr)
    *pMeanSquaredError = mse;

  if (pMaxPossibleSquaredError != nullptr)
    *pMaxPossibleSquaredError = (double)maxError;

  return psnr;
}
