#ifndef limg_factorization_h__
#define limg_factorization_h__

#include "limg_internal.h"
#include "limg_simd.h"

//////////////////////////////////////////////////////////////////////////

template <size_t channels>
static LIMG_INLINE void limg_color_error_state_3d_get_factors(const float color[channels], const limg_encode_3d_output<channels> &in, const limg_color_error_state_3d<channels> &state, float &fac_a, float &fac_b, float &fac_c)
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
LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_3_sse41(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_3d_output<3> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 3;

  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128 sign_bit = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
  const __m128 inv_sign_bit = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
  const __m128 zero_alpha = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
  const __m128 preferenceBias = _mm_set_ps(0, FLT_EPSILON * 1, FLT_EPSILON * 2, FLT_EPSILON * 3);

  uint32_t avg[channels];

  for (size_t i = 0; i < channels; i++)
    avg[i] = (uint32_t)state.sum[i];

  const __m128 inv_count_ = _mm_set1_ps(1.f / (float)(size));
  const __m128 avg_ = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(avg))), inv_count_);

  __m128 diff_xi_dirA_ = _mm_setzero_ps();

  {
    for (size_t i = 0; i < size; i++)
    {
      const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
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
      for (size_t i = 0; i < size; i++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
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

      diff_xi_dirB_ = _mm_mul_ps(diff_xi_dirB_, inv_count_);
    }

    // diff_xi_dirC = diff_xi_dirA x diff_xi_dirB
    {
      const __m128 shufA = _mm_shuffle_ps(diff_xi_dirA_, diff_xi_dirA_, _MM_SHUFFLE(3, 0, 2, 1));
      const __m128 shufB = _mm_shuffle_ps(diff_xi_dirB_, diff_xi_dirB_, _MM_SHUFFLE(3, 1, 0, 2));
      const __m128 mul0 = _mm_mul_ps(shufA, shufB);
      const __m128 shufA1 = _mm_shuffle_ps(shufA, shufA, _MM_SHUFFLE(3, 0, 2, 1));
      const __m128 shufB1 = _mm_shuffle_ps(shufB, shufB, _MM_SHUFFLE(3, 1, 0, 2));

      diff_xi_dirC_ = _mm_sub_ps(mul0, _mm_mul_ps(shufA1, shufB1));
    }

    pEstimate = reinterpret_cast<__m128 *>(pScratch);

    const __m128 inv_length_diff_xi_dirB = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirB_, diff_xi_dirB_, 0x7F)); // should be 0xFF with 4 channels.
    const __m128 inv_length_diff_xi_dirC = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirC_, diff_xi_dirC_, 0x7F)); // should be 0xFF with 4 channels.

    min_dirC = min_dirB = _mm_set1_ps(FLT_MAX);
    max_dirC = max_dirB = _mm_set1_ps(-FLT_MAX);

    {
      for (size_t i = 0; i < size; i++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));

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

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_4_sse41(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_3d_output<4> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 4;

  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  const __m128 sign_bit = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
  const __m128 inv_sign_bit = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
  const __m128 preferenceBias = _mm_set_ps(0, FLT_EPSILON * 1, FLT_EPSILON * 2, FLT_EPSILON * 3);

  uint32_t avg[channels];

  for (size_t i = 0; i < channels; i++)
    avg[i] = (uint32_t)state.sum[i];

  const __m128 inv_count_ = _mm_set1_ps(1.f / (float)(size));
  const __m128 avg_ = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(avg))), inv_count_);

  __m128 diff_xi_dirA_ = _mm_setzero_ps();

  {
    for (size_t i = 0; i < size; i++)
    {
      const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
      const __m128 corrected = _mm_sub_ps(px, avg_);

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

        const __m128 invLength = _mm_xor_ps(flip_sign, _mm_rsqrt_ps(_mm_dp_ps(corrected, corrected, 0xFF)));
        const __m128 invLength4 = _mm_shuffle_ps(invLength, invLength, _MM_SHUFFLE(0, 0, 0, 0));

        const __m128 val = _mm_mul_ps(corrected, invLength4);
        diff_xi_dirA_ = _mm_add_ps(diff_xi_dirA_, val);

      }
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

    const __m128 inv_length_diff_xi_dirA = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirA_, diff_xi_dirA_, 0xFF));

    {
      for (size_t i = 0; i < size; i++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
        const __m128 lineOriginToPx = _mm_sub_ps(px, avg_);

        const __m128 facA = _mm_mul_ps(_mm_dp_ps(lineOriginToPx, diff_xi_dirA_, 0xFF), inv_length_diff_xi_dirA);
        const __m128 facA_full = _mm_shuffle_ps(facA, facA, _MM_SHUFFLE(0, 0, 0, 0));

        min_dirA = _mm_min_ps(min_dirA, facA);
        max_dirA = _mm_max_ps(max_dirA, facA);

        const __m128 estimateA = _mm_add_ps(avg_, _mm_mul_ps(facA_full, diff_xi_dirA_));
        const __m128 error_vec_dirA = _mm_sub_ps(px, estimateA);

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

          const __m128 invLength = _mm_xor_ps(flip_sign, _mm_rsqrt_ps(_mm_dp_ps(error_vec_dirA, error_vec_dirA, 0xFF)));
          const __m128 invLength4 = _mm_shuffle_ps(invLength, invLength, _MM_SHUFFLE(0, 0, 0, 0));

          diff_xi_dirB_ = _mm_add_ps(diff_xi_dirB_, _mm_mul_ps(error_vec_dirA, invLength4));
        }
      }

      diff_xi_dirB_ = _mm_mul_ps(diff_xi_dirB_, inv_count_);
    }

    pEstimate = reinterpret_cast<__m128 *>(pScratch);

    const __m128 inv_length_diff_xi_dirB = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirB_, diff_xi_dirB_, 0xFF));

    min_dirC = min_dirB = _mm_set1_ps(FLT_MAX);
    max_dirC = max_dirB = _mm_set1_ps(-FLT_MAX);

    {
      for (size_t i = 0; i < size; i++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
        const __m128 estimateA = _mm_loadu_ps(reinterpret_cast<float *>(pEstimate));
        const __m128 lineOriginToPx = _mm_sub_ps(px, estimateA);

        const __m128 facB = _mm_mul_ps(_mm_dp_ps(lineOriginToPx, diff_xi_dirB_, 0xFF), inv_length_diff_xi_dirB);
        const __m128 facB_full = _mm_shuffle_ps(facB, facB, _MM_SHUFFLE(0, 0, 0, 0));

        min_dirB = _mm_min_ps(min_dirB, facB);
        max_dirB = _mm_max_ps(max_dirB, facB);

        const __m128 estimateB = _mm_add_ps(estimateA, _mm_mul_ps(facB_full, diff_xi_dirB_));
        const __m128 error_vec_dirAB = _mm_sub_ps(px, estimateB);

        _mm_storeu_ps(reinterpret_cast<float *>(pEstimate), estimateB);
        pEstimate++;

        const int32_t mask = _mm_movemask_ps(_mm_cmpeq_ps(error_vec_dirAB, _mm_setzero_ps()));

        if (0b1111 != mask) // can we use a different kind of `cmp` here? maybe `epi32` works fine as well?
        {
          const __m128 minBiased = _mm_sub_ps(error_vec_dirAB, preferenceBias);
          const __m128 maxBiased = _mm_add_ps(error_vec_dirAB, preferenceBias);
          const __m128 min_23 = _mm_shuffle_ps(minBiased, minBiased, _MM_SHUFFLE(0, 0, 3, 2));
          const __m128 max_23 = _mm_shuffle_ps(maxBiased, maxBiased, _MM_SHUFFLE(0, 0, 3, 2));
          const __m128 half_min = _mm_min_ps(minBiased, min_23);
          const __m128 half_max = _mm_max_ps(maxBiased, max_23);
          const __m128 abs_min = _mm_and_ps(inv_sign_bit, _mm_min_ps(half_min, _mm_shuffle_ps(half_min, half_min, _MM_SHUFFLE(0, 0, 0, 1))));
          const __m128 max = _mm_max_ps(half_max, _mm_shuffle_ps(half_max, half_max, _MM_SHUFFLE(0, 0, 0, 1)));
          const __m128 flip_sign = _mm_and_ps(sign_bit, _mm_cmpgt_ps(abs_min, max));

          const __m128 invLength = _mm_xor_ps(flip_sign, _mm_rsqrt_ps(_mm_dp_ps(error_vec_dirAB, error_vec_dirAB, 0xFF)));
          const __m128 invLength4 = _mm_shuffle_ps(invLength, invLength, _MM_SHUFFLE(0, 0, 0, 0));

          diff_xi_dirC_ = _mm_add_ps(diff_xi_dirC_, _mm_mul_ps(error_vec_dirAB, invLength4));
        }
      }
      
      diff_xi_dirC_ = _mm_mul_ps(diff_xi_dirC_, inv_count_);
    }

    const __m128 inv_length_diff_xi_dirC = _mm_div_ps(_mm_set1_ps(1.f), _mm_dp_ps(diff_xi_dirC_, diff_xi_dirC_, 0xFF));

    pEstimate = reinterpret_cast<__m128 *>(pScratch);

    {
      for (size_t i = 0; i < size; i++)
      {
        const __m128 px = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pPixels[i]))));
        const __m128 estimateAB = _mm_loadu_ps(reinterpret_cast<float *>(pEstimate));
        const __m128 lineOriginToPx = _mm_sub_ps(px, estimateAB);

        const __m128 facC = _mm_mul_ps(_mm_dp_ps(lineOriginToPx, diff_xi_dirC_, 0xFF), inv_length_diff_xi_dirC);

        min_dirC = _mm_min_ps(min_dirC, facC);
        max_dirC = _mm_max_ps(max_dirC, facC);
      }
    }
  }

  const __m128i _32_to_16 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0);

  __m128i dirAmin = _mm_cvtps_epi32(_mm_add_ps(avg_, _mm_mul_ps(_mm_shuffle_ps(min_dirA, min_dirA, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirA_)));
  __m128i dirAmax = _mm_cvtps_epi32(_mm_add_ps(avg_, _mm_mul_ps(_mm_shuffle_ps(max_dirA, max_dirA, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirA_)));
  __m128i dirBmin = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(min_dirB, min_dirB, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirB_));
  __m128i dirBmax = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(max_dirB, max_dirB, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirB_));
  __m128i dirCmin = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(min_dirC, min_dirC, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirC_));
  __m128i dirCmax = _mm_cvtps_epi32(_mm_mul_ps(_mm_shuffle_ps(max_dirC, max_dirC, _MM_SHUFFLE(0, 0, 0, 0)), diff_xi_dirC_));

  dirAmin = _mm_shuffle_epi8(dirAmin, _32_to_16);
  dirAmax = _mm_shuffle_epi8(dirAmax, _32_to_16);
  dirBmin = _mm_shuffle_epi8(dirBmin, _32_to_16);
  dirBmax = _mm_shuffle_epi8(dirBmax, _32_to_16);
  dirCmin = _mm_shuffle_epi8(dirCmin, _32_to_16);
  dirCmax = _mm_shuffle_epi8(dirCmax, _32_to_16);

  _mm_storeu_ps(out.avg, avg_);


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
  *reinterpret_cast<uint64_t *>(out.dirA_min) = _mm_extract_epi64(dirAmin, 0);
  *reinterpret_cast<uint64_t *>(out.dirA_max) = _mm_extract_epi64(dirAmax, 0);
  *reinterpret_cast<uint64_t *>(out.dirB_offset) = _mm_extract_epi64(dirBmin, 0);
  *reinterpret_cast<uint64_t *>(out.dirB_mag) = _mm_extract_epi64(dirBmax, 0);
  *reinterpret_cast<uint64_t *>(out.dirC_offset) = _mm_extract_epi64(dirCmin, 0);
  *reinterpret_cast<uint64_t *>(out.dirC_mag) = _mm_extract_epi64(dirCmax, 0);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_3(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_3d_output<3> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 3;

  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pPixels);

  float avg[channels];

  const float inv_count = 1.f / (float)(size);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi_dirA[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] = 0;

  for (size_t i = 0; i < size; i++)
  {
    const limg_ui8_4 px = pStart[i];

    float corrected[channels];
    float max_abs = 0;
    size_t max_idx = 0;

    for (size_t j = 0; j < channels; j++)
    {
      corrected[j] = (float)px[j] - avg[j];

      const float abs_val = fabsf(corrected[j]);

      if (abs_val > max_abs)
      {
        max_abs = abs_val;
        max_idx = j;
      }
    }

    if (max_abs != 0)
    {
      max_abs = corrected[max_idx];

      float lengthSquared = 0;

      for (size_t j = 0; j < channels; j++)
        lengthSquared += corrected[j] * corrected[j];

      const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

      for (size_t j = 0; j < channels; j++)
        diff_xi_dirA[j] += corrected[j] * inv_length;
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

    for (size_t i = 0; i < size; i++)
    {
      const limg_ui8_4 px = pStart[i];

      float lineOriginToPx[channels];

      for (size_t j = 0; j < channels; j++)
        lineOriginToPx[j] = (float)px[j] - avg[j];

      const float facA = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirA) * inv_length_diff_xi_dirA;

      min_dirA = limgMin(min_dirA, facA);
      max_dirA = limgMax(max_dirA, facA);

      float error_vec_dirA[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t j = 0; j < channels; j++)
      {
        pEstimate[j] = avg[j] + facA * diff_xi_dirA[j];
        error_vec_dirA[j] = (float)px[j] - pEstimate[j];

        const float abs_val = fabsf(error_vec_dirA[j]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = j;
        }
      }

      pEstimate += channels;

      if (max_abs != 0)
      {
        max_abs = error_vec_dirA[max_idx];

        float lengthSquared = 0;

        for (size_t j = 0; j < channels; j++)
          lengthSquared += error_vec_dirA[j] * error_vec_dirA[j];

        const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

        for (size_t j = 0; j < channels; j++)
          diff_xi_dirB[j] += error_vec_dirA[j] * inv_length;
      }
    }

    for (size_t j = 0; j < channels; j++)
      diff_xi_dirB[j] *= inv_count;

    limg_cross(diff_xi_dirA, diff_xi_dirB, diff_xi_dirC);

    pEstimate = pScratch;

    const float inv_length_diff_xi_dirB = 1.f / limg_dot<float, channels>(diff_xi_dirB, diff_xi_dirB);
    const float inv_length_diff_xi_dirC = 1.f / limg_dot<float, channels>(diff_xi_dirC, diff_xi_dirC);

    min_dirB = FLT_MAX;
    max_dirB = -FLT_MAX;
    min_dirC = FLT_MAX;
    max_dirC = -FLT_MAX;

    for (size_t i = 0; i < size; i++)
    {
      const limg_ui8_4 px = pStart[i];

      float lineOriginToPx[channels];

      for (size_t j = 0; j < channels; j++)
        lineOriginToPx[j] = px[j] - pEstimate[j];

      const float facB = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirB) * inv_length_diff_xi_dirB;

      min_dirB = limgMin(min_dirB, facB);
      max_dirB = limgMax(max_dirB, facB);

      float error_vec_dirAB[channels];

      for (size_t j = 0; j < channels; j++)
      {
        const float estimage = (pEstimate[j] + facB * diff_xi_dirB[j]);
        error_vec_dirAB[j] = (float)px[j] - estimage;
      }

      const float facC = limg_dot<float, channels>(error_vec_dirAB, diff_xi_dirC) * inv_length_diff_xi_dirC;

      min_dirC = limgMin(min_dirC, facC);
      max_dirC = limgMax(max_dirC, facC);

      pEstimate += channels;
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
LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d_4(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_3d_output<4> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  constexpr size_t channels = 4;

  const limg_ui8_4 *pStart = reinterpret_cast<const limg_ui8_4 *>(pPixels);

  float avg[channels];

  const float inv_count = 1.f / (float)(size);

  for (size_t i = 0; i < channels; i++)
    avg[i] = state.sum[i] * inv_count;

  // Calculate average yi/xi, zi/xi (and potentially wi/xi).
  // but rather than always choosing xi, we choose the largest of xi, yi, zi (,wi).
  float diff_xi_dirA[channels];

  for (size_t i = 0; i < channels; i++)
    diff_xi_dirA[i] = 0;

  for (size_t i = 0; i < size; i++)
  {
    const limg_ui8_4 px = pStart[i];

    float corrected[channels];
    float max_abs = 0;
    size_t max_idx = 0;

    for (size_t j = 0; j < channels; j++)
    {
      corrected[j] = (float)px[j] - avg[j];

      const float abs_val = fabsf(corrected[j]);

      if (abs_val > max_abs)
      {
        max_abs = abs_val;
        max_idx = j;
      }
    }

    if (max_abs != 0)
    {
      max_abs = corrected[max_idx];

      float lengthSquared = 0;

      for (size_t j = 0; j < channels; j++)
        lengthSquared += corrected[j] * corrected[j];

      const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

      for (size_t j = 0; j < channels; j++)
        diff_xi_dirA[j] += corrected[j] * inv_length;
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

    for (size_t i = 0; i < size; i++)
    {
      const limg_ui8_4 px = pStart[i];

      float lineOriginToPx[channels];

      for (size_t j = 0; j < channels; j++)
        lineOriginToPx[j] = (float)px[j] - avg[j];

      const float facA = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirA) * inv_length_diff_xi_dirA;

      min_dirA = limgMin(min_dirA, facA);
      max_dirA = limgMax(max_dirA, facA);

      float error_vec_dirA[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t j = 0; j < channels; j++)
      {
        pEstimate[j] = avg[j] + facA * diff_xi_dirA[j];
        error_vec_dirA[j] = (float)px[j] - pEstimate[j];

        const float abs_val = fabsf(error_vec_dirA[j]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = j;
        }
      }

      pEstimate += channels;

      if (max_abs != 0)
      {
        max_abs = error_vec_dirA[max_idx];

        float lengthSquared = 0;

        for (size_t j = 0; j < channels; j++)
          lengthSquared += error_vec_dirA[j] * error_vec_dirA[j];

        const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

        for (size_t j = 0; j < channels; j++)
          diff_xi_dirB[j] += error_vec_dirA[j] * inv_length;
      }
    }

    for (size_t i = 0; i < channels; i++)
      diff_xi_dirB[i] *= inv_count;

    const float inv_length_diff_xi_dirB = 1.f / limg_dot<float, channels>(diff_xi_dirB, diff_xi_dirB);

    min_dirB = FLT_MAX;
    max_dirB = -FLT_MAX;

    pEstimate = pScratch;

    for (size_t i = 0; i < size; i++)
    {
      const limg_ui8_4 px = pStart[i];

      float lineOriginToPx[channels];

      for (size_t j = 0; j < channels; j++)
        lineOriginToPx[j] = px[j] - pEstimate[j];

      const float facB = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirB) * inv_length_diff_xi_dirB;

      min_dirB = limgMin(min_dirB, facB);
      max_dirB = limgMax(max_dirB, facB);

      float error_vec_dirAB[channels];
      float max_abs = 0;
      size_t max_idx = 0;

      for (size_t j = 0; j < channels; j++)
      {
        pEstimate[j] = (pEstimate[j] + facB * diff_xi_dirB[j]);
        error_vec_dirAB[j] = (float)px[j] - pEstimate[j];

        const float abs_val = fabsf(error_vec_dirAB[j]);

        if (abs_val > max_abs)
        {
          max_abs = abs_val;
          max_idx = j;
        }
      }

      pEstimate += channels;

      if (max_abs != 0)
      {
        max_abs = error_vec_dirAB[max_idx];

        float lengthSquared = 0;

        for (size_t j = 0; j < channels; j++)
          lengthSquared += error_vec_dirAB[j] * error_vec_dirAB[j];

        const float inv_length = copysignf(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lengthSquared))), max_abs);

        for (size_t j = 0; j < channels; j++)
          diff_xi_dirC[j] += error_vec_dirAB[j] * inv_length;
      }
    }

    for (size_t j = 0; j < channels; j++)
      diff_xi_dirC[j] *= inv_count;

    const float inv_length_diff_xi_dirC = 1.f / limg_dot<float, channels>(diff_xi_dirC, diff_xi_dirC);

    pEstimate = pScratch;

    min_dirC = FLT_MAX;
    max_dirC = -FLT_MAX;

    for (size_t i = 0; i < size; i++)
    {
      const limg_ui8_4 px = pStart[i];

      float lineOriginToPx[channels];

      for (size_t j = 0; j < channels; j++)
        lineOriginToPx[j] = px[j] - pEstimate[j];

      const float facC = limg_dot<float, channels>(lineOriginToPx, diff_xi_dirC) * inv_length_diff_xi_dirC;

      min_dirC = limgMin(min_dirC, facC);
      max_dirC = limgMax(max_dirC, facC);

      pEstimate += channels;
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
static LIMG_DEBUG_NO_INLINE void limg_encode_get_block_factors_accurate_from_state_3d(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, limg_encode_3d_output<channels> &out, limg_encode_decomposition_state &state, float *pScratch)
{
  if constexpr (channels == 3)
  {
    if (sse41Supported)
      limg_encode_get_block_factors_accurate_from_state_3d_3_sse41(pCtx, pPixels, size, out, state, pScratch);
    else
      limg_encode_get_block_factors_accurate_from_state_3d_3(pCtx, pPixels, size, out, state, pScratch);
  }
  else
  {
    if (sse41Supported)
      limg_encode_get_block_factors_accurate_from_state_3d_4_sse41(pCtx, pPixels, size, out, state, pScratch);
    else
      limg_encode_get_block_factors_accurate_from_state_3d_4(pCtx, pPixels, size, out, state, pScratch);
  }
}

//////////////////////////////////////////////////////////////////////////

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

#endif // limg_factorization_h__
