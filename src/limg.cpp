#include "limg.h"

#include "limg_simd.h"
#include "limg_bit_crush.h"
#include "limg_decode.h"
#include "limg_factorization.h"

//////////////////////////////////////////////////////////////////////////

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
static LIMG_INLINE void limg_encode_sum_to_decomposition_state_(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_decomposition_state &state)
{
  for (size_t i = 0; i < channels; i++)
    state.sum[i] = 0;

  const limg_ui8_4 *pPx = reinterpret_cast<const limg_ui8_4 *>(pPixels);

  for (size_t i = 0; i < size; i++)
  {
    for (size_t j = 0; j < channels; j++)
      state.sum[j] += (*pPx)[j];

    pPx++;
  }
}

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
static LIMG_INLINE void limg_encode_sum_to_decomposition_state_sse41(limg_encode_context *, const uint32_t *pPixels, const size_t size, limg_encode_decomposition_state &state) // this could be made SSSE3 compatible with very little effort.
{
  __m128i sum = _mm_setzero_si128();

  const limg_ui8_4 *pLine = reinterpret_cast<const limg_ui8_4 *>(pPixels);

  const __m128i shuffle_8lo_16 = _mm_set_epi8(-1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0);
  const __m128i shuffle_8hi_16 = _mm_set_epi8(-1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8);

  const size_t size_si128 = (size_t)limgMax(0LL, (int64_t)(size - sizeof(__m128i) / sizeof(uint32_t)));

  size_t i = 0;

  for (; i <= size_si128; i += sizeof(__m128i) / sizeof(uint32_t))
  {
    const __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[i]));

    const __m128i sum16 = _mm_add_epi16(_mm_shuffle_epi8(val, shuffle_8lo_16), _mm_shuffle_epi8(val, shuffle_8hi_16));
    const __m128i sum32 = _mm_cvtepi16_epi32(_mm_hadd_epi16(sum16, sum16));

    sum = _mm_add_epi32(sum, sum32);
  }

  for (; i < size; i++)
    sum = _mm_add_epi32(sum, _mm_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&pLine[i]))));

  _mm_storeu_si128(reinterpret_cast<__m128i *>(&state.sum[0]), _mm_cvtepu32_epi64(sum));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(&state.sum[2]), _mm_cvtepu32_epi64(_mm_bsrli_si128(sum, 8)));
}

template <size_t channels>
static LIMG_INLINE void limg_encode_sum_to_decomposition_state(limg_encode_context *pCtx, const uint32_t *pPixels, const size_t size, limg_encode_decomposition_state &state)
{
  if (sse41Supported)
    limg_encode_sum_to_decomposition_state_sse41(pCtx, pPixels, size, state);
  else
    limg_encode_sum_to_decomposition_state_<channels>(pCtx, pPixels, size, state);
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
LIMG_INLINE static uint64_t limg_encode_dither_(const uint8_t shift, const size_t rangeSize, uint64_t ditherHash, uint8_t *pFactorsU8)
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

LIMG_INLINE static uint64_t limg_encode_dither_aes_sse41(const uint8_t shift, const size_t rangeSize, uint64_t ditherHash, uint8_t *pFactorsU8)
{
  if (shift > 7)
    return ditherHash;

  uint64_t ditherHashes[2] = { ditherHash, ~ditherHash };

  const uint16_t ditherSize = (1 << shift) - 1;
  const int16_t ditherOffset = 1 << (shift - 1);

  const __m128i ditherSize_ = _mm_set1_epi16((int16_t)ditherSize);
  const __m128i ditherOffset_ = _mm_set1_epi16(ditherOffset);
  __m128i state = _mm_loadu_si128(reinterpret_cast<__m128i *>(ditherHashes));
  const __m128i state2 = _mm_set_epi64x(0x2A76E98006CB4CADLL, 0x824A73EAAB705E1DLL);
  const __m128i FFhex_ = _mm_set1_epi16(0xFF);
  const __m128i shuffle16to8 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);

  constexpr size_t blockSize = sizeof(__m128i) / sizeof(int16_t);

  size_t i = 0;

  if (rangeSize >= blockSize)
  {
    const size_t rangeSize128 = rangeSize - blockSize + 1;

    for (; i < rangeSize128; i += blockSize)
    {
      const __m128i val = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<__m128i *>(&pFactorsU8[i])));

      state = _mm_aesdec_si128(state, state2);

      const __m128i dith = _mm_sub_epi16(_mm_and_si128(state, ditherSize_), ditherOffset_);
      const __m128i out = _mm_shuffle_epi8(_mm_srli_epi16(_mm_max_epi16(_mm_setzero_si128(), _mm_min_epi16(FFhex_, _mm_add_epi16(dith, val))), shift), shuffle16to8);

      *reinterpret_cast<uint64_t *>(&pFactorsU8[i]) = _mm_extract_epi64(out, 0);
    }

    _mm_storeu_si128(reinterpret_cast<__m128i *>(ditherHashes), state);
  }

  for (; i < rangeSize; i++)
  {
    ditherHashes[0] = ditherHashes[0] * 6364136223846793005ULL + 1;

    const uint32_t xorshifted_hi = (uint32_t)(((ditherHashes[0] >> 18) ^ ditherHashes[0]) >> 27);
    const uint32_t rot_hi = (uint32_t)(ditherHashes[0] >> 59);

    const uint32_t hi = (xorshifted_hi >> rot_hi) | (xorshifted_hi << (uint32_t)((-(int32_t)rot_hi) & 31));

    const int32_t rand = (int32_t)(hi & ditherSize) - ditherOffset;

    pFactorsU8[i] = (uint8_t)limgClamp((int32_t)pFactorsU8[i] + rand, 0, 0xFF) >> shift;
  }

  return ditherHashes[0];
}

LIMG_INLINE static uint64_t limg_encode_dither(const uint8_t shift, const size_t rangeSize, uint64_t ditherHash, uint8_t *pFactorsU8)
{
  if (sse41Supported && aesNiSupported)
    return limg_encode_dither_aes_sse41(shift, rangeSize, ditherHash, pFactorsU8);
  else
    return limg_encode_dither_(shift, rangeSize, ditherHash, pFactorsU8);
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

template <size_t channels>
void limg_encode3d_blocked_test_y_range(limg_encode_context *pCtx, const size_t y_start, const size_t y_end)
{
  float scratchBuffer[limg_MinBlockSize * limg_MinBlockSize * 4];
  uint32_t pixels[limg_MinBlockSize * limg_MinBlockSize];

  limg_encode_3d_output<channels> *pOut = reinterpret_cast<limg_encode_3d_output<channels> *>(pCtx->pBlockColorDecompositions);
  pOut += (y_start / limg_MinBlockSize) * pCtx->blockX;

  for (size_t y = y_start; y < y_end; y += limg_MinBlockSize)
  {
    for (size_t x = 0; x < pCtx->sizeX; x += limg_MinBlockSize)
    {
      const size_t rx = limgMin(pCtx->sizeX - x, limg_MinBlockSize);
      const size_t ry = limgMin(pCtx->sizeY - y, limg_MinBlockSize);

      const size_t rangeSize = rx * ry;

      for (size_t yy = 0; yy < ry; yy++)
        memcpy(pixels + yy * rx, pCtx->pSourceImage + (y + yy) * pCtx->sizeX + x, rx * sizeof(uint32_t));

      limg_encode_decomposition_state encode_state;
      limg_encode_sum_to_decomposition_state<channels>(pCtx, pixels, rangeSize, encode_state);

      limg_encode_3d_output<channels> decomposition;
      limg_encode_get_block_factors_accurate_from_state_3d<channels>(pCtx, pixels, rangeSize, decomposition, encode_state, scratchBuffer);

      *pOut = decomposition;
      pOut++;
    }
  }
}

bool LIMG_INLINE limg_encode_check_block_unused_3d(limg_encode_context *pCtx, const size_t ox, const size_t oy, const size_t rx, const size_t ry)
{
  const uint32_t *pBlockInfoLine = &pCtx->pBlockInfo[oy * pCtx->blockX + ox];

  for (size_t y = 0; y < ry; y++)
  {
    for (size_t x = 0; x < rx; x++)
      if (!!(pBlockInfoLine[x] & BlockInfo_InUse))
        return false;

    pBlockInfoLine += pCtx->blockX;
  }

  return true;
}

template <size_t channels>
bool LIMG_DEBUG_NO_INLINE limg_encode_3d_matches_sse2(limg_encode_context *pCtx, const limg_encode_3d_output<channels> &a, const limg_encode_3d_output<channels> &b)
{
  limg_color_error_state_3d<channels> stateA, stateB;
  limg_init_color_error_state_3d<channels>(a, stateA);
  limg_init_color_error_state_3d<channels>(b, stateB);

  float avgDiffSq = 0;
  float LIMG_ALIGN(16) lenSqDirA[4] = { 3, 3, 3 }; // just 4 because of SIMD.
  float LIMG_ALIGN(16) lenSqDirB[4] = { 3, 3, 3 }; // just 4 because of SIMD.

  const float_t colorDiffFactors[4] = { 2, 4, 3, 3 }; // for a very basic perceptual model

  for (size_t i = 0; i < channels; i++)
  {
    const float avgDiff = a.avg[i] - b.avg[i];
    avgDiffSq += avgDiff * avgDiff * colorDiffFactors[i];

    lenSqDirA[0] += (stateA.normalA[i] * stateA.normalA[i]) * colorDiffFactors[i];
    lenSqDirB[0] += (stateB.normalA[i] * stateB.normalA[i]) * colorDiffFactors[i];
    lenSqDirA[1] += (stateA.normalB[i] * stateA.normalB[i]) * colorDiffFactors[i];
    lenSqDirB[1] += (stateB.normalB[i] * stateB.normalB[i]) * colorDiffFactors[i];
    lenSqDirA[2] += (stateA.normalC[i] * stateA.normalC[i]) * colorDiffFactors[i];
    lenSqDirB[2] += (stateB.normalC[i] * stateB.normalC[i]) * colorDiffFactors[i];
  }

  const float sumLenSqDirA = lenSqDirA[0] + lenSqDirA[1] + lenSqDirA[2];
  const float sumLenSqDirB = lenSqDirB[0] + lenSqDirB[1] + lenSqDirB[2];
  const float sumLenRatio = (sumLenSqDirA + 1) / (sumLenSqDirB + 1);

  constexpr float maxAcceptAvgDiff = 16 * 3 * channels; // 3 -> average colorDiffFactor.
  constexpr float maxAcceptRange = 200 * 3 * channels; // 3 -> average colorDiffFactor.

  if (avgDiffSq < maxAcceptAvgDiff && sumLenSqDirA < maxAcceptRange && sumLenSqDirB < maxAcceptRange)
    return true;

  if constexpr (limg_DiagnoseCulprits)
  {
    if (avgDiffSq >= maxAcceptAvgDiff)
    {
      pCtx->culprits++;
      pCtx->culpritWasFastBlockMergeAvgDiffError++;
    }

    if (!(sumLenSqDirA < maxAcceptRange && sumLenSqDirB < maxAcceptRange))
    {
      pCtx->culprits++;
      pCtx->culpritWasFastBlockMergeRangeError++;
    }
  }

  constexpr float maxRatio = 1.8f;

  if (sumLenRatio > maxRatio || sumLenRatio < (1.f / maxRatio))
  {
    if constexpr (limg_DiagnoseCulprits)
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockExpandSizeMismatchError++;
    }
    
    return false;
  }

  float LIMG_ALIGN(16) invLenDirA[4]; // just 4 because of SIMD.
  float LIMG_ALIGN(16) invLenDirB[4]; // just 4 because of SIMD.
  const __m128 one = _mm_set1_ps(1.f);

  _mm_store_ps(invLenDirA, /*_mm_rsqrt_ps*/_mm_div_ps(one, (_mm_load_ps(lenSqDirA)))); // 1 / sqrt(len) wouldn't have the perceptual properties we're looking for.
  _mm_store_ps(invLenDirB, /*_mm_rsqrt_ps*/_mm_div_ps(one, (_mm_load_ps(lenSqDirB)))); // 1 / sqrt(len) wouldn't have the perceptual properties we're looking for.

  for (size_t i = 1; i < 3; i++)
  {
    invLenDirA[i] *= 2.f;
    invLenDirB[i] *= 2.f;
  }

  float color[channels];
  float fac_a, fac_b, fac_c;
  float sumFactors = 0;

  for (size_t z = 0; z < 3; z++)
  {
    const float zf = z * 0.5f;

    for (size_t y = 0; y < 3; y++)
    {
      const float yf = y * 0.5f;

      for (size_t x = 0; x < 3; x++)
      {
        const float xf = x * 0.5f;

        for (size_t i = 0; i < channels; i++)
          color[i] = stateB.normalA[i] * xf + stateB.normalB[i] * yf + stateB.normalC[i] * zf;

        limg_color_error_state_3d_get_factors<channels>(color, a, stateA, fac_a, fac_b, fac_c);
        sumFactors += fabsf(fac_a) * invLenDirA[0] + fabsf(0.5f - fac_b) * invLenDirA[1] + fabsf(0.5f - fac_c) * invLenDirA[2];

        for (size_t i = 0; i < channels; i++)
          color[i] = stateA.normalA[i] * xf + stateA.normalB[i] * yf + stateA.normalC[i] * zf;

        limg_color_error_state_3d_get_factors<channels>(a.avg, b, stateB, fac_a, fac_b, fac_c);
        sumFactors += fabsf(fac_a) * invLenDirB[0] + fabsf(0.5f - fac_b) * invLenDirB[1] + fabsf(0.5f - fac_c) * invLenDirB[2];

      }
    }
  }

  constexpr float divToAvg = 1.f / (3 * 3 * 3);
  const float sumFactorsAvg = sumFactors * divToAvg;

  constexpr float maxFactorSumCombine = 3.0f;

  if constexpr (limg_DiagnoseCulprits)
  {
    if (sumFactorsAvg < maxFactorSumCombine)
    {
      return true;
    }
    else
    {
      pCtx->culprits++;
      pCtx->culpritWasBlockExpandValueMismatchError++;

      return false;
    }
  }
  else
  {
    return (sumFactorsAvg < maxFactorSumCombine);
  }
}

template <size_t channels>
bool limg_encode_3d_matches(limg_encode_context *pCtx, limg_encode_3d_output<channels> &a, const limg_encode_3d_output<channels> &b)
{
  return limg_encode_3d_matches_sse2<channels>(pCtx, a, b);
}

template <size_t channels>
bool LIMG_INLINE limg_encode_3d_check_area(limg_encode_context *pCtx, limg_encode_3d_output<channels> *pDecomp, const size_t ox, const size_t oy, const size_t rx, const size_t ry, limg_encode_3d_output<channels> &out)
{
  const limg_encode_3d_output<channels> *pDecompLine = &pDecomp[oy * pCtx->blockX + ox];

  for (size_t y = 0; y < ry; y++)
  {
    for (size_t x = 0; x < rx; x++)
      if (!limg_encode_3d_matches(pCtx, out, pDecompLine[x]))
        return false;

    pDecompLine += pCtx->blockX;
  }

  return true;
}

template <size_t channels, bool canGrowUpLeft>
bool LIMG_DEBUG_NO_INLINE limg_encode_find_block_3d_expand(limg_encode_context *pCtx, limg_encode_3d_output<channels> *pDecomp, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, const bool up, const bool down, const bool left, const bool right, limg_encode_3d_output<channels> &out_decomp)
{
  size_t ox = *pOffsetX;
  size_t oy = *pOffsetY;
  size_t rx = *pRangeX;
  size_t ry = *pRangeY;

  bool canGrowUp = up;
  bool canGrowDown = down;
  bool canGrowLeft = left;
  bool canGrowRight = right;

  if constexpr (!canGrowUpLeft)
  {
    canGrowUp = false;
    canGrowLeft = false;
  }

  limg_encode_3d_output<channels> decomp = pDecomp[ox + oy * pCtx->blockX];

  while (canGrowUp || canGrowDown || canGrowLeft || canGrowRight)
  {
    if (canGrowRight)
    {
      const size_t newRx = rx + 1;

      if (ox + newRx < pCtx->blockX && limg_encode_check_block_unused_3d(pCtx, ox + rx, oy, newRx - rx, ry) && limg_encode_3d_check_area<channels>(pCtx, pDecomp, ox + rx, oy, newRx - rx, ry, decomp))
      {
        rx = newRx;
      }
      else
      {
        canGrowRight = false;
      }
    }

    if (canGrowDown)
    {
      const size_t newRy = ry + 1;

      if (oy + newRy < pCtx->blockY && limg_encode_check_block_unused_3d(pCtx, ox, oy + ry, rx, newRy - ry) && limg_encode_3d_check_area<channels>(pCtx, pDecomp, ox, oy + ry, rx, newRy - ry, decomp))
      {
        ry = newRy;
      }
      else
      {
        canGrowDown = false;
      }
    }

    if constexpr (canGrowUpLeft)
    {
      if (canGrowUp)
      {
        const size_t newOy = oy - 1;
        const size_t newRy = ry + 1;

        if (oy > 0 && limg_encode_check_block_unused_3d(pCtx, ox, newOy, rx, newRy - ry) && limg_encode_3d_check_area<channels>(pCtx, pDecomp, ox, newOy, rx, newRy - ry, decomp))
        {
          oy = newOy;
          ry = newRy;
        }
        else
        {
          canGrowUp = false;
        }
      }

      if (canGrowLeft)
      {
        const size_t newOx = ox - 1;
        const size_t newRx = rx + 1;

        if (ox > 0 && limg_encode_check_block_unused_3d(pCtx, newOx, oy, newRx - rx, ry) && limg_encode_3d_check_area<channels>(pCtx, pDecomp, newOx, oy, newRx - rx, ry, decomp))
        {
          ox = newOx;
          rx = newRx;
        }
        else
        {
          canGrowLeft = false;
        }
      }
    }
  }

  *pOffsetX = ox;
  *pOffsetY = oy;
  *pRangeX = rx;
  *pRangeY = ry;
  out_decomp = decomp;

  return true;
}

template <size_t channels, bool acceptTinyBlocks>
bool LIMG_DEBUG_NO_INLINE limg_encode_find_block_3d(limg_encode_context *pCtx, limg_encode_3d_output<channels> *pDecomp, size_t &staticX, size_t &staticY, size_t *pOffsetX, size_t *pOffsetY, size_t *pRangeX, size_t *pRangeY, limg_encode_3d_output<channels> &decomp)
{
  size_t ox = staticX;
  size_t oy = staticY;

  for (; oy < pCtx->blockY; oy++)
  {
    const uint32_t *pBlockInfoLine = &pCtx->pBlockInfo[oy * pCtx->blockX];

    for (; ox < pCtx->blockX; ox++)
    {
      if (!!(pBlockInfoLine[ox] & BlockInfo_InUse))
        continue;

      size_t rx = 1;
      size_t ry = 1;

      *pOffsetX = ox;
      *pOffsetY = oy;
      *pRangeX = rx;
      *pRangeY = ry;

      if (!limg_encode_find_block_3d_expand<channels, false>(pCtx, pDecomp, pOffsetX, pOffsetY, pRangeX, pRangeY, false, true, false, true, decomp))
        continue;

      if (*pRangeX == 1 && *pRangeY == 1)
        continue;

      rx = *pRangeX;
      ry = *pRangeY;

      limg_encode_3d_output<channels> sDecomp = decomp;

      if constexpr (!acceptTinyBlocks)
      {
        if (rx >= 3 && ry >= 3) // let's try expanding from the center third.
        {
          *pOffsetX = ox + rx / 3;
          *pOffsetY = oy + ry / 3;
          *pRangeX = rx / 3;
          *pRangeY = ry / 3;

          if (limg_encode_find_block_3d_expand<channels, true>(pCtx, pDecomp, pOffsetX, pOffsetY, pRangeX, pRangeY, true, true, true, true, decomp) && *pRangeX * *pRangeY > rx * ry)
          {
            staticX = ox;
            staticY = oy;

            return true;
          }

          *pOffsetX = ox;
          *pOffsetY = oy;
          *pRangeX = rx;
          *pRangeY = ry;

          staticX = ox + rx;
          staticY = oy;

          decomp = sDecomp;

          return true;
        }
        else
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasLargeBlockMergeResultingBlockSizeError++;
          }
        }
      }
      else
      {
        if (rx > 1 || ry > 1) // do we want to reject terrible matches? not sure.
        {
          *pOffsetX = ox;
          *pOffsetY = oy;
          *pRangeX = rx;
          *pRangeY = ry;

          staticX = ox + rx;
          staticY = oy;

          decomp = sDecomp;

          return true;
        }
        else
        {
          if constexpr (limg_DiagnoseCulprits)
          {
            pCtx->culprits++;
            pCtx->culpritWasSmallBlockMergeResultingBlockSizeError++;
          }
        }
      }
    }

    ox = 0;
  }

  staticX = ox;
  staticY = oy;

  return false;
}

template <size_t channels>
void limg_encode3d_encode_block_from_decomposition(limg_encode_context *pCtx, uint32_t *pPixels, const limg_encode_3d_output<channels> &decomposition, const size_t ox, const size_t oy, const size_t rx, const size_t ry, float *pScratch, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, uint32_t *pBlockIndex, uint8_t *pBlockError, uint8_t *pBitsPerPixel, size_t accum_bits[3 + 3 * 8], uint64_t &ditherLast, const uint32_t blockIndex)
{
  const size_t rangeSize = rx * ry;

  limg_color_error_state_3d<channels> color_error_state;
  limg_init_color_error_state_3d<channels>(decomposition, color_error_state);

  uint8_t *pAu8 = reinterpret_cast<uint8_t *>(pScratch);
  uint8_t *pBu8 = pAu8 + rangeSize;
  uint8_t *pCu8 = pBu8 + rangeSize;

  limg_color_error_state_3d_get_all_factors(pCtx, decomposition, color_error_state, pPixels, rangeSize, pAu8, pBu8, pCu8);

  uint8_t shift[3] = { 0, 0, 0 };

  if (pCtx->crushBits)
  {
    if (pCtx->errorPixelRetainingBitCrush)
    {
      if (pCtx->coarseFineBitCrush)
        limg_encode_find_shift_for_block_error_pixel_preference_stepwise_3d<channels>(pCtx, pPixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift);
      else
        limg_encode_find_shift_for_block_error_pixel_preference_3d<channels>(pCtx, pPixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift);
    }
    else
    {
      size_t minBlockError = (size_t)-1;

      // Try best guesses.
      if (pCtx->guessCrush)
        limg_encode_guess_shift_for_block_3d<channels>(pCtx, pPixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, &minBlockError);

      if (pCtx->coarseFineBitCrush)
        limg_encode_find_shift_for_block_stepwise_3d<channels>(pCtx, pPixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, minBlockError);
      else
        limg_encode_find_shift_for_block_3d<channels>(pCtx, pPixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, minBlockError);
    }

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

      //if constexpr (store_accum_bits)
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
      //if constexpr (store_accum_bits)
      {
        for (size_t i = 0; i < 3; i++)
        {
          accum_bits[i] += 8 * rangeSize;
          accum_bits[3 + i * 9] += rangeSize;
        }
      }
    }
  }
  else
  {
    //if constexpr (store_accum_bits)
    {
      for (size_t i = 0; i < 3; i++)
      {
        accum_bits[i] += 8 * rangeSize;
        accum_bits[3 + i * 9] += rangeSize;
      }
    }
  }

  //if constexpr (store_factors_shift)
  {
    const uint8_t bit_to_pattern[9] = { 0, 0x22, 0x44, 0x66, 0x88, 0xAA, 0xCC, 0xEE, 0xFF };

    const uint32_t shift_val = 0xFF000000 | (bit_to_pattern[shift[0]] << 16) | (bit_to_pattern[shift[1]] << 8) | bit_to_pattern[shift[2]];

    uint32_t colAMin = 0;
    uint32_t colAMax = 0;
    uint32_t colBMin = 0;
    uint32_t colBMax = 0;
    uint32_t colCMin = 0;
    uint32_t colCMax = 0;

    const uint8_t channelOffset[4] = { 0, 8, 16, 24 };

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

    constexpr size_t rawPixelBits = channels * CHAR_BIT;
    constexpr size_t staticPessimisticBlockBits = (channels * (CHAR_BIT + 1) * 2 + channels * CHAR_BIT + 2 * sizeof(uint16_t) * CHAR_BIT); // 110 (3) or 136 (4 channels).
    const size_t pixelBits = rangeSize * ((8 - shift[0]) + (8 - shift[1]) + (8 - shift[2]));
    const size_t bits = staticPessimisticBlockBits + pixelBits;
    const float bitsPerPixel = bits / (float)rangeSize;
    constexpr float rawPixelScaleFac = (255.f / (float)rawPixelBits);
    const uint8_t scaledBitsPerPixel = (uint8_t)limgMin(0xFF, (size_t)(bitsPerPixel * rawPixelScaleFac));
    const uint8_t bitsPerPixelU8 = (uint8_t)((bits + rangeSize / 2 /* rounding! */) / rangeSize);

    for (size_t offsetY = 0; offsetY < ry; offsetY++)
    {
      uint8_t *pFactorsALine = pFactorsA + (oy + offsetY) * pCtx->sizeX + ox;
      uint8_t *pFactorsBLine = pFactorsB + (oy + offsetY) * pCtx->sizeX + ox;
      uint8_t *pFactorsCLine = pFactorsC + (oy + offsetY) * pCtx->sizeX + ox;
      uint8_t *pBitsPerPixelLine = pBitsPerPixel + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pShiftLine = pShiftABCX + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColAMinLine = pColAMin + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColAMaxLine = pColAMax + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColBMinLine = pColBMin + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColBMaxLine = pColBMax + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColCMinLine = pColCMin + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pColCMaxLine = pColCMax + (oy + offsetY) * pCtx->sizeX + ox;
      uint32_t *pBlockIndexLine = pBlockIndex + (oy + offsetY) * pCtx->sizeX + ox;

      for (size_t offsetX = 0; offsetX < rx; offsetX++)
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

        if constexpr (limg_ScaledBitsPerPixelOutput)
        {
          pBitsPerPixelLine++;
          *pBitsPerPixelLine = (scaledBitsPerPixel == 0xFF && !!((offsetX + offsetY) & 2)) ? 0 : scaledBitsPerPixel;
        }
        else
        {
          *pBitsPerPixelLine = bitsPerPixelU8;
          pBitsPerPixelLine++;
        }

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

        *pBlockIndexLine = (0xFF000000 | blockIndex);
        pBlockIndexLine++;
      }
    }

    pAu8 = reinterpret_cast<uint8_t *>(pScratch);
    pBu8 = pAu8 + rangeSize;
    pCu8 = pBu8 + rangeSize;
  }

  //if constexpr (decode)
  {
    uint32_t *pDecodedStart = pDecoded + oy * pCtx->sizeX + ox;

    limg_decode_block_from_factors_3d<channels>(pDecodedStart, pCtx->sizeX, rx, ry, pAu8, pBu8, pCu8, decomposition, shift);
  }

  (void)pBlockError;
}

template <size_t channels, bool keepDecomposition>
limg_result limg_encode_region_from_3d_output(limg_encode_context *pCtx, const size_t ox, const size_t oy, const size_t rx, const size_t ry, const limg_encode_3d_output<channels> &decomp, uint32_t **ppPixels, size_t *pPixelCapacity, float **ppScratch, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, uint32_t *pBlockIndex, uint8_t *pBlockError, uint8_t *pBitsPerPixel, size_t accum_bits[3 + 3 * 8], uint64_t &ditherLast, const uint32_t blockIndex)
{
  limg_result result = limg_success;

  size_t x_px = rx * limg_MinBlockSize;
  size_t y_px = ry * limg_MinBlockSize;

  if (ox + rx == pCtx->blockX)
    x_px = x_px - limg_MinBlockSize + (pCtx->sizeX % limg_MinBlockSize);

  if (oy + ry == pCtx->blockY)
    y_px = y_px - limg_MinBlockSize + (pCtx->sizeY % limg_MinBlockSize);

  const size_t rangeSize = x_px * y_px;

  if (rangeSize > *pPixelCapacity)
  {
    *ppPixels = reinterpret_cast<uint32_t *>(realloc(*ppPixels, rangeSize * sizeof(uint32_t)));
    *ppScratch = reinterpret_cast<float *>(realloc(*ppScratch, rangeSize * 4 * sizeof(float)));
    *pPixelCapacity = rangeSize;

    LIMG_ERROR_IF(*ppPixels == nullptr || *ppScratch == nullptr, limg_error_MemoryAllocationFailure);
  }

  for (size_t yy = 0; yy < y_px; yy++)
    memcpy(*ppPixels + yy * x_px, pCtx->pSourceImage + (oy * limg_MinBlockSize + yy) * pCtx->sizeX + ox * limg_MinBlockSize, x_px * sizeof(uint32_t));

  limg_encode_decomposition_state encode_state;
  limg_encode_sum_to_decomposition_state<channels>(pCtx, *ppPixels, rangeSize, encode_state);

  if constexpr (!keepDecomposition)
  {
    limg_encode_3d_output<channels> decomposition;
    limg_encode_get_block_factors_accurate_from_state_3d<channels>(pCtx, *ppPixels, rangeSize, decomposition, encode_state, *ppScratch);

    limg_encode3d_encode_block_from_decomposition<channels>(pCtx, *ppPixels, decomposition, ox * limg_MinBlockSize, oy * limg_MinBlockSize, x_px, y_px, *ppScratch, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, ditherLast, blockIndex);
  }
  else
  {
    limg_encode3d_encode_block_from_decomposition<channels>(pCtx, *ppPixels, decomp, ox * limg_MinBlockSize, oy * limg_MinBlockSize, x_px, y_px, *ppScratch, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, ditherLast, blockIndex);
  }

epilogue:
  return limg_success;
}

template <size_t channels>
limg_result limg_encode3d_blocked_test_(limg_encode_context *pCtx, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, uint32_t *pBlockIndex, uint8_t *pBlockError, uint8_t *pBitsPerPixel, size_t accum_bits[3 + 3 * 8], limg_thread_pool *pThreadPool)
{
  limg_result result = limg_success;

  if (pThreadPool == nullptr)
  {
    limg_encode3d_blocked_test_y_range<channels>(pCtx, 0, pCtx->sizeY);
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

      limg_thread_pool_add(pThreadPool, [=]() { limg_encode3d_blocked_test_y_range<channels>(pCtx, start, end); });
    }

    limg_thread_pool_add(pThreadPool, [=]() { limg_encode3d_blocked_test_y_range<channels>(pCtx, y_start, pCtx->sizeY); });

    limg_thread_pool_await(pThreadPool);
  }

  limg_encode_3d_output<channels> *pDecomposition = reinterpret_cast<limg_encode_3d_output<channels> *>(pCtx->pBlockColorDecompositions);

  uint32_t *pPixels = nullptr;
  float *pScratch = nullptr;
  size_t pixelCapacity = 0;
  uint64_t ditherLast = 0xCA7F00D15BADF00D;
  uint32_t blockIndex = 0;

  // Attempt to Merge Large Blocks.
  {
    size_t blockFindStaticX = 0;
    size_t blockFindStaticY = 0;

    while (true)
    {
      size_t ox, oy, rx, ry;
      limg_encode_3d_output<channels> decomp;

      if (!limg_encode_find_block_3d<channels, false>(pCtx, pDecomposition, blockFindStaticX, blockFindStaticY, &ox, &oy, &rx, &ry, decomp))
        break;

      blockIndex++;

      for (size_t y = oy; y < oy + ry; y++)
        for (size_t x = ox; x < ox + rx; x++)
          pCtx->pBlockInfo[x + y * pCtx->blockX] = BlockInfo_InUse;

      LIMG_ERROR_CHECK((limg_encode_region_from_3d_output<channels, false>(pCtx, ox, oy, rx, ry, decomp, &pPixels, &pixelCapacity, &pScratch, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, ditherLast, blockIndex)));
    }
  }

  // Attempt to Merge Remaining Blocks.
  {
    size_t blockFindStaticX = 0;
    size_t blockFindStaticY = 0;

    while (true)
    {
      size_t ox, oy, rx, ry;
      limg_encode_3d_output<channels> decomp;

      if (!limg_encode_find_block_3d<channels, true>(pCtx, pDecomposition, blockFindStaticX, blockFindStaticY, &ox, &oy, &rx, &ry, decomp))
        break;

      blockIndex++;

      for (size_t y = oy; y < oy + ry; y++)
        for (size_t x = ox; x < ox + rx; x++)
          pCtx->pBlockInfo[x + y * pCtx->blockX] = BlockInfo_InUse;

      LIMG_ERROR_CHECK((limg_encode_region_from_3d_output<channels, false>(pCtx, ox, oy, rx, ry, decomp, &pPixels, &pixelCapacity, &pScratch, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, ditherLast, blockIndex)));
    }
  }

  // Encode Still Remaining Blocks.
  {
    const size_t rx = 1, ry = 1;

    for (size_t y = 0; y < pCtx->blockY; y++)
    {
      for (size_t x = 0; x < pCtx->blockX; x++)
      {
        if (pCtx->pBlockInfo[x + y * pCtx->blockX] & BlockInfo_InUse)
          continue;

        limg_encode_3d_output<channels> decomp = pDecomposition[x + y * pCtx->blockX];
        pCtx->pBlockInfo[x + y * pCtx->blockX] = BlockInfo_InUse;
        blockIndex++;

        LIMG_ERROR_CHECK((limg_encode_region_from_3d_output<channels, true>(pCtx, x, y, rx, ry, decomp, &pPixels, &pixelCapacity, &pScratch, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, ditherLast, blockIndex)));
      }
    }
  }
  
epilogue:
  free(pPixels);
  free(pScratch);

  return result;
}

template <size_t channels, bool store_factors_shift, bool decode, bool store_accum_bits>
void limg_encode3d_test_y_range(limg_encode_context *pCtx, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, size_t accum_bits[3 + 3 * 8], const size_t y_start, const size_t y_end)
{
  float scratchBuffer[limg_MinBlockSize * limg_MinBlockSize * 4];
  uint32_t pixels[limg_MinBlockSize * limg_MinBlockSize];

  uint64_t ditherLast = 0xCA7F00D15BADF00D;

  for (size_t y = y_start; y < y_end; y += limg_MinBlockSize)
  {
    for (size_t x = 0; x < pCtx->sizeX; x += limg_MinBlockSize)
    {
      const size_t rx = limgMin(pCtx->sizeX - x, limg_MinBlockSize);
      const size_t ry = limgMin(pCtx->sizeY - y, limg_MinBlockSize);

      const size_t rangeSize = rx * ry;

      for (size_t yy = 0; yy < ry; yy++)
        memcpy(pixels + yy * rx, pCtx->pSourceImage + (y + yy) * pCtx->sizeX + x, rx * sizeof(uint32_t));

      limg_encode_decomposition_state encode_state;
      limg_encode_sum_to_decomposition_state<channels>(pCtx, pixels, rangeSize, encode_state);

      limg_encode_3d_output<channels> decomposition;
      limg_encode_get_block_factors_accurate_from_state_3d<channels>(pCtx, pixels, rangeSize, decomposition, encode_state, scratchBuffer);

      limg_color_error_state_3d<channels> color_error_state;
      limg_init_color_error_state_3d<channels>(decomposition, color_error_state);

      uint8_t *pAu8 = reinterpret_cast<uint8_t *>(scratchBuffer);
      uint8_t *pBu8 = pAu8 + rangeSize;
      uint8_t *pCu8 = pBu8 + rangeSize;

      limg_color_error_state_3d_get_all_factors(pCtx, decomposition, color_error_state, pixels, rangeSize, pAu8, pBu8, pCu8);

      uint8_t shift[3] = { 0, 0, 0 };

      if (pCtx->crushBits)
      {
        if (pCtx->errorPixelRetainingBitCrush)
        {
          if (pCtx->coarseFineBitCrush)
            limg_encode_find_shift_for_block_error_pixel_preference_stepwise_3d<channels>(pCtx, pixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift);
          else
            limg_encode_find_shift_for_block_error_pixel_preference_3d<channels>(pCtx, pixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift); 
        }
        else
        {
          size_t minBlockError = (size_t)-1;
        
          // Try best guesses.
          if (pCtx->guessCrush)
            limg_encode_guess_shift_for_block_3d<channels>(pCtx, pixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, &minBlockError);
        
          if (pCtx->coarseFineBitCrush)
            limg_encode_find_shift_for_block_stepwise_3d<channels>(pCtx, pixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, minBlockError);
          else
            limg_encode_find_shift_for_block_3d<channels>(pCtx, pixels, rangeSize, decomposition, pAu8, pBu8, pCu8, shift, minBlockError);
        }

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

        pAu8 = reinterpret_cast<uint8_t *>(scratchBuffer);
        pBu8 = pAu8 + rangeSize;
        pCu8 = pBu8 + rangeSize;
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
  ctx.crushBits = errorFactor != 0;
  ctx.errorPixelRetainingBitCrush = !fastBitCrushing;
  ctx.coarseFineBitCrush = !ctx.errorPixelRetainingBitCrush; // faster & less accurate. faster with `ctx.errorPixelRetainingBitCrush` false.

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
  ctx.crushBits = errorFactor != 0;
  ctx.errorPixelRetainingBitCrush = !fastBitCrushing;
  ctx.coarseFineBitCrush = !ctx.errorPixelRetainingBitCrush; // faster & less accurate. faster with `ctx.errorPixelRetainingBitCrush` false.

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

limg_result limg_encode3d_blocked_test(const uint32_t *pIn, const size_t sizeX, const size_t sizeY, uint32_t *pDecoded, uint8_t *pFactorsA, uint8_t *pFactorsB, uint8_t *pFactorsC, uint32_t *pShiftABCX, uint32_t *pColAMin, uint32_t *pColAMax, uint32_t *pColBMin, uint32_t *pColBMax, uint32_t *pColCMin, uint32_t *pColCMax, uint32_t *pBlockIndex, uint8_t *pBlockError, uint8_t *pBitsPerPixel, const bool hasAlpha, const uint32_t errorFactor, limg_thread_pool *pThreadPool, const bool fastBitCrushing)
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
  ctx.crushBits = errorFactor != 0;
  ctx.errorPixelRetainingBitCrush = !fastBitCrushing;
  ctx.coarseFineBitCrush = !ctx.errorPixelRetainingBitCrush; // faster & less accurate. faster with `ctx.errorPixelRetainingBitCrush` false.

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

  ctx.blockX = (ctx.sizeX + (limg_MinBlockSize - 1)) / limg_MinBlockSize;
  ctx.blockY = (ctx.sizeY + (limg_MinBlockSize - 1)) / limg_MinBlockSize;
  ctx.pBlockColorDecompositions = malloc(ctx.blockX * ctx.blockY * (hasAlpha ? sizeof(limg_encode_3d_output<4>) : sizeof(limg_encode_3d_output<3>)));
  LIMG_ERROR_IF(ctx.pBlockColorDecompositions == nullptr, limg_error_MemoryAllocationFailure);

  ctx.pBlockInfo = reinterpret_cast<uint32_t *>(calloc(ctx.blockX * ctx.blockY, sizeof(uint32_t)));
  LIMG_ERROR_IF(ctx.pBlockInfo == nullptr, limg_error_MemoryAllocationFailure);

  _DetectCPUFeatures();

  size_t accum_bits[3 + 3 * 9] = { 0 };

  if (ctx.hasAlpha)
    LIMG_ERROR_CHECK(limg_encode3d_blocked_test_<4>(&ctx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, pThreadPool));
  else
    LIMG_ERROR_CHECK(limg_encode3d_blocked_test_<3>(&ctx, pDecoded, pFactorsA, pFactorsB, pFactorsC, pShiftABCX, pColAMin, pColAMax, pColBMin, pColBMax, pColCMin, pColCMax, pBlockIndex, pBlockError, pBitsPerPixel, accum_bits, pThreadPool));

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
    puts("-- Bit Crush -----------------------------------------");
    printf("PixelBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasPixelBitCrushError, (ctx.culpritWasPixelBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasPixelBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    printf("BlockBitCrushError    : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockBitCrushError, (ctx.culpritWasBlockBitCrushError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockBitCrushError / (double)(ctx.culpritWasPixelBitCrushError + ctx.culpritWasBlockBitCrushError)) * 100.0);
    puts("-- Block Merge ---------------------------------------");
    printf("BlockMergeSizeError   : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockExpandSizeMismatchError, (ctx.culpritWasBlockExpandSizeMismatchError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockExpandSizeMismatchError / (double)(ctx.culpritWasBlockExpandSizeMismatchError + ctx.culpritWasBlockExpandValueMismatchError)) * 100.0);
    printf("BlockMergeValueError  : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasBlockExpandValueMismatchError, (ctx.culpritWasBlockExpandValueMismatchError / (double)ctx.culprits) * 100.0, (ctx.culpritWasBlockExpandValueMismatchError / (double)(ctx.culpritWasBlockExpandSizeMismatchError + ctx.culpritWasBlockExpandValueMismatchError)) * 100.0);
    puts("-- Fast Block Merge ----------------------------------");
    printf("FastMergeAvgDiffError : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasFastBlockMergeAvgDiffError, (ctx.culpritWasFastBlockMergeAvgDiffError / (double)ctx.culprits) * 100.0, (ctx.culpritWasFastBlockMergeAvgDiffError / (double)(ctx.culpritWasFastBlockMergeAvgDiffError + ctx.culpritWasFastBlockMergeRangeError)) * 100.0);
    printf("FastMergeRangeError   : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasFastBlockMergeRangeError, (ctx.culpritWasFastBlockMergeRangeError / (double)ctx.culprits) * 100.0, (ctx.culpritWasFastBlockMergeRangeError / (double)(ctx.culpritWasFastBlockMergeAvgDiffError + ctx.culpritWasFastBlockMergeRangeError)) * 100.0);
    puts("-- Block Search --------------------------------------");
    printf("BlockSizeRejectLarge  : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasLargeBlockMergeResultingBlockSizeError, (ctx.culpritWasLargeBlockMergeResultingBlockSizeError / (double)ctx.culprits) * 100.0, (ctx.culpritWasLargeBlockMergeResultingBlockSizeError / (double)(ctx.culpritWasLargeBlockMergeResultingBlockSizeError + ctx.culpritWasSmallBlockMergeResultingBlockSizeError)) * 100.0);
    printf("BlockSizeRejectSmall  : %8" PRIu64 " (%7.3f%% / %7.3f%%)\n", ctx.culpritWasSmallBlockMergeResultingBlockSizeError, (ctx.culpritWasSmallBlockMergeResultingBlockSizeError / (double)ctx.culprits) * 100.0, (ctx.culpritWasSmallBlockMergeResultingBlockSizeError / (double)(ctx.culpritWasLargeBlockMergeResultingBlockSizeError + ctx.culpritWasSmallBlockMergeResultingBlockSizeError)) * 100.0);
    puts("");
  }

  if constexpr (!limg_ScaledBitsPerPixelOutput)
  {
    size_t bits = 0;

    for (size_t i = 0; i < ctx.sizeX * ctx.sizeY; i++)
      bits += pBitsPerPixel[i];

    printf("Compression Average: ~%7.4f bits per pixel\n\n", bits / (double)(ctx.sizeX * ctx.sizeY));
  }

#endif

  goto epilogue;

epilogue:

  if (ctx.pBlockColorDecompositions != nullptr)
    free(ctx.pBlockColorDecompositions);

  if (ctx.pBlockInfo != nullptr)
    free(ctx.pBlockInfo);

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
