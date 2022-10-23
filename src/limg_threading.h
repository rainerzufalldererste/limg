#ifndef limg_threading_h__
#define limg_threading_h__

#include <stddef.h>
#include <functional>

struct limg_thread_pool;

limg_thread_pool * limg_thread_pool_new(const size_t threads);
void limg_thread_pool_destroy(limg_thread_pool **ppThreadPool);

size_t limg_thread_pool_thread_count(limg_thread_pool *pThreadPool);

void limg_thread_pool_add(limg_thread_pool *pThreadPool, const std::function<void(void)> &func);
void limg_thread_pool_await(limg_thread_pool *pThreadPool);

size_t limg_threading_max_threads();

#endif // limg_threading_h__
