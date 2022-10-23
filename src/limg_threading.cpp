// Improved Version of https://github.com/rainerzufalldererste/slapcodec/blob/master/slapcodec/src/threadpool.cpp

#include "limg_threading.h"

#include "limg.h"

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

#ifdef _WIN32
#include <windows.h>
#endif

struct limg_thread_pool
{
  std::queue<std::function<void(void)>> tasks;
  
  std::thread *pThreads;
  size_t threadCount;

  std::atomic<size_t> taskCount;
  std::atomic<bool> isRunning;
  std::mutex mutex;
  std::condition_variable condition_var;

  limg_thread_pool(const size_t threadCount);
  ~limg_thread_pool();
};

void limg_thread_pool_ThreadFunc(limg_thread_pool *pThreadPool, const size_t index)
{
#ifdef _WIN32
  SetThreadIdealProcessor(GetCurrentThread(), (DWORD)index);
#else
  (void)index;
#endif

  while (pThreadPool->isRunning)
  {
    std::function<void(void)> task = nullptr;

    {
      std::unique_lock<std::mutex> lock(pThreadPool->mutex);
      pThreadPool->condition_var.wait_for(lock, std::chrono::milliseconds(1));

      if (!pThreadPool->tasks.empty())
      {
        task = pThreadPool->tasks.front();
        pThreadPool->tasks.pop();
      }
    }

    if (task)
    {
      task();
      --pThreadPool->taskCount;
      continue;
    }
  }
}

limg_thread_pool::limg_thread_pool(const size_t threads) :
  tasks(),
  isRunning(true),
  pThreads(nullptr),
  threadCount(threads),
  mutex(),
  condition_var(),
  taskCount(0)
{
  pThreads = reinterpret_cast<std::thread *>(malloc(sizeof(std::thread) * threads));

  for (size_t i = 0; i < threads; i++)
    new (&pThreads[i]) std::thread(limg_thread_pool_ThreadFunc, this, i);
}

limg_thread_pool::~limg_thread_pool()
{
  limg_thread_pool_await(this);

  isRunning = false;
  condition_var.notify_all();

  for (size_t i = 0; i < threadCount; i++)
  {
    pThreads[i].join();
    pThreads[i].~thread();
  }

  free(pThreads);
}

limg_thread_pool * limg_thread_pool_new(const size_t threads)
{
  return new limg_thread_pool(threads);
}

void limg_thread_pool_destroy(limg_thread_pool **ppThreadPool)
{
  if (ppThreadPool == nullptr || *ppThreadPool == nullptr)
    return;

  delete *ppThreadPool;
}

size_t limg_thread_pool_thread_count(limg_thread_pool *pPool)
{
  if (pPool == nullptr)
    return 1;

  return pPool->threadCount == 0 ? 1 : pPool->threadCount;
}

void limg_thread_pool_add(limg_thread_pool *pThreadPool, const std::function<void(void)> &task)
{
  pThreadPool->taskCount++;

  pThreadPool->mutex.lock();
  pThreadPool->tasks.push(task);
  pThreadPool->mutex.unlock();

  pThreadPool->condition_var.notify_one();
}

void limg_thread_pool_await(limg_thread_pool *pThreadPool)
{
  while (true)
  {
    std::function<void(void)> task = nullptr;

    // Locked by mutex.
    {
      pThreadPool->mutex.lock();

      if (!pThreadPool->tasks.empty())
      {
        task = pThreadPool->tasks.front();
        pThreadPool->tasks.pop();
      }

      pThreadPool->mutex.unlock();
    }

    if (task)
    {
      task();
      pThreadPool->taskCount--;
    }
    else
    {
      break;
    }
  }

  while (pThreadPool->taskCount > 0)
    std::this_thread::yield(); // Wait for all other threads to finish their tasks.
}

size_t limg_threading_max_threads()
{
  return std::thread::hardware_concurrency();
}
