/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:00:38
 * @LastEditTime: 2024-11-26 09:29:20
 * @FilePath: /EasyDeploy/deploy_core/include/deploy_core/block_queue.h
 */
#ifndef __EASY_DEPLOY_BLOCK_QUEUE_H
#define __EASY_DEPLOY_BLOCK_QUEUE_H

#include <atomic>
#include <condition_variable>
#include <optional>
#include <queue>

/**
 * @brief A simple implementation of block queue.
 *
 * @tparam T
 */
template <typename T>
class BlockQueue {
public:
  BlockQueue<T>(const size_t max_size) : max_size_(max_size)
  {}

  /**
   * @brief Push a obj into the queue. Will block the thread if the queue is full.
   *
   * @param obj
   * @return true
   * @return false
   */
  bool BlockPush(const T &obj) noexcept;

  /**
   * @brief Push a obj into the queue. Will cover the oldest element if the queue is full.
   *
   * @param obj
   * @return true
   * @return false
   */
  bool CoverPush(const T &obj) noexcept;

  /**
   * @brief Get and pop the oldest element in the queue. Will block the thread if the queue is
   * empty.
   *
   * @return std::optional<T>
   */
  std::optional<T> Take() noexcept;

  /**
   * @brief Get and pop the oldest element in the queue. Will return `nullopt` if the queue is
   * empty.
   *
   * @return std::optional<T>
   */
  std::optional<T> TryTake() noexcept;

  /**
   * @brief Get the size of the queue.
   *
   * @return int
   */
  int Size() noexcept;

  /**
   * @brief Return if the queue is empty.
   *
   * @return true
   * @return false
   */
  bool Empty() noexcept;

  /**
   * @brief Set the `push` process disabled. After called this method, all `push` calling will
   * return `false`, which means this block queue no longer accept new elements.
   *
   */
  void DisablePush() noexcept;

  /**
   * @brief Set the `push` process enabled.
   *
   */
  void EnablePush() noexcept;

  /**
   * @brief Set the `take` process disabled. After called this method, all `take` calling will
   * return `false`, which means this block queue no longer provides elements.
   *
   */
  void DisableTake() noexcept;

  /**
   * @brief Set the `take` process enabled.
   *
   */
  void EnableTake() noexcept;

  /**
   * @brief Set the `push` and `take` process disabled.
   *
   */
  void Disable() noexcept;

  /**
   * @brief Get the max size of the block queue.
   *
   * @return int
   */
  int GetMaxSize() const noexcept;

  /**
   * @brief Set the `push` and `take` process disabled, and clear all elements in it.
   *
   */
  void DisableAndClear() noexcept;

  /**
   * @brief Set the `push` process will no longer be called. The consumer threads which were
   * blocked will be notified and quit blocking, when this method is called.
   *
   */
  void SetNoMoreInput() noexcept;

  ~BlockQueue() noexcept;

private:
  const size_t            max_size_;
  std::queue<T>           q_;
  std::atomic<bool>       push_enabled_{true};
  std::atomic<bool>       take_enabled_{true};
  std::condition_variable producer_cv_;
  std::condition_variable consumer_cv_;
  std::mutex              lck_;

  std::atomic<bool> no_more_input_{false};
};

template <typename T>
BlockQueue<T>::~BlockQueue() noexcept
{
  Disable();
}

template <typename T>
bool BlockQueue<T>::BlockPush(const T &obj) noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  while (q_.size() >= max_size_ && push_enabled_.load())
  {
    producer_cv_.wait(u_lck);
  }
  if (!push_enabled_.load())
  {
    return false;
  }
  q_.push(obj);
  consumer_cv_.notify_one();
  return true;
}

template <typename T>
bool BlockQueue<T>::CoverPush(const T &obj) noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  if (!push_enabled_.load())
  {
    return false;
  }
  if (q_.size() == max_size_)
  {
    q_.pop();
  }
  q_.push(obj);
  consumer_cv_.notify_one();
  return true;
}

template <typename T>
std::optional<T> BlockQueue<T>::Take() noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  // block until: 1. take disabled; 2. no more input set; 3. new elements
  while (q_.size() == 0 && take_enabled_ && no_more_input_ == false)
  {
    consumer_cv_.wait(u_lck);
  }
  if (!take_enabled_ || (no_more_input_ && q_.size() == 0))
  {
    return std::nullopt;
  }
  T ret = q_.front();
  q_.pop();
  producer_cv_.notify_one();

  if (no_more_input_)
  {
    consumer_cv_.notify_all();
  }
  return ret;
}

template <typename T>
std::optional<T> BlockQueue<T>::TryTake() noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  if (q_.size() == 0)
  {
    return std::nullopt;
  } else
  {
    T ret = q_.front();
    q_.pop();
    producer_cv_.notify_all();
    if (no_more_input_)
    {
      consumer_cv_.notify_all();
    }
    return ret;
  }
}

template <typename T>
int BlockQueue<T>::Size() noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  return q_.size();
}

template <typename T>
bool BlockQueue<T>::Empty() noexcept
{
  std::unique_lock<std::mutex> u_lck(lck_);
  return q_.size() == 0;
}

template <typename T>
int BlockQueue<T>::GetMaxSize() const noexcept
{
  return max_size_;
}

template <typename T>
void BlockQueue<T>::Disable() noexcept
{
  DisablePush();
  DisableTake();
}

template <typename T>
void BlockQueue<T>::DisableAndClear() noexcept
{
  Disable();
  std::unique_lock<std::mutex> u_lck(lck_);
  while (!q_.empty()) q_.pop();
}

template <typename T>
void BlockQueue<T>::DisablePush() noexcept
{
  push_enabled_.store(false);
  producer_cv_.notify_all();
}

template <typename T>
void BlockQueue<T>::EnablePush() noexcept
{
  push_enabled_.store(true);
}

template <typename T>
void BlockQueue<T>::DisableTake() noexcept
{
  take_enabled_.store(false);
  consumer_cv_.notify_all();
}

template <typename T>
void BlockQueue<T>::EnableTake() noexcept
{
  take_enabled_.store(true);
}

template <typename T>
void BlockQueue<T>::SetNoMoreInput() noexcept
{
  no_more_input_.store(true);
  consumer_cv_.notify_all();
}

#endif