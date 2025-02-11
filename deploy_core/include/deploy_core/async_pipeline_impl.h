/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:00:38
 * @LastEditTime: 2024-11-26 21:50:48
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/async_pipeline_impl.h
 */
#ifndef __EASY_DEPLOY_ASYNC_PIPELINE_IMPL_H
#define __EASY_DEPLOY_ASYNC_PIPELINE_IMPL_H

#include <functional>
#include <future>
#include <vector>

#include <glog/log_severity.h>
#include <glog/logging.h>

#include "deploy_core/block_queue.h"

namespace async_pipeline {

/**
 * @brief Async Pipeline Block
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class AsyncPipelineBlock {
public:
  AsyncPipelineBlock() = default;
  AsyncPipelineBlock(const AsyncPipelineBlock &block)
      : func_(block.func_), block_name_(block.block_name_)
  {}

  AsyncPipelineBlock &operator=(const AsyncPipelineBlock &block)
  {
    func_       = block.func_;
    block_name_ = block.block_name_;
    return *this;
  }

  AsyncPipelineBlock(const std::function<bool(ParsingType)> &func) : func_(func)
  {}

  AsyncPipelineBlock(const std::function<bool(ParsingType)> &func, const std::string &block_name)
      : func_(func), block_name_(block_name)
  {}

  const std::string &GetName() const
  {
    return block_name_;
  }

  bool operator()(const ParsingType &pipeline_unit) const
  {
    return func_(pipeline_unit);
  }

private:
  std::function<bool(ParsingType)> func_;
  std::string                      block_name_;
};

/**
 * @brief Async Pipeline Context
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class AsyncPipelineContext {
  using Block_t   = AsyncPipelineBlock<ParsingType>;
  using Context_t = AsyncPipelineContext<ParsingType>;

public:
  AsyncPipelineContext() = default;

  AsyncPipelineContext(const Block_t &block) : blocks_({block})
  {}

  AsyncPipelineContext(const std::vector<Block_t> &block_vec)
  {
    for (const auto &block : block_vec)
    {
      blocks_.push_back(block);
    }
  }

  AsyncPipelineContext &operator=(const std::vector<Block_t> &block_vec)
  {
    for (const auto &block : block_vec)
    {
      blocks_.push_back(block);
    }
    return *this;
  }

  AsyncPipelineContext(const Context_t &context) : blocks_(context.blocks_)
  {}

  AsyncPipelineContext(const std::vector<Context_t> &context_vec)
  {
    for (const auto &context : context_vec)
    {
      for (const auto &block : context.blocks_)
      {
        blocks_.push_back(block);
      }
    }
  }

  AsyncPipelineContext &operator=(const std::vector<Context_t> &context_vec)
  {
    for (const auto &context : context_vec)
    {
      for (const auto &block : context.blocks_)
      {
        blocks_.push_back(block);
      }
    }
    return *this;
  }

  AsyncPipelineContext &operator=(const Context_t &context)
  {
    for (const auto &block : context.blocks_)
    {
      blocks_.push_back(block);
    }
    return *this;
  }

public:
  std::vector<Block_t> blocks_;
};

/**
 * @brief Async Pipeline Processing Instance
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class PipelineInstance {
  using Block_t    = AsyncPipelineBlock<ParsingType>;
  using Context_t  = AsyncPipelineContext<ParsingType>;
  using Callback_t = std::function<bool(const ParsingType &)>;

  // for inner processing
  struct _InnerPackage {
    ParsingType package;
    Callback_t  callback;
  };
  using InnerParsingType = std::shared_ptr<_InnerPackage>;
  using InnerBlock_t     = AsyncPipelineBlock<InnerParsingType>;
  using InnerContext_t   = AsyncPipelineContext<InnerParsingType>;

public:
  PipelineInstance() = default;

  PipelineInstance(const std::vector<Context_t> &block_list) : context_(block_list)
  {
    // initialize inner context
    std::vector<InnerBlock_t> inner_block_list;
    for (const auto &block : context_.blocks_)
    {
      auto         func = [&](InnerParsingType p) -> bool { return block(p->package); };
      InnerBlock_t inner_block(func, block.GetName());
      inner_block_list.push_back(inner_block);
    }
    inner_context_ = InnerContext_t(inner_block_list);
  }

  ~PipelineInstance()
  {
    ClosePipeline();
  }

  void Init(int bq_max_size = 100)
  {
    // 1. for `n` blocks, construct `n+1` block queues
    const auto blocks = inner_context_.blocks_;
    const int  n      = blocks.size();
    LOG(INFO) << "[AsyncPipelineInstance] Total {" << n << "} Pipeline Blocks";
    for (int i = 0; i < n + 1; ++i)
    {
      block_queue_.emplace_back(std::make_shared<BlockQueue<InnerParsingType>>(bq_max_size));
    }
    pipeline_close_flag_.store(false);

    async_futures_.resize(n + 1);
    // 2. open `n` async threads to execute blocks
    for (int i = 0; i < n; ++i)
    {
      async_futures_[i] = std::async(&PipelineInstance::ThreadExcuteEntry, this, block_queue_[i],
                                     block_queue_[i + 1], blocks[i]);
    }
    // 3. open output threads to execute callback
    async_futures_[n] = std::async(&PipelineInstance::ThreadOutputEntry, this, block_queue_[n]);

    pipeline_initialized_.store(true);
  }

  void ClosePipeline()
  {
    if (pipeline_initialized_)
    {
      LOG(INFO) << "[AsyncPipelineInstance] Closing pipeline ...";
      for (const auto &bq : block_queue_)
      {
        bq->DisableAndClear();
      }
      LOG(INFO) << "[AsyncPipelineInstance] Disabled all block queue ...";
      pipeline_close_flag_.store(true);

      for (auto &future : async_futures_)
      {
        auto res = future.get();
      }
      LOG(INFO) << "[AsyncPipelineInstance] Join all block queue ...";
      block_queue_.clear();
      LOG(INFO) << "[AsyncPipelineInstance] Async pipeline is released successfully!!";
      pipeline_initialized_ = false;
      pipeline_close_flag_.store(true);
      pipeline_no_more_input_.store(true);
    }
  }

  void StopPipeline()
  {
    if (pipeline_initialized_)
    {
      pipeline_no_more_input_.store(true);
      block_queue_[0]->SetNoMoreInput();
    }
  }

  bool IsInitialized() const
  {
    return pipeline_initialized_;
  }

  const Context_t &GetContext() const
  {
    return context_;
  }

  void PushPipeline(const ParsingType &obj, const Callback_t &callback)
  {
    auto inner_pack      = std::make_shared<_InnerPackage>();
    inner_pack->package  = obj;
    inner_pack->callback = callback;

    block_queue_[0]->BlockPush(inner_pack);
  }

private:
  bool ThreadExcuteEntry(std::shared_ptr<BlockQueue<InnerParsingType>> bq_input,
                         std::shared_ptr<BlockQueue<InnerParsingType>> bq_output,
                         const InnerBlock_t                           &pipeline_block)
  {
    LOG(INFO) << "[AsyncPipelineInstance] {" << pipeline_block.GetName() << "} thread start!";
    while (!pipeline_close_flag_)
    {
      auto data = bq_input->Take();
      if (!data.has_value())
      {
        if (pipeline_no_more_input_)
        {
          LOG(INFO) << "[AsyncPipelineInstance] {" << pipeline_block.GetName()
                    << "} set no more output ...";
          bq_output->SetNoMoreInput();
          break;
        } else
        {
          continue;
        }
      }
      auto start  = std::chrono::high_resolution_clock::now();
      bool status = pipeline_block(data.value());
      auto end    = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "[AsyncPipelineInstance] {" << pipeline_block.GetName() << "} cost (us) : "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

      if (!status)
      {
        LOG(WARNING) << "[AsyncPipelineInstance] {" << pipeline_block.GetName()
                     << "}, excute block function failed! Drop package.";
        continue;
      }

      bq_output->BlockPush(data.value());
    }
    LOG(INFO) << "[AsyncPipelineInstance] {" << pipeline_block.GetName() << "} thread quit!";
    return true;
  }

  bool ThreadOutputEntry(std::shared_ptr<BlockQueue<InnerParsingType>> bq_input)
  {
    LOG(INFO) << "[AsyncPipelineInstance] {Output} thread start!";
    while (!pipeline_close_flag_)
    {
      auto data = bq_input->Take();
      if (!data.has_value())
      {
        if (pipeline_no_more_input_)
        {
          LOG(INFO) << "[AsyncPipelineInstance] {Output} set no more output ...";
          break;
        } else
        {
          continue;
        }
      }
      const auto &inner_pack = data.value();
      if (inner_pack != nullptr && inner_pack->callback != nullptr)
      {
        inner_pack->callback(inner_pack->package);
      } else
      {
        LOG(WARNING)
            << "[AsyncPipelineInstance] {Output} package without valid callback will be dropped!!!";
      }
    }
    LOG(INFO) << "[AsyncPipelineInstance] {Output} thread quit!";

    return true;
  }

private:
  Context_t context_;

  InnerContext_t inner_context_;

  std::vector<std::shared_ptr<BlockQueue<InnerParsingType>>> block_queue_;
  std::vector<std::future<bool>>                             async_futures_;

  std::atomic<bool> pipeline_close_flag_{true};
  std::atomic<bool> pipeline_no_more_input_{true};
  std::atomic<bool> pipeline_initialized_{false};
};

} // namespace async_pipeline

#endif
