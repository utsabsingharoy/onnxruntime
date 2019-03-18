// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/session.h"
#include "core/session/inference_session.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

std::unique_ptr<Session> Session::Create(const SessionOptions& session_options,
                                         logging::LoggingManager* logging_manager,
                                         SessionType session_type) {
  switch (session_type) {
    case onnxruntime::Session::SessionType::Inference:
      return std::unique_ptr<InferenceSession>(new InferenceSession(session_options, logging_manager));
      break;
    default:
      break;
  }
  return nullptr;
}

common::Status Session::Run(const RunOptions& run_options,
                            const NameMLValMap& feeds_map,
                            const std::vector<std::string>& output_names,
                            std::vector<MLValue>* p_fetches) {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;

  auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches);
}
}  // namespace onnxruntime
