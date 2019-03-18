#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/framework/environment.h"
#include "core/session/session.h"

namespace onnxruntime {
namespace test {
TEST(InferenceSessionWithoutEnvironment, UninitializedEnvironment) {
  EXPECT_FALSE(onnxruntime::Environment::IsInitialized());

  onnxruntime::SessionOptions session_options{};
  EXPECT_THROW(onnxruntime::Session::Create(session_options),
               onnxruntime::OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime
