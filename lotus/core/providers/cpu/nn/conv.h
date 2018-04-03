/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/providers/cpu/nn/conv_base.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {
namespace {
// helper function
inline void ComputeSizeAndPad(
    const int64_t in_dim,
    const int64_t stride,
    const int64_t kernel,
    const int64_t dilation,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_dim) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;

  if (pad_type == AutoPadType::NOTSET) {
    *out_dim = static_cast<int64_t>(static_cast<float>(in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
  } else {
    switch (pad_type) {
      case AutoPadType::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_dim = (in_dim - dkernel) / stride + 1;
        break;
      case AutoPadType::SAME_UPPER:
      case AutoPadType::SAME_LOWER:
        LOTUS_ENFORCE(dilation == 1, "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        *pad_tail = pad_needed - *pad_head;
        *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

        if (pad_type == AutoPadType::SAME_UPPER) {
          *pad_head = (pad_needed + 1) / 2;
        } else {
          *pad_head = pad_needed / 2;
        }

        break;
    }
  }
}
}  // namespace

template <typename T>
class Conv : public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    size_t num_inputs = OpKernel::Node().InputDefs().size();
    bool Is2DKernel = kernel_shape_.size() == 2;
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* W = context->Input<Tensor>(1);
    const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;
    const int64_t N = X->Shape()[0];
    const int64_t C = X->Shape()[1];
    const int64_t M = W->Shape()[0];

    vector<int64_t> pads(pads_);  // copy pads since it can be modified by InferOutputShape
    vector<int64_t> Y_dims;
    Y_dims.insert(Y_dims.begin(), {N, M});
    TensorShape input_shape = X->Shape().Slice(2);
    InferOutputShape(input_shape, &pads, &Y_dims);
    Tensor* Y = context->Output(0, TensorShape(Y_dims));
    TensorShape output_shape = Y->Shape().Slice(2);

    const int64_t input_image_size = input_shape.Size();
    const int64_t output_image_size = output_shape.Size();
    const int64_t kernel_size = TensorShape(kernel_shape_).Size();
    const int64_t X_offset = C / group_ * input_image_size;
    const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / group_;
    const int64_t W_offset = W->Shape().Size() / group_;
    const int64_t kernel_dim = C / group_ * kernel_size;
    const int64_t col_buffer_size = kernel_dim * output_image_size;

    auto& info = OpKernel::Allocator();
    auto& alloc = AllocatorManager::Instance().GetArena(info.name, info.id);

    auto col_data = alloc.Alloc(sizeof(T) * col_buffer_size);
    BufferUniquePtr col_buffer(col_data, BufferDeleter(&alloc));
    T* col_buffer_data = static_cast<T*>(col_buffer.get());

    const T* Xdata = X->template Data<T>();
    T* Ydata = Y->template MutableData<T>();

    TensorShape image_shape = X->Shape().Slice(1);
    vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                            output_shape.GetDims().end());

    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (Is2DKernel) {
          Math::Im2col<T, CPUMathUtil, StorageOrder::NCHW>(
              Xdata + group_id * X_offset,
              C / group_,
              input_shape[0],
              input_shape[1],
              kernel_shape_[0],
              kernel_shape_[1],
              dilations_[0],
              dilations_[1],
              pads[0],
              pads[1],
              pads[2],
              pads[3],
              strides_[0],
              strides_[1],
              col_buffer_data,
              &CPUMathUtil::Instance());
        } else {
          Math::Im2colNd<T, CPUMathUtil, StorageOrder::NCHW>(
              Xdata + group_id * X_offset,
              image_shape.GetDims().data(),
              col_buffer_shape.data(),
              C * input_image_size,
              col_buffer_size,
              kernel_shape_.data(),
              strides_.data(),
              dilations_.data(),
              pads.data(),
              static_cast<int>(kernel_shape_.size()),
              col_buffer_data,
              &CPUMathUtil::Instance());
        }

        Math::Gemm<T, CPUMathUtil>(
            CblasNoTrans,
            CblasNoTrans,
            M / group_,
            output_image_size,
            kernel_dim,
            1,
            W->template Data<T>() + group_id * W_offset,
            col_buffer_data,
            0,
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }

      if (B != nullptr) {
        auto Ymatrix = EigenMatrixMap<T>(Ydata, output_image_size, M);
        auto Bvec = ConstEigenVectorMap<T>(B->template Data<T>(), M);
        Ymatrix.rowwise() += Bvec.transpose();
      }

      Xdata += X_offset * group_;
      Ydata += Y_offset * group_;
    }

    return Status::OK();
  }

 private:
  void InferOutputShape(const TensorShape input_shape, vector<int64_t>* pads, vector<int64_t>* output_shape) const {
    for (int dim = 0; dim < input_shape.NumDimensions(); ++dim) {
      int64_t dim_size = 0;
      ComputeSizeAndPad(input_shape[dim],
                        strides_[dim],
                        kernel_shape_[dim],
                        dilations_[dim],
                        auto_pad_,
                        &pads->at(dim),
                        &pads->at(input_shape.NumDimensions() + dim),
                        &dim_size);
      output_shape->push_back(dim_size);
    }
  }
};

}  // namespace Lotus