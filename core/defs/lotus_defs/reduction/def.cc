#include "core/graph/op.h"

namespace LotusIR {

    #define REGISTER_REDUCE_OPERATOR_SCHEMA(OpName)                                                         \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                        \
        .Description("Computes the "#OpName" of the input tensor's element along the provided axes. "       \
            "The resulted tensor has the same shape as the input if keepdims equal 1. If keepdims "         \
            "equal 0, then the resulted tensor have the reduced dimension pruned. "                         \
                                                                                                            \
            "The above behavior is similar to numpy, with the exception that numpy default keepdims "       \
            "to False instead of True.")                                                                    \
        .Input("input", "Input tensor to be reduced.", "T")                                                 \
        .Output("output", "Output tensor.", "T")                                                            \
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },                                              \
            "Constrain input and output types to float tensors.")                                                  \
        .Attr("axis", "A list of axes to reduce into.", AttrType::AttributeProto_AttributeType_INTS)                                     \
        .Attr("keepdims", "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",      \
            AttrType::AttributeProto_AttributeType_INT, int64_t(1));

    // Taken from ONNX
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceSum)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceMean)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceProd)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceMax)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceMin)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceLogSumExp)

    // Taken from RS4
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceLogSum)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceSumSquare)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceL1)
    REGISTER_REDUCE_OPERATOR_SCHEMA(ReduceL2)


    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Argmax)
        .Description("Computes the indices of the max elements of the input tensor's element "
            "along the provided axes. The resulted tensor has the same shape as the input if "
            "keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced "
            "dimension pruned. The type of the output tensor is integer.")
        .Input("input", "Input tensor.", "T")
        .Output("output", "Output tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.")
        .Attr("axis", "A list of integers, along which to reduce max.", AttrType::AttributeProto_AttributeType_INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Argmin)
        .Description("Computes the indices of the min elements of the input tensor's element "
            "along the provided axes. The resulted tensor has the same shape as the input if "
            "keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced "
            "dimension pruned. The type of the output tensor is integer.")
        .Input("input", "Input tensor.", "T")
        .Output("output", "Output tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.")
        .Attr("axis", "A list of integers, along which to reduce min.", AttrType::AttributeProto_AttributeType_INT);

}
