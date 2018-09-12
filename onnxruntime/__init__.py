"""
*ONNX Runtime* is a critical component for platforms that 
enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low. 
Building on Microsoft's dedication to the 
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well 
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.
"""
__version__ = "0.1.0"
__author__ = "Microsoft"

from onnxruntime.python.session import InferenceSession
from onnxruntime.python._pybind_state import RunOptions, SessionOptions, get_device
from onnxruntime.python import datasets