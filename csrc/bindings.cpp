#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA operators";
  m.def("layer_norm_fwd", &cuda_ops::layer_norm_fwd, "LayerNorm forward (CUDA)", py::arg("x"),
        py::arg("gamma"), py::arg("beta"), py::arg("eps"));
  m.def("matmul", &cuda_ops::matmul, "Matrix multiplication (CUDA)", py::arg("a"), py::arg("b"));
}
