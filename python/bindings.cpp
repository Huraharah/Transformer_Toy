#include <pybind11/pybind11.h>

#include "core/cuda_utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_transformer_toy, module) {
	module.doc() =
		"Native Python bindings for Transformer_Toy";

	module.def(
		"cuda_available",
		&isCudaAvailable,
		"Return True when a compatible CUDA device is available");
}