#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/cuda_utils.h"
#include "core/tensor.h"
#include "data/tokenizer.h"
#include "layers/config.h"
#include "training/training_config.h"
#include "training/learning_rate_scheduler_callback.h"


namespace py = pybind11;


namespace {

    void bindEnums(py::module_& module) {
        py::enum_<Device>(module, "Device")
            .value("CPU", Device::CPU)
            .value("CUDA", Device::CUDA)
            .value("AUTO", Device::AUTO)
            .export_values();

        py::enum_<TokenizerType>(module, "TokenizerType")
            .value("CHARACTER", TokenizerType::Character)
            .value("WORD", TokenizerType::Word)
            .value("BPE", TokenizerType::BPE)
            .export_values();

        py::enum_<AttentionType>(module, "AttentionType")
            .value("SINGLE_HEAD", AttentionType::SingleHead)
            .value("MULTI_HEAD", AttentionType::MultiHead)
            .export_values();

        py::enum_<LearningRateSchedule>(module, "LearningRateSchedule")
            // Replace these with the exact enum members in your header.
            .value("CONSTANT", LearningRateSchedule::Constant)
            .value("STEP_DECAY", LearningRateSchedule::StepDecay)
            .value("EXPONENTIAL_DECAY", LearningRateSchedule::ExponentialDecay)
            .value("COSINE_DECAY", LearningRateSchedule::CosineDecay)
            .export_values();
    }



} // namespace

PYBIND11_MODULE(_transformer_toy, module) {
	module.doc() =
		"Native Python bindings for Transformer_Toy";

	module.def(
		"cuda_available",
		&isCudaAvailable,
		"Return True when a compatible CUDA device is available");

    bindEnums(module);

}