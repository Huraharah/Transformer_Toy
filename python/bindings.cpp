#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/cuda_utils.h"
#include "core/tensor.h"
#include "data/tokenizer.h"
#include "layers/config.h"
#include "training/training_config.h"
#include "training/learning_rate_scheduler_callback.h"
#include "data/tokenizer.h"
#include "core/tensor.h"

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

    void bindGenerationConfig(py::module_& module)
    {
        py::class_<GenerationConfig>(module, "GenerationConfig")
            .def(py::init<>())

            .def(
                py::init<
                std::size_t,
                float,
                std::size_t,
                bool,
                bool,
                bool,
                unsigned int
                >(),
                py::arg("max_new_tokens"),
                py::arg("temperature") = 1.0f,
                py::arg("top_k") = 0,
                py::arg("greedy_sampling") = false,
                py::arg("print_generated_text") = true,
                py::arg("print_token_ids") = false,
                py::arg("random_seed") = 42
            )

            .def_readwrite(
                "max_new_tokens",
                &GenerationConfig::maxNewTokens
            )
            .def_readwrite(
                "temperature",
                &GenerationConfig::temperature
            )
            .def_readwrite(
                "top_k",
                &GenerationConfig::topK
            )
            .def_readwrite(
                "greedy_sampling",
                &GenerationConfig::greedySampling
            )
            .def_readwrite(
                "print_generated_text",
                &GenerationConfig::printGeneratedText
            )
            .def_readwrite(
                "print_token_ids",
                &GenerationConfig::printTokenIds
            )
            .def_readwrite(
                "random_seed",
                &GenerationConfig::randomSeed
            )

            .def(
                "validate",
                &GenerationConfig::validate
            );
    }

    void bindTokenizerConfig(py::module_& module)
    {
        py::class_<TokenizerConfig>(module, "TokenizerConfig")
            .def(py::init<>())

            .def(
                py::init<
                TokenizerType,
                std::size_t,
                std::size_t,
                std::string,
                bool,
                bool,
                bool
                >(),
                py::arg("type"),
                py::arg("vocab_size") = 512,
                py::arg("min_frequency") = 2,
                py::arg("model_path") = "",
                py::arg("train_if_missing") = true,
                py::arg("preserve_whitespace") = true,
                py::arg("preserve_punctuation") = true
            )

            .def_readwrite(
                "type",
                &TokenizerConfig::type
            )
            .def_readwrite(
                "vocab_size",
                &TokenizerConfig::vocabSize
            )
            .def_readwrite(
                "min_frequency",
                &TokenizerConfig::minFrequency
            )
            .def_readwrite(
                "model_path",
                &TokenizerConfig::modelPath
            )
            .def_readwrite(
                "train_if_missing",
                &TokenizerConfig::trainIfMissing
            )
            .def_readwrite(
                "preserve_whitespace",
                &TokenizerConfig::preserveWhitespace
            )
            .def_readwrite(
                "preserve_punctuation",
                &TokenizerConfig::preservePunctuation
            );
    }

    void bindAttentionConfig(py::module_& module)
    {
        py::class_<AttentionConfig>(module, "AttentionConfig")
            .def(py::init<>())

            .def(
                py::init<
                size_t,
                size_t,
                bool,
                bool,
                float,
                float
                >(),
                py::arg("embed_dim"),
                py::arg("num_heads"),
                py::arg("causal") = false,
                py::arg("use_bias") = true,
                py::arg("attention_dropout") = 0.0f,
                py::arg("projection_dropout") = 0.0f
            )

            .def_readwrite(
                "embed_dim",
                &AttentionConfig::embedDim
            )
            .def_readwrite(
                "num_heads",
                &AttentionConfig::numHeads
            )
            .def_property_readonly(
                "head_dim",
                &AttentionConfig::headDim
            )
            .def_readwrite(
                "causal",
                &AttentionConfig::causal
            )
            .def_readwrite(
                "use_bias",
                &AttentionConfig::use_bias
            )
            .def_readwrite(
                "attention_dropout",
                &AttentionConfig::attention_dropout
            )
            .def_readwrite(
                "projection_dropout",
                &AttentionConfig::projection_dropout
            )
            .def(
                "validate",
                &AttentionConfig::validate
            );
    }

    void bindTransformerBlockConfig(py::module_& module)
    {
        py::class_<TransformerBlockConfig>(module, "TransformerBlockConfig")
            .def(py::init<>())

            .def(
                py::init <
                int,
                int,
                float,
                float,
                bool,
                bool
                >(),
                py::arg("d_model"),
                py::arg("d_ff"),
                py::arg("residual_dropout") = 0.0f,
                py::arg("ffn_dropout") = 0.0f,
                py::arg("pre_norm") = true,
                py::arg("use_bias") = true
            )
            .def_readwrite(
                "d_model",
                &TransformerBlockConfig::d_model
            )
            .def_readwrite(
                "d_ff",
                &TransformerBlockConfig::d_ff
            )
            .def_readwrite(
                "pre_norm",
                &TransformerBlockConfig::pre_norm
            )
            .def_readwrite(
                "use_bias",
                &TransformerBlockConfig::use_bias
            )
            .def_readwrite(
                "residual_dropout",
                &TransformerBlockConfig::residual_dropout
            )
            .def_readwrite(
                "ffn_dropout",
                &TransformerBlockConfig::ffn_dropout
            )
            .def_readwrite(
                "attention_type",
                &TransformerBlockConfig::attentionType
            )
            .def_readwrite(
                "num_heads",
                &TransformerBlockConfig::numHeads
            )
            .def(
                "validate",
                &TransformerBlockConfig::validate
            );
    }

    void bindTransformerModelConfig(py::module_& module)
    {
        py::class_<TransformerModelConfig>(module, "TransformerModelConfig")
            .def(py::init<>())

            .def(
                py::init <
                int,
                int,
                int,
                const TransformerBlockConfig&,
                bool,
                float
                >(),
                py::arg("vocab_size"),
                py::arg("max_seq_len"),
                py::arg("num_layers"),
                py::arg("block"),
                py::arg("learned_positional_embeddings") = true,
                py::arg("embedding_dropout") = 0.0f
            )
            .def_readwrite(
                "vocab_size",
                &TransformerModelConfig::vocab_size
            )
            .def_readwrite(
                "max_seq_len",
                &TransformerModelConfig::max_seq_len
            )
            .def_readwrite(
                "num_layers",
                &TransformerModelConfig::num_layers
            )
            .def_readwrite(
                "block",
                &TransformerModelConfig::block
            )
            .def_readwrite(
                "learned_positional_embeddings",
                &TransformerModelConfig::learned_positional_embeddings
            )
            .def_readwrite(
                "embedding_dropout",
                &TransformerModelConfig::embedding_dropout
            )
            .def(
                "validate",
                &TransformerModelConfig::validate
            );

    }

    void bindTrainingConfig(py::module_& module)
    {
        py::class_<TrainingConfig>(
            module,
            "TrainingConfig"
        )
            .def(py::init<>())

            // Core training
            .def_readwrite(
                "epochs",
                &TrainingConfig::epochs
            )
            .def_readwrite(
                "batch_size",
                &TrainingConfig::batchSize
            )
            .def_readwrite(
                "device",
                &TrainingConfig::device
            )

            // Logging
            .def_readwrite(
                "log_every_steps",
                &TrainingConfig::logEverySteps
            )
            .def_readwrite(
                "run_name",
                &TrainingConfig::runName
            )

            // Checkpointing
            .def_readwrite(
                "enable_checkpointing",
                &TrainingConfig::enableCheckpointing
            )
            .def_readwrite(
                "enable_best_checkpoint",
                &TrainingConfig::enableBestCheckpoint
            )
            .def_readwrite(
                "checkpoint_every_epochs",
                &TrainingConfig::checkpointEveryEpochs
            )
            .def_readwrite(
                "best_checkpoint_min_delta",
                &TrainingConfig::bestCheckpointMinDelta
            )
            .def_readwrite(
                "checkpoint_directory",
                &TrainingConfig::checkpointDirectory
            )

            // Early stopping
            .def_readwrite(
                "enable_early_stopping",
                &TrainingConfig::enableEarlyStopping
            )
            .def_readwrite(
                "early_stopping_patience",
                &TrainingConfig::earlyStoppingPatience
            )
            .def_readwrite(
                "early_stopping_min_delta",
                &TrainingConfig::earlyStoppingMinDelta
            )
            .def_readwrite(
                "prefer_validation_loss",
                &TrainingConfig::preferValidationLoss
            )

            // Learning-rate scheduler
            .def_readwrite(
                "enable_lr_scheduler",
                &TrainingConfig::enableLRScheduler
            )
            .def_readwrite(
                "lr_schedule",
                &TrainingConfig::lrSchedule
            )
            .def_readwrite(
                "lr_step_size",
                &TrainingConfig::lrStepSize
            )
            .def_readwrite(
                "lr_gamma",
                &TrainingConfig::lrGamma
            )
            .def_readwrite(
                "minimum_learning_rate",
                &TrainingConfig::minimumLearningRate
            )

            // Generation snapshots
            .def_readwrite(
                "enable_generation_snapshots",
                &TrainingConfig::enableGenerationSnapshots
            )
            .def_readwrite(
                "generation_every_epochs",
                &TrainingConfig::generationEveryEpochs
            )
            .def_readwrite(
                "generation_config",
                &TrainingConfig::generationConfig
            )
            .def_readwrite(
                "generation_prompt",
                &TrainingConfig::generationPrompt
            )

            // Profiling
            .def_readwrite(
                "enable_profiling",
                &TrainingConfig::enableProfiling
            )
            .def_readwrite(
                "print_profile_summary",
                &TrainingConfig::printProfileSummary
            )
            .def_readwrite(
                "synchronize_profiling_phases",
                &TrainingConfig::synchronizeProfilingPhases
            )
            .def_readwrite(
                "profile_warmup_steps",
                &TrainingConfig::profileWarmupSteps
            )
            .def_readwrite(
                "profile_every_steps",
                &TrainingConfig::profileEverySteps
            )

            .def(
                "validate",
                &TrainingConfig::validate
            );
    }

    void bindTokenizerInterface(py::module_& module)
    {
        py::class_<Tokenizer>(
            module,
            "Tokenizer",
            "Abstract base interface for text tokenizers."
        )
            .def(
                "name",
                &Tokenizer::name,
                "Return the tokenizer's descriptive name."
            )
            .def(
                "type",
                &Tokenizer::type,
                "Return the tokenizer type."
            )
            .def(
                "train",
                &Tokenizer::train,
                py::arg("corpus"),
                "Train the tokenizer from a text corpus."
            )
            .def(
                "encode",
                &Tokenizer::encode,
                py::arg("text"),
                "Encode text into token IDs."
            )
            .def(
                "decode",
                &Tokenizer::decode,
                py::arg("token_ids"),
                "Decode token IDs into text."
            )
            .def_property_readonly(
                "vocab_size",
                &Tokenizer::vocabSize,
                "Number of tokens in the trained vocabulary."
            )
            .def(
                "save",
                &Tokenizer::save,
                py::arg("path"),
                "Save tokenizer state to a file."
            )
            .def(
                "load",
                &Tokenizer::load,
                py::arg("path"),
                "Load tokenizer state from a file."
            )
            .def_property_readonly(
                "is_trained",
                &Tokenizer::isTrained,
                "Whether the tokenizer has a trained or loaded vocabulary."
            );
    }

    void bindCharTokenizer(py::module_& module)
    {
        py::class_<
            CharTokenizer,
            Tokenizer
        >(
            module,
            "CharTokenizer",
            "Character-level tokenizer."
        )
            .def(py::init<>())

            .def(
                "build_from_text",
                &CharTokenizer::buildFromText,
                py::arg("text"),
                "Build the character vocabulary from text."
            )
            .def(
                "id_to_char",
                &CharTokenizer::idToChar,
                py::arg("token_id"),
                "Return the character associated with a token ID."
            )
            .def(
                "char_to_id",
                &CharTokenizer::charToId,
                py::arg("character"),
                "Return the token ID associated with a character."
            );
    }

    void bindWordTokenizer(py::module_& module)
    {
        py::class_<
            WordTokenizer,
            Tokenizer
        >(
            module,
            "WordTokenizer",
            "Word-level tokenizer with configurable vocabulary behavior."
        )
            .def(
                py::init<
                std::size_t,
                std::size_t,
                bool,
                bool
                >(),
                py::arg("target_vocab_size") = 0,
                py::arg("min_frequency") = 1,
                py::arg("preserve_whitespace") = true,
                py::arg("preserve_punctuation") = true
            )

            .def(
                "id_to_token",
                [](const WordTokenizer& tokenizer, int tokenId) {
                    return std::string(
                        tokenizer.idToToken(tokenId)
                    );
                },
                py::arg("token_id"),
                "Return the token associated with a token ID."
            )
            .def(
                "token_to_id",
                &WordTokenizer::tokenToId,
                py::arg("token"),
                "Return the ID associated with a token."
            );
    }

    void bindBPETokenizer(py::module_& module)
    {
        py::class_<
            BPETokenizer,
            Tokenizer
        >(
            module,
            "BPETokenizer",
            "Byte Pair Encoding subword tokenizer."
        )
            .def(
                py::init<
                std::size_t,
                std::size_t
                >(),
                py::arg("target_vocab_size") = 512,
                py::arg("min_frequency") = 2
            )

            .def(
                "id_to_token",
                [](const BPETokenizer& tokenizer, int tokenId) {
                    return std::string(
                        tokenizer.idToToken(tokenId)
                    );
                },
                py::arg("token_id"),
                "Return the subword token associated with a token ID."
            )
            .def(
                "token_to_id",
                &BPETokenizer::tokenToId,
                py::arg("token"),
                "Return the ID associated with a subword token."
            )
            .def_property_readonly(
                "merges",
                [](const BPETokenizer& tokenizer) {
                    return tokenizer.merges();
                },
                "Return the learned BPE merge sequence."
            );
    }

    void bindTokenizerFactory(py::module_& module)
    {
        module.def(
            "create_tokenizer",
            &createTokenizer,
            py::arg("config"),
            "Create a tokenizer from a TokenizerConfig."
        );
    }

    Tensor tensorFromNumPy(
        py::array_t<
        float,
        py::array::c_style | py::array::forcecast
        > array
    )
    {
        py::buffer_info buffer = array.request();

        if (buffer.ndim == 0) {
            throw py::value_error(
                "Tensor NumPy input must have at least one dimension."
            );
        }

        std::vector<size_t> shape;
        shape.reserve(static_cast<size_t>(buffer.ndim));

        for (py::ssize_t dimension : buffer.shape) {
            if (dimension < 0) {
                throw py::value_error(
                    "Tensor dimensions cannot be negative."
                );
            }

            shape.push_back(
                static_cast<size_t>(dimension)
            );
        }

        Tensor tensor(shape);

        const auto* source =
            static_cast<const float*>(buffer.ptr);

        std::copy(
            source,
            source + tensor.size(),
            tensor.data().begin()
        );

        return tensor;
    }


    Tensor tensorFromData(
        const std::vector<float>& values,
        const std::vector<size_t>& shape
    )
    {
        Tensor tensor(shape);

        if (values.size() != tensor.size()) {
            throw py::value_error(
                "Number of values does not match the requested tensor shape."
            );
        }

        std::copy(
            values.begin(),
            values.end(),
            tensor.data().begin()
        );

        return tensor;
    }


    py::array_t<float> tensorToNumPy(Tensor& tensor)
    {
        // Synchronize device-generated data to the host first.
        tensor.toCPU();

        std::vector<py::ssize_t> shape;
        shape.reserve(tensor.rank());

        for (size_t dimension : tensor.shape()) {
            shape.push_back(
                static_cast<py::ssize_t>(dimension)
            );
        }

        py::array_t<float> output(shape);
        py::buffer_info buffer = output.request();

        auto* destination =
            static_cast<float*>(buffer.ptr);

        const std::vector<float>& values =
            static_cast<const Tensor&>(tensor).data();

        std::copy(
            values.begin(),
            values.end(),
            destination
        );

        return output;
    }
    
    void bindTensor(py::module_& module)
    {
        py::class_<Tensor>(
            module,
            "Tensor",
            "Contiguous float32 tensor with CPU and CUDA storage support."
        )
            // Empty tensor
            .def(
                py::init<>(),
                "Create an empty tensor."
            )

            // Shape-only tensor, zero initialized
            .def(
                py::init<const std::vector<size_t>&>(),
                py::arg("shape"),
                "Create a zero-initialized tensor with the given shape."
            )

            // Shape + fill value
            .def(
                py::init<
                const std::vector<size_t>&,
                float
                >(),
                py::arg("shape"),
                py::arg("fill_value"),
                "Create a tensor filled with a scalar value."
            )

            // NumPy constructor
            .def(
                py::init(
                    [](py::array_t<
                        float,
                        py::array::c_style |
                        py::array::forcecast
                    > array)
                    {
                        return tensorFromNumPy(
                            std::move(array)
                        );
                    }
                ),
                py::arg("array"),
                "Create a tensor by copying a NumPy-compatible array."
            )

            .def(
                "__array__",
                [](Tensor& tensor, py::object dtype, py::object copy)
                {
                    py::array array = tensorToNumPy(tensor);

                    if (!dtype.is_none()) {
                        array = array.attr("astype")(
                            dtype,
                            py::arg("copy") = false
                            );
                    }

                    return array;
                },
                py::arg("dtype") = py::none(),
                py::arg("copy") = py::none()
            )

            // Explicit raw-data factory
            .def_static(
                "from_data",
                &tensorFromData,
                py::arg("values"),
                py::arg("shape"),
                "Create a tensor from flat Python data and a shape."
            )

            // NumPy output
            .def(
                "numpy",
                &tensorToNumPy,
                "Return a float32 NumPy copy of the tensor."
            )

            // Allow np.asarray(tensor)
            .def(
                "__array__",
                [](Tensor& tensor, py::object dtype)
                {
                    py::array array = tensorToNumPy(tensor);

                    if (!dtype.is_none()) {
                        return array.attr("astype")(
                            dtype,
                            py::arg("copy") = false
                            );
                    }

                    return py::object(array);
                },
                py::arg("dtype") = py::none()
            )

            // Structural properties
            .def_property_readonly(
                "shape",
                [](const Tensor& tensor)
                {
                    return tensor.shape();
                },
                "Tensor dimensions."
            )
            .def_property_readonly(
                "strides",
                [](const Tensor& tensor)
                {
                    return tensor.strides();
                },
                "Tensor strides measured in elements."
            )
            .def_property_readonly(
                "rank",
                &Tensor::rank,
                "Number of tensor dimensions."
            )
            .def_property_readonly(
                "size",
                &Tensor::size,
                "Total number of elements."
            )
            .def_property_readonly(
                "empty",
                &Tensor::empty,
                "Whether the tensor contains no elements."
            )
            .def_property_readonly(
                "device",
                &Tensor::device,
                "Current tensor device."
            )
            .def_property_readonly(
                "has_device_data",
                &Tensor::hasDeviceData,
                "Whether CUDA device storage has been allocated."
            )

            // Device movement
            .def(
                "to_cuda",
                &Tensor::toCUDA,
                "Allocate CUDA storage and synchronize host data to it."
            )
            .def(
                "to_cpu",
                &Tensor::toCPU,
                "Synchronize CUDA data back to host memory."
            )

            // Flat element access
            .def(
                "__getitem__",
                [](const Tensor& tensor, size_t index)
                {
                    return tensor[index];
                },
                py::arg("index")
            )
            .def(
                "__setitem__",
                [](Tensor& tensor, size_t index, float value)
                {
                    tensor[index] = value;
                },
                py::arg("index"),
                py::arg("value")
            )

            // Multidimensional access
            .def(
                "at",
                [](const Tensor& tensor,
                    const std::vector<size_t>& indices)
                {
                    return tensor.at(indices);
                },
                py::arg("indices"),
                "Read an element using multidimensional indices."
            )
            .def(
                "set_at",
                [](Tensor& tensor,
                    const std::vector<size_t>& indices,
                    float value)
                {
                    tensor.at(indices) = value;
                },
                py::arg("indices"),
                py::arg("value"),
                "Write an element using multidimensional indices."
            )

            // Tensor mutation
            .def(
                "fill",
                &Tensor::fill,
                py::arg("value"),
                "Fill every tensor element with a scalar value."
            )
            .def(
                "reshape",
                &Tensor::reshape,
                py::arg("shape"),
                "Change the tensor shape without changing its data."
            )

            // Python conveniences
            .def(
                "__len__",
                &Tensor::size
            )
            .def(
                "__repr__",
                [](const Tensor& tensor)
                {
                    return
                        "Tensor(shape=" +
                        tensor.shapeString() +
                        ", device=" +
                        (
                            tensor.device() == Device::CUDA
                            ? std::string("CUDA")
                            : std::string("CPU")
                            ) +
                        ")";
                }
            );
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
    bindGenerationConfig(module);
    bindTokenizerConfig(module);
    bindAttentionConfig(module);
    bindTransformerBlockConfig(module);
    bindTransformerModelConfig(module);
    bindTrainingConfig(module);
    bindTokenizerInterface(module);
    bindCharTokenizer(module);
    bindWordTokenizer(module);
    bindBPETokenizer(module);
    bindTokenizerFactory(module);
    bindTensor(module);
}