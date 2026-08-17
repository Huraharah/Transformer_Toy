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
#include "core/random.h"
#include "data/dataset.h"
#include "core/parameter.h"
#include "layers/linear.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/ffn.h"
#include "layers/attention.h"
#include "layers/transformer_block.h"
#include "model/transformer.h"
#include "training/loss.h"
#include "training/cross_entropy_loss.h"
#include "training/optimizer.h"
#include "training/sgd_optimizer.h"
#include "training/adam_optimizer.h"
#include "training/training_history.h"
#include "training/trainer.h"
#include "training/checkpoint.h"
#include "training/epoch_callback.h"
#include "training/best_checkpoint_callback.h"
#include "training/early_stopping_callback.h"
#include "training/generation_callback.h"
#include "training/learning_rate_scheduler_callback.h"
#include "training/checkpoint_callback.h"


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

    void bindRandom(py::module_& module)
    {
        py::class_<Random>(
            module,
            "Random",
            "Seedable random-number generator used by model initialization and batching."
        )
            .def(
                py::init<uint32_t>(),
                py::arg("seed")
            )
            .def(
                "set_seed",
                &Random::setSeed,
                py::arg("seed")
            )
            .def(
                "uniform",
                &Random::uniform,
                py::arg("min_value"),
                py::arg("max_value")
            )
            .def(
                "normal",
                &Random::normal,
                py::arg("mean"),
                py::arg("stddev")
            )
            .def(
                "randint",
                &Random::randint,
                py::arg("min_value"),
                py::arg("max_value")
            );
    }

    void bindTextDataset(py::module_& module)
    {
        py::class_<TextDataset>(
            module,
            "TextDataset",
            "Tokenized text dataset providing autoregressive training windows."
        )
            .def(
                py::init<
                const std::string&,
                std::size_t
                >(),
                py::arg("file_path"),
                py::arg("context_length")
            )

            .def(
                py::init<
                const std::string&,
                std::size_t,
                const TokenizerConfig&
                >(),
                py::arg("file_path"),
                py::arg("context_length"),
                py::arg("tokenizer_config")
            )

            .def(
                "get_input_window",
                &TextDataset::getInputWindow,
                py::arg("start_index")
            )
            .def(
                "get_target_window",
                &TextDataset::getTargetWindow,
                py::arg("start_index")
            )

            .def(
                "get_batch",
                [](
                    const TextDataset& dataset,
                    std::size_t batchSize,
                    Random& rng
                    )
                {
                    Tensor inputs;
                    Tensor targets;

                    dataset.getBatch(
                        batchSize,
                        rng,
                        inputs,
                        targets
                    );

                    return py::make_tuple(
                        std::move(inputs),
                        std::move(targets)
                    );
                },
                py::arg("batch_size"),
                py::arg("rng")
            )

            .def_property_readonly(
                "num_windows",
                &TextDataset::numWindows
            )
            .def_property_readonly(
                "context_length",
                &TextDataset::contextLength
            )
            .def_property_readonly(
                "vocab_size",
                &TextDataset::vocabSize
            )
            .def_property_readonly(
                "token_count",
                &TextDataset::tokenCount
            )

            .def_property_readonly(
                "tokenizer",
                static_cast<Tokenizer & (TextDataset::*)()>(
                    &TextDataset::tokenizer
                    ),
                py::return_value_policy::reference_internal
            )

            .def_property_readonly(
                "token_ids",
                [](const TextDataset& dataset)
                {
                    return dataset.tokenIds();
                }
            )
            .def_property_readonly(
                "raw_text",
                [](const TextDataset& dataset)
                {
                    return dataset.rawText();
                }
            )

            .def(
                "__len__",
                &TextDataset::numWindows
            )

            .def(
                "__repr__",
                [](const TextDataset& dataset)
                {
                    return
                        "TextDataset("
                        "token_count=" +
                        std::to_string(dataset.tokenCount()) +
                        ", context_length=" +
                        std::to_string(dataset.contextLength()) +
                        ", num_windows=" +
                        std::to_string(dataset.numWindows()) +
                        ", vocab_size=" +
                        std::to_string(dataset.vocabSize()) +
                        ")";
                }
            );
    }

    void bindParameter(py::module_& module)
    {
        py::class_<Parameter>(
            module,
            "Parameter",
            "A trainable tensor value and its associated gradient."
        )
            .def(
                py::init<>(),
                "Create an empty parameter that requires gradients."
            )

            .def(
                py::init<
                const Tensor&,
                const std::string&,
                bool
                >(),
                py::arg("value"),
                py::arg("name") = "",
                py::arg("requires_grad") = true,
                "Create a parameter from a tensor value."
            )

            .def_property(
                "value",
                [](Parameter& parameter) -> Tensor&
                {
                    return parameter.value;
                },
                [](Parameter& parameter, const Tensor& value)
                {
                    parameter.value = value;
                },
                py::return_value_policy::reference_internal,
                "The parameter value tensor."
            )

            .def_property(
                "grad",
                [](Parameter& parameter) -> Tensor&
                {
                    return parameter.grad;
                },
                [](Parameter& parameter, const Tensor& grad)
                {
                    parameter.grad = grad;
                },
                py::return_value_policy::reference_internal,
                "The gradient tensor associated with the parameter."
            )

            .def_readwrite(
                "name",
                &Parameter::name,
                "The descriptive parameter name."
            )

            .def_readwrite(
                "requires_grad",
                &Parameter::requires_grad,
                "Whether gradient computation and clearing are enabled."
            )

            .def(
                "zero_grad",
                &Parameter::zeroGrad,
                "Fill the gradient tensor with zeros when requires_grad is true."
            )

            .def_property_readonly(
                "has_grad",
                &Parameter::hasGrad,
                "Whether this parameter is configured to require gradients."
            )

            .def_property_readonly(
                "size",
                &Parameter::size,
                "Number of elements in the parameter value."
            )

            .def(
                "validate",
                &Parameter::validate,
                "Raise RuntimeError if value and gradient sizes do not match."
            )

            .def(
                "__repr__",
                [](const Parameter& parameter)
                {
                    return
                        "Parameter("
                        "name='" +
                        parameter.name +
                        "', shape=" +
                        parameter.value.shapeString() +
                        ", requires_grad=" +
                        std::string(
                            parameter.requires_grad
                            ? "True"
                            : "False"
                        ) +
                        ")";
                }
            );
    }

    void bindLinear(py::module_& module)
    {
        py::class_<Linear>(
            module,
            "Linear",
            "Fully connected linear transformation."
        )
            .def(
                py::init<
                size_t,
                size_t,
                Random&
                >(),
                py::arg("in_features"),
                py::arg("out_features"),
                py::arg("rng"),
                "Create a linear layer with randomly initialized weights."
            )

            .def(
                "forward",
                &Linear::forward,
                py::arg("input"),
                "Apply the linear transformation to a rank-2 tensor."
            )

            .def(
                "backward",
                &Linear::backward,
                py::arg("grad_output"),
                "Backpropagate through the layer and return the input gradient."
            )

            /*
                Return copies for convenient inspection.

                Live trainable state is available through parameters().
            */
            .def_property_readonly(
                "weights",
                [](const Linear& layer)
                {
                    return layer.weights();
                },
                "Copy of the current weight tensor."
            )

            .def_property_readonly(
                "bias",
                [](const Linear& layer)
                {
                    return layer.bias();
                },
                "Copy of the current bias tensor."
            )

            /*
                Return live Parameter references owned by the layer.
            */
            .def(
                "parameters",
                [](Linear& layer)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &layer,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : layer.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return the layer's weight and bias Parameters."
            )

            .def(
                "__repr__",
                [](const Linear& layer)
                {
                    const std::vector<size_t>& shape =
                        layer.weights().shape();

                    return
                        "Linear(in_features=" +
                        std::to_string(shape[1]) +
                        ", out_features=" +
                        std::to_string(shape[0]) +
                        ")";
                }
            );
    }

    void bindEmbedding(py::module_& module)
    {
        py::class_<Embedding>(
            module,
            "Embedding",
            "Trainable token embedding lookup table."
        )
            .def(
                py::init<
                size_t,
                size_t,
                Random&
                >(),
                py::arg("vocab_size"),
                py::arg("embedding_dim"),
                py::arg("rng"),
                "Create an embedding layer with randomly initialized values."
            )

            .def(
                "forward",
                &Embedding::forward,
                py::arg("token_ids"),
                "Look up embeddings for a rank-2 token-ID tensor."
            )

            .def(
                "backward",
                &Embedding::backward,
                py::arg("grad_output"),
                "Accumulate embedding-table gradients and return a zero token gradient."
            )

            .def_property_readonly(
                "table",
                [](const Embedding& embedding)
                {
                    return embedding.table();
                },
                "Copy of the current embedding table."
            )

            .def(
                "parameters",
                [](Embedding& embedding)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &embedding,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : embedding.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return the embedding-table Parameter."
            )

            .def(
                "__repr__",
                [](const Embedding& embedding)
                {
                    const auto& shape = embedding.table().shape();

                    return
                        "Embedding(vocab_size=" +
                        std::to_string(shape[0]) +
                        ", embedding_dim=" +
                        std::to_string(shape[1]) +
                        ")";
                }
            );
    }

    void bindLayerNorm(py::module_& module)
    {
        py::class_<LayerNorm>(
            module,
            "LayerNorm",
            "Layer normalization over the final tensor dimension."
        )
            .def(
                py::init<
                size_t,
                float
                >(),
                py::arg("feature_dim"),
                py::arg("epsilon") = 1.0e-5f,
                "Create a LayerNorm layer."
            )

            .def(
                "forward",
                &LayerNorm::forward,
                py::arg("input"),
                "Normalize a rank-3 tensor over its final dimension."
            )

            .def(
                "backward",
                &LayerNorm::backward,
                py::arg("grad_output"),
                "Backpropagate through LayerNorm."
            )

            .def(
                "parameters",
                [](LayerNorm& layer)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &layer,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : layer.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return the gamma and beta parameters."
            )

            .def(
                "__repr__",
                [](LayerNorm& layer)
                {
                    const auto parameters = layer.parameters();

                    return
                        "LayerNorm(feature_dim=" +
                        std::to_string(
                            parameters[0]->value.size()
                        ) +
                        ")";
                }
            );
    }

    void bindFFN(py::module_& module)
    {
        py::class_<FFN>(
            module,
            "FFN",
            "Two-layer feed-forward network with GELU activation."
        )
            .def(
                py::init<size_t, size_t, float, Random&>(),
                py::arg("embed_dim"),
                py::arg("hidden_dim"),
                py::arg("dropout_probability"),
                py::arg("rng"),
                "Create a feed-forward network."
            )

            .def(
                "forward",
                &FFN::forward,
                py::arg("input"),
                "Apply the FFN to a rank-3 tensor."
            )

            .def(
                "backward",
                &FFN::backward,
                py::arg("grad_output"),
                "Backpropagate through the FFN."
            )

            .def(
                "parameters",
                [](FFN& ffn)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &ffn,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : ffn.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return parameters from both internal linear layers."
            )

            .def(
                "__repr__",
                [](FFN& ffn)
                {
                    const auto parameters = ffn.parameters();

                    const size_t hiddenDim =
                        parameters[0]->value.shape()[0];

                    const size_t embedDim =
                        parameters[0]->value.shape()[1];

                    return
                        "FFN(embed_dim=" +
                        std::to_string(embedDim) +
                        ", hidden_dim=" +
                        std::to_string(hiddenDim) +
                        ")";
                }
            );
    }

    void bindAttentionLayers(py::module_& module)
    {
        py::class_<SelfAttention>(
            module,
            "SelfAttention",
            "Single-head causal self-attention."
        )
            .def(
                py::init<const AttentionConfig&, Random&>(),
                py::arg("config"),
                py::arg("rng"),
                "Create a single-head causal self-attention layer."
            )

            .def(
                "forward",
                &SelfAttention::forward,
                py::arg("input"),
                "Apply causal self-attention to a rank-3 tensor."
            )

            .def(
                "backward",
                &SelfAttention::backward,
                py::arg("grad_output"),
                "Backpropagate through single-head self-attention."
            )

            .def(
                "parameters",
                [](SelfAttention& attention)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &attention,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter :
                        attention.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return parameters from the Q, K, V, and output projections."
            )

            .def(
                "__repr__",
                [](SelfAttention& attention)
                {
                    const auto parameters =
                        attention.parameters();

                    const size_t embedDim =
                        parameters[0]->value.shape()[1];

                    return
                        "SelfAttention(embed_dim=" +
                        std::to_string(embedDim) +
                        ")";
                }
            );


        py::class_<MultiHeadAttention>(
            module,
            "MultiHeadAttention",
            "Multi-head causal self-attention."
        )
            .def(
                py::init<
                const AttentionConfig&,
                Random&
                >(),
                py::arg("config"),
                py::arg("rng"),
                "Create a multi-head attention layer from an AttentionConfig."
            )

            .def(
                "forward",
                &MultiHeadAttention::forward,
                py::arg("input"),
                "Apply multi-head causal self-attention to a rank-3 tensor."
            )

            .def(
                "backward",
                &MultiHeadAttention::backward,
                py::arg("grad_output"),
                "Backpropagate through multi-head self-attention."
            )

            .def(
                "parameters",
                [](MultiHeadAttention& attention)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &attention,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter :
                        attention.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return parameters from the Q, K, V, and output projections."
            )

            .def(
                "__repr__",
                [](MultiHeadAttention& attention)
                {
                    const auto parameters =
                        attention.parameters();

                    const size_t embedDim =
                        parameters[0]->value.shape()[1];

                    return
                        "MultiHeadAttention(embed_dim=" +
                        std::to_string(embedDim) +
                        ")";
                }
            );
    }

    void bindTransformerBlock(py::module_& module)
    {
        py::class_<TransformerBlock>(
            module,
            "TransformerBlock",
            "Transformer block containing attention, normalization, "
            "feed-forward mixing, and residual connections."
        )
            .def(
                py::init<
                const TransformerBlockConfig&,
                Random&
                >(),
                py::arg("config"),
                py::arg("rng"),
                "Create a Transformer block from its configuration."
            )

            .def(
                "forward",
                &TransformerBlock::forward,
                py::arg("input"),
                "Apply the Transformer block to a rank-3 tensor."
            )

            .def(
                "backward",
                &TransformerBlock::backward,
                py::arg("grad_output"),
                "Backpropagate through the Transformer block."
            )

            .def(
                "parameters",
                [](TransformerBlock& block)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &block,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : block.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                },
                "Return all trainable parameters owned by the block."
            )

            .def(
                "__repr__",
                [](TransformerBlock& block)
                {
                    const auto parameters = block.parameters();

                    /*
                        The first parameter belongs to the attention query
                        projection and has shape [d_model, d_model].
                    */
                    const size_t dModel =
                        parameters.empty()
                        ? 0
                        : parameters[0]->value.shape()[1];

                    return
                        "TransformerBlock(d_model=" +
                        std::to_string(dModel) +
                        ", parameters=" +
                        std::to_string(parameters.size()) +
                        ")";
                }
            );
    }

    void bindTransformer(py::module_& module)
    {
        py::class_<Transformer, TrainableModel>(module, "Transformer")

            .def(
                py::init<
                const TransformerModelConfig&,
                Random&
                >(),
                py::arg("config"),
                py::arg("rng")
            )

            .def(
                py::init<
                size_t,
                size_t,
                size_t,
                size_t,
                size_t,
                Random&
                >(),
                py::arg("vocab_size"),
                py::arg("context_length"),
                py::arg("embed_dim"),
                py::arg("hidden_dim"),
                py::arg("num_layers"),
                py::arg("rng")
            )

            .def(
                "forward",
                &Transformer::forward
            )

            .def(
                "backward",
                &Transformer::backward
            )

            .def(
                "generate",
                py::overload_cast<
                const std::string&,
                const Tokenizer&,
                const GenerationConfig&,
                Random&
                >(
                    &Transformer::generate
                ),
                py::arg("prompt"),
                py::arg("tokenizer"),
                py::arg("config"),
                py::arg("rng")
            )

            .def(
                "generate",
                py::overload_cast<
                const std::string&,
                const Tokenizer&,
                size_t,
                float,
                size_t,
                Random&
                >(
                    &Transformer::generate
                ),
                py::arg("prompt"),
                py::arg("tokenizer"),
                py::arg("max_new_tokens"),
                py::arg("temperature"),
                py::arg("top_k"),
                py::arg("rng")
            )

            .def(
                "parameters",
                [](Transformer& model)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &model,
                        py::return_value_policy::reference
                    );

                    for (Parameter* p : model.parameters())
                    {
                        result.append(
                            py::cast(
                                p,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                }
            )

            .def(
                "__repr__",
                [](Transformer& model)
                {
                    return
                        "Transformer(parameters="
                        + std::to_string(
                            model.parameters().size()
                        )
                        + ")";
                }
            );
    }

    void bindLosses(py::module_& module)
    {
        py::class_<Loss>(
            module,
            "Loss",
            "Abstract base interface for loss functions."
        )
            .def(
                "forward",
                &Loss::forward,
                py::arg("predictions"),
                py::arg("targets"),
                "Compute the loss and cache the gradient."
            )
            .def(
                "backward",
                &Loss::backward,
                "Return the gradient cached by the previous forward pass."
            );


        py::class_<
            CrossEntropyLoss,
            Loss
        >(
            module,
            "CrossEntropyLoss",
            "Mean cross-entropy loss for rank-2 logits and rank-1 class targets."
        )
            .def(
                py::init<>(),
                "Create a cross-entropy loss object."
            )

            .def(
                "forward",
                &CrossEntropyLoss::forward,
                py::arg("logits"),
                py::arg("targets"),
                "Compute mean cross-entropy loss and cache the logits gradient."
            )

            .def(
                "backward",
                &CrossEntropyLoss::backward,
                "Return the gradient cached during the previous forward pass."
            )

            .def(
                "__repr__",
                [](const CrossEntropyLoss&)
                {
                    return "CrossEntropyLoss()";
                }
            );
    }

    std::vector<Parameter*> collectParameterPointers(
        const py::iterable& parameters
    )
    {
        std::vector<Parameter*> result;

        for (py::handle item : parameters) {
            try {
                Parameter& parameter =
                    py::cast<Parameter&>(item);

                result.push_back(&parameter);
            }
            catch (const py::cast_error&) {
                throw py::type_error(
                    "Optimizer parameters must contain "
                    "transformer_toy.Parameter objects."
                );
            }
        }

        return result;
    }

    std::vector<TrainingBatch> collectTrainingBatches(
        const py::iterable& batches
    )
    {
        std::vector<TrainingBatch> result;

        for (py::handle item : batches) {
            try {
                result.push_back(
                    py::cast<TrainingBatch>(item)
                );
            }
            catch (const py::cast_error&) {
                throw py::type_error(
                    "Trainer batches must contain "
                    "transformer_toy.TrainingBatch objects."
                );
            }
        }

        return result;
    }

    void bindOptimizers(py::module_& module)
    {
        py::class_<Optimizer>(
            module,
            "Optimizer",
            "Abstract base interface for parameter optimizers."
        )
            .def(
                "step",
                [](
                    Optimizer& optimizer,
                    const py::iterable& parameters
                    )
                {
                    std::vector<Parameter*> pointers =
                        collectParameterPointers(parameters);

                    optimizer.step(pointers);
                },
                py::arg("parameters"),
                "Update the supplied parameters."
            )

            .def(
                "zero_grad",
                [](
                    Optimizer& optimizer,
                    const py::iterable& parameters
                    )
                {
                    std::vector<Parameter*> pointers =
                        collectParameterPointers(parameters);

                    optimizer.zeroGrad(pointers);
                },
                py::arg("parameters"),
                "Zero gradients for parameters that require gradients."
            )

            .def_property(
                "learning_rate",
                &Optimizer::getLearningRate,
                &Optimizer::setLearningRate,
                "Current optimizer learning rate."
            );


        py::class_<
            SGDOptimizer,
            Optimizer
        >(
            module,
            "SGDOptimizer",
            "Stochastic gradient descent with optional weight decay."
        )
            .def(
                py::init<
                float,
                float
                >(),
                py::arg("learning_rate"),
                py::arg("weight_decay") = 0.0f
            )

            .def(
                "__repr__",
                [](const SGDOptimizer& optimizer)
                {
                    return
                        "SGDOptimizer(learning_rate=" +
                        std::to_string(
                            optimizer.getLearningRate()
                        ) +
                        ")";
                }
            );


        py::class_<
            AdamOptimizer,
            Optimizer
        >(
            module,
            "AdamOptimizer",
            "Adam optimizer with optional weight decay."
        )
            .def(
                py::init<
                float,
                float,
                float,
                float,
                float
                >(),
                py::arg("learning_rate") = 0.001f,
                py::arg("beta1") = 0.9f,
                py::arg("beta2") = 0.999f,
                py::arg("epsilon") = 1.0e-8f,
                py::arg("weight_decay") = 0.0f
            )

            .def(
                "__repr__",
                [](const AdamOptimizer& optimizer)
                {
                    return
                        "AdamOptimizer(learning_rate=" +
                        std::to_string(
                            optimizer.getLearningRate()
                        ) +
                        ")";
                }
            );
    }

    void bindTrainingHistory(py::module_& module)
    {
        py::class_<TrainingHistory>(
            module,
            "TrainingHistory",
            "Recorded training and validation losses."
        )
            .def(
                py::init<>(),
                "Create an empty training history."
            )

            .def_property(
                "train_losses",
                [](const TrainingHistory& history)
                {
                    return history.trainLosses;
                },
                [](TrainingHistory& history,
                    const std::vector<float>& losses)
                {
                    history.trainLosses = losses;
                },
                "Recorded training losses."
            )

            .def_property(
                "validation_losses",
                [](const TrainingHistory& history)
                {
                    return history.validationLosses;
                },
                [](TrainingHistory& history,
                    const std::vector<float>& losses)
                {
                    history.validationLosses = losses;
                },
                "Recorded validation losses."
            )

            .def(
                "add_train_loss",
                &TrainingHistory::addTrainLoss,
                py::arg("loss"),
                "Append a training loss."
            )

            .def(
                "add_validation_loss",
                &TrainingHistory::addValidationLoss,
                py::arg("loss"),
                "Append a validation loss."
            )

            .def_property_readonly(
                "latest_train_loss",
                &TrainingHistory::latestTrainLoss,
                "Most recently recorded training loss."
            )

            .def_property_readonly(
                "latest_validation_loss",
                &TrainingHistory::latestValidationLoss,
                "Most recently recorded validation loss."
            )

            .def(
                "save_csv",
                &TrainingHistory::saveCsv,
                py::arg("path"),
                "Save training and validation losses to a CSV file."
            )

            .def(
                "clear",
                &TrainingHistory::clear,
                "Remove all recorded losses."
            )

            .def_property_readonly(
                "train_count",
                [](const TrainingHistory& history)
                {
                    return history.trainLosses.size();
                }
            )

            .def_property_readonly(
                "validation_count",
                [](const TrainingHistory& history)
                {
                    return history.validationLosses.size();
                }
            )

            .def(
                "__len__",
                [](const TrainingHistory& history)
                {
                    return history.trainLosses.size();
                }
            )

            .def(
                "__repr__",
                [](const TrainingHistory& history)
                {
                    return
                        "TrainingHistory(train_count=" +
                        std::to_string(
                            history.trainLosses.size()
                        ) +
                        ", validation_count=" +
                        std::to_string(
                            history.validationLosses.size()
                        ) +
                        ")";
                }
            );
    }

    void bindTrainingBatch(py::module_& module)
    {
        py::class_<TrainingBatch>(
            module,
            "TrainingBatch",
            "A pair of input and target tensors used by Trainer."
        )
            .def(
                py::init<>(),
                "Create an empty training batch."
            )

            .def(
                py::init(
                    [](
                        const Tensor& inputs,
                        const Tensor& targets
                        )
                    {
                        TrainingBatch batch;
                        batch.inputs = inputs;
                        batch.targets = targets;
                        return batch;
                    }
                ),
                py::arg("inputs"),
                py::arg("targets"),
                "Create a training batch from input and target tensors."
            )

            .def_property(
                "inputs",
                [](TrainingBatch& batch) -> Tensor&
                {
                    return batch.inputs;
                },
                [](TrainingBatch& batch, const Tensor& inputs)
                {
                    batch.inputs = inputs;
                },
                py::return_value_policy::reference_internal,
                "Input tensor."
            )

            .def_property(
                "targets",
                [](TrainingBatch& batch) -> Tensor&
                {
                    return batch.targets;
                },
                [](TrainingBatch& batch, const Tensor& targets)
                {
                    batch.targets = targets;
                },
                py::return_value_policy::reference_internal,
                "Target tensor."
            )

            .def(
                "__repr__",
                [](const TrainingBatch& batch)
                {
                    return
                        "TrainingBatch(inputs_shape=" +
                        batch.inputs.shapeString() +
                        ", targets_shape=" +
                        batch.targets.shapeString() +
                        ")";
                }
            );
    }

    void bindCheckpoint(py::module_& module)
    {
        py::class_<CheckpointMetadata>(
            module,
            "CheckpointMetadata",
            "Metadata stored alongside model parameters in a checkpoint."
        )
            .def(
                py::init<>(),
                "Create default checkpoint metadata."
            )

            .def_readwrite(
                "epoch",
                &CheckpointMetadata::epoch,
                "Epoch associated with the checkpoint."
            )

            .def_readwrite(
                "global_step",
                &CheckpointMetadata::globalStep,
                "Global training step associated with the checkpoint."
            )

            .def_readwrite(
                "run_name",
                &CheckpointMetadata::runName,
                "Training run name."
            )

            .def(
                "__repr__",
                [](const CheckpointMetadata& metadata)
                {
                    return
                        "CheckpointMetadata("
                        "epoch=" +
                        std::to_string(metadata.epoch) +
                        ", global_step=" +
                        std::to_string(metadata.globalStep) +
                        ", run_name='" +
                        metadata.runName +
                        "')";
                }
            );


        py::class_<Checkpoint>(
            module,
            "Checkpoint",
            "Static checkpoint save and load operations."
        )
            .def_static(
                "save",
                [](
                    const std::string& path,
                    const py::iterable& parameters,
                    const CheckpointMetadata& metadata,
                    const TrainingHistory& history
                    )
                {
                    std::vector<Parameter*> parameterPointers =
                        collectParameterPointers(parameters);

                    /*
                        The checkpoint implementation reads host-side tensor
                        storage. Synchronize any CUDA-resident values first.
                    */
                    for (Parameter* parameter : parameterPointers) {
                        parameter->value.toCPU();
                        parameter->grad.toCPU();
                    }

                    Checkpoint::save(
                        path,
                        parameterPointers,
                        metadata,
                        history
                    );
                },
                py::arg("path"),
                py::arg("parameters"),
                py::arg("metadata"),
                py::arg("history"),
                "Save parameters, metadata, and training history."
            )

            .def_static(
                "load",
                [](
                    const std::string& path,
                    const py::iterable& parameters,
                    CheckpointMetadata& metadata,
                    TrainingHistory& history
                    )
                {
                    std::vector<Parameter*> parameterPointers =
                        collectParameterPointers(parameters);

                    Checkpoint::load(
                        path,
                        parameterPointers,
                        metadata,
                        history
                    );
                },
                py::arg("path"),
                py::arg("parameters"),
                py::arg("metadata"),
                py::arg("history"),
                "Load checkpoint state into existing parameters, metadata, and history."
            );
    }

    void bindTrainableModel(py::module_& module)
    {
        py::class_<TrainableModel>(
            module,
            "TrainableModel",
            "Abstract interface implemented by trainable native models."
        )
            .def(
                "forward",
                &TrainableModel::forward,
                py::arg("inputs")
            )
            .def(
                "backward",
                &TrainableModel::backward,
                py::arg("grad_output")
            )

            .def(
                "parameters",
                [](TrainableModel& model)
                {
                    py::list result;

                    py::object owner = py::cast(
                        &model,
                        py::return_value_policy::reference
                    );

                    for (Parameter* parameter : model.parameters()) {
                        result.append(
                            py::cast(
                                parameter,
                                py::return_value_policy::reference_internal,
                                owner
                            )
                        );
                    }

                    return result;
                }
            );
    }

    void bindEpochContext(py::module_& module)
    {
        py::class_<EpochContext>(
            module,
            "EpochContext",
            "Training state supplied to callbacks at the end of an epoch."
        )
            .def(py::init<>())

            .def_readwrite(
                "epoch",
                &EpochContext::epoch
            )
            .def_readwrite(
                "total_epochs",
                &EpochContext::totalEpochs
            )
            .def_readwrite(
                "global_step",
                &EpochContext::globalStep
            )
            .def_readwrite(
                "train_loss",
                &EpochContext::trainLoss
            )
            .def_readwrite(
                "validation_loss",
                &EpochContext::validationLoss
            )
            .def_readwrite(
                "has_validation_loss",
                &EpochContext::hasValidationLoss
            )
            .def_readwrite(
                "device",
                &EpochContext::device
            )
            .def_readwrite(
                "run_name",
                &EpochContext::runName
            )
            .def_readwrite(
                "checkpoint_every_epochs",
                &EpochContext::checkpointEveryEpochs
            )
            .def_readwrite(
                "checkpoint_directory",
                &EpochContext::checkpointDirectory
            )

            .def(
                "__repr__",
                [](const EpochContext& context)
                {
                    return
                        "EpochContext(epoch=" +
                        std::to_string(context.epoch) +
                        ", total_epochs=" +
                        std::to_string(context.totalEpochs) +
                        ", global_step=" +
                        std::to_string(context.globalStep) +
                        ", train_loss=" +
                        std::to_string(context.trainLoss) +
                        ")";
                }
            );
    }

    void bindEpochCallback(py::module_& module)
    {
        py::class_<EpochCallback>(
            module,
            "EpochCallback",
            "Abstract callback invoked at the end of a training epoch."
        )
            .def(
                "on_epoch_end",
                &EpochCallback::onEpochEnd,
                py::arg("context"),
                py::arg("model"),
                py::arg("optimizer"),
                py::arg("history")
            )

            .def_property_readonly(
                "should_stop_training",
                &EpochCallback::shouldStopTraining
            );
    }

    void bindCheckpointCallback(py::module_& module)
    {
        py::class_<
            CheckpointCallback,
            EpochCallback
        >(
            module,
            "CheckpointCallback",
            "Save an epoch checkpoint at a fixed interval."
        )
            .def(
                py::init<
                const std::string&,
                int
                >(),
                py::arg("checkpoint_directory"),
                py::arg("every_n_epochs")
            );
    }

    void bindBestCheckpointCallback(py::module_& module)
    {
        py::class_<
            BestCheckpointCallback,
            EpochCallback
        >(
            module,
            "BestCheckpointCallback",
            "Save a checkpoint whenever the monitored loss improves."
        )
            .def(
                py::init<
                const std::string&,
                float,
                bool
                >(),
                py::arg("checkpoint_directory"),
                py::arg("min_delta") = 0.0f,
                py::arg("prefer_validation_loss") = true
            )

            .def_property_readonly(
                "has_best_checkpoint",
                &BestCheckpointCallback::hasBestCheckpoint
            )
            .def_property_readonly(
                "best_metric",
                &BestCheckpointCallback::getBestMetric
            )
            .def_property_readonly(
                "best_epoch",
                &BestCheckpointCallback::getBestEpoch
            )
            .def_property_readonly(
                "best_checkpoint_path",
                [](const BestCheckpointCallback& callback)
                {
                    return callback.getBestCheckpointPath();
                }
            );
    }

    void bindEarlyStoppingCallback(py::module_& module)
    {
        py::class_<
            EarlyStoppingCallback,
            EpochCallback
        >(
            module,
            "EarlyStoppingCallback",
            "Request training termination after a loss metric stops improving."
        )
            .def(
                py::init<
                int,
                float,
                bool
                >(),
                py::arg("patience"),
                py::arg("min_delta") = 0.0f,
                py::arg("prefer_validation_loss") = true
            )

            .def_property_readonly(
                "has_best_metric",
                &EarlyStoppingCallback::hasBestMetric
            )
            .def_property_readonly(
                "best_metric",
                &EarlyStoppingCallback::getBestMetric
            )
            .def_property_readonly(
                "best_epoch",
                &EarlyStoppingCallback::getBestEpoch
            )
            .def_property_readonly(
                "epochs_without_improvement",
                &EarlyStoppingCallback::getEpochsWithoutImprovement
            );
    }

    void bindLearningRateSchedulerCallback(py::module_& module)
    {
        py::class_<
            LearningRateSchedulerCallback,
            EpochCallback
        >(
            module,
            "LearningRateSchedulerCallback",
            "Update optimizer learning rate at epoch boundaries."
        )
            .def(
                py::init<
                LearningRateSchedule,
                int,
                float,
                float
                >(),
                py::arg("schedule"),
                py::arg("step_size") = 1,
                py::arg("gamma") = 0.1f,
                py::arg("minimum_learning_rate") = 0.0f
            )

            .def_property_readonly(
                "initial_learning_rate",
                &LearningRateSchedulerCallback::
                getInitialLearningRate
            )
            .def_property_readonly(
                "current_learning_rate",
                &LearningRateSchedulerCallback::
                getCurrentLearningRate
            );
    }

    void bindGenerationCallback(py::module_& module)
    {
        py::class_<GenerationSnapshot>(
            module,
            "GenerationSnapshot",
            "Generated text captured at the end of an epoch."
        )
            .def(py::init<>())

            .def_readwrite(
                "epoch",
                &GenerationSnapshot::epoch
            )
            .def_readwrite(
                "text",
                &GenerationSnapshot::text
            )

            .def(
                "__repr__",
                [](const GenerationSnapshot& snapshot)
                {
                    return
                        "GenerationSnapshot(epoch=" +
                        std::to_string(snapshot.epoch) +
                        ", text_length=" +
                        std::to_string(snapshot.text.size()) +
                        ")";
                }
            );


        py::class_<
            GenerationCallback,
            EpochCallback
        >(
            module,
            "GenerationCallback",
            "Generate and retain text snapshots during training."
        )
            .def(
                py::init<
                const Tokenizer&,
                const std::string&,
                const GenerationConfig&,
                int,
                bool,
                uint32_t
                >(),
                py::arg("tokenizer"),
                py::arg("prompt"),
                py::arg("generation_config"),
                py::arg("every_n_epochs") = 1,
                py::arg("print_generated_text") = true,
                py::arg("random_seed") = 1234,

                /*
                    GenerationCallback stores a tokenizer reference.
                    Keep the tokenizer alive as long as the callback lives.
                */
                py::keep_alive<1, 2>()
            )

            .def_property_readonly(
                "snapshots",
                [](const GenerationCallback& callback)
                {
                    return callback.getSnapshots();
                }
            )

            .def_property_readonly(
                "latest_snapshot",
                [](const GenerationCallback& callback)
                {
                    return callback.getLatestSnapshot();
                }
            )

            .def(
                "clear_snapshots",
                &GenerationCallback::clearSnapshots
            );
    }

    void bindCallbacks(py::module_& module)
    {
        bindEpochContext(module);
        bindEpochCallback(module);

        bindCheckpointCallback(module);
        bindBestCheckpointCallback(module);
        bindEarlyStoppingCallback(module);
        bindLearningRateSchedulerCallback(module);
        bindGenerationCallback(module);
    }

    void bindTrainer(py::module_& module)
    {
        py::class_<Trainer>(
            module,
            "Trainer",
            "Native training loop for a TrainableModel."
        )
            .def(
                py::init<
                TrainableModel&,
                Loss&,
                Optimizer&,
                const TrainingConfig&
                >(),
                py::arg("model"),
                py::arg("loss_function"),
                py::arg("optimizer"),
                py::arg("config"),

                /*
                    Trainer stores references to these three objects.
                    Keep them alive for at least as long as the Trainer.
                */
                py::keep_alive<1, 2>(),
                py::keep_alive<1, 3>(),
                py::keep_alive<1, 4>()
            )

            .def_property_readonly(
                "history",
                static_cast<TrainingHistory & (Trainer::*)()>(
                    &Trainer::getHistory
                    ),
                py::return_value_policy::reference_internal,
                "Training history owned by this Trainer."
            )

            .def(
                "train",
                [](
                    Trainer& trainer,
                    const py::iterable& trainBatches,
                    py::object validationBatches
                    )
                {
                    std::vector<TrainingBatch> training =
                        collectTrainingBatches(
                            trainBatches
                        );

                    std::vector<TrainingBatch> validation;

                    const std::vector<TrainingBatch>*
                        validationPointer = nullptr;

                    if (!validationBatches.is_none()) {
                        validation =
                            collectTrainingBatches(
                                validationBatches.cast<
                                py::iterable
                                >()
                            );

                        validationPointer = &validation;
                    }

                    /*
                        Training is entirely native once inputs have been
                        converted. Release the GIL so notebooks and other
                        Python threads are not blocked unnecessarily.
                    */
                    py::gil_scoped_release release;

                    trainer.train(
                        training,
                        validationPointer
                    );
                },
                py::arg("train_batches"),
                py::arg("validation_batches") = py::none(),
                "Train for the configured number of epochs."
            )

            .def(
                "evaluate",
                [](
                    Trainer& trainer,
                    const py::iterable& validationBatches
                    )
                {
                    std::vector<TrainingBatch> validation =
                        collectTrainingBatches(
                            validationBatches
                        );

                    py::gil_scoped_release release;

                    return trainer.evaluate(
                        validation
                    );
                },
                py::arg("validation_batches"),
                "Return mean loss across validation batches."
            )

            .def(
                "add_callback",
                &Trainer::addCallback,
                py::arg("callback"),

                /*
                    Trainer stores only an EpochCallback pointer.
                    Keep the Python callback object alive with Trainer.
                */
                py::keep_alive<1, 2>(),
                "Register an epoch callback."
            )

            .def(
                "clear_callbacks",
                &Trainer::clearCallbacks,
                "Remove all registered callbacks."
            )

            .def(
                "__repr__",
                [](const Trainer& trainer)
                {
                    return
                        "Trainer(history_entries=" +
                        std::to_string(
                            trainer.getHistory()
                            .trainLosses.size()
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
    bindRandom(module);
	bindTextDataset(module);
	bindParameter(module);
    bindLinear(module);
	bindEmbedding(module);
	bindLayerNorm(module);
    bindFFN(module);
    bindAttentionLayers(module);
    bindTransformerBlock(module);
    bindTrainableModel(module);
	bindTransformer(module);
    bindLosses(module);
    bindOptimizers(module);
	bindTrainingHistory(module);
	bindTrainingBatch(module);
	bindCheckpoint(module);
    bindCallbacks(module);
	bindTrainer(module);
}