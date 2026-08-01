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
}