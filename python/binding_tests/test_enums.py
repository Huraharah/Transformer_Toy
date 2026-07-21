import transformer_toy as tt


def main() -> None:
    print(f"CUDA available: {tt.cuda_available()}")

    print(f"CPU: {tt.Device.CPU}")
    print(f"CUDA: {tt.Device.CUDA}")
    print(f"AUTO: {tt.Device.AUTO}")

    print(f"Character tokenizer: {tt.TokenizerType.CHARACTER}")
    print(f"Word tokenizer: {tt.TokenizerType.WORD}")
    print(f"BPE tokenizer: {tt.TokenizerType.BPE}")

    print(f"Single-head attention: {tt.AttentionType.SINGLE_HEAD}")
    print(f"Multi-head attention: {tt.AttentionType.MULTI_HEAD}")

    print(f"Constant LR: {tt.LearningRateSchedule.CONSTANT}")
    print(f"Step decay: {tt.LearningRateSchedule.STEP_DECAY}")
    print(
        "Exponential decay: "
        f"{tt.LearningRateSchedule.EXPONENTIAL_DECAY}"
    )
    print(f"Cosine decay: {tt.LearningRateSchedule.COSINE_DECAY}")

    device = tt.Device.CUDA
    assert device == tt.Device.CUDA
    assert device != tt.Device.CPU

    schedule = tt.LearningRateSchedule.COSINE_DECAY
    assert schedule == tt.LearningRateSchedule.COSINE_DECAY

    print("Enum tests passed.")


if __name__ == "__main__":
    main()