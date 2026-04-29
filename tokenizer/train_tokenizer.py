
import sentencepiece as spm
import os

INPUT_FILE = "data/splits/train.txt"
OUTPUT_PREFIX = "tokenizer/svg_bpe"
VOCAB_SIZE = 2048

os.makedirs("tokenizer", exist_ok=True)


def main():
    print(f"Training tokenizer with vocab_size={VOCAB_SIZE}")

    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=OUTPUT_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        shuffle_input_sentence=True
    )

    print("Tokenizer training complete.")


if __name__ == "__main__":
    main()
