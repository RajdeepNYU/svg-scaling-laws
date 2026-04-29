
import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

MODEL_PATH = "tokenizer/svg_bpe.model"
INPUT_DIR = "data/splits"
OUTPUT_DIR = "data/tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)


def encode_file_to_bin(in_file, out_file):
    all_ids = []

    with open(in_file, "r") as f:
        for line in tqdm(f, desc=f"Encoding {in_file}"):
            line = line.strip()
            if not line:
                continue

            ids = sp.encode(line, out_type=int)
            ids = [sp.bos_id()] + ids + [sp.eos_id()]
            all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_file)

    print(f"Saved {out_file} ({len(arr)} tokens)")


def main():
    encode_file_to_bin(
        os.path.join(INPUT_DIR, "train.txt"),
        os.path.join(OUTPUT_DIR, "train.bin")
    )
    encode_file_to_bin(
        os.path.join(INPUT_DIR, "val.txt"),
        os.path.join(OUTPUT_DIR, "val.bin")
    )
    encode_file_to_bin(
        os.path.join(INPUT_DIR, "test.txt"),
        os.path.join(OUTPUT_DIR, "test.bin")
    )


if __name__ == "__main__":
    main()
