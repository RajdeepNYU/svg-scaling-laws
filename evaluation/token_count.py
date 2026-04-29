import numpy as np
import os

TOKENIZED_DIR = "data/tokenized"


def count_tokens(file_path):
    arr = np.fromfile(file_path, dtype=np.uint16)
    return len(arr)


def main():
    train_tokens = count_tokens(os.path.join(TOKENIZED_DIR, "train.bin"))
    val_tokens   = count_tokens(os.path.join(TOKENIZED_DIR, "val.bin"))
    test_tokens  = count_tokens(os.path.join(TOKENIZED_DIR, "test.bin"))

    total = train_tokens + val_tokens + test_tokens

    print(f"Train tokens: {train_tokens}")
    print(f"Val tokens:   {val_tokens}")
    print(f"Test tokens:  {test_tokens}")
    print(f"Total tokens: {total}")


if __name__ == "__main__":
    main()
