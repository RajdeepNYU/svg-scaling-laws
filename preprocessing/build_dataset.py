print("BUILD_DATASET_VERSION_3")

import os
import random
from tqdm import tqdm
from datasets import load_dataset

from preprocessing.clean_svg import (
    clean_svg,
    is_valid,
    is_reasonable_length,
    canonicalize
)

OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Loading svg-icons-simple...")
    icons = load_dataset("starvector/svg-icons-simple", split="train")

    print("Loading svg-emoji-simple...")
    emoji = load_dataset("starvector/svg-emoji-simple", split="train")

    print("Loading 10% of svg-fonts-simple...")
    fonts = load_dataset(
        "starvector/svg-fonts-simple",
        split="train[:27%]"
    )

    all_svgs = []

    for ds in [icons, emoji, fonts]:
        for item in ds:
            all_svgs.append(item["Svg"])

    print(f"Total raw SVG samples: {len(all_svgs)}")
    return all_svgs

def process_dataset(svg_list):
    cleaned = []

    for svg in tqdm(svg_list):
        svg = clean_svg(svg)

        if not is_valid(svg):
            continue

        if not is_reasonable_length(svg):
            continue

        svg = canonicalize(svg)
        cleaned.append(svg)

    return cleaned

def split_dataset(data):
    random.shuffle(data)

    n = len(data)
    train = data[:int(0.98 * n)]
    val   = data[int(0.98 * n):int(0.99 * n)]
    test  = data[int(0.99 * n):]

    return train, val, test

def write_split(name, data):
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path, "w") as f:
        for svg in data:
            svg = svg.replace("\n", " ")
            f.write(svg + "\n")

def main():
    raw_svgs = load_data()
    cleaned = process_dataset(raw_svgs)

    print(f"Total cleaned SVGs: {len(cleaned)}")

    train, val, test = split_dataset(cleaned)

    write_split("train", train)
    write_split("val", val)
    write_split("test", test)

    print("Dataset splits saved.")

if __name__ == "__main__":
    main()
