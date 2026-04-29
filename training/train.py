import os
import json
import math
import argparse
import time
import numpy as np
import torch
from torch.optim import AdamW

from models.transformer import GPT, GPTConfig


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64))
        for i in ix
    ])

    x = x.pin_memory().to(device, non_blocking=True) if device == "cuda" else x.to(device)
    y = y.pin_memory().to(device, non_blocking=True) if device == "cuda" else y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device, use_amp):
    model.eval()
    out = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                _, loss = model(x, y)

            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def get_lr(it, learning_rate, warmup_iters, max_iters):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/tokenized")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate

    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir

    os.makedirs(cfg["out_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.get("use_amp", True) and device == "cuda"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("AMP enabled:", use_amp)

    train_data = np.memmap(
        os.path.join(args.dataset, "train.bin"),
        dtype=np.uint16,
        mode="r"
    )

    val_data = np.memmap(
        os.path.join(args.dataset, "val.bin"),
        dtype=np.uint16,
        mode="r"
    )

    model_config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )

    model = GPT(model_config).to(device)

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    print("Parameters:", model.count_params())

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.1
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    start_time = time.time()

    for it in range(cfg["max_iters"] + 1):
        lr = get_lr(
            it,
            cfg["learning_rate"],
            cfg["warmup_iters"],
            cfg["max_iters"]
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % cfg["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                cfg["batch_size"],
                cfg["block_size"],
                cfg["eval_iters"],
                device,
                use_amp
            )

            elapsed = time.time() - start_time

            print(
                f"iter {it}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {lr:.2e}, "
                f"time {elapsed:.1f}s"
            )

            if device == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"GPU memory used: {mem_gb:.2f} GB")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                ckpt_path = os.path.join(cfg["out_dir"], "best.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "params": model.count_params(),
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_path
                )

        x, y = get_batch(
            train_data,
            cfg["batch_size"],
            cfg["block_size"],
            device
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
