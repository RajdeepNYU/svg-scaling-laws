import argparse
import torch
import sentencepiece as spm
from models.transformer import GPT, GPTConfig

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/base/best.pt")
    parser.add_argument("--tokenizer", default="tokenizer/svg_bpe.model")
    parser.add_argument("--out", default="generated_samples.txt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prefix", type=str, default="<svg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model_config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )

    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    outputs = []

    for i in range(args.num_samples):
        ids = sp.encode(args.prefix, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        out = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

        text = sp.decode(out[0].tolist())

        if "</svg>" in text:
            text = text[:text.index("</svg>") + len("</svg>")]

        outputs.append(text)

    with open(args.out, "w") as f:
        for i, sample in enumerate(outputs):
            f.write(f"--- SAMPLE {i} ---\n")
            f.write(sample + "\n\n")

    print(f"Saved {len(outputs)} samples to {args.out}")

if __name__ == "__main__":
    main()
