import subprocess

learning_rates = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3
]

for lr in learning_rates:
    out_dir = f"outputs/lr_sweep_tiny_mup/lr_{lr}"

    cmd = [
        "python",
        "training/train_mup.py",
        "--config",
        "configs/tiny.json",
        "--dataset",
        "data/tokenized",
        "--learning_rate",
        str(lr),
        "--out_dir",
        out_dir,
    ]

    print("=" * 80)
    print(f"Running µP LR = {lr}")
    print("=" * 80)

    subprocess.run(cmd, check=True)
