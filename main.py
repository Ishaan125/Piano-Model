import argparse
import os
from pathlib import Path
import torch

from preprocess import run_preprocessing
from train import train_main
from load_model import load_model, autoregressive_generate
from output import to_midi

def ensure_and_preprocess(folder_path: str, seq_length: int = 20, is_test: bool = False):
    p = Path(folder_path)
    if not p.exists():
        print(f"Warning: folder does not exist: {p.resolve()}")
        return False
    print(f"Preprocessing folder: {p.resolve()} (test={is_test})")
    run_preprocessing(str(p), seq_length=seq_length, save=True, test=is_test)
    return True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-folder', default=os.path.join("Datasets","Classical","Classical"))
    p.add_argument('--test-folder', default=None)
    p.add_argument('--preprocess', default=False, action='store_true', help='Run preprocessing before training')
    p.add_argument('--train', default=False, action='store_true', help='Run training')
    p.add_argument('--generate', default=True, action='store_true', help='Run generation from checkpoint')
    p.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    p.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--seed', nargs='*', type=int, default=[60,62,64,65,67])
    p.add_argument('--gen-steps', type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    print("Using device:", args.device)
    # Preprocess
    if args.preprocess:
        ensure_and_preprocess(args.train_folder, seq_length=20, is_test=False)
        if args.test_folder:
            ensure_and_preprocess(args.test_folder, seq_length=20, is_test=True)

    # Train
    if args.train:
        print("Starting training...")
        train_main()

    # Generate
    if args.generate:
        print("Loading model and generating...")
        model = load_model(args.checkpoint, device=args.device)
        seq = args.seed
        out = autoregressive_generate(model, seq, gen_steps=args.gen_steps, device=args.device, repeat_penalty=1.3)
        print("Generated tokens (first 20):", out[:20])
        to_midi(out, out_path='gen2.mid')
        print("MIDI saved to gen2.mid")

if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()