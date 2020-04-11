import os
import argparse
from model import GenericLM
from text_utils import *
from file_utils import *

def main(args):
    raw_text = load_data(args.corpus)
    raw_text = text_cleaner(raw_text)
    mapping = generate_mapping(raw_text)
    vocab_size = len(mapping)

    generic_lm = GenericLM(vocab_size, mapping, seq_length=args.seq_length,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)

    if args.mode == 'left2right':
        generic_lm.fit(raw_text, epochs=args.epochs)
    elif args.mode == 'right2left':
        raw_text = raw_text[::-1]
        generic_lm.fit(raw_text, epochs=args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='truyen_kieu.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='left2right')

    args = parser.parse_args()

    assert args.mode in ['left2right', 'right2left'], "Choose one of these mode: left2right, right2left."
    main(args)
