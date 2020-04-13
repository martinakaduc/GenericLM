import os
import argparse
from model import GenericLM, GenericLM_multi
from text_utils import *
from file_utils import *
from _thread import *

def train_left(args, vocab_size, mapping, raw_text):
    generic_lm_l2r = GenericLM(vocab_size, mapping, seq_length=args.seq_length,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path+'/left', model_path=args.model_path, mode_name=args.mode)
    generic_lm_l2r.fit(raw_text, epochs=args.epochs, ckpt_period=args.ckpt_period)

def train_right(args, vocab_size, mapping, raw_text):
    generic_lm_r2l = GenericLM(vocab_size, mapping, seq_length=args.seq_length,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path+'/right', model_path=args.model_path, mode_name=args.mode)

    generic_lm_r2l.fit(raw_text[::-1], epochs=args.epochs, ckpt_period=args.ckpt_period)

def main(args):
    raw_text = load_data(args.corpus)
    raw_text = text_cleaner(raw_text)
    mapping = generate_mapping(raw_text)
    vocab_size = len(mapping)

    if args.mode == 'both':
        start_new_thread(train_left, (args, vocab_size, mapping, raw_text))
        start_new_thread(train_right, (args, vocab_size, mapping, raw_text))

    else:
        if not args.multi_gpu:
            generic_lm = GenericLM(vocab_size, mapping, seq_length=args.seq_length,
                        batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)
        else:
            generic_lm = GenericLM_multi(vocab_size, mapping, seq_length=args.seq_length,
                        batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)

        if args.mode == 'left2right':
            generic_lm.fit(raw_text, epochs=args.epochs, ckpt_period=args.ckpt_period)
        elif args.mode == 'right2left':
            raw_text = raw_text[::-1]
            generic_lm.fit(raw_text, epochs=args.epochs, ckpt_period=args.ckpt_period)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='truyen_kieu.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seq_length', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='left2right')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--ckpt_period', type=int, default=1)

    args = parser.parse_args()

    assert args.mode in ['left2right', 'right2left', 'both'], "Choose one of these mode: left2right, right2left."
    main(args)
