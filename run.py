import argparse
import pickle
from model import GenericLM
from text_utils import *

def main(args):
    mapping = pickle.load(open('generic_mapping.pkl', 'rb'))
    vocab_size = len(mapping)

    generic_lm = GenericLM(vocab_size, mapping, seq_length=args.seq_length, multi_gpu=args.multi_gpu,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)

    while True:
        input_text = input('Input: ')
        if input_text.lower() == 'exit':
            break

        if (args.mode == 'right2left'):
            input_text = input_text[::-1]

        for _ in range(args.predict_length):
            encoded_text = encode_sequence(mapping, input_text.lower(), padding=True, seq_length=args.seq_length)
            next_char = generic_lm.predict(encoded_text, return_prob_table=False, return_label=True)[0]
            input_text += next_char

        sentence_prob = 1
        for i, char in enumerate(input_text):
            encoded_text = encode_sequence(mapping, input_text[:i].lower(), padding=True, seq_length=args.seq_length)
            prob_table = generic_lm.predict(encoded_text, return_prob_table=True)[0]
            cur_idx = mapping[char]
            sentence_prob *= prob_table[cur_idx]

        print("Sentence probability: %.10f" % sentence_prob)

        if (args.mode == 'right2left'):
            input_text = input_text[::-1]

        print('Predict: %s\n' % input_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='left2right')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--predict_length', type=int, default=10)

    args = parser.parse_args()

    assert args.mode in ['left2right', 'right2left'], "Choose one of these mode: left2right, right2left."
    main(args)
