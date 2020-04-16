import re
import pickle
from keras.preprocessing.sequence import pad_sequences

def text_cleaner(text):
    # lower case text
    text = text.lower()
    text = re.sub(r"'s\b","",text)
    # remove punctuations
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    text = re.sub("[^a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]", " ", text)
    return text

def encode_sequence(mapping, text, padding=False, seq_length=40):
    encoded_seq = [mapping[char] for char in text]
    if padding:
        encoded_seq = pad_sequences([encoded_seq], maxlen=seq_length, truncating='pre')
    return encoded_seq

def generate_mapping(text, mapping_file='generic_mapping.pkl'):
    chars = sorted(list(set(text)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    # save the mapping
    pickle.dump(mapping, open(mapping_file, 'wb'))
    return mapping

def decode_sequence(mapping, text):
    out_text = ""
    for i in range(len(text)):
        for char, index in mapping.items():
            if index == text[i]:
                out_text += char
                break
    return out_text
