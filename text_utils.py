import re
import pickle
from keras.preprocessing.sequence import pad_sequences

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    newString = re.sub("[^a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]", " ", newString)
    return newString

def encode_sequence(mapping, text, padding=False):
    encoded_seq = [mapping[char] for char in text]
    if padding:
        encoded_seq = pad_sequences([encoded_seq], maxlen=len(mapping), truncating='pre')
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
            if index == in_text[i]:
                out_text += char
                break
    return out_text
