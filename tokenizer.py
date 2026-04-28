import re

with open("data/raw/the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
# print("Total number of characters: ", len(raw_text))
# print(raw_text[:10])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])
# print(len(preprocessed))

all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
# print(vocab_size)

vocab={token:integer for integer,token in enumerate(all_words)}
# for i,item in enumerate(vocab.items()):
#     print(item)

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text=" ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text