# implementing a data loader that implements inut target pairs using sliding window approach
import tiktoken
with open("data/raw/the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()


tokenizer=tiktoken.get_encoding("gpt2") #we got byte-pair encoding used by gpt-2
enc_text=tokenizer.encode(raw_text)

enc_sample=enc_text[50:]

context_size=4 #number of words needed to predict the next word

x=enc_sample[:context_size]
y=enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y:      {y}")

for i in range(1 ,context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    # print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    #print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

from torch.utils.data import Dataset, DataLoader
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt ,tokenizer ,  max_length , stride):
        self.input_ids=[]
        self.target_ids=[]
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0 ,len(token_ids) - max_length , stride): #We want the last start index such that i + max_length + 1 <= len(token_ids)
            input_chunks=token_ids[i:i+max_length]
            target_chunks=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    #  dataloader will help us do parallel processing

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


#creating vector embeddings

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer(torch.tensor([3]))) 
print(embedding_layer(input_ids)) 