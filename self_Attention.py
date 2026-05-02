#simplified self attention on entire sentence
import torch

inputs=torch.tensor(
[
   [0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55] # step     (x^6)
]
)
query = inputs[1] # 2nd input token is the query

#calculating attention scores through loops
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

#making use of linear algebra and performing simplified self attention
# since loops are slower we gonna do matrix multiplication btw input matrix and input transpose , gotta make use of linear algebra
attn_scores = inputs @ inputs.T
print(attn_scores)
#normalization
attn_weights = torch.softmax(attn_scores, dim=-1) #dim=-1 means we gotta normalize along the columns
print(attn_weights)
#finally multiply the attention weigth with inputs and get the context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)