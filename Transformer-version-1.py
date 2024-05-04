# External Library
import torch
import numpy
import torch.nn as nn
from torch.nn import functional as f
import requests


# Hyper-parameters

batch_size = 4
block_size = 8
device = 'cuda' if torch.cuda.is_available() else "cpu"
iteration = 30000
learning_rate = 1e-3
torch.manual_seed(1337)
eval_interval = 300
eval_iter = 200
vocab_size = 65
n_emb= 32




# Data Preparation

# Let's download the File
# wget = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# req = requests.get(wget)
# with open("input.txt","w",encoding="utf-8") as file:
#     file.write(req.text)

# Let's Read the File

with open("input.txt","r") as file:
    text = file.read()

print("The total length of document:",len(text))

# Let Design the Encoder and Decoder
vocabulary = sorted(list(set(text)))
vocal_size = len(vocabulary)
print("The vocal_size:",vocal_size)

itostr = {i:ch for i,ch in enumerate(vocabulary)}
strtoi = {ch:i for i,ch in enumerate(vocabulary)}

Encoder  = lambda x : [strtoi[i] for i in x]
Decoder = lambda x : "".join([itostr[i] for i in x ])


data = torch.tensor(Encoder(text),dtype=torch.long)

# Data Split

n = int(0.90 * len(data))

train_data = data[:n]
val_data = data[n:]


x = train_data[:block_size]
y = train_data[1:block_size+1]

for i in range(block_size):
    context = x[:i+1]
    target = y[i]


# Let's Create the batch

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x,y = x.to(device),y.to(device)
    return x,y

xb,yb = get_batch("train")
print(xb)
print()
print(yb)

@torch.no_grad()
def estimation_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X,Y = get_batch("train")
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# Let's Build the Model

class bigrammodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_emb)
        self.positonal_embedding = nn.Embedding(block_size,n_emb)
        self.ln_head = nn.Linear(n_emb,vocab_size)

    def forward(self,idx,target=None):
        B,T = idx.shape
        token_emb = self.embedding_layer(idx)   # (B,T,C)
        postional_embedding = self.positonal_embedding(torch.arange(T,device=device)) # (T,C)
        x = token_emb + postional_embedding #(B,T,C)
        logits = self.ln_head(x) #(B,T,vocab_size)
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)

            target = target.view(B*T)
            loss = f.cross_entropy(logits,target)
        return logits,loss


    def generate(self,idx,maximum_num_token):
        for _ in range(maximum_num_token):
            # Let get the prediction

            logits,loss = self(idx)


            # Let Get the last step
            logits = logits[:,-1,:]

            prob = f.softmax(logits,dim=1)
            idx_next = torch.multinomial(prob,num_samples=1)

            idx = torch.cat([idx,idx_next],dim=1)

        return idx
model = bigrammodel()
model = model.to(device)

logits,loss = model(xb,yb)
print(loss)
#print(Decoder(model.generate(idx = torch.zeros((1,1),dtype=torch.long,device=device),maximum_num_token=1000)[0].tolist()))


# Model Training

# Let's create optimizer

Optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(iteration):

    # Every once in a while evaluate the model loss  on train  and val dataset

    if iter % eval_interval ==0:
        losses = estimation_loss()
        print(f'Step {iter} model the train loss {losses["train"]} and val loss {losses["val"]:.4f}')


    xb,yb = get_batch(train_data)
    logits,loss = model(xb,yb)
    Optimizer.zero_grad(set_to_none=True)
    loss.backward()
    Optimizer.step()
print(loss.item())

























