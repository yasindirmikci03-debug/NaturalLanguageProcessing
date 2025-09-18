import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size=32
block_size=8
max_iters=5000
eval_interval=500
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd = 32
B,T,C = 4,8,2

# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt','r',encoding="utf-8") as f:
    text = f.read()

print("length of the dataset in characters:", len(text))

print(text[:1000])

## here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#### Tokenization,train/val split

# creating a mapping f  m characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]     # encoder: take a string, output list of integers
decode = lambda l: ''.join(itos[i] for i in l)      # decoder :  take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.

data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000])

# Let's now split up the data into train and validation sets
n = int(0.9*len(data))      # first 90% will be train,rest val
train_data = data[:n]
val_data = data[n:]


##### Data Loader : Batches of chunks of data

block_size = 8
train_data[: block_size + 1]


x = train_data[:block_size]   # x are the inputs to the transformers it will just be the first block size char
y = train_data[1:block_size+1]  # y will be the next block size characters so it's offset by one , y are the targets for each position
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target : {target}")


torch.manual_seed(1337)
batch_size = 4      # how many independent sequences will we process in parallel?
block_size = 8      # what is the maximum context length for predictions ?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

xb,yb = get_batch('train')
print("inputs :") 
print(xb.shape)
print(xb)
print("targets : ")
print(yb.shape)
print(yb)

print("----")

for b in range(batch_size):     # batch dimension
    for t in range(block_size):     # time dimension
        context = xb[b, : t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target : {target}")


#### Simple baseline : bigram language model, loss, generation
torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5       # (B,T,C) @ (B,C,T) -> (B,T,T)  
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))     # (B,T,T)
        wei = F.softmax(wei,dim=-1) 
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in  range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block : communication followed by computation"""

    def __init__(self,n_embd,n_head):
        # n_embd = embedding dimension, n_head = the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):

        B, T = idx.shape

        # idx and targets are both(B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)


        return logits , loss
    
    def generate(self,idx,max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits,loss = self.forward(idx)
            # focus only on the last time step
            logits = logits[:,-1, :] # Becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1) # (B,C)
            # sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1)   #(B, T+1) 
        return idx
    


m= BigramLanguageModel().to(device)
logits, loss= m(xb,yb)
print(logits.shape)
print(loss)
print(decode(m.generate(torch.zeros((1,1), dtype = torch.long),max_new_tokens=100)[0].tolist()))

##### Training the bigram model

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

batch_size = 32
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb,yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(loss.item())

context = torch.zeros((1,1), dtype = torch.long)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))

### The mathematical trick in self-attention

# consider the following toy example:
# B,T,C = 4,8,2
# x = torch.randnd(B,T,C)

# # We want x[b,t] = mean_{i<=t} x[b,i]
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, : t+1] # (t,C)
#         xbow[b,t] = torch.mean(xprev,0)

# version 2:
# wei = torch.tril(torch.ones(T,T))
# wei = wei / wei.sum(1,keepdim=True)
# xbow2 = wei @ x     # (B,T,T) @ (B,T,C) ----> (B,T,C)


# version 3: adding softmax
# tril = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = F.softmax(wei,dim=-1)
# xbow3 = wei @ x

# version 4 : self-attention!
# B,T,C = 4,8,32
# x = torch.randn(B,T,C)

# # let's see a single Head perform self-attention
# head_size = 16
# key = nn.Linear(C,head_size,bias = False)
# query = nn.Linear(C,head_size,bias = False)
# value = nn.Linear(C,head_size,bias = False)
# k = key(x)  # (B,T,16)
# q = query(x)  # (B,T,16)
# wei = q @ k.transpose(-2,-1)    # (B,T,16) @ (B,16,T) ---> (B,T,T)

# tril = torch.tril(torch.ones(T,T), diagonal=0).to(device)
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = F.softmax(wei,dim=-1)

# v = value(x)
# out = wei @ v

# print(wei[0])
