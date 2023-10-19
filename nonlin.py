import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
from random import randint as r_int, getrandbits, shuffle, choice

chars = "0123456789.=+-*"[:]
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)


## Create data
block_size = 24

def build_data(examples:int, digits = 6):
  data = []
  X,Y =[], []

  for i in range(examples):
    a, b = r_int(0, 10**digits), r_int(0, 10**digits)
    a_negative, b_negative = bool(getrandbits(1)), bool(getrandbits(1))
    c = (-a if a_negative else a) + (-b if b_negative else b)
    x = ("-" if a_negative else "") + str(a)[::-1] + ("-" if b_negative else "+") + str(b)[::-1] + "="
    y = ("-" if c < 0 else "") + str(abs(c))[::-1] + "."
    data.append([x,y])

  for x,y in data:
    context = (block_size - len(x)) * [10] + [stoi[ix] for ix in x]
    for char in y:
      iy = stoi[char]
      X.append(context)
      Y.append(iy)
      context = context[1:] + [iy]

  combined = list(zip(X, Y))
  shuffle(combined)
  X, Y = zip(*combined)
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

Xtr, Ytr = build_data(10_000)
Xte, Yte = build_data(10_000)


## NN classes
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    batch_size = IX.shape[0]
    self.out = self.weight[IX].view(batch_size,-1)
    return self.out
  
  def parameters(self):
    return [self.weight]

# -----------------------------------------------------------------------------------------------
class Nonlin():
  def __call__(self, x):
    self.out = 2*x/(x**2+1)
    return self.out
  def parameters(self):
    return []
  # -----------------------------------------------------------------------------------------------
class Linear:
  
  def __init__(self, fan_in, fan_out):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
  
  def __call__(self, x):
    self.out = x @ self.weight
    return self.out
  
  def parameters(self):
    return [self.weight]
# -----------------------------------------------------------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

# -----------------------------------------------------------------------------------------------
class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]


## Creating NN
# hierarchical network
n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of the MLP
model = Sequential([
  Embedding(vocab_size, n_embd),
  Linear(n_embd*block_size, n_hidden), Nonlin(),
  Linear(n_hidden, n_hidden), Nonlin(),
  Linear(n_hidden, n_hidden), Nonlin(),
  Linear(n_hidden, vocab_size),
])


## NN init
with torch.no_grad():
  model.layers[-1].weight *= 0.1 # last layer make less confident

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
  
## Training NN
# same optimization as last time
max_steps = 25_000
batch_size = 32
lossi = []
avg_loss = 2.7
lrs = [0.1, 0.075, 0.05, 0.025, 0.01, 0.005]

for i in range(max_steps):
  # minibatch construct
  Xb, Yb = build_data(batch_size) # batch X,Y
  
  # forward pass
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb) # loss function
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  avg_loss = avg_loss * 0.99 + loss.item() * 0.01
  
  # update: simple SGD
  for p in parameters:
    p.data += -lrs[(i*6)//max_steps] * p.grad

  # track stats
  if i % 5_000 == 0: # print every once in a while
    if i == 0:
      print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    else:
      print(f'{i:7d}/{max_steps:7d}: {avg_loss:.4f}')
  lossi.append(loss.item())
  

## Visualisations
# put layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
  layer.training = False
  
# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'test': (Xte, Yte),
  }[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('test')

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()