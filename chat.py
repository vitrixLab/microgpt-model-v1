import os, math, random, pickle, time
import numpy as np
from wordnet_loader import build_definition_cache

# ---------- Config – MUST match training ----------
n_layer = 1
n_embd = 16
block_size = 32
n_head = 4
head_dim = n_embd // n_head

# ---------- Load vocabulary ----------
if not os.path.exists("vocab.pkl"):
    print("Error: vocab.pkl missing. Run microgpt.py first.")
    exit()
with open("vocab.pkl", "rb") as f:
    uchars = pickle.load(f)
BOS = len(uchars)
vocab_size = len(uchars) + 1

# ---------- Autograd engine (same iterative backward) ----------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, 0)]
        while stack:
            v, idx = stack[-1]
            if idx == 0:
                visited.add(v)
            if idx < len(v._children):
                child = v._children[idx]
                stack[-1] = (v, idx+1)
                if child not in visited:
                    stack.append((child, 0))
            else:
                topo.append(v)
                stack.pop()
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------- Rebuild model parameters ----------
random.seed(42)
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# ---------- Load saved weights ----------
if not os.path.exists("model_weights.pkl"):
    print("Error: model_weights.pkl missing. Train first.")
    exit()
with open("model_weights.pkl", "rb") as f:
    saved = pickle.load(f)
if len(saved) != len(params):
    print(f"Error: weight count mismatch ({len(saved)} vs {len(params)}). Model config differs.")
    exit()
for p, val in zip(params, saved):
    p.data = val
print(f"Loaded {len(params)} parameters.")

# ---------- Weight statistics ----------
weights = np.array([p.data for p in params])
print("\n--- Weight Statistics ---")
print(f"Min: {weights.min():.4f}  Max: {weights.max():.4f}")
print(f"Mean: {weights.mean():.4f}  Std: {weights.std():.4f}")
hist, bins = np.histogram(weights, bins=5)
print(f"Histogram: {hist} (bins: {bins})\n")

# ---------- Helper functions (same as training) ----------
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, state_dict['lm_head'])

# ---------- WordNet cache ----------
definition_cache = build_definition_cache()

def extract_lookup_word(question):
    q = question.lower().strip().translate(str.maketrans('', '', '?,.!'))
    for phrase in ["what does", "define", "meaning of", "definition of"]:
        if phrase in q:
            after = q.split(phrase)[-1].strip()
            if after:
                word = after.split()[0]
                if word in definition_cache:
                    return word
    for w in q.split():
        if w in definition_cache:
            return w
    return None

def rule_based_answer(question):
    word = extract_lookup_word(question)
    if word:
        return f"{word}: {definition_cache[word]}"
    return None

# ---------- SLM generation ----------
def ask(question, temperature=0.5, max_new_tokens=50):
    prompt = f" Q: {question} A:"
    tokens = [BOS] + [uchars.index(c) for c in prompt if c in uchars]
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    logits = None
    for pos, tok in enumerate(tokens):
        logits = gpt(tok, pos, keys, values)

    answer_tokens = []
    # Stop when we reach block_size-1 (max position is block_size-1)
    while len(tokens) + len(answer_tokens) < block_size:
        scaled = [l / temperature for l in logits]
        probs = softmax(scaled)
        next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if next_token == BOS:
            break
        answer_tokens.append(uchars[next_token])
        logits = gpt(next_token, len(tokens) + len(answer_tokens), keys, values)
        if len(tokens) + len(answer_tokens) >= block_size:
            break   # safety, should already be caught by while condition

    return ''.join(answer_tokens)

# ---------- Interactive chat (CLI) ----------
print("Chatbot ready. Ask a question (or type 'quit' to exit).\n")
while True:
    user = input("You: ")
    if user.lower() in ("quit", "exit"):
        break
    cached = rule_based_answer(user)
    if cached:
        print(f"Bot (cache): {cached}")
    else:
        print("Bot (SLM): ", end="")
        answer = ask(user, temperature=0.4)
        print(answer)