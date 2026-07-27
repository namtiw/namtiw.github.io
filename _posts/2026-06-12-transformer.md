---
layout: post
title: Assembling the Transformer 
subtitle: How language models learn to look back
thumbnail-img: /assets/img/ml.jpg
mathjax: true
mermaid: true
published: false
tags: [Machine-Learning, Transformer, Self-attention, JAX]
---

The [previous post]({% post_url 2026-05-02-attention %}) introduced self-attention: Queries, Keys, Values and a softmax. Self-attention enables the model to capture semantic correlation between far apart words, allowing longer inputs. That in turn makes a rich context available to predict the output tokens. Transformers, as introduced in "Attention is all you need"[^vaswani], followed an encoder-decoder architecture with residual connections.

## Encoder-decoder architecture 

The architecture is an encoder that reads the source sentence and a decoder that writes the target sentence one token at a time. Both are built from the same two sublayers, self-attention followed by a multilayer perceptron. The decoder has a third sublayer which attends over the encoder's output, the cross-attention from the previous post. The best performing networks stack several such encoder and decoder layers, six of each in the paper. The paper also introduces the following additions to the self-attention layer to improve overall model performance.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 55, "padding": 30}}}%%
flowchart TB
  SRC["Source tokens<br/>+ positional encoding"]
  TGT["Target tokens so far<br/>+ positional encoding"]

  subgraph ENC["Encoder layer (× 6)"]
    direction TB
    ESA["Self-attention<br/>every position sees every other"] --> EFF["Feed-forward network"]
  end

  subgraph DEC["Decoder layer (× 6)"]
    direction TB
    DSA["Masked self-attention<br/>each position sees only its prefix"] --> DCA["Cross-attention<br/>queries from the decoder,<br/>keys and values from the encoder"] --> DFF["Feed-forward network"]
  end

  SRC ==> ESA
  TGT ==> DSA
  EFF ==>|"encoder output"| DCA
  DFF ==> OUT["Linear to vocabulary<br/>→ next target token"]

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
  classDef attn fill:#4c1d95,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
  classDef cross fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef ff fill:#065f46,stroke:#34d399,stroke-width:1.5px,color:#f8fafc
  classDef out fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff

  class SRC,TGT input
  class ESA,DSA attn
  class DCA cross
  class EFF,DFF ff
  class OUT out
</pre>
<figcaption>The two stacks of the original architecture. Each sublayer shown here is wrapped in a residual connection and a layer normalization, left out to keep the diagram readable; they are the subject of the next section.</figcaption>
</figure>

## Residual connections, layer normalization and Scaled dot product attention

Deep neural networks (a high number of hidden layers) are prone to the exploding/vanishing gradient problem.

$$\text{output} = \text{LayerNorm}\left(x + \text{Sublayer}(x)\right)$$

Residual connections, $x +$ [^resnet], adding the input back to the output allows the model to use gradients from deeper in the stack to apply meaningful updates during gradient descent. Layer normalization[^layernorm] transform the output of a layer such that its mean is 0 and variance 1. Scaling numbers into this range keeps the activation function, softmax in this case, out of the regions with really low gradient values.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Post-norm and pre-norm</div>
<div class="collapsiblecontent" markdown="1">
The formula above applies the normalization after the residual addition, referred to as post-norm. $x + \text{Sublayer}(\text{LayerNorm}(x))$, pre-norm, adding the residual input to normalized output of a layer. Post norm produces disproportionately large gradients during initialization [^xiong]. The Transformer paper uses a learning rate warmup schedule during training to combat this issue. The implementation later in this post uses pre-norm which does not have this limitation.
</div>
</div>


Attention scores are scaled by $\sqrt{d_k}$ before the softmax to solve for the same issue:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


## Multi-head attention

 Multi-head attention runs $h$ copies of the mechanism in parallel, each with its own query, key and value projection, and concatenates their outputs. This gives the model more independent representation spaces to allow capturing more context from the input. Intuitively, a sentence consists of multiple relevant contexts to convey full meaning. For instance, resolving a pronoun, tracking which noun a verb agrees with, and finding the object of a preposition are separate lookups.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 60}}}%%
flowchart TB
  H["Input sequence H<br/>(n × d_model)"]

  A1["head₁<br/>own W_Q, W_K, W_V<br/>(n × dₖ)"]
  A2["head₂<br/>own W_Q, W_K, W_V<br/>(n × dₖ)"]
  A3["…"]
  A4["headₕ<br/>own W_Q, W_K, W_V<br/>(n × dₖ)"]

  CAT["Concatenate<br/>(n × h·dₖ = n × d_model)"]
  WO["× W_O<br/>(d_model × d_model)"]
  OUT["Output<br/>(n × d_model)"]

  H ==> A1 & A2 & A3 & A4
  A1 & A2 & A3 & A4 ==> CAT ==> WO ==> OUT

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
  classDef head fill:#4c1d95,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
  classDef inter fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef out fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff

  class H input
  class A1,A2,A3,A4 head
  class CAT,WO inter
  class OUT out
</pre>
<figcaption>Every head is the attention module from the previous post, run at width dₖ = d_model / h.</figcaption>
</figure>

## Masked self-attention

Self-attention as built so far lets every position see every other position, which is wrong for the sentence generation use case. A language model generates one token at a time, each conditioned on the tokens before it. If during training position $i$ attends to position $i+1$, the model learns to read the answer off its own input, which will be unavailable when generating on unseen inputs. Masking fixes this: position $i$ sees positions $1$ through $i$ and nothing else, exactly the information it will have when generating.

The mask is applied before the softmax, with the masked entries set to $-\infty$. Then $e^{-\infty} = 0$ falls out of the exponential, the denominator only ever sums the legal entries, and each row renormalizes itself over the positions it is allowed to see.

<figure markdown="1">

$$
\underbrace{
\begin{bmatrix}
s_{11} & {\color{#F59E0B}-\infty} & {\color{#F59E0B}-\infty} & {\color{#F59E0B}-\infty} \\
s_{21} & s_{22} & {\color{#F59E0B}-\infty} & {\color{#F59E0B}-\infty} \\
s_{31} & s_{32} & s_{33} & {\color{#F59E0B}-\infty} \\
s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix}
}_{\text{masked scores}}
\;\xrightarrow{\;\text{softmax by row}\;}\;
\underbrace{
\begin{bmatrix}
1 & {\color{#F59E0B}0} & {\color{#F59E0B}0} & {\color{#F59E0B}0} \\
\alpha_{21} & \alpha_{22} & {\color{#F59E0B}0} & {\color{#F59E0B}0} \\
\alpha_{31} & \alpha_{32} & \alpha_{33} & {\color{#F59E0B}0} \\
\alpha_{41} & \alpha_{42} & \alpha_{43} & \alpha_{44}
\end{bmatrix}
}_{\text{attention weights, rows sum to 1}}
$$

<figcaption>The mask is applied to the scores. Each row of the result is a probability distribution, over a shorter prefix</figcaption>
</figure>

## Positional encoding

The sequential nature of RNNs let them learn that aspect of language for free, but self-attention reads everything at once, and the mechanism is permutation invariant: shuffling the rows of the input causes the rows of the output to shuffle identically, with the same values ($QK^T$), so "dog bites man" and "man bites dog" would produce the same set of representations.

The paper adds positional embeddings to the input of the network. Using sinusoids allows the model to capture relative distances between tokens alongside the semantic relationships.

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### Rotary Position Embeddings

Adding a position signal to the embedding passes on the task of learning the semantics of a token and its position in the sentence upto to the network. RoPE[^rope] avoids that. Instead of adding anything to the input, it rotates each query and key vector, in two-dimensional slices, by an angle proportional to its position. Because a dot product between two rotated vectors depends only on the difference of their angles, the score between positions $m$ and $n$ comes out as a function of $m - n$ automatically, so relative position is built into the attention computation rather than something the network has to learn to extract. Most current open models use it. EleutherAI's [write-up](https://blog.eleuther.ai/rotary-embeddings/) is a good resource for further reading.

## The feed-forward network

Also known as a multilayer perceptron [^first]: two linear layers with a non-linearity between them.

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

The inner width $d_{ff}$ is conventionally four times $d_{model}$, so the vector is projected up, passed through a ReLU, and projected back down. The division of labour is easy to lose in the diagram: attention is the only part of the architecture that moves information between positions, and the feed-forward network is the only part that does non-linear processing on what a position has collected. Attention on its own is a weighted average, which is a linear operation, so without this sublayer stacking layers would collapse into something not much more expressive than a single one.

It is also where most of the parameters are. In one layer, four attention projections of $d_{model} \times d_{model}$ come to $4d_{model}^2$, while the feed-forward network's two matrices come to $8d_{model}^2$, roughly two thirds of the layer, in the architecture named after attention.

## Decoder-only transformers

The encoder-decoder split makes sense for translation, where there are two distinct sequences: a finished source sentence to read and a target sentence to write. For plain autoregressive generation, continuing a sequence one token at a time, there is only ever one sequence, so the encoder and the cross-attention that reads from it have nothing to do. Dropping both leaves a stack of blocks containing masked self-attention and a feed-forward network. That is the decoder-only transformer, the architecture behind GPT[^radford2018] [^radford2019] and essentially every large language model since.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 55, "padding": 30}}}%%
flowchart TB
  TOK["Token ids"] --> EMB["Token embedding<br/>+ positional encoding"]

  subgraph BLK["Block (x N)"]
    direction TB
    N1["LayerNorm"] --> SA["Masked multi-head self-attention"] --> R1["+ residual"]
    R1 --> N2["LayerNorm"] --> FF["Feed-forward network"] --> R2["+ residual"]
  end

  EMB ==> N1
  R2 ==> FN["Final LayerNorm"] ==> UNEMB["Linear to vocabulary"] ==> LOG["Logits<br/>one score per token in the vocabulary,<br/>for every position"]

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
  classDef attn fill:#4c1d95,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
  classDef norm fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef ff fill:#065f46,stroke:#34d399,stroke-width:1.5px,color:#f8fafc
  classDef out fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff

  class TOK,EMB input
  class SA attn
  class N1,N2,R1,R2,FN norm
  class FF ff
  class UNEMB,LOG out
</pre>
<figcaption>The decoder-only stack. No encoder, no cross-attention, and the normalizations sit inside the residual branches in the pre-norm arrangement.</figcaption>
</figure>


**The training objective** is next-token prediction: given a sequence, predict the token that follows at every position at once, which the causal mask makes possible in a single forward pass. That objective is why the same model generalizes across code generation, translation and arithmetic without a task-specific output head. Predicting the next token well enough, over a large enough corpus, requires learning whatever structure the corpus contains. **Post-training** is everything done to that base model afterwards to make it a usable assistant: supervised fine-tuning on curated examples, reinforcement learning from feedback, and at the very top, a system prompt, which is just tokens prepended to the context. ChatGPT and Gemini are base models plus these layers and additional training.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>What a "high effort" setting on a chatbot actually does</div>
<div class="collapsiblecontent" markdown="1">
A base model trained only on next-token prediction imitates its corpus. Getting it to solve problems reliably takes a second training stage, and that stage is reinforcement learning rather than more imitation. The loop is: give the model a problem, let it sample a complete response from its own output distribution, score that response, and update the weights so that behaviour leading to higher scores becomes more likely. The score comes either from a reward model trained on human preference comparisons between pairs of responses[^ouyang], or, for domains where correctness can be checked mechanically, from the check itself, unit tests passing or a math answer matching. That second, verifiable kind of reward is what drove the recent jump in reasoning ability.[^deepseekr1]

The update rule is a policy optimization algorithm. PPO[^ppo] maximizes expected reward while clipping how far the updated policy is allowed to move from the previous one on any single step, because an unconstrained step taken on a noisy sampled reward can destroy the fluency acquired during pretraining. GRPO[^grpo] is the variant common in current reasoning models: sample a group of responses to the same prompt, score each one relative to the group's mean, and use that relative advantage directly, which removes the separately trained value network PPO needs.

What the model learns from this is to spend tokens before committing to an answer: restate the problem, break it into parts, work through them, notice a contradiction and backtrack. Those intermediate tokens are ordinary generated tokens, produced by exactly the stack built in this post, with nothing added to the architecture. An "effort" or "thinking budget" control is a dial on how many of them the model may spend before it has to answer, and it helps only because reinforcement learning taught the model to use them productively. Turned up, the model explores more of that intermediate work and usually does better on problems that decompose into steps; turned down, it answers sooner and costs less. This is a sketch of a large and fast-moving field, and the specifics of any particular commercial system are not public.
</div>
</div>

## Implementing it in JAX

The pieces above amount to little code. As in the previous post the implementation uses Flax[^flax], and again it reimplements something the library already provides, [`nnx.MultiHeadAttention`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html), for learning purposes.


```python
import jax
import jax.numpy as jnp
from flax import nnx


class Head(nnx.Module):
    """One attention head: the previous post's module, scaled and masked."""

    def __init__(self, d_model: int, d_k: int, *, rngs: nnx.Rngs):
        self.d_k = d_k
        self.w_q = nnx.Linear(d_model, d_k, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_k, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_k, use_bias=False, rngs=rngs)

    def __call__(self, x, mask):
        queries, keys, values = self.w_q(x), self.w_k(x), self.w_v(x)   # (b, n, d_k)

        scores = queries @ keys.swapaxes(-1, -2) / jnp.sqrt(self.d_k)   # (b, n, n)
        scores = jnp.where(mask, scores, -jnp.inf)                      # before the softmax

        weights = jax.nn.softmax(scores, axis=-1)   # each row sums over its prefix
        return weights @ values                     # (b, n, d_k)
```

Three things changed from the previous post's version. The scores are divided by $\sqrt{d_k}$; the mask is applied to them, and applied *before* `jax.nn.softmax` runs, which is what makes each row renormalize over its visible prefix; and `keys.T` became `keys.swapaxes(-1, -2)`, because these inputs carry a leading batch axis and `.T` would reverse it along with the rest.

A multi-head layer is then just several of those, run on the same input, with their outputs laid side by side:

```python
class MultiHeadAttention(nnx.Module):
    """Multi-head causal self-attention over a batch of sequences."""

    def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs):
        assert d_model % n_heads == 0
        self.heads = nnx.List([
            Head(d_model, d_model // n_heads, rngs=rngs) for _ in range(n_heads)
        ])
        self.w_o = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x):
        n = x.shape[-2]
        causal = jnp.tril(jnp.ones((n, n), dtype=bool))   # True on and below the diagonal

        outputs = [head(x, causal) for head in self.heads]   # each (b, n, d_k)
        return self.w_o(jnp.concatenate(outputs, axis=-1))   # (b, n, d_model)
```


A block wraps that in the residual and normalization pattern, with the feed-forward network:

```python
class Block(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff_in = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff_out = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))                          # mix across positions
        return x + self.ff_out(jax.nn.relu(self.ff_in(self.norm2(x))))   # process each position
```

Both lines have the same shape, `x = x + sublayer(norm(x))`, the pre-norm arrangement from the collapsible above. The feed-forward layers keep their biases, unlike the attention projections, where a bias would only shift every score by a constant that the softmax normalizes away.

The positional encoding is a fixed table, computed once and never trained:

```python
def positional_encoding(n: int, d_model: int):
    pos = jnp.arange(n)[:, None]                       # (n, 1)
    i = jnp.arange(0, d_model, 2)[None, :]             # (1, d_model/2)
    angle = pos / jnp.power(10000, i / d_model)
    pe = jnp.zeros((n, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(angle))            # even dimensions
    pe = pe.at[:, 1::2].set(jnp.cos(angle))            # odd dimensions
    return pe
```

And the model is the stack, plus the two layers that convert between token ids and vectors:

```python
class DecoderOnlyTransformer(nnx.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, *, rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos = positional_encoding(max_len, d_model)
        self.blocks = nnx.List([
            Block(d_model, n_heads, d_ff, rngs=rngs) for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.unembed = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, token_ids):
        n = token_ids.shape[-1]
        x = self.embed(token_ids) + self.pos[:n]   # (b, n, d_model)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))          # (b, n, vocab_size)
```

`nnx.Embed` is a lookup table with one learned row per vocabulary entry, and `unembed` projects back to one score per vocabulary entry, per position. Both are placeholders for a later post: how text becomes token ids, and what those embedding rows come to represent, is a subject of its own. Here they exist to make the shapes line up at the two ends of the stack. The blocks are held in an `nnx.List` rather than a plain Python list, which Flax rejects: the container has to be one it can traverse to find the parameters inside it.

```python
model = DecoderOnlyTransformer(
    vocab_size=1000, d_model=64, n_heads=4, d_ff=256,
    n_layers=2, max_len=128, rngs=nnx.Rngs(0),
)

tokens = jnp.zeros((1, 12), dtype=jnp.int32)   # one sequence of 12 token ids
logits = model(tokens)

n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(logits.shape, n_params)   # (1, 12, 1000) 227584
```

A forward pass returns a score for every vocabulary entry at every position, `(1, 12, 1000)`. The softmax of the last position's row is the model's distribution over the next token; the other eleven rows are the predictions it would have made at each earlier point, which is what makes training on a whole sequence in one pass possible. This toy has 227,584 parameters against the paper's 65 million, and untrained it predicts noise. Making it produce anything is the next post's problem.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 60}}}%%
flowchart LR
  T["token_ids<br/>(1 × 12)"]
  E["embed + pos<br/>(1 × 12 × 64)"]
  Q["queries, keys, values<br/>per head, ×4<br/>(1 × 12 × 16)"]
  S["scores, masked<br/>per head, ×4<br/>(1 × 12 × 12)"]
  M["concatenated heads<br/>(1 × 12 × 64)"]
  F["feed-forward<br/>(1 × 12 × 256) → (1 × 12 × 64)"]
  L["logits<br/>(1 × 12 × 1000)"]

  T ==> E ==> Q ==> S ==> M ==> F ==> L

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
  classDef attn fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#f8fafc
  classDef inter fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef ff fill:#065f46,stroke:#34d399,stroke-width:1.5px,color:#f8fafc
  classDef out fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff

  class T,E input
  class Q,S attn
  class M inter
  class F ff
  class L out
</pre>
<figcaption>Shapes through one block of the model instantiated above, d_model = 64 split into four heads of 16. The sequence length 12 appears twice in the score matrix, the quadratic cost from the previous post.</figcaption>
</figure>


- **KV caching**, an inference-time change rather than an architectural one. Generating token $n+1$ recomputes attention over the same prefix the previous step already processed, so the keys and values for that prefix are cached and reused instead. This is also why the memory cost of a long conversation grows the way it does.

## Takeaways

- Most of the architecture is scaffolding around the Attention primitive: residual connections so gradients survive depth, normalization so activations stay in range, several heads so a layer can attend to more than one thing, and a feed-forward sublayer to supply the non-linearity.
- The two adjustments to the primitive itself, scaling by $\sqrt{d_k}$ and the causal mask, both follow from the shape of the softmax. One keeps it out of its flat region, the other keeps it normalized over a prefix. 
- Parallelism was obtained by giving up sequential reading, so order has to be reintroduced explicitly as positional encoding. This trade off allows the architecture to scale better than RNNs.
- Decoder only for autoregressive generation.

## Up next

The model above runs but predicts noise, and two of its pieces are still hand-wavy: how text is converted to token ids, and the embedding table. The next post covers tokenization and embeddings, then trains this stack on a corpus with a next-token objective and samples text out of it.

<figure>
<img src="/assets/img/three-transformers.png" alt="Three technical drawings side by side in the same line-art style: William Stanley's 1886 induction coil patent figure, the patent drawing of the Optimus Prime toy in robot form, and the decoder-only stack from this post drawn as a numbered patent figure">
<figcaption markdown="1">Transformer is a loaded term now.
Left, W. Stanley Jr., *Induction Coil*, [US 349,611](https://commons.wikimedia.org/wiki/File:StanleyTransformer2.png) (1886). Centre, H. Obara, reconfigurable toy assembly, [US 4,516,948](https://commons.wikimedia.org/wiki/File:Optimus_Prime_patent.png) (filed 1984)
</figcaption>
</figure>

## References

Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY) builds an equivalent model in PyTorch, Harvard NLP's [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) annotates the original paper with running code, and the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) runs a real GPT-2 in the browser with every intermediate tensor exposed.

[^vaswani]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [*Attention Is All You Need.*](https://arxiv.org/abs/1706.03762) Advances in Neural Information Processing Systems (NeurIPS). For a highly visual walkthrough of the architecture, see [Jay Alammar's *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
[^resnet]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). [*Deep Residual Learning for Image Recognition.*](https://arxiv.org/abs/1512.03385) IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
[^layernorm]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [*Layer Normalization.*](https://arxiv.org/abs/1607.06450)
[^xiong]: Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T.-Y. (2020). [*On Layer Normalization in the Transformer Architecture.*](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) Proceedings of the 37th International Conference on Machine Learning (ICML).
[^rope]: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). [*RoFormer: Enhanced Transformer with Rotary Position Embedding.*](https://arxiv.org/abs/2104.09864)
[^radford2018]: Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [*Improving Language Understanding by Generative Pre-Training.*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) OpenAI.
[^radford2019]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). [*Language Models are Unsupervised Multitask Learners.*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) OpenAI. For a visual walkthrough, see [Jay Alammar's *The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/).
[^flax]: [Flax documentation](https://flax.readthedocs.io/en/stable/key_concepts.html)
[^rmsnorm]: Zhang, B., & Sennrich, R. (2019). [*Root Mean Square Layer Normalization.*](https://arxiv.org/abs/1910.07467) Advances in Neural Information Processing Systems (NeurIPS).
[^shazeer]: Shazeer, N. (2020). [*GLU Variants Improve Transformer.*](https://arxiv.org/abs/2002.05202)
[^gqa]: Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*](https://arxiv.org/abs/2305.13245) Conference on Empirical Methods in Natural Language Processing (EMNLP).
[^ouyang]: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback.*](https://arxiv.org/abs/2203.02155) Advances in Neural Information Processing Systems (NeurIPS).
[^ppo]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [*Proximal Policy Optimization Algorithms.*](https://arxiv.org/abs/1707.06347)
[^grpo]: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*](https://arxiv.org/abs/2402.03300) The paper that introduced GRPO.
[^deepseekr1]: DeepSeek-AI (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*](https://arxiv.org/abs/2501.12948)
[^first]: The multilayer perceptron was built in [the first post in this series]({% post_url 2026-04-07-MachineLearningBeginning %}).

