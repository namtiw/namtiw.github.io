---
layout: post
title: Implementing Attention in JAX
subtitle: How language models learn to look back
thumbnail-img: /assets/img/ml.jpg
mathjax: true
mermaid: true
published: false
tags: [Machine-Learning, RNN, Attention, Self-attention, JAX]
---
All neural networks are built from the same primitive we saw in the [multilayer perceptron]({% post_url 2026-04-07-MachineLearningBeginning %}), a linear map followed by an activation. The architectures vary primarily in how these primitives are interconnected to achieve desired results. Convolutional neural networks[^lecun] wire it into filters that slide across an image, learning relationships between neighboring pixels, and reach human-level accuracy at image classification[^he]. This post develops the next building block of modern language models, Attention. True to the pedagogical bent of this series, it is developed here the way it emerged historically: as a fix for the limits of the architectures first applied to the sequence to sequence generation problem.

<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 90}}}%%
flowchart TB 
  subgraph PRIM["Perceptron Primitive: f(Wx + b)"]
    direction LR
    subgraph INP1["Inputs"]
      X1["x₁"]
      X2["x₂"]
    end
    SUM["Σ<br/>w₁x₁ + w₂x₂ + b"]
    ACT["Activation<br/>f(z)"]
    Y1["Output y"]

    X1 -- "w₁" --> SUM
    X2 -- "w₂" --> SUM
    SUM --> ACT --> Y1
  end

  subgraph NET["Network Topology: Wired Primitives"]
    direction LR
    subgraph INP2["Input Layer"]
      I1["x₁"]
      I2["x₂"]
    end
    subgraph HID["Hidden Layer"]
      H1(("f"))
      H2(("f"))
      H3(("f"))
    end
    subgraph OUT["Output Layer"]
      O1(("f"))
      O2["y′"]
    end

    I1 --> H1 & H2 & H3
    I2 --> H1 & H2 & H3
    H1 & H2 & H3 --> O1 --> O2
  end

  PRIM -. "Every node f is an instance of the primitive" .-> H2

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
  classDef math fill:#1e293b,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
  classDef active fill:#0284c7,stroke:#38bdf8,stroke-width:2px,color:#ffffff
  classDef focus fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff
  classDef output fill:#4c0519,stroke:#f43f5e,stroke-width:1.5px,color:#f8fafc

  class X1,X2,I1,I2 input
  class SUM math
  class ACT active
  class H2 focus
  class Y1,O2 output
</pre>

## Sequence to sequence generation

Given an input sequence, the task is to generate another sequence satisfying some requirement. This spans an amazingly vast range of real world problems: text generation such as autocomplete or summarization, and video game frame generation, where a renderer predicts frames it hasn't actually computed from the ones that came before. For the purposes of this post, consider language translation: for a sentence in one language, generate the sequence of words that is its translation in another.

Neural networks are graphs of weight matrices, and they work best on numerical representations of their inputs[^bengio]. The text sequence is therefore translated into numerical vectors in a high-dimensional space, namely embeddings. The process of generating these embeddings is discussed in a later post. In a nutshell, for each word in the vocabulary there is a corresponding $d$-dimensional vector representation of it, the same idea as the two features, a wedding ending or a character's death, that separated comedies from tragedies in the first post, albeit scaled up: instead of two hand-picked features, a word is represented typically using hundreds of features, each capturing some aspect of its meaning or usage. These weight assigned to each feature are parameters of the network and are learnt during the training process. Embeddings typically represent, tokens, a unique ID for each atom of the text (characters, words or a set of words) instead of the words directly, this nuance can be overlooked to keep things simple here. Thus, although not entirely accurate, _word_ and _token_ are used interchangeably for the rest of this post.

<figure markdown="1">

$$
\begin{array}{r}
{\color{#F59E0B}\text{all}} \\
{\color{#F59E0B}\text{attention}} \\
\vdots \\
{\color{#F59E0B}\text{is}} \\
\vdots \\
{\color{#F59E0B}\text{need}} \\
\text{river} \\
\vdots \\
{\color{#F59E0B}\text{you}} \\
\text{zebra}
\end{array}
\left[
\overbrace{
\begin{array}{ccccc}
{\color{#F59E0B}0.77} & {\color{#F59E0B}0.35} & {\color{#F59E0B}-0.28} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}0.60} \\
{\color{#F59E0B}0.21} & {\color{#F59E0B}-1.04} & {\color{#F59E0B}0.66} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}0.87} \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}-0.53} & {\color{#F59E0B}0.12} & {\color{#F59E0B}1.41} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}-0.09} \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}0.31} & {\color{#F59E0B}-0.45} & {\color{#F59E0B}0.08} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}1.12} \\
0.44 & 0.19 & -0.87 & \cdots & 0.05 \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}-0.72} & {\color{#F59E0B}0.63} & {\color{#F59E0B}0.24} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}-0.31} \\
-0.16 & 0.92 & 0.33 & \cdots & -0.41
\end{array}
}^{d_{model}\ \text{columns}}
\right]
$$

<figcaption>Each input word is represented as a row of the embedding matrix.</figcaption>
</figure>


### Recurrent Neural Networks
A toy five word phrase surfaces the complexity of the translation problem, since it isn't a word by word lookup: consider `Attention is all you need` and its Spanish translation, `La atención es todo lo que necesitas`. The output is seven words long, and "all you need" has no single counterpart to line up against, it is spread across "todo lo que necesitas". The model must learn to accumulate context over the whole input and use it to predict the output. Recurrent neural network architecture encodes this context using hidden states: the network reads in one word's embedding at a time and folds it into a hidden state that summarizes everything seen so far. A distinction to note here: an embedding is a vector representation of a word in isolation, while a hidden state is what that word means (to the network) once combined with everything the network has read before it.

#### The Encoder-Decoder architecture

The sequence generation problem is solved in two steps, encoding the input sequence to gather context and then decoding the encoder's representation to construct the output sequence. Architecturally, tying this back to the [perceptrons]({% post_url 2026-04-07-MachineLearningBeginning %}), the Encoder and Decoder are each computational graphs of neurons, each with learnable weights, biases, and activations, with an additional feedback loop. A hidden state is the intermediate value in the neural network that are used to calculate the output. A layer's own hidden state at one time step becomes part of its input at the next. Therefore, for the encoder layer it is projection of its inputs, current input token and previous hidden state. Mathematically, the encoder's recurrence is:

$$
h_i^{enc} = f_{enc}\left(x_i, h_{i-1}^{enc}\right) \tag{1}
$$

where, $h_i^{enc}$ is encoder output for the $i^{th}$ token in the input sequence, $x_i$. $f_{enc}$ represents a linear transformation of the inputs similar to a single layer Perceptron ($y = xW + b$ operations). Once the entire input sequence has been read, the final hidden state is given as input context to the decoder. The decoder, 

$$
h_i^{dec} = f_{dec}\left(h_{i-1}^{dec}, y_{i-1}, c \right)
$$

uses its previous output $y_{i-1}$, its previous hidden state and the context from the encoder $c$ to predict its next hidden state. $h_i{dec}$ is in turn, used to predict the next output token $y_i$. For instance consider the encoder has finished reading all five English words and produced hidden states $h_1^{enc}, \dots, h_5^{enc}$; the final one, $h_5^{enc}$, becomes the context $c$ handed off to the decoder. Generating the fourth Spanish word, "todo", is decoder step $i = 4$:

<figure>
<pre class="mermaid">{% raw %}
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 70}}}%%
flowchart TB
  subgraph ENCODER ["Encoder Phase (Reading)"]
      X1["x₁ (attention)<br/>t=1"] --> H1(("h₁ᵉⁿᶜ<br/>(t=1)"))
      X2["x₂ (is)<br/>t=2"] --> H2(("h₂ᵉⁿᶜ<br/>(t=2)"))
      DOTS["⋯ x₃, x₄ ⋯"]
      X5["x₅ (need)<br/>t=5"] --> H5(("h₅ᵉⁿᶜ<br/>(t=5)"))

      H1 -- "State updates" --> H2 -- "State updates" --> DOTS -- "State updates" --> H5
  end

  H5 ==>|"State freezes"| C{{"Context (c)"}}

  subgraph DECODER ["Decoder Phase (Generating)"]
      D1(("h₁ᵈᵉᶜ<br/>(t=6)")) --> Y1["y₁ (la)<br/>(t=6)"]
      D2(("h₂ᵈᵉᶜ<br/>(t=7)")) --> Y2["y₂ (atención)<br/>(t=7)"]
      
      D1 -- "State updates" --> D2
      Y1 -. "Feedback" .-> D2
  end

  C ==>|"Read by every step"| D1
  C ==>|"Read by every step"| D2

  %% Styling
  classDef input fill:#0f172a,stroke:#38bdf8,color:#f8fafc
  classDef encNode fill:#1e293b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc
  classDef bottleneck fill:#7c2d12,stroke:#f97316,stroke-width:3px,color:#fff
  classDef decNode fill:#312e81,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
  classDef output fill:#4c0519,stroke:#f43f5e,color:#f8fafc
  
  class X1,X2,X5 input
  class H1,H2,H5,DOTS encNode
  class C bottleneck
  class D1,D2 decNode
  class Y1,Y2 output
{% endraw %}</pre>
<figcaption>One Encoder-Decoder time step. Note the bottleneck, fixed length context vector <em>c</em> .</figcaption>
</figure>
The order in which words appear matters for any language task, and because the network reads them one at a time in sequence, that order is naturally preserved.Processing input tokens in a loop allows one fixed set of weights process a variable-length sequence one token at a time, instead of needing a separate set of weights per position. Two such loops of processing one token at a time, one reading the input sequence and one writing an output, the encoder-decoder architecture, performs reasonably well on the machine translation task[^sutskever] [^cho].

#### The bottleneck

This architecture has three noteworthy limitations.

**Everything funnels through one vector.** Every input, however long, is compressed into one fixed-size context vector $c$. As the input grows, that vector has to represent more and more of it in the same amount of space, so semantic detail from early tokens gets diluted or lost. Cho et al. measured this dilution, and the BLEU score of a fixed-context encoder-decoder falls off steeply as source sentences get longer[^chodegrade].

**It is sequential by construction.** Each hidden state depends on the one before it, so the network cannot process a whole sequence in parallel, only token at a time, which does not scale well.

**Gradients are unstable over long sequences.**
When training neural networks using backpropagation, we use the chain rule to calculate how much a small change in an early layer's weights affects the final loss. For deep feedforward networks (networks with many hidden layers) or recurrent neural networks (RNNs), this means the gradients of earlier weights are calculated using increasingly many multiplications by the layer's weight matrix (e.g., $W$ in an RNN) for every layer or time step. If the matrix operation inherently "stretches" the gradient vectors, this repeated multiplication causes the weight gradients at earlier layers to grow exponentially. The result is that the optimization algorithm takes massive, unstable steps that can introduce instability in the training process, slow it down, or halt it entirely.


Chronologically, gradient clipping and gated architectures like the LSTM[^lstm] were the first real fixes for the third limitation. For our purposes, we jump straight to attention, which targets the context dilution problem.

## Attention

Bahdanau et al.[^bahdanau], modified the encoder-decoder architecture above to give the decoder direct access to every encoder hidden state, so it can learn which parts of the input matter most for the word it's generating during the current time step, instead of relying on everything being routed through one fixed context vector. 

<!--TODO: add diagram for the input hidden state, aligment score, attention to context to output diagram for the network -->
A hidden state $h_i$ is calculated for each input token $x_i$. Comparing the decoder's current state, $h_{i-1}$, against each encoder state gives a measure of relevance. Hence, these alignment scores are given by:

$$
e_{ij} = f_{alignment}\left(h_{i-1}^{dec}, h_j^{enc}\right)\tag{3}
$$

the dot product is large when the two vectors point in a similar direction, i.e. when encoder position $j$ is relevant to what the decoder is about to generate at step $i$. Normalizing these scores with softmax turns them into weights that sum to 1:

$$
\alpha_{ij} = \text{softmax}(e_{i1}, \dots, e_{i5})_j \tag{4}
$$

and the new context is a weighted sum of the encoder states, weighted exactly by how relevant each one is right now:

$$
c_i = \sum_{j=1}^5 \alpha_{ij} h_j^{enc} \tag{5}
$$

$c_i$ replaces the static $c$ in equation (2), $h_i^{dec} = f\left(h_{i-1}^{dec}, y_{i-1}, c_i\right)$, hence at each step the output of the Recurrent network relies on a distilled representation of the entire input. 

The interesting object here is the intermediate matrix $\alpha$: one row per output word, one column per input word. When represented as a grid, it shows which source words the model looked at to produce each target word. Bahdanau et al. demonstrated that a network trained only on sentence pairs learns something recognisable as word alignment, without ever being told what alignment is.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Softmax</div>
<div class="collapsiblecontent" markdown="1">
Softmax is a function that converts a vector of arbitrary real-valued scores into a probability distribution. For a vector $z$ of length $n$:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

Two properties fall out of this: every output is strictly positive, and all outputs sum to exactly 1. Together these make the output a valid probability distribution over the $n$ inputs, and with more than one input each individual weight is strictly less than 1. Hence, softmax is useful here: given a set of scores, it turns them into weights that can be used to compute a weighted average,as done in equation (5) above.

</div>
</div>

### Carving attention out
The $\alpha$ matrix generated above using alignment check, then blend mechanism is useful as a standalone primitive, independent of the RNN it lives inside above. 

#### Queries, Keys and Values

Queries, Keys and Values are core vector components of the Attention mechanism. They act like a fuzzy lookup system that allows each token to assign a weight, an Attention score, to every other token in the sequence. Queries are decoder hidden states. Keys and Values are projections of the input sequence $x_i$. Similar to the above architecture, Softmax of aligment scores between Queries and Keys weighted by the projection of the Values give Attention scores. The projections are calculated using learnable weights, thus enabling the model at each generation step to be able to extract relevant information from the input sequence. Equations (3), (4) and (5) then collapse into two matrix multiplications with a softmax between them, computing the context at each step:

$$ \text{Encoder States}: K = W_k \cdot x_i, V = W_v \cdot x_i $$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(QK^\top\right)V \tag{8}$$

<!--TODO: attention diagram -->

One historical note on equation (3). Bahdanau's original aligment function, $f_{aligment}$ above was not a dot product but a small learned network, $e_{ij} = v_a^\top \tanh(W_a h_{i-1}^{dec} + U_a h_j^{enc})$, trained jointly with everything else. Luong et al. later compared several scoring functions and found the plain dot product to work about as well while being considerably cheaper[^luong].

### Self-attention

In the Attention mechanism in equation (8), Queries are decoder states and independent of input projections Keys and Values, this is called Cross Attention, since aligment is checked between queries generated independent of the input sequence. In Self Attention, Queries too are generated as projections of the input sequence, so a sequence attends to itself.

$$q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V \tag{9}$$

There's no encoder feeding a separate decoder anymore, one sequence generates all three roles from itself, then attends across its own positions to build a representation that accounts for the rest of the sentence. This is a powerful primitive as without relying on recurrence it accumulates context from the entire sequence.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 120}}}%%
flowchart TD
  Q["Queries (Q)<br/>(n × dₖ)"]
  K["Keys (K)<br/>(n × dₖ)"]
  V["Values (V)<br/>(n × dᵥ)"]

  SCORE["Scores from Matrix Multiply (Q × Kᵀ)<br/>(n × n)"]
  ATTN_W["Attention Weights<br/>(n × n)"]
  OUT["Contextualized Matrix<br/>(n × dᵥ)"]

  Q ==> SCORE
  K -.->|"Transposed"| SCORE
  
  SCORE ==>|"Softmax Normalization"| ATTN_W
  
  ATTN_W ==>|"Weighted Combination (Weights × V)"| OUT
  V -.->|"Multiplied by Weights"| OUT

  %% Styling
  classDef queryMat fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#fff
  classDef keyMat fill:#065f46,stroke:#34d399,stroke-width:2px,color:#fff
  classDef valueMat fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#fff
  classDef interMat fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef outMat fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#fff

  class Q queryMat
  class K keyMat
  class V valueMat
  class SCORE,ATTN_W interMat
  class OUT outMat
</pre>
<figcaption>In self-attention, the same input sequence plays both roles: every token, as a query, attends to every token, including itself, as a key/value. </figcaption>
</figure>

### Implementing attention in JAX

JAX enables high performance mathematical computations across devices, CPU, GPU and TPU using the same code. Only functions with no side effects can "jitted", annotated with `@jax.jit`, to enable (Just-In-Time compilation)[https://docs.jax.dev/en/latest/jit-compilation.html] of the function which makes it run operations highly efficiently (in parallel). The following implementation uses Flax[^flax], a library with handy abstractions for commonly used primitives in machine learning built for JAX.

As seen above, attention is simply a series of matrix multiplications. The linear layers used here are single layer perceptrons hence essentially a matrix multiplication operation: [`nnx.Linear`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/linear.html) computes $y = xW + b$, and constructing it with `use_bias=False` drops the offset and leaves $y = xW$, a single learned weight matrix used to calculate projections. Notably, Flax also provides an implementation of [Attention](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html), the implementation below is simply a learning exercise. 

<!-- TODO: implement linear layer in jax as well below just for show -->


```python
import jax
import jax.numpy as jnp
from flax import nnx


class Attention(nnx.Module):
    """Single-head attention: three bias-free projections and two matrix multiplies."""

    def __init__(self, d_model: int, d_k: int, d_v: int, *, rngs: nnx.Rngs):
        # use_bias=False makes each projection exactly x @ W, with no affine offset
        self.w_q = nnx.Linear(d_model, d_k, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_k, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_v, use_bias=False, rngs=rngs)

    def __call__(self, x_query, x_context):
        queries = self.w_q(x_query)      # (n, d_k)
        keys = self.w_k(x_context)       # (m, d_k)
        values = self.w_v(x_context)     # (m, d_v)

        scores = queries @ keys.T                    # (n, m), one score per pair
        weights = jax.nn.softmax(scores, axis=-1)    # each row sums to 1
        return weights @ values                      # (n, d_v)


model = Attention(d_model=16, d_k=8, d_v=12, rngs=nnx.Rngs(0))
context = jnp.ones((7, 16))
sequence = jnp.ones((5, 16))

cross = model(sequence, context)   # (5, 12), cross-attention
self_attended = model(sequence, sequence)   # (5, 12), self-attention
```

## Key Takeaways

- Attention enables the model to capture semantic relationships between far away tokens in the sequence.
- Self-attention uses input projections: Queries, Keys and Values to generate attention scores. This helps overcome the sequential recurrence limitation of RNN and allows parallel processing of input tokens.

## References

[^lecun]: LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). [*Backpropagation Applied to Handwritten Zip Code Recognition.*](https://direct.mit.edu/neco/article/1/4/541/5515/Backpropagation-Applied-to-Handwritten-Zip-Code) Neural Computation, 1(4), 541-551.
[^he]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). [*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.*](https://arxiv.org/abs/1502.01852) IEEE International Conference on Computer Vision (ICCV).
[^bengio]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). [*A Neural Probabilistic Language Model.*](https://www.jmlr.org/papers/v3/bengio03a.html) Journal of Machine Learning Research, 3, 1137-1155. The paper that introduced learning distributed word representations jointly with the model that uses them.
[^elman]: Elman, J. L. (1990). [*Finding Structure in Time.*](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) Cognitive Science, 14(2), 179-211. The recurrent network with a hidden state fed back as input, in the form used here.
[^sutskever]: Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [*Sequence to Sequence Learning with Neural Networks.*](https://arxiv.org/abs/1409.3215) Advances in Neural Information Processing Systems (NeurIPS).
[^cho]: Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). [*Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.*](https://arxiv.org/abs/1406.1078) Conference on Empirical Methods in Natural Language Processing (EMNLP).
[^chodegrade]: Cho, K., van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). [*On the Properties of Neural Machine Translation: Encoder-Decoder Approaches.*](https://arxiv.org/abs/1409.1259) Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation (SSST-8). Section 3.2 plots BLEU against source sentence length.
[^pascanu]: Pascanu, R., Mikolov, T., & Bengio, Y. (2013). [*On the Difficulty of Training Recurrent Neural Networks.*](https://arxiv.org/abs/1211.5063) Proceedings of the 30th International Conference on Machine Learning (ICML). Section 2.1 gives the conditions on $\lambda_1$; section 3.2 introduces gradient clipping.
[^lstm]: Hochreiter, S., & Schmidhuber, J. (1997). [*Long Short-Term Memory.*](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) Neural Computation, 9(8), 1735-1780. For a gentler walkthrough, see [Christopher Olah's *Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
[^bahdanau]: Bahdanau, D., Cho, K., & Bengio, Y. (2015). [*Neural Machine Translation by Jointly Learning to Align and Translate.*](https://arxiv.org/abs/1409.0473) International Conference on Learning Representations (ICLR). The paper that introduced attention, and the alignment plots in section 6.3 are worth looking at.
[^luong]: Luong, M.-T., Pham, H., & Manning, C. D. (2015). [*Effective Approaches to Attention-based Neural Machine Translation.*](https://arxiv.org/abs/1508.04025) Conference on Empirical Methods in Natural Language Processing (EMNLP). Compares dot product, bilinear and additive scoring functions.
[^flash]: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*](https://arxiv.org/abs/2205.14135) Advances in Neural Information Processing Systems (NeurIPS).
[^vaswani]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [*Attention Is All You Need.*](https://arxiv.org/abs/1706.03762) Advances in Neural Information Processing Systems (NeurIPS). For a highly visual walkthrough of the resulting architecture, see [Jay Alammar's *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
[^flax]: [Flax documentation](https://flax.readthedocs.io/en/stable/key_concepts.html)
[^adam]: Kingma, D. P., & Ba, J. (2015). [*Adam: A Method for Stochastic Optimization.*](https://arxiv.org/abs/1412.6980) International Conference on Learning Representations (ICLR).
[^adamw]: Loshchilov, I., & Hutter, F. (2019). [*Decoupled Weight Decay Regularization.*](https://arxiv.org/abs/1711.05101) International Conference on Learning Representations (ICLR). The Optax implementation used here is documented under [`optax.adamw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw).
[^ruder]: Ruder, S. (2016). [*An Overview of Gradient Descent Optimization Algorithms.*](https://www.ruder.io/optimizing-gradient-descent/) A walkthrough of how each optimizer, from vanilla SGD through momentum, Adagrad, RMSprop and Adam, changes the way the learning rate is applied.
