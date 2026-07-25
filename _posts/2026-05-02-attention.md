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
All neural networks are built from the same primitive we saw in the [multilayer perceptron]({% post_url 2026-04-07-MachineLearningBeginning %}), a linear map followed by an activation. What separates architectures is how that primitive is wired for the task at hand. Convolutional neural networks[^lecun] wire it into filters that slide across an image, learning relationships between neighboring pixels, and reach human-level accuracy at image classification[^he]. This post develops the next building block, attention, the wiring behind modern language models. True to the pedagogical bent of this series, it is developed here the way it emerged historically: as a fix for the limits of the architectures first applied to the sequence to sequence generation problem.

<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 90}}}%%
flowchart TB
  subgraph PRIM["Shared network primitive"]
    direction LR
    X1["x₁"] -- "w₁" --> S["Σ<br/>w₁x₁ + w₂x₂ + b"]
    X2["x₂"] -- "w₂" --> S
    S -- "z" --> F["activation<br/>f(z)"]
    F --> Y["y"]
  end
  subgraph NET["Copies of the primitive wired together: a neural network"]
    direction LR
    I1["x₁"] --> A1 & A2 & A3
    I2["x₂"] --> A1 & A2 & A3
    A1(("f")) --> B1 & B2
    A2(("f")) --> B1 & B2
    A3(("f")) --> B1 & B2
    B1(("f")) --> O["y′"]
    B2(("f")) --> O
  end
  PRIM -. "each f circle is one copy of this unit" .-> A2
  classDef focus stroke:#F59E0B,stroke-width:2px
  class A2 focus
  classDef inp fill:#0C2A3F,stroke:#38BDF8
  classDef outp fill:#3A1D24,stroke:#F87171
  class X1,X2,I1,I2 inp
  class Y,O outp
  linkStyle default stroke:#475569,stroke-width:1px
  linkStyle 0,1,2,3 stroke:#38BDF8,stroke-width:1.5px
  linkStyle 18 stroke:#F59E0B,stroke-width:2px
</pre>

### Sequence to sequence generation
Given an input sequence, the task is to generate another sequence satisfying some requirement. This spans an amazingly vast range of real world problems: text generation such as autocomplete, and video game frame generation, where a renderer predicts frames it hasn't actually computed from the ones that came before. For the purposes of this post, consider language translation: for a sentence in one language, generate the sequence of words that is its translation in another. Neural networks are graphs of weight matrices, and they work best on numerical representations of their inputs[^bengio]. The text sequence is therefore translated into numerical vectors in a high-dimensional space, namely embeddings. The process of generating these embeddings is discussed in a later post. In a nutshell, for each word in the vocabulary there is a corresponding $d$-dimensional vector representation of it, the same idea as the two features, a wedding ending or a character's death, that separated comedies from tragedies in the first post, just scaled up: instead of two hand-picked features, a word gets hundreds of learned ones, each capturing some aspect of its meaning or usage. These values are parameters of the network and are learnt during the training process.


<figure markdown="1">

$$
\begin{array}{r}
\text{river} \\
\vdots \\
{\color{#F59E0B}\text{attention}} \\
\vdots \\
{\color{#F59E0B}\text{is}} \\
\vdots \\
{\color{#F59E0B}\text{all}} \\
\vdots \\
\text{zebra}
\end{array}
\left[
\overbrace{
\begin{array}{ccccc}
0.44 & 0.19 & -0.87 & \cdots & 0.05 \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}0.21} & {\color{#F59E0B}-1.04} & {\color{#F59E0B}0.66} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}0.87} \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}-0.53} & {\color{#F59E0B}0.12} & {\color{#F59E0B}1.41} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}-0.09} \\
\vdots & \vdots & \vdots & & \vdots \\
{\color{#F59E0B}0.77} & {\color{#F59E0B}0.35} & {\color{#F59E0B}-0.28} & {\color{#F59E0B}\cdots} & {\color{#F59E0B}0.60} \\
\vdots & \vdots & \vdots & & \vdots \\
-0.16 & 0.92 & 0.33 & \cdots & -0.41
\end{array}
}^{d\ \text{columns}}
\right]
$$

<figcaption>Each input word is represented as a row of the embedding matrix.</figcaption>
</figure>


### Recurrent Neural Networks
A toy five word phrase surfaces the complexity of the translation problem, since it isn't a word by word lookup: consider `Attention is all you need` and its Spanish translation, `la atención es todo lo que necesitas`. The output is seven words long, and "all you need" has no single counterpart to line up against, it is spread across "todo lo que necesitas". The model must learn to accumulate context over the whole input and use it to predict the output. Recurrent neural network architecture encodes this context using hidden states: the network reads in one word's embedding at a time and folds it into a hidden state that summarizes everything seen so far. An embedding is a fixed vector for a word in isolation, while a hidden state is what that word means once combined with everything the network has read before it, an important distinction we'll lean on again shortly. The order in which words appear matters for any language task, and because the network reads them one at a time in sequence, that order is naturally preserved. Mathematically, the encoder's recurrence is:

$$h_i^{enc} = f\left(W_h h_{i-1}^{enc} + W_x x_i + b\right)$$

It helps to zoom into a single decoder step to see exactly how an output word gets generated. Say the encoder has finished reading all five English words and produced hidden states $h_1^{enc}, \dots, h_5^{enc}$; the final one, $h_5^{enc}$, becomes the context $c$ handed off to the decoder. Generating the fourth Spanish word, "todo", is decoder step $i = 4$:

$$h_i^{dec} = f\left(W_h h_{i-1}^{dec} + W_y y_{i-1} + W_c c + b\right), \qquad y_i = g\left(h_i^{dec}\right)$$

Each weight matrix specializes in one source of information, and all three are learned jointly during training: $W_h$ controls how much of the previous hidden state carries forward, $W_y$ controls how much weight the previous output word gets, and $W_c$ controls how much the broader context contributes. Concretely, the decoder computes a new hidden state $h_4^{dec}$ from its previous hidden state $h_3^{dec}$ (through $W_h$), the previous output token $y_3$ ("es") (through $W_y$), and the context $c$ (through $W_c$), then reads the next word $y_4$ off of that hidden state. The same context $c$ is reused, unchanged, at every single decoder step, no matter the length of the output. The mechanics of translating words into vectors ($x_i$, $y_i$) and reading a word back out of a hidden state ($g$) are deliberately handwaved here; they're covered properly in a later post on embeddings and output projections.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 70}}}%%
flowchart LR
  subgraph ENC["Encoder"]
    direction LR
    X1["x₁"] --> H1E(("h₁"))
    X2["x₂"] --> H2E(("h₂"))
    X3["x₃"] --> H3E(("h₃"))
    X4["x₄"] --> H4E(("h₄"))
    X5["x₅"] --> H5E(("h₅"))
    H1E --> H2E --> H3E --> H4E --> H5E
  end
  H5E -- "context" --> C(("c"))
  subgraph DEC["Decoder: generating y₄"]
    direction LR
    H1D(("h₁")) --> H2D(("h₂")) --> H3D(("h₃")) --> H4D(("h₄"))
    Y1["y₁"] -.-> H2D
    Y2["y₂"] -.-> H3D
    Y3["y₃"] -.-> H4D
    H4D --> Y4["y₄"]
  end
  C -.-> H1D
  C -.-> H2D
  C -.-> H3D
  C -.-> H4D
  classDef inp fill:#0C2A3F,stroke:#38BDF8
  classDef outp fill:#3A1D24,stroke:#F87171
  classDef focus fill:#3A2A0C,stroke:#F59E0B,stroke-width:2px
  class X1,X2,X3,X4,X5,Y1,Y2,Y3 inp
  class Y4 outp
  class C focus
  linkStyle default stroke:#475569,stroke-width:1px
  linkStyle 0,1,2,3,4 stroke:#38BDF8,stroke-width:1.5px
  linkStyle 9,17,18,19,20 stroke:#F59E0B,stroke-width:2px
</pre>
<figcaption>The encoder folds the five English words into hidden states, ending in a single context vector <em>c</em>. The decoder reuses that same, unchanged context at every step to generate each Spanish word, shown here generating the fourth word, "todo".</figcaption>
</figure>

#### The encoder decoder architecture

Architecturally, tying this back to the perceptrons from [the first post in this series]({% post_url 2026-04-07-MachineLearningBeginning %}), the encoder and decoder are each ordinary layers of neurons, complete with weights, biases, and activations, wired together with one addition: a feedback loop, where a layer's own hidden state at one time step becomes part of its input at the next. That loop is what lets one fixed set of weights process a variable-length sequence one token at a time, instead of needing a separate set of weights per position. The diagram below unrolls this loop over time, showing the same encoder and decoder weights reused at every step as the five-word English sentence is read in and the Spanish translation is read out.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 70}}}%%
flowchart LR
  subgraph ENC["Encoder (same weights reused every step)"]
    direction LR
    X1["x₁"] --> H1(("h₁"))
    X2["x₂"] --> H2(("h₂"))
    X3["x₃"] --> H3(("h₃"))
    X4["x₄"] --> H4(("h₄"))
    X5["x₅"] --> H5(("h₅"))
    H1 --> H2 --> H3 --> H4 --> H5
  end
  H5 -- "context" --> C(("c"))
  subgraph DEC["Decoder (same weights reused every step)"]
    direction LR
    D1(("h₁")) --> D2(("h₂")) --> D3(("h₃")) --> D4(("h₄")) --> D5(("h₅"))
    D1 --> Y1["y₁"]
    D2 --> Y2["y₂"]
    D3 --> Y3["y₃"]
    D4 --> Y4["y₄"]
    D5 --> Y5["y₅"]
    Y1 -.-> D2
    Y2 -.-> D3
    Y3 -.-> D4
    Y4 -.-> D5
  end
  C -.-> D1
  C -.-> D2
  C -.-> D3
  C -.-> D4
  C -.-> D5
  classDef inp fill:#0C2A3F,stroke:#38BDF8
  classDef outp fill:#3A1D24,stroke:#F87171
  classDef focus fill:#3A2A0C,stroke:#F59E0B,stroke-width:2px
  class X1,X2,X3,X4,X5 inp
  class Y1,Y2,Y3,Y4,Y5 outp
  class C focus
  linkStyle default stroke:#475569,stroke-width:1px
  linkStyle 0,1,2,3,4 stroke:#38BDF8,stroke-width:1.5px
  linkStyle 9 stroke:#F59E0B,stroke-width:2px
  linkStyle 23,24,25,26,27 stroke:#F59E0B,stroke-width:2px
</pre>
<figcaption>The encoder and decoder each reuse one fixed set of weights at every time step, the feedback loop from the recurrence above, unrolled here in time. Everything the network knows about the input has to funnel through the single context vector <em>c</em>, a bottleneck the next section addresses.</figcaption>
</figure>

#### The bottleneck

This architecture has two noteworthy limitations. First, it is sequential by construction: each hidden state depends on the one before it, so the network cannot process a whole sequence in parallel, only token by token, which makes both training and inference slow and hard to batch efficiently. Second, every input, however long, is compressed into one fixed-size context vector $c$. As the input grows, that vector has to represent more and more of it in the same amount of space, so semantic detail from early tokens gets diluted or lost by the time the decoder needs it.

The encoder-decoder layout of the RNN is assembled from all differentiable steps, hence it can be trained using gradient descent and learn parameters using backpropagation. The second limitation: training the network with gradient descent runs backpropagation through every time step, repeatedly multiplying by the same weight matrix $W_h$. If the dominant eigenvalue (the spectral radius) of $W_h$ is larger than 1, this repeated multiplication makes gradients grow exponentially with sequence length, the exploding gradient problem, causing gradient descent to take wild, unstable steps instead of settling toward a minimum.[^pascanu] A knee jerk fix is shrinking $W_h$'s spectral radius below 1, which after a number of time steps leads the gradients to shrink instead, the vanishing gradient problem, and the training stalls due to lack of meaningful updates. Chronologically, architectures like LSTM[^lstm] were the first real fix for this. For our purposes, we jump straight to attention, currently the best solution to both the compute and long-range dependency problems.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Exploding gradient problem</div>
<div class="collapsiblecontent" markdown="1">
When training neural networks using backpropagation, we use the chain rule to calculate how much a small change in an early layer's weights affects the final loss. For deep feedforward networks or recurrent neural networks (RNNs), this means the gradients of earlier weights are calculated using increasingly many multiplications by the layer's weight matrix (e.g., $W$ in an RNN) for every layer or time step. If the matrix operation inherently "stretches" the gradient vectors, this repeated multiplication causes the weight gradients at earlier layers to grow exponentially. The result is that the optimization algorithm takes massive, unstable steps that can introduce instability in the training process, slow it down, or halt it entirely.
</div>
</div>
### Attention

The next logical step up, Attention, sandwiched in a encoder-decoder layout, primarily targets the second limitation: it gives the decoder direct access to every encoder hidden state, not just the final one, so it can learn which parts of the input matter most for the word it's generating right now, instead of relying on everything being routed through one fixed context vector. The encoder here still reads the input sequentially, exactly as before, attention changes what the decoder can look at, not how the input is read.

#### An aside, Softmax
Softmax is a function that converts a vector of arbitrary real-valued scores into a probability distribution. For a vector $z$ of length $n$:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

Two properties fall out of this: every output is strictly positive and less than 1 (each term is an exponential divided by a sum that includes itself), and all outputs sum to exactly 1. Together these make the output a valid probability distribution over the $n$ inputs. That's what makes softmax useful here: given a set of scores, it turns them into weights that can be used to compute a weighted average, the mechanism attention leans on directly, as the next section shows.

The fixed context $c$ handed to the decoder is one snapshot, frozen after the encoder finishes reading. But the decoder needs different context at different steps. For instance, generating "todo" leans on different words than generating "necesitas". So let the decoder search back over the encoder's hidden states itself, at every step, and build a context suited to that step alone.

Comparing the decoder's current state against each encoder state gives a measure of relevance:

$$e_{ij} = h_{i-1}^{dec} \cdot h_j^{enc}$$

the dot product is large when the two vectors point in a similar direction, i.e. when encoder position $j$ is relevant to what the decoder is about to generate at step $i$. Normalizing these scores with softmax turns them into weights that sum to 1:

$$\alpha_{ij} = \text{softmax}(e_{i1}, \dots, e_{i5})_j$$

and the new context is a weighted sum of the encoder states, weighted exactly by how relevant each one is right now:

$$c_i = \sum_{j=1}^5 \alpha_{ij} h_j^{enc}$$

$c_i$ replaces the static $c$ in the decoder recurrence, $h_i^{dec} = f\left(W_h h_{i-1}^{dec} + W_y y_{i-1} + W_c c_i + b\right)$, recomputed fresh at every step instead of fixed once. That's how the network reaches across the whole input without squeezing it through one bottleneck.

#### Carving attention out
This search, compare, then blend mechanism is useful enough to name and carve out as a standalone primitive, independent of the RNN it lives inside here. The vector doing the searching, $h_{i-1}^{dec}$, is a query. The vectors being searched against, $h_j^{enc}$, are keys; the vectors being summed are values. Here keys and values are the same encoder hidden states, the next section projects them independently instead. For linguistic tasks, this is what lets the network capture far-away dependencies, idioms are the clearest example, understanding one often depends on the whole phrase rather than any single word in it.

#### Queries, keys and values

In the walkthrough above, queries, keys, and values were the raw hidden states themselves, $h_{i-1}^{dec}$ doubling as the query, $h_j^{enc}$ doubling as both key and value. In practice each role gets its own learned projection: three weight matrices $W_Q$, $W_K$, $W_V$ map the underlying vectors into a query space, a key space, and a value space, so the network can learn what to search for and what to search on separately, instead of reusing one vector for every role.

$$q_i = W_Q h_i, \qquad k_j = W_K h_j, \qquad v_j = W_V h_j$$

These weights are learnable, exactly like any other weight matrix in the network, trained by gradient descent alongside everything else. There's nothing hand-designed about what a query or key ends up representing, the search-match-collect intuition is a way to reason about the role each projection plays, not a constraint imposed on it.

Computing one query, key, and value at a time is wasteful once there are many positions to process, so in practice the hidden states are stacked into a matrix $H \in \mathbb{R}^{n \times d}$, one row per position, and projected all at once:

$$Q = HW_Q, \qquad K = HW_K, \qquad V = HW_V$$

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 50, "rankSpacing": 70}}}%%
flowchart LR
  H["H<br/>(n × d)<br/>stacked hidden states"] -- "× W_Q" --> Q["Q<br/>(n × d_k)"]
  H -- "× W_K" --> K["K<br/>(n × d_k)"]
  H -- "× W_V" --> V["V<br/>(n × d_v)"]
  classDef inp fill:#0C2A3F,stroke:#38BDF8
  classDef outp fill:#3A1D24,stroke:#F87171
  class H inp
  class Q,K,V outp
  linkStyle default stroke:#475569,stroke-width:1.5px
</pre>
<figcaption>The stacked hidden states $H$ are projected through three independent, learnable weight matrices to produce the query, key, and value matrices.</figcaption>
</figure>

With $Q$ and $K$ each shaped $(n \times d_k)$, the score matrix $QK^T$ is $(n \times n)$, one score per pair of positions. That's quadratic in sequence length: doubling $n$ quadruples the memory needed to hold the score matrix, the reason attention gets expensive on long sequences and a real constraint on GPU memory as context windows grow.

#### Self-attention
So far attention has connected two separate sequences, cross-attention: queries came from the decoder's hidden state, keys and values from the encoder's. Self-attention is the same score-then-blend mechanism, but every projection reads from the same sequence, so a sequence attends to itself instead of to some other sequence.

$$q_i = W_Q x_i, \qquad k_i = W_K x_i, \qquad v_i = W_V x_i$$

There's no encoder feeding a separate decoder anymore, one sequence generates all three roles from itself, then attends across its own positions to build a representation that already accounts for the rest of the sentence. This is the mechanism the rest of the series builds on: no recurrence is needed to accumulate context, every position can see every other position directly, in one step.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 120}}}%%
flowchart LR
  subgraph Q["as queries"]
    direction TB
    QX1["x₁"]
    QX2["x₂"]
    QX3["x₃"]
    QX4["x₄"]
  end
  subgraph K["as keys / values"]
    direction TB
    KX1["x₁"]
    KX2["x₂"]
    KX3["x₃"]
    KX4["x₄"]
  end
  QX1 --> KX1 & KX2 & KX3 & KX4
  QX2 --> KX1 & KX2 & KX3 & KX4
  QX3 --> KX1 & KX2 & KX3 & KX4
  QX4 --> KX1 & KX2 & KX3 & KX4
  classDef inp fill:#0C2A3F,stroke:#38BDF8
  class QX1,QX2,QX3,QX4,KX1,KX2,KX3,KX4 inp
  linkStyle default stroke:#475569,stroke-width:1px
</pre>
<figcaption>In self-attention, the same input sequence plays both roles: every token, as a query, attends to every token, including itself, as a key/value. There's no separate encoder feeding this, and no recurrence, every pairwise interaction happens in one step.</figcaption>
</figure>

### Implementing attention in JAX

With the pieces above in place, here's a minimal, unscaled, unmasked self-attention layer in JAX, operating on one hand-crafted sequence so every intermediate matrix stays small enough to inspect directly. Scaling and masking are covered once the transformer block is assembled in the next post.

<!-- TODO: code block, single-sequence single-head unscaled/unmasked attention, matches companion notebook -->

<!-- TODO: attention-matrix heatmap for this example, computed in the notebook -->

The full notebook is available <!-- TODO: link once notebook exists -->.

### Where this leads

Vaswani et al. (2017)[^vaswani] showed that attention alone, stacked with simple feed-forward layers and no recurrence at all, reaches state-of-the-art translation quality, measured in BLEU score, a standard machine-translation metric. That architecture is the transformer, the subject of the next post. <!-- next post starts with transformer and then moves on to decoder only after explaining why and then implements it-->

The takeaway from this post is narrower than that: attention removes the fixed-context bottleneck, and because self-attention needs no recurrence, it lets a sequence be processed in parallel instead of one token at a time. That parallelism is what makes training at the scale modern language models run at even feasible; training genuinely deep stacks of these layers is a separate problem, solved with residual connections and normalization, covered once the full transformer block is assembled. What follows from here, the fluency and apparent reasoning of large language models, comes from stacking this simple primitive, matrix multiplications and softmax, at scale, trained on enormous amounts of curated data, not from any single new mechanical trick.

#### References

[^lecun]: LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). [*Backpropagation Applied to Handwritten Zip Code Recognition.*](https://direct.mit.edu/neco/article/1/4/541/5515/Backpropagation-Applied-to-Handwritten-Zip-Code) Neural Computation, 1(4), 541-551.
[^he]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). [*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.*](https://arxiv.org/abs/1502.01852) IEEE International Conference on Computer Vision (ICCV).
[^bengio]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). [*A Neural Probabilistic Language Model.*](https://www.jmlr.org/papers/v3/bengio03a.html) Journal of Machine Learning Research, 3, 1137-1155.
[^pascanu]: Pascanu, R., Mikolov, T., & Bengio, Y. (2013). [*On the Difficulty of Training Recurrent Neural Networks.*](https://arxiv.org/abs/1211.5063) Proceedings of the 30th International Conference on Machine Learning (ICML).
[^lstm]: Hochreiter, S., & Schmidhuber, J. (1997). [*Long Short-Term Memory.*](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) Neural Computation, 9(8), 1735-1780. For a gentler walkthrough, see [Christopher Olah's *Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
[^vaswani]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [*Attention Is All You Need.*](https://arxiv.org/abs/1706.03762) Advances in Neural Information Processing Systems (NeurIPS). For a highly visual walkthrough of the resulting architecture, see [Jay Alammar's *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).


