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

Given an input sequence, the task is to generate another sequence satisfying some requirement. This spans an amazingly vast range of real world problems: text generation such as autocomplete, and video game frame generation, where a renderer predicts frames it hasn't actually computed from the ones that came before. For the purposes of this post, consider language translation: for a sentence in one language, generate the sequence of words that is its translation in another. Neural networks are graphs of weight matrices, and they work best on numerical representations of their inputs[^bengio]. The text sequence is therefore translated into numerical vectors in a high-dimensional space, namely embeddings. The process of generating these embeddings is discussed in a later post. In a nutshell, for each word in the vocabulary there is a corresponding $d$-dimensional vector representation of it, the same idea as the two features, a wedding ending or a character's death, that separated comedies from tragedies in the first post, just scaled up: instead of two hand-picked features, a word gets hundreds of learned ones, each capturing some aspect of its meaning or usage. These values are parameters of the network and are learnt during the training process.

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
A toy five word phrase surfaces the complexity of the translation problem, since it isn't a word by word lookup: consider `Attention is all you need` and its Spanish translation, `la atención es todo lo que necesitas`. The output is seven words long, and "all you need" has no single counterpart to line up against, it is spread across "todo lo que necesitas". The model must learn to accumulate context over the whole input and use it to predict the output. Recurrent neural network architecture encodes this context using hidden states: the network reads in one word's embedding at a time and folds it into a hidden state that summarizes everything seen so far. An embedding is a fixed vector for a word in isolation, while a hidden state is what that word means once combined with everything the network has read before it, an important distinction we'll lean on again shortly. The order in which words appear matters for any language task, and because the network reads them one at a time in sequence, that order is naturally preserved. Mathematically, the encoder's recurrence is:

$$
h_i^{enc} = f\left(W_h h_{i-1}^{enc} + W_x x_i + b\right) \tag{1}
$$

It helps to zoom into a single decoder step to see exactly how an output word gets generated. Say the encoder has finished reading all five English words and produced hidden states $h_1^{enc}, \dots, h_5^{enc}$; the final one, $h_5^{enc}$, becomes the context $c$ handed off to the decoder. Generating the fourth Spanish word, "todo", is decoder step $i = 4$:

$$
h_i^{dec} = f\left(W_h h_{i-1}^{dec} + W_y y_{i-1} + W_c c + b\right), \qquad y_i = g\left(h_i^{dec}\right) \tag{2}
$$

Each weight matrix specializes in one source of information, and all three are learned jointly during training: $W_h$ controls how much of the previous hidden state carries forward, $W_y$ controls how much weight the previous output word gets, and $W_c$ controls how much the broader context contributes. Concretely, the decoder computes a new hidden state $h_4^{dec}$ from its previous hidden state $h_3^{dec}$ (through $W_h$), the previous output token $y_3$ ("es") (through $W_y$), and the context $c$ (through $W_c$), then reads the next word $y_4$ off of that hidden state. The same context $c$ is reused, unchanged, at every single decoder step, no matter the length of the output. The mechanics of translating words into vectors ($x_i$, $y_i$) and reading a word back out of a hidden state ($g$) are deliberately handwaved here; they're covered properly in a later post on embeddings and output projections.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 70}}}%%
flowchart TB
  subgraph ENC["Encoder Sequence (Processed)"]
    direction LR
    X1["x₁"] --> E1(("h₁"))
    X2["x₂"] --> E2(("h₂"))
    X3["x₃"] --> E3(("h₃"))
    X4["x₄"] --> E4(("h₄"))
    X5["x₅"] --> E5(("h₅"))
    E1 --> E2 --> E3 --> E4 --> E5
  end

  E5 ==> C["Final hidden state h₅ᵉⁿᶜ as context c to decoder"]

  subgraph DEC["Decoder Step i=4 (Generating 'todo')"]
    direction LR
    D3(("h₃ᵈᵉᶜ<br/>(prev hidden)"))
    D4(("h₄ᵈᵉᶜ<br/>(current state)"))
    Y3["y₃ ('es')<br/>(prev token)"]
    Y4["y₄ ('todo')<br/>(generated)"]

    D3 -- "W_h" --> D4
    Y3 -- "W_y" --> D4
    D4 -- "g(h₄)" --> Y4
  end

  C ==>|"W_c"| D4

  %% Styling
  classDef encNode fill:#0f172a,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
  classDef decNode fill:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc
  classDef context fill:#451a03,stroke:#f97316,stroke-width:2px,color:#fff
  classDef outNode fill:#4c0519,stroke:#f43f5e,stroke-width:2px,color:#fff

  class X1,X2,X3,X4,X5,E1,E2,E3,E4,E5 encNode
  class D3,D4,Y3 decNode
  class C context
  class Y4 outNode
</pre>
<figcaption>The encoder folds the five English words into hidden states, ending in a single context vector <em>c</em>. The decoder reuses that same, unchanged context at every step to generate each Spanish word, shown here generating the fourth word, "todo".</figcaption>
</figure>

#### The encoder decoder architecture

Architecturally, tying this back to the perceptrons from [the first post in this series]({% post_url 2026-04-07-MachineLearningBeginning %}), the encoder and decoder are each ordinary layers of neurons, complete with weights, biases, and activations, wired together with one addition: a feedback loop, where a layer's own hidden state at one time step becomes part of its input at the next. That loop is what lets one fixed set of weights process a variable-length sequence one token at a time, instead of needing a separate set of weights per position. Pairing two of them, one reading and one writing, is the encoder-decoder architecture that machine translation was built on before attention[^sutskever] [^cho]. The diagram below unrolls this loop over time, showing the same encoder and decoder weights reused at every step as the five-word English sentence is read in and the Spanish translation is read out.

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

  H5 ==>|"State freezes"| C{{"Context Bottleneck (c)"}}

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
<figcaption>The encoder and decoder each reuse one fixed set of weights at every time step, the feedback loop from the recurrence above, unrolled here in time. Everything the network knows about the input has to funnel through the single context vector <em>c</em> .</figcaption>
</figure>


#### The bottleneck

This architecture has three noteworthy limitations.

**It is sequential by construction.** Each hidden state depends on the one before it, so the network cannot process a whole sequence in parallel, only token by token, which makes both training and inference slow and hard to batch efficiently.

**Everything funnels through one vector.** Every input, however long, is compressed into one fixed-size context vector $c$. As the input grows, that vector has to represent more and more of it in the same amount of space, so semantic detail from early tokens gets diluted or lost by the time the decoder needs it. This is not just an intuition: Cho et al. measured it, and the BLEU score of a fixed-context encoder-decoder falls off steeply as source sentences get longer[^chodegrade], while the attention-based model of the next section holds its score across lengths.

**Gradients are unstable over long sequences.** The encoder-decoder layout is assembled from differentiable steps, so it can be trained by gradient descent with backpropagation, and running backpropagation through every time step means repeatedly multiplying by the same recurrent weight matrix $W_h$ and by the derivative of the activation. Pascanu et al. state the condition precisely: with $\gamma$ a bound on that derivative ($\gamma = 1$ for tanh), $\lambda_1 < 1/\gamma$ for the largest eigenvalue of $W_h$ is *sufficient* for gradients to shrink exponentially with sequence length, the vanishing gradient problem, and training stalls for lack of meaningful updates. The mirror condition, $\lambda_1 > 1/\gamma$, is *necessary* for the exploding gradient problem, though not sufficient; when gradients do explode, gradient descent takes wild, unstable steps instead of settling toward a minimum.[^pascanu] Shrinking $W_h$ to escape one problem walks straight into the other.

Chronologically, gradient clipping and gated architectures like the LSTM[^lstm] were the first real fixes for the third limitation. For our purposes, we jump straight to attention, which targets the second.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Exploding gradient problem</div>
<div class="collapsiblecontent" markdown="1">
When training neural networks using backpropagation, we use the chain rule to calculate how much a small change in an early layer's weights affects the final loss. For deep feedforward networks (networks with many hidden layers) or recurrent neural networks (RNNs), this means the gradients of earlier weights are calculated using increasingly many multiplications by the layer's weight matrix (e.g., $W$ in an RNN) for every layer or time step. If the matrix operation inherently "stretches" the gradient vectors, this repeated multiplication causes the weight gradients at earlier layers to grow exponentially. The result is that the optimization algorithm takes massive, unstable steps that can introduce instability in the training process, slow it down, or halt it entirely.

</div>
</div>

## Attention

The next logical step up, attention[^bahdanau], sandwiched in an encoder-decoder layout, targets the second limitation: it gives the decoder direct access to every encoder hidden state, not just the final one, so it can learn which parts of the input matter most for the word it's generating right now, instead of relying on everything being routed through one fixed context vector. The encoder here still reads the input sequentially, exactly as before, so the first limitation stands for now; attention changes what the decoder can look at, not how the input is read.

The fixed context $c$ handed to the decoder is one snapshot, frozen after the encoder finishes reading. But the decoder needs different context at different steps. For instance, generating "todo" leans on different words than generating "necesitas". So let the decoder search back over the encoder's hidden states itself, at every step, and build a context suited to that step alone.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Softmax</div>
<div class="collapsiblecontent" markdown="1">
Softmax is a function that converts a vector of arbitrary real-valued scores into a probability distribution. For a vector $z$ of length $n$:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

Two properties fall out of this: every output is strictly positive (each term is an exponential, and the denominator is a sum of exponentials), and all outputs sum to exactly 1. Together these make the output a valid probability distribution over the $n$ inputs, and with more than one input each individual weight is strictly less than 1. That's what makes softmax useful here: given a set of scores, it turns them into weights that can be used to compute a weighted average, which is what equation (5) below does.

The exponential also means the largest score dominates: differences between scores are amplified, so a set of similar scores produces a near-uniform blend while one clear winner produces something close to a hard selection. That property comes back in the next post, where the scale of the scores going in turns out to matter.
</div>
</div>

Comparing the decoder's current state against each encoder state gives a measure of relevance:

$$
e_{ij} = h_{i-1}^{dec} \cdot h_j^{enc} \tag{3}
$$

the dot product is large when the two vectors point in a similar direction, i.e. when encoder position $j$ is relevant to what the decoder is about to generate at step $i$. Normalizing these scores with softmax turns them into weights that sum to 1:

$$
\alpha_{ij} = \text{softmax}(e_{i1}, \dots, e_{i5})_j \tag{4}
$$

and the new context is a weighted sum of the encoder states, weighted exactly by how relevant each one is right now:

$$
c_i = \sum_{j=1}^5 \alpha_{ij} h_j^{enc} \tag{5}
$$

$c_i$ replaces the static $c$ in equation (2), $h_i^{dec} = f\left(W_h h_{i-1}^{dec} + W_y y_{i-1} + W_c c_i + b\right)$, recomputed fresh at every step instead of fixed once. That's how the network reaches across the whole input without squeezing it through one bottleneck.

The interesting object here is the matrix $\alpha$ itself: one row per output word, one column per input word, each row summing to 1. Read as a grid it shows which source words the model looked at to produce each target word, which is how Bahdanau et al. demonstrated that a network trained only on sentence pairs learns something recognisable as word alignment, without ever being told what alignment is. Plotting a real one requires a trained model, so that figure waits for the post that does the training.

### Carving attention out
This search, compare, then blend mechanism is useful enough to name and carve out as a standalone primitive, independent of the RNN it lives inside here. The vector doing the searching, $h_{i-1}^{dec}$, is a query. The vectors being searched against, $h_j^{enc}$, are keys; the vectors being summed are values. Here keys and values are the same encoder hidden states, the next section projects them independently instead. For linguistic tasks, this is what lets the network capture far-away dependencies, idioms are the clearest example, understanding one often depends on the whole phrase rather than any single word in it.

One historical note on equation (3). Bahdanau's original scoring function was not a dot product but a small learned network, $e_{ij} = v_a^\top \tanh(W_a h_{i-1}^{dec} + U_a h_j^{enc})$, trained jointly with everything else. Luong et al. later compared several scoring functions and found the plain dot product to work about as well while being considerably cheaper[^luong], and that is the form used above, and the form that survived into the transformer. The dot product does impose one constraint the learned version does not: the two vectors have to have the same dimension to be multiplied together at all. The projections in the next section remove that constraint.

### Queries, keys and values

In the walkthrough above, queries, keys, and values were the raw hidden states themselves, $h_{i-1}^{dec}$ doubling as the query, $h_j^{enc}$ doubling as both key and value. In practice each role gets its own learned projection: three weight matrices $W_Q$, $W_K$, $W_V$ map the underlying vectors into a query space, a key space, and a value space, so the network can learn what to search for and what to search on separately, instead of reusing one vector for every role.

$$q_i = h_i W_Q, \qquad k_j = h_j W_K, \qquad v_j = h_j W_V \tag{6}$$

Hidden states are treated as row vectors here, so a projection is a right multiplication. This matches how the matrices are laid out in code, where a batch of vectors is a matrix with one row per vector.

These weights are learnable, exactly like any other weight matrix in the network, trained by gradient descent alongside everything else. There's nothing hand-designed about what a query or key ends up representing, the search-match-collect intuition is a way to reason about the role each projection plays, not a constraint imposed on it.

Computing one query, key, and value at a time is wasteful once there are many positions to process, so in practice the hidden states are stacked into a matrix $H \in \mathbb{R}^{n \times d_{model}}$, one row per position, and projected all at once:

$$Q = HW_Q, \qquad K = HW_K, \qquad V = HW_V \tag{7}$$

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 50, "rankSpacing": 70}}}%%
flowchart TD
  H["Stacked Hidden States H<br/>(n × d_model)"]

  WQ["Linear Projection<br/>× W_Q (d_model × dₖ)"]
  WK["Linear Projection<br/>× W_K (d_model × dₖ)"]
  WV["Linear Projection<br/>× W_V (d_model × dᵥ)"]

  Q["Queries (Q)<br/>(n × dₖ)<br/><i>'What am I looking for?'</i>"]
  K["Keys (K)<br/>(m × dₖ)<br/><i>'What information do I offer?'</i>"]
  V["Values (V)<br/>(m × dᵥ)<br/><i>'What is my actual content?'</i>"]

  H ==> WQ ==> Q
  H ==> WK ==> K
  H ==> WV ==> V

  %% Styling
  classDef inputMat fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
  classDef weightMat fill:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc
  classDef queryMat fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#fff
  classDef keyMat fill:#065f46,stroke:#34d399,stroke-width:2px,color:#fff
  classDef valueMat fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#fff

  class H inputMat
  class WQ,WK,WV weightMat
  class Q queryMat
  class K keyMat
  class V valueMat
</pre>
<figcaption>The stacked hidden states are projected through three independent, learnable weight matrices to produce the query, key, and value matrices. Queries come from the sequence being generated and keys and values from the sequence being read, so the two can differ in length, n and m; in self-attention below all three come from the same sequence and n = m.</figcaption>
</figure>

Steps (3), (4) and (5) then collapse into two matrix multiplications with a softmax between them, computing every step's context at once:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(QK^\top\right)V \tag{8}$$

The standard form of this equation carries a $1/\sqrt{d_k}$ factor inside the softmax, for reasons that have nothing to do with the mechanism itself and everything to do with the scale of the numbers going in. It is motivated in the next post.

With $Q$ shaped $(n \times d_k)$ and $K$ shaped $(m \times d_k)$, the score matrix $QK^\top$ is $(n \times m)$, one score per query-key pair. When queries and keys come from the same sequence, as in the next section, that is $(n \times n)$: quadratic in sequence length, so doubling $n$ quadruples the number of scores to compute. That is the reason attention gets expensive on long sequences. Whether it also quadruples memory is an implementation question rather than a property of equation (8): written the direct way, as it is below, the entire score matrix is materialized, while fused kernels like FlashAttention compute it in tiles and never hold it in full, keeping memory linear in $n$ even though the compute stays quadratic[^flash].

### Self-attention
So far attention has connected two separate sequences, cross-attention: queries came from the decoder's hidden state, keys and values from the encoder's. Self-attention is the same score-then-blend mechanism, but every projection reads from the same sequence, so a sequence attends to itself instead of to some other sequence.

$$q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V \tag{9}$$

There's no encoder feeding a separate decoder anymore, one sequence generates all three roles from itself, then attends across its own positions to build a representation that already accounts for the rest of the sentence. This is the mechanism the rest of the series builds on: no recurrence is needed to accumulate context, every position can see every other position directly, in one step.

That step comes at a price the recurrence was paying silently. An RNN gets word order for free, because it reads words in order. Equation (8) has no notion of order at all: shuffle the rows of $H$ and the rows of the output shuffle identically, with the same values, so "dog bites man" and "man bites dog" produce the same set of representations. Order has to be put back in explicitly, which is the job of positional encoding in the next post.

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

As seen above, attention is simply a series of matrix multiplications. The following implementation uses Flax[^flax], a library with handy abstractions for commonly used primitives in machine learning. The linear layers used here are not merely analogous to a matrix multiplication, they are exactly one: [`nnx.Linear`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/linear.html) computes $y = xW + b$, and constructing it with `use_bias=False` drops the offset and leaves $y = xW$, a single learned weight matrix and nothing else. Notably, Flax also provides an implementation of [Attention](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html), the implementation below is simply a learning exercise. The code in this post runs on jax 0.10.2, flax 0.12.8 and optax 0.2.8; the `nnx.Optimizer` signature in particular changed in recent Flax releases.

<!-- TODO: explain why nnx.Module used in class declaration, also, use same dimensions for queries,keys and values say its convention -->
<!-- TODO: explain rngs and jax and pure functions and why it is needed for efficient computations -->
<!-- TODO: encoder makes embeddings decoder predicts next token -->
<!-- TODO: attention, amount of weight for each element in a sequence -->
<!-- TODO: use bertviz on jupyter for visualization of attention, comes with BERT model too -->


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

Keys and values take a single `x_context` argument rather than one each, because key $j$ and value $j$ are two projections of the same token: the thing being matched against and the thing contributed when it matches. Self-attention is then not a separate mechanism at all, it is the same call with the same sequence passed in both roles. The `axis=-1` on the softmax is load-bearing: normalizing along the key axis is what makes each query's weights sum to 1, and therefore what makes the final line a weighted average rather than an arbitrary sum.

Every arrow below is one line of that `__call__`, with the shapes the constructor above actually produces: a five-token query sequence attending over a seven-token context, `d_model = 16`, `dₖ = 8`, `dᵥ = 12`.

<figure>
<pre class="mermaid">
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 95}}}%%
flowchart LR
  XQ["x_query<br/>(5 × 16)"]
  XC["x_context<br/>(7 × 16)"]

  Q["queries<br/>(5 × 8)"]
  K["keys<br/>(7 × 8)"]
  V["values<br/>(7 × 12)"]

  S["scores<br/>(5 × 7)"]
  W["weights<br/>(5 × 7)<br/>rows sum to 1"]
  O["output<br/>(5 × 12)"]

  XQ -- "@ w_q kernel (16 × 8)" --> Q
  XC -- "@ w_k kernel (16 × 8)" --> K
  XC -- "@ w_v kernel (16 × 12)" --> V

  Q == "@ keys.T" ==> S
  K -.-> S
  S == "softmax(axis=-1)" ==> W
  W == "@ values" ==> O
  V -.-> O

  classDef input fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
  classDef queryMat fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#fff
  classDef keyMat fill:#065f46,stroke:#34d399,stroke-width:2px,color:#fff
  classDef valueMat fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#fff
  classDef interMat fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
  classDef outMat fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#fff

  class XQ,XC input
  class Q queryMat
  class K keyMat
  class V valueMat
  class S,W interMat
  class O outMat
</pre>
<figcaption>The implemented module, end to end. Every labelled arrow is a matrix multiplication and the three kernels named on them are the entire parameter set, no biases anywhere; the only step that is not a matmul is the softmax. Training adjusts nothing but those three matrices.</figcaption>
</figure>

The forward pass really is only that: two matrix multiplications with a softmax between them. Nothing in it needed a hand-written derivative, which is the point of building it in JAX, the backward pass is asking for gradients of a loss with respect to the three weight matrices, and handing them to an optimizer:

```python
import optax

optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)


def loss_fn(model, x, context, target):
    return jnp.mean((model(x, context) - target) ** 2)


target = jnp.zeros((5, 12))   # stand-in for whatever the task supervises

loss, grads = nnx.value_and_grad(loss_fn)(model, sequence, context, target)
optimizer.update(model, grads)
```

The gradient that comes back for `w_q` is shaped $(16, 8)$, exactly the shape of its weight matrix, and `update` steps that matrix. Learning attention is nothing more than repeating those two lines: differentiate the matrix multiplications, nudge the three matrices, repeat. The full notebook is available [here](/assets/notebooks/attention.ipynb).

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>What <code>optax.adamw</code> is doing</div>
<div class="collapsiblecontent" markdown="1">
An optimizer is the rule that turns gradients into an actual change in the weights. The gradient only says which direction increases the loss; it says nothing about how far to move. The simplest rule, plain gradient descent, is the one from the [first post in this series]({% post_url 2026-04-07-MachineLearningBeginning %}):

$$W \leftarrow W - \eta \nabla_W L$$

The step size $\eta$ is the learning rate, and it is unforgiving. Too large and each update overshoots the minimum it was aiming at, so the loss oscillates or diverges outright. Too small and training crawls, or settles into the first shallow dip it encounters. Worse, no single $\eta$ suits every parameter: a weight receiving consistently large gradients wants smaller steps than one that is barely updated at all, and the right scale changes as training progresses.

Adam[^adam] keeps a running estimate of the mean of each parameter's recent gradients and of their mean square, the uncentered second moment, and divides the step by the square root of that second moment, so every weight gets its own effective step size: parameters with large or erratic gradients are damped, parameters with small steady gradients are amplified. The `1e-3` passed above is still a learning rate and still sets the overall scale, so it has not disappeared, but the optimizer absorbs most of the per-parameter tuning that would otherwise have to be done by hand. The `W` in AdamW[^adamw] is decoupled weight decay, which shrinks weights directly during the update rather than by adding a penalty to the loss, the version that behaves correctly alongside Adam's per-parameter scaling. For a survey of these rules and what each one fixes about plain gradient descent, Ruder's overview[^ruder] is the clearest single reference.
</div>
</div>

## Takeaways

- Attention removes the fixed-context bottleneck. Instead of one snapshot reused at every decoder step, each step builds its own context as a weighted sum over all of the encoder's hidden states, with the weights computed fresh from what that step is trying to generate.
- The primitive underneath is three roles and two matrix multiplications: a query searching, keys being searched against, values being blended, scored by a dot product and normalized by a softmax. Each role is an ordinary learned projection; nothing about what a query or key represents is hand-designed.
- The cost is one score per query-key pair, quadratic in sequence length, which is the constraint every long-context technique since has been working against.
- Self-attention drops the recurrence entirely, so a sequence can be processed in parallel rather than token by token, at the cost of losing word order, which has to be reintroduced explicitly.

## Up next

That parallelism is what makes training at the scale modern language models are trained at feasible in the first place, and the fluency and emergent reasoning those models show comes from stacking this one simple primitive, matrix multiplications and a softmax, at that scale, on enormous amounts of filtered text. Generation is still sequential, one token at a time; it is training that parallelizes.

Vaswani et al. (2017)[^vaswani] showed that attention alone, wired together with simple feed-forward layers and no recurrence at all, reaches state-of-the-art translation quality, measured in BLEU score (a standard machine-translation metric). That architecture is the transformer, and assembling it, along with the residual connections, normalization and positional encoding it needs to work, is the subject of [the next post]({% post_url 2026-06-12-transformer %}).

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
