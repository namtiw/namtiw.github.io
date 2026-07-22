---
layout: post
title: Implementing Attention in JAX
subtitle: The atom 
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
Given an input sequence, the task is to generate another sequence satisfying some requirement. This spans an amazingly vast range of real world problems: text generation such as autocomplete, and video game frame generation, where a renderer predicts frames it hasn't actually computed from the ones that came before. For the purposes of this post, consider language translation: for a sentence in one language, generate the sequence of words that is its translation in another. Neural networks are graphs of weight matrices, and they work best on numerical representations of their inputs[^bengio]. The text sequence is therefore translated into numerical vectors in a high dimensional space, namely embeddings. The process of generating these embeddings is discussed in a later post. In a nutshell, for each word in the vocabulary there is a corresponding $d$ dimensional vector representation of it. These values are parameters of the network and are learnt during the training process.


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
A toy five word phrase surfaces the complexity of the translation problem, since it isn't a word by word lookup: consider `Attention is all you need` and its Spanish translation, `la atención es todo lo que necesitas`. The output is seven words long, and "all you need" has no single counterpart to line up against, it is spread across "todo lo que necesitas". The model must learn to accumulate context over the whole input and use it to predict the output. Recurrent neural network architecture encodes this context using hidden states: the network reads in one word's embedding at a time and folds it into a hidden state that summarizes everything seen so far. An embedding is a fixed vector for a word in isolation, while a hidden state is what that word means once combined with everything the network has read before it, an important distinction we'll lean on again shortly. The order in which words appear matters for any language task, and because the network reads them one at a time in sequence, that order is naturally preserved without any extra bookkeeping. Mathematically, the encoder's recurrence is:

$$h_i^{enc} = f\left(W_h h_{i-1}^{enc} + W_x x_i + b\right)$$

It helps to zoom into a single decoder step to see exactly how an output word gets generated. Say the encoder has finished reading all five English words and produced hidden states $h_1^{enc}, \dots, h_5^{enc}$; the final one, $h_5^{enc}$, becomes the context $c$ handed off to the decoder. Generating the fourth Spanish word, "todo", is decoder step $i = 4$:

$$h_i^{dec} = f\left(W_h h_{i-1}^{dec} + W_y y_{i-1} + W_c c + b\right), \qquad y_i = g\left(h_i^{dec}\right)$$

the decoder computes a new hidden state $h_4^{dec}$ from its previous hidden state $h_3^{dec}$, the previous output token $y_3$ ("es"), and the context $c$, then reads the next word $y_4$ off of that hidden state. The same context $c$ is reused, unchanged, at every single decoder step, no matter how long the output gets, a limitation the next section digs into. The mechanics of translating words into vectors ($x_i$, $y_i$) and reading a word back out of a hidden state ($g$) are deliberately handwaved here; they're covered properly in a later post on embeddings and output projections.

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

Architecturally, tying this to the introduction's neural network layout, this can be thought of encoder neurons with their weights, biases and activations <link to post 1 where this is discussed as inherent to neuron>, similar decoder neurons and a feedback loop that allows the network to use outputs from layers as input to previous layers at next time step. Note, in the diagram below the architecture is unrolled over time and shows how an input flows through the network. Each $f$ in the mathematical representation above can be considered as a network primitive shown above with its own learnable weights, biases and activations to learn complex non linear relationships between its inputs and outputs.
The hidden states are then used to construct the context. The decoder part of the network uses the last hidden state as $context$. That is, at the current time step $h_i$ is input to the decoder. give the formula for generating $y_i$ output from the query vector and context and the previous time step output.ss

<!-- NOTE: encoder-decoder diagram unrolled over time, hidden states flowing, context vector highlighted as the bottleneck -->

#### The bottleneck

This architecture has two limitations. The network has a high number of parameters to learn but it is not sequential in nature and only processes one token (think word of a sentence) at a time. which makes training a computationally expensive process memory constraints limit batching across examples. Also, the fixed context length limits the look back capability of the network to only a certain number of tokens hence, semantic context will be lost as the size of the input and output scale. Concretely, this idea of context being lost is the exploding gradient problem. Training the network using gradient descent implies weight matrices ($W_h$) are updated at each time step and with each multiplication operation with input vector the numerical values in these matrices grows to a large number (could be positive or negative depending on the data). As these values become larger, gradient descent, instead of taking small steps towards a minima applies large scale updates to the weights each time thus jumping around on the loss curve violently instead of settling. A knee jerk response could be to make the values in the matrix $< 1$ so multiplication brings numerical values down, although this runs into the vanishing gradient issue, wherein the weight values become so insignificantly small that gradient based updates have no effect on its values thus stalling the learning process. Chronolgically, these problems have been solved using architectures like LSTM (link out to a good resource on this) but, for our purposes we jump to `Attention`, currently regarded as the best solution to this problem.

<!-- NOTE: show the diagram for the architecture showing neuron feedback loop and unrolled network clearly calling out the time axis -->
### Attention

The Attention primitive primarily solves the increasing computations problem and learning dependencies over long distances to maintain a richer context of input seen by the network. Instead of sequentially reading the input one token at a time, it is used entirely to generate a Attention matrix which captures the $context$ and then sequentially generate output using the previous output token and some hidden state. 

#### An aside, Softmax
Softmax is a mathematical function often used in machine learning. In a nutshell, it takes in random scaled numerical value vectors and converts them to a probability distribution, ie, the output has two special characteristics all values are $< 1$ and they sum to 1. The internals of softmax are inconsequential here, a key takeaway is this operation proves to be really uses in converting outputs from an arbitrary output of a layer in a network into probability scores.  

<!-- NOTE: first appearance of softmax in the series, define properly: equation, all positive, sums to one, weighted average interpretation -->

As shown in the diagram below, generation of output $y_4$ uses the context $c$ generated using entire input sequence alongwith some hidden states. Poke at each one at a time. Starting with the alignment scores, hidden state $s_0$ and dot products of $h_i$s give matrix of scores which can be thought of as the alignment of the vectors $h_i$s with the networks hidden state. These alignment scores are then passed through a softmax to get attention weights. The attention weights are then scaled by the initial hidden states $h_i$ and summed to get the final context to use for output generation.
attention primitive is carved out from this neural network encoder decoder layout and it solves for the limited context problem by using hidden state of every input token until this time step to create context. Similar to RNNs, the query vector are used to calculate alignment scores which are then softmaxed to get attention score which are multiplied by values to scale which gives context.< add diagram to show the whole network. a diagram as heatmap of attention matrix what it will look like we will write code for it later>
the context is then used along with the query state and previous output token to calculate next output token. This allows the network to look back at the entire input so far summarized in the context and use it to predict the output. Thus, the length of the context vector does not bottle neck the semantic meaning the network can capture.

<!-- NOTE: alignment score calculation, decide additive (Bahdanau) vs dot product simplification -->



#### Carving attention out
Attention can be carved out from the encoder-decoder layout shown above. The hidden states are called query, key and values which teases the intuition of how the model is able to the capture the context over large distances in a sequence.  Inputs/data vectors $x_i$ are projected into keys and values. Alignment scores are generated from the key and query dot products. Queries in this case, are independent matrices, learnable during training. The dot product as alluded to earlier, allows the network to capture the alignment of input to any queries that the network deemed relevant during training. The softmax alignment scores are then dotted with values to get Attention scores. The final attention matrix can be thought of as the network's representation of the entire input. This can be read as the weight the network assigns to each input when using it to generate the output, thus, the influence of each input dimension. For linguistic tasks, this enables the network to capture far away words in a sentence that affect the meaning of the entire phrase, for instance, understanding idioms requires the knowledge of the whole phrase and less about the individual words. 
<!-- NOTE: walk one full step through the attention-augmented network BEFORE renaming anything, then the reveal: score-reweight-sum is a standalone primitive -->
<!-- NOTE: cross attention FIRST (queries from decoder, keys/values from encoder side), self attention comes later by making queries depend on x too -->

#### Queries, keys and values

<!-- NOTE: from query and data vector, project the data vector to key and value using learnable parameters ensure everything is covered above well -->
<!-- NOTE: batch into Q, K, V matrices; diagram and equations for these new operations -->
<!-- NOTE: intuition of searching the space with queries, finding alignment, collecting values -->
<!-- NOTE: callout - intuition only, in practice these are learnable weights and everything is differentiable so gradient descent learns them -->

#### Scaled dot product attention

<!-- NOTE: do this in next post when we actually develop the transformer architecture and explain why the paper does it why scale by sqrt(d_k): large dot products push softmax into flat regions, gradients die -->
<!-- NOTE: collapsible on the vanishing gradient problem and why it happens; distinguish from the deep-stack vanishing gradients saved for post 4 -->

#### Self-attention
It is a slight variation in the architecture above wherein, queries are also calculated using projections of the data values. <show the new diagram>

<!-- NOTE: convert queries to depend on x as well, introduce their parameters; input alone now generates queries, keys, values -->
<!-- NOTE: softmax over the score matrix gives the attention matrix -->
<!-- NOTE: visualize attention heatmap for an example to see correlation learnt, code for it in the implementation section -->

### Multi-head attention
convs2s, bytenet all solve for the sequential computational bottleneck that attention does and try to keep context over long sequences. Attention, reduces the number of operations at the cost of some loss in resolution of captured context due to averaging of weights. This can be resolved by multi head attention.

### Masked attention
<!-- NOTE:we will explain this in the next article when it becomes relevant for encoding sentences -->

### Implementing attention in JAX

then we will impelement a layer of attention in codek

<!-- NOTE: hand-crafted small vectors so every matrix is inspectable; companion notebook; decide whether jax.vmap (blueprint) is in or out -->

### Where this leads

Attention is all you need paper tells us that just the attention values passed to a mlp and a network using this gives sota performance bleu scores<write this better> that whole arch is transformer next lecture. <!-- next post starts with transformer and then moves on to decoder only after explaining why and then implements it-->
A major takeaway is how attention solves for computational constraints allowing training deep networks (high number of hidden layers) at scale while solving the context capturing problem between far away neighbors in a sequence. Hence, the success they have recently seen, know that at their core it is just matrix multiplications and the emergent properties can be attributed to the large amounts of curated data for training, the architecture design and the ingenuity that goes into training these models at scale.

[^lecun]: LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). *Backpropagation Applied to Handwritten Zip Code Recognition.* Neural Computation, 1(4), 541-551.
[^he]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* IEEE International Conference on Computer Vision (ICCV).
[^bengio]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research, 3, 1137-1155.


