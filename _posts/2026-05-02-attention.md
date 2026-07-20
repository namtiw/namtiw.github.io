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
Given an input sequence, the task is to generate another sequence satisfying some requirement. This an amazingly vast range of real world problems: text generation such as autocomplete, and video generation, where given the past frames of a video the model generates the next frame (frame generation in videogame rendering is a sequence to sequence problem at heart <link to dlss explanation>). For the purposes of this post, consider language translation, for a sentence in one language generating the sequence of words that is its translation in another. Neural networks are graphs of weight matrices and work best with numerical representations of inputs <back that with a reference>. The text sequence is translated into numerical vectors in a high dimensional space, namely embeddings. The process of generating these embeddings is discussed in a later post. In a nutshell, for each word in the vocabulary there is a corresponding $d$ dimensional vector representation of it. These values are parameters of the network and learnt during the training process. 


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

<figcaption>Each input word is represented as a row the Embedding matrix.</figcaption>
</figure>


### Recurrent Neural Networks
 A toy five word phrase surfaces the complexity as it is not a word by word lookup: consider `Attention is all you need` and its spanish translation, `<insert translation here>`. The output is seven words long, and "all you need" has no counterpart to line up against, it is spread across "todo lo que necesitas". The model must learn to accumulate context over a list of words and use it to predict the output. Recurrent neural networks encode this context using hidden states. The network takes in embeddings of one word at a time and uses it to generate a hidden state. Mathematically,
$$

h_i = <func>f(W_hh * x_i, h_(i-1))

$$

<!-- NOTE: reinstate the tokens -> model -> tokens seq2seq mermaid figure here, reused when walking through a time step -->
<!-- NOTE: one sentence distinguishing embeddings (word without context) from hidden states (sequence so far) -->

#### The encoder decoder architecture

The hidden states are then used to construct the context. The decoder part of the network uses the last hidden state as $context$. That is, at the current time step $h_i$ is input to the decoder. give the formula for generating $y_i$ output from the query vector and context and the previous time step output.

<!-- NOTE: encoder-decoder diagram unrolled over time, hidden states flowing, context vector highlighted as the bottleneck -->

#### The bottleneck

This architecture has two limitations. The network has a high number of parameters to learn but it is not sequential in nature and only processes one token (think word of a sentence) at a time. which makes training a computationally expensive process. Also, the fixed context length limits the look back capability of the network to only a certain number of tokens hence, semantic context will be lost as the size of the input and output scale.

### Attention

attention primitive is carved out from this neural network encoder decoder layout and it solves for the limited context problem by using hidden state of every input token until this time step to create context. Similar to RNNs, the query vector are used to calculate alignment scores which are then softmaxed to get attention score which are multiplied by values to scale which gives context.< add diagram to show the whole network. a diagram as heatmap of attention matrix what it will look like we will write code for it later>
the context is then used along with the query state and previous output token to calculate next output token. This allows the network to look back at the entire input so far summarized in the context and use it to predict the output. Thus, the length of the context vector does not bottle neck the semantic meaning the network can capture.

<!-- NOTE: alignment score calculation, decide additive (Bahdanau) vs dot product simplification -->

#### From scores to weights: softmax

<!-- NOTE: first appearance of softmax in the series, define properly: equation, all positive, sums to one, weighted average interpretation -->

#### Carving attention out

<!-- NOTE: walk one full step through the attention-augmented network BEFORE renaming anything, then the reveal: score-reweight-sum is a standalone primitive -->
<!-- NOTE: cross attention FIRST (queries from decoder, keys/values from encoder side), self attention comes later by making queries depend on x too -->

#### Queries, keys and values

<!-- NOTE: from query and data vector, project the data vector to key and value using learnable parameters -->
<!-- NOTE: batch into Q, K, V matrices; diagram and equations for these new operations -->
<!-- NOTE: intuition of searching the space with queries, finding alignment, collecting values -->
<!-- NOTE: callout - intuition only, in practice these are learnable weights and everything is differentiable so gradient descent learns them -->

#### Scaled dot product attention

<!-- NOTE: why scale by sqrt(d_k): large dot products push softmax into flat regions, gradients die -->
<!-- NOTE: collapsible on the vanishing gradient problem and why it happens; distinguish from the deep-stack vanishing gradients saved for post 4 -->

### Self-attention

<!-- NOTE: convert queries to depend on x as well, introduce their parameters; input alone now generates queries, keys, values -->
<!-- NOTE: softmax over the score matrix gives the attention matrix -->
<!-- NOTE: visualize attention heatmap for an example to see correlation learnt, code for it in the implementation section -->

### Multi-head attention

### Masked attention

### Implementing attention in JAX

then we will impelement a layer of attention in codek

<!-- NOTE: hand-crafted small vectors so every matrix is inspectable; companion notebook; decide whether jax.vmap (blueprint) is in or out -->

### Where this leads

Attention is all you need paper tells us that just the attention values passed to a mlp and a network using this gives sota performance bleu scores<write this better> that whole arch is transformer next lecture.

[^lecun]: LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). *Backpropagation Applied to Handwritten Zip Code Recognition.* Neural Computation, 1(4), 541-551.
[^he]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* IEEE International Conference on Computer Vision (ICCV).


