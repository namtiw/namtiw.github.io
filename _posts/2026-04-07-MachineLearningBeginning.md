---
layout: post
title: Machine Learning
subtitle: From scratch in JAX
thumbnail-img: /assets/img/ml.jpg
mathjax: true
mermaid: true
published: false
tags: [Machine-Learning, Perceptron, Transformer, JAX]
---
Traditional computer programming involves encoding real world problems into decision tree like representations in code. Such programs are deterministic, for a given input state the program follows the same route in the tree to get to identical output every time. Machine learning introduces the possibility of enabling computers to learn this decision tree given a set of curated input and output states (ignoring [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) to simplify here). At face value it is a steep promise, given enough data a computer is able to learn any decision tree and perform any real world task. In practice, for any problem worth solving, collecting the required labelled data and setting up a conducive learning architecture is non trivial. Neural networks are one such architecture: they have proven to be on par with or better than humans at image classification[^he], and the latest Transformer based networks match human performance on a range of language tasks when trained on reasonably sized datasets.

<pre class="mermaid">
flowchart LR
  subgraph TP["Traditional programming: encoding human expertise"]
    direction LR
    E["Human expertise"] --> RW["Hand-coded rules<br/>every situation anticipated,<br/>each rule coded, debugged, refined"] --> P1["Program"]
    IN1([Input]) --> P1 --> OUT1([Output])
  end
</pre>

<pre class="mermaid">
flowchart LR
  subgraph ML["Machine learning: learning from data"]
    direction LR
    EX["Labelled examples"] --> LA["Learning algorithm<br/>predict, compare,<br/>adjust on error"] --> W["Learned parameters"]
    IN2([Unseen input]) --> W --> OUT2([Output])
  end
</pre>

The next few posts document my pedagogical exercise of building a decoder only Transformer network from scratch. The aim is to develop an intuitive understanding of the architecture most modern LLMs use and implement it from scratch in JAX, to reason about how massive models are designed piece by piece and what the learning process looks like for these networks. ***Intended as my notes for ML concepts, starting from basic math and computer science principles.*** To follow along, a basic understanding of linear algebra (matrix multiplication), calculus (gradients and partial derivatives), and Python programming is a prerequisite; no prior knowledge of machine learning is required.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>A note on terminology: model, network, LLM</div>
<div class="collapsiblecontent" markdown="1">
These posts use *model*, *network*, and *LLM* somewhat interchangeably. Strictly: the network is the architecture, the graph of operations and its parameter slots; a model is a network together with a specific set of learned parameter values; an LLM (large language model) is a model with a Transformer architecture whose parameters were trained on large text corpora.
</div>
</div>

### Perceptron, the first learning machine

The Perceptron, introduced by Frank Rosenblatt in 1958[^rosenblatt], is a binary classifier. He developed an iterative learning algorithm which _trains_ the network on a set of datapoints such that, when given an unseen example, the network produces the expected result, eliminating the need to encode every possible example and its outcome[^cornell]. In fact, starting from an arbitrary initial state, the algorithm is guaranteed to eventually converge to a solution that classifies 100% of the examples correctly, given such a solution exists[^novikoff].

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Historical context: from hand-coded rules to learned weights</div>
<div class="collapsiblecontent" markdown="1">
The perceptron was not the first mathematical model of a neuron. Warren McCulloch and Walter Pitts published one in 1943, but their neurons had fixed, hand-set connections; it was a theory of computation, not of learning. Rosenblatt's contribution at Cornell Aeronautical Laboratory was a working system that acquired its parameters from labelled examples[^brenndoerfer].

The urgency and funding came from the Cold War. Military agencies wanted machines that could pick out aircraft or tanks in aerial photographs without teams of human analysts, and a pattern recognizer that adapts from examples promised exactly that. The contrast with the programming practice of the day was stark: an optical character recognizer, for instance, was built from hand-crafted rules about the loops and strokes of each letter, every rule individually coded, debugged, and refined. The perceptron replaced rule writing with data curation, supply labelled examples and let the algorithm assign the feature weights. It also gave the field its first formal learning guarantee, the convergence theorem above.

The design borrowed loosely from biology: a neuron collects signals through its dendrites and fires down its axon once the combined stimulation crosses a threshold. The weighted sum and step function mirror exactly this.
</div>
</div>

#### Problem statement

In order to motivate this topic, imagine developing a (traditional, deterministic) program to solve a classification problem of Shakespearean plays into Comedy and Not Comedy. For our purposes, assume we have a dataset with true/false binary labels for each work. A viable approach, rooted in familiarity with his work, is to check if the story [ends with at least one wedding](https://www.bellshakespeare.com.au/genres-of-shakespeares-plays). Given the volume of Shakespearean work it is expected this strategy only gives correct output for a small subset of all available examples. Making the strategy slightly more competent, let's add: if no main character dies, it is a comedy. It is evident that solving this problem with respectable levels of accuracy spanning all of Shakespeare's works will require a rigorous analysis of the dataset to discover emerging themes and develop corresponding rules.

Let's break this down into a simpler subproblem. Assume our data has two identified notable features (ending with a wedding, death of a character in our analogy) $x_1$ and $x_2$ that cleanly divide it into two classes. Mathematically, this can be represented as points in a two dimensional space with a clear boundary between two classes of points. Taking a geometry based shortcut to the solution, the general purpose [equation of a line](https://mathworld.wolfram.com/Line.html) in our two dimensional feature space is given by

$$w_1 x_1 + w_2 x_2 + b = 0 \tag{1}$$

Hence the objective, given a set of points in 2D space learn the parameters $w_1$, $w_2$, $b$ such that it divides the space with different class of points on either side.

+a diagram here to represent the dataset of points and explain the equation of plane+
points in 2d space two classes represented as blue and red

The variable names are chosen intentionally to tie into ML naming conventions.

#### The architecture

The original [photoperceptron](https://com-cog-book.github.io/com-cog-book/features/perceptron.html) developed by Rosenblatt drew interesting inspirations from biology; we discuss a slight variation that is today most widely recognised as the Perceptron.

The notion of a linear function that outputs the class of a 2d point can be represented as a network of mathematical operations on the input coordinates.

<pre class="mermaid">
flowchart LR
  X1(["x₁"]) -- "w₁" --> S["Σ&nbsp;&nbsp;weighted sum<br/>w₁x₁ + w₂x₂ + b"]
  X2(["x₂"]) -- "w₂" --> S
  B(["b"]) --> S
  S -- "z" --> F["step function<br/>f(z), threshold T"]
  F --> Y(["y′ = 0 or 1"])
</pre>

$$y' = f(w_1 x_1 + w_2 x_2 + b) \tag{2}$$

where $f$ is the step function

$$f(z) = \begin{cases} 1 & \text{if } z \geq T \\ 0 & \text{if } z < T \end{cases} \tag{3}$$

The step function helps translate the continuous output of the linear function into discrete values which can be interpreted as classes. Note that the threshold $T$ is redundant given the bias: rewriting $z \geq T$ as $z - T \geq 0$ folds $T$ into $b$, so by convention the threshold is set to $0$.
Intuitively, this can be thought of as a weighted sum of features that govern a dataset. In the Shakespearean analogy, a wedding finale contributes differently to a particular work being a comedy than the death of a protagonist, and the combined effect must pass a certain threshold (step function) to qualify as a comedy.

+run one pass through the network using two plays as examples, assume some weights+
For a given input, the network's decision depends on $w_1$, $w_2$, $b$, and the threshold.
#### Learning algorithm

Generalizing over the dataset, for the $i$-th point $(x_{1i}, x_{2i})$, $y_i$ is the true label from the dataset and $y'_i$ is the network output:

$$y'_i = f(w_1 x_{1i} + w_2 x_{2i} + b) \tag{4}$$

<pre class="mermaid">
flowchart LR
  DS[("Dataset")] -- "(x₁ᵢ, x₂ᵢ)" --> N["Network"]
  N --> YP(["y′ᵢ&nbsp;&nbsp;prediction"])
  DS -- "label" --> Y(["yᵢ&nbsp;&nbsp;true value"])
  YP --> E["error<br/>yᵢ − y′ᵢ"]
  Y --> E
</pre>

The difference between the network output (prediction) and the true value, $y_i - y'_i$, represents the error in our classification. The algorithm states that for each datapoint, a correction equal to this error, scaled by a factor $\eta$, is applied to the parameters:

$$w_j \leftarrow w_j + \eta \, (y_i - y'_i) \, x_{ji} \qquad b \leftarrow b + \eta \, (y_i - y'_i) \tag{5}$$

The scaling factor $\eta$ controls the effect each wrong classification has on the parameter update. This is known as the learning rate, since increasing it means the program takes faster steps towards the right answers in the dataset. Eventually, after applying similar nudges for enough datapoints, the parameters converge to stable values that correctly classify the dataset. One adjustment step in this iterative process is illustrated in the following figure.

<pre class="mermaid">
flowchart TD
  DS[("Dataset")] -- "next point<br/>(x₁ᵢ, x₂ᵢ), yᵢ" --> N["Network<br/>w₁, w₂, b"]
  N -- "y′ᵢ" --> C["Compare<br/>error = yᵢ − y′ᵢ"]
  C -- "scale by η" --> U["Adjust w₁, w₂, b<br/>eq. (5)"]
  U -. "repeat for each point" .-> DS
</pre>

##### Learning visualized

The mathematical operations in the above pass through the network are succinctly represented by matrix multiplications (ideal for hardware acceleration when running this code, more on this later). Collecting the weights into a matrix $W$ and the inputs into a vector $\mathbf{x}$, equation (2) becomes:

$$y' = f(W\mathbf{x} + b), \qquad W = \begin{bmatrix} w_1 & w_2 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \tag{6}$$

For a single perceptron $W$ has one row. Each additional neuron in a layer adds a row, and the same matrix product computes every neuron's weighted sum at once.

+ talk is cheap show me the code+
Translating from English to Python gives us the following. Note, each iteration of the loop takes a dataset input value, passes it through the network, and then compares the result to the expected output. The difference dictates the adjustment in network parameters. Thus, with each step the network's decision boundary is nudged towards the correct final state.
+JAX concepts used here, see below+
+show a slider with animation so each example step and nudge can be visualized+

Take a moment to appreciate that from a simple loop in Python and some datapoints emerges spatial reasoning capability, in the feature dimensions, in a deterministic digital machine. The New York Times called this the "embryo of an electronic computer" at the time[^nyt].
+insert image of nyt snippet+

It is evident from equation (2) that this network is capable of capturing (learning to represent) a linear relationship between the inputs and outputs. Most real world problems do not fit into this constraint. Marvin Minsky, interestingly a schoolmate of Rosenblatt[^bronx], along with Seymour Papert proved this limitation in their 1969 book *Perceptrons*[^minsky], showing a single layer perceptron cannot solve problems that are not linearly separable (the XOR gate being the canonical example), which led to the first of the AI winters in history.

Non-linearity in the network's representation is necessary to solve richer problems with it. In other words, real world problems often have a large number of features (think a higher dimensional feature space, instead of 2D) that affect the output, so the mathematical relation is a higher order function. A (surprisingly) straightforward avenue is adding more than one layer of the above matrix multiplications to the network.

### Multilayer perceptrons

There are two levers to pull: adding more layers to the network and increasing the number of neurons per layer.

<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>What is a neuron?</div>
<div class="collapsiblecontent" markdown="1">
A neuron (also called a unit or node) is the smallest building block of a network: it computes a weighted sum of its inputs, adds a bias, and applies an activation function, exactly equation (2). A layer is a set of neurons reading the same inputs; in the matrix form of equation (6), each neuron is one row of $W$. The name is a loose biological metaphor and carries no further meaning here.
</div>
</div>

The $w_i$ are parameters of the model. As seen above, these control the contribution of a feature to the weighted sum, hence they are known as the weights of a model. The decision tree branching is controlled by the bias ($b$) and activation (step function) terms of a neuron. For instance, +from the diagram explain neuron 3 layer 2 has affinity to be activated, output 1 whenever neuron 4 in layer 1 is active will have a higher weight associated with the corresponding link+ For an individual neuron, its affinity to activate is controlled by the bias term and activation: a higher bias implies a higher weighted sum, and a lower activation threshold implies a lower value of the sum activates the neuron.

Two notable aspects of the learning algorithm are the measure of the error in its output and the incremental corrections to reduce this error. The error in ML papers is often referred to as the Loss. It has been empirically shown[^goodfellow] that as models get deeper (higher number of layers, higher number of weights to be learnt) a simple difference based loss is too choppy, and functions like root mean square result in smoother updates to the network at each step. Mathematically, the relation between input and output is represented by a complex higher order function. Unlike the linear case, each neuron (weight term) in a layer contributes to the loss differently. In order to correctly update each weight according to its effect on the loss, backpropagation is used. Think of the loss as a curve: for any given point on the curve, the direction of a minimum can be deduced using the slope of the tangent (gradient) at that point. +show diagram with a curve and a gradient at a point show both a negative and positive gradient+ Hence, the partial derivative of the loss with respect to each weight gives the gradient at a point with respect to that weight, exactly what is needed here to update the weight. The weights in layers deeper into the network can be similarly derived using partial derivatives of the loss. Since the derivatives with respect to the weights of the following layer are already known, the chain rule can be used to avoid recalculations. Hence, the process of moving backwards in the network and calculating partial derivatives at each step using derivatives from a later step is known as backpropagation. The gradients can then be scaled and used to update the weights of the network during training. Andrej Karpathy's Zero to Hero series is a great resource to understand this concept, as he manually shows the forward pass to get the loss and then the backward pass to propagate the gradients through a network.

#### JAX
[JAX](https://docs.jax.dev/en/latest/) is a Python framework which makes developing new network architectures and training them easier by abstracting away tedious parts like handrolling gradient calculations at each layer and weight updates.
<div class="collapsiblecontainer">
<div class="collapsibleheader"><span>+ </span>Why JAX over PyTorch?</div>
<div class="collapsiblecontent" markdown="1">
Both frameworks provide the two things that are tedious to handroll: automatic differentiation and accelerator-ready array operations. PyTorch goes further, packaging layers, parameter state, and training utilities into `nn.Module` objects. JAX stays close to NumPy: parameters live in explicit data structures the programmer passes around, and a model is a pure function of parameters and inputs. For a from-scratch exercise this is the point, nothing is hidden and every weight update is visible in code, which also permits more control over complex network implementations. Fewer abstractions additionally mean fewer places for implicit behavior to hide: shape pitfalls such as the `keepdims` dilemma in reduction operations still exist, but they sit in code you wrote and can inspect, rather than inside a framework module where they lead to sneaky training bugs.
</div>
</div>

#### Implementation and Training
+show one perceptron and then make it multi layer+
+two moons dataset for training example in JAX+

### Conclusion

Given enough data and a capable model architecture, interesting emergent properties are observed from the training process: without explicit mention of two classes or a classification problem, the network trains to fit the dataset and becomes capable of classifying unseen inputs correctly (assuming the unseen input follows the same patterns). Some key takeaways from the exercise here are:
- The number of features, the size of a network, and its architecture (how layers are structured and connected) dictate the nature of problems that it will be able to learn to solve with respectable accuracy.
- Non linearities in the network are important to enable it to model complex real world problems.
- The training process is more hands off than the original Bard example. For instance, selecting sarcasm and puns as features is not required. However, a good understanding of the major features that relate inputs to outputs will aid in better architecture of the network. This highlights two main problems with training capable networks: availability of clean, ample datasets for the problem at hand, and understanding of the dataset to be able to architect a model conducive to training.
- Since they are not explicitly encoded, the features a model identifies in a dataset as important (represented by higher weights assigned to the neuron) are non deterministic. In a sense, the model is a black box.
- As models get deeper, the choice of architecture, activations, and loss function dictates the model's ability to follow the gradient smoothly to a minimum on the loss curve. Various strategies to increase the likelihood of this convergence using a reasonable amount of data have been developed over the years. More on these in later posts as we use each to improve the network and its training.
+Anthropic discovering consciousness like properties in networks shows this black box nature and emergent properties, a sentence on this and a link out+

### References
This series is heavily inspired by Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series, which is worth watching in full. These posts document a similar journey, building a Transformer based model from scratch, but in JAX, with a different set of examples for intuition and a different final product.

[^he]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852). IEEE International Conference on Computer Vision (ICCV).
[^rosenblatt]: Rosenblatt, F. (1958). [The perceptron: A probabilistic model for information storage and organization in the brain](https://doi.org/10.1037/h0042519). Psychological Review, 65(6), 386-408.
[^cornell]: [Professor's perceptron paved the way for AI, 60 years too soon](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon). Cornell Chronicle, September 2019.
[^brenndoerfer]: Brenndoerfer, M. (2025). [The Perceptron, Foundation of Modern Neural Networks](https://mbrenndoerfer.com/writing/history-perceptron-neural-network-foundation).
[^novikoff]: Novikoff, A. B. (1962). On convergence proofs on perceptrons. Proceedings of the Symposium on the Mathematical Theory of Automata, 12, 615-622. Known as the perceptron convergence theorem.
[^nyt]: [New Navy Device Learns By Doing](https://www.nytimes.com/1958/07/08/archives/new-navy-device-learns-by-doing-psychologist-shows-embryo-of.html). The New York Times, July 8, 1958.
[^bronx]: Both attended the Bronx High School of Science, one year apart. See [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), Wikipedia.
[^minsky]: Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
[^goodfellow]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press. Chapters 6 and 8 cover loss functions and the optimization difficulties of deep models.
