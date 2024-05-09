# Medusa

> Medusa is a new paper which introduces a new method for speculative decoding. Using Medusa, we're able to achieve up to 2.2-2.6x speedups and ~80% accuracy when we allow for top-5 choices on these new modelling heads. More importantly, the paper introduces a new method to help prevent model drift when training these new language heads.

## Introduction

> LM inference is predominantly memory-bound [Shazeer, 2019, Kim et al., 2023], with the main latency bottleneck stemming from accelerators’ memory bandwidth rather than arith- metic computations. This bottleneck is inherent to the sequential nature of auto-regressive decoding, where each forward pass requires transferring the complete model parameters from High-Bandwidth Memory (HBM) to the accelerator’s cache. This process, which generates only a single token, un- derutilizes the arithmetic computation potential of modern accelerators, leading to inefficiency.

The bottleneck in the forward pass comes from loading the model layer weights into the computation cores of your device, not from performing the computations themselves. Medusa provides a way for us to generate multiple tokens on a single forward pass.

There are a few broad categories of optimisations that we can perform to make model inference faster

1. Hardware-Specific optimisations: We can use custom kernels like Flash Attention or reduce the memory footprint of the model using Quantization
2. Batching : We can process a larger number of inputs at a single time, therefore increasing the throughput of our model
3. Parallelize the workload: Distribute the workload using multiple GPUs so that we can speed up the computation

## Speculative Decoding

### Draft Models

Prior Work with speculative decoding involves using a smaller draft model to generate tokens that are subsequently verified by a main model.

First, we take our initial prompt and generate $n$ tokens auto-regressively with a smaller draft model

![Speculative Decoding](/speculative-decoding.png)

If our draft model has predicted these tokens correctly, then we will see the same predictions in our larger model for the same inputs.

![Speculative Decoding](/speculative-decoding-verification.png)

We run inference using these completed sequences through the model for each token

```
Prompt: The Capital Of France is

Predicted: The Capital Of France is Paris and it is a beautiful city

Inference:
The Capital Of France is -> Paris?
The Capital Of France is Paris -> And?
...
```

We are thus able to get more tokens out of a single inference call. However, this introduces some additional challenges such as

1. Ensuring that the draft model is able to approximate the larger model
2. Inference Optimizations ( How do we configure the hyper-parameters )

## Medusa

### Overview

Medusa proposes a different approach. Instead of using a draft model, we can instead just use the hidden state of the last token to predict the next few tokens.

![Medusa Heads](/medusa-heads.png)

These heads aren't anything special, they're just MLP networks that generate a distribution over the entire vocabulary

![Medusa Head Eqn](/medusa-head-eqn.png)

If we have a final hidden layer that is $1xd$ where $d$ is the hidden state dimension of the transformer model, then this produces a single $1xv$ vector at the end. **Note the use of a residual layer in the equation**.

Each head produces a probability distribution over $v$ different possible choices. This means that each head is going to produce $s_k$ different options for each token. Therefore we will have $(s_{k})^n$ possible completions where $n$ is the number of heads.

![Speculative Decoding List](/speculative-decoding-tree.png)

> Therefore, we can greedily add nodes to the tree by choosing the node that is connected to the current tree and has the highest accuracy. This process can be repeated until the total number of nodes reaches the desired number. In this way, we can construct a tree structure that maximizes the expectation of the acceptance length.

We can use top-k or we can use our training dataset to learn a unique top-k parameter for each individual node so that we maximise the accuracy.

### Training

#### Freeze LLM, only heads

The easiest way to train Medusa is by using a frozen base LLM and using it's hidden state to predict the next $t+1$ tokens. We then compute a modified cross-entropy loss for each individual head.

![Medusa Loss](/medusa-loss.png)

Note here that $\lambda_k$ is simply a constant taken to the power of $k$. This means that the further the token is from the last predicted token by the original language modelling head, the less it's loss weighs in to the total loss.

This makes sense, since the task of predicting tokens become harder the further we get from the original language modelling head's token. This takes ~5 hours to do on a 7B model with 60k ShareGPT samples.

#### LLM and Heads

A slightly harder way that yields better results is going to be training both the LLM and the individual heads. This results in the new loss equation of

![Medusa Loss v2](/medusa-loss-v2.png)

Note that we have a loss term $\lambda_0$ to balance the loss of the backbone model and that of the new heads. Note that this term is going to small because the medusa heads will have horrible predictions at the start.

During training, the way that they optimize training is by

- Training the base model first on a dataset
- Training medusa heads + base model on dataset while slowly increasing the value of $\lambda_0$ over time

### Dataset

It's important for us to be able to have a dataset that reflects the original dataset the model was trained on. We can do so using a self-distillation method.

> We first take a public seed dataset from a domain similar to the target model; for example, using the ShareGPT [ShareGPT, 2023] dataset for chat models. Then, we simply take the prompts from the dataset and ask the model to reply to the prompts. In order to obtain multi-turn conversation samples, we can sequentially feed the prompts from the seed dataset to the model. Or, for models like Zephyr 7B [Tunstall et al., 2023], which are trained on both roles of the conversation, they have the ability to self-talk, and we can simply feed the first prompt and let the model generate multiple rounds of conversation

In short, we take prompts from a similar dataset, then get the model to generate completions. This then becomes our new dataset. However, if we're doing training using the language heads AND the original model, then we need to factor in a KL divergence for the base model loss.

![KL Divergence](/medusa-kl-divergence.png)

## Sources

Here are some relevant sources which I referred to while writing this page

- [Assisted Generation](https://huggingface.co/blog/assisted-generation)
- [Whisper Speculative Decoding](https://huggingface.co/blog/whisper-speculative-decoding)
