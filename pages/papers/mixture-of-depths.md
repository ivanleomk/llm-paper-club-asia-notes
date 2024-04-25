# Mixture Of Depths

> Original Paper: https://arxiv.org/abs/2404.02258

## Introduction

The Mixture Of Depths paper introduces a way for us to dynamically route tokens through a transformer layer by adapting an existing MOE routing mechanism called Expert Choice Routing to the entire transformer layer itself. By integrating these new MOD layers in an alternating fashion with normal transformer layers, they speed up training convergence by 2x among a host of other improvements.

## Why Care?

Transformers are great but they're difficult to scale to larger sequence length. This is because of the nature of computation of attention which scales quadratically with the length of the sequence since every token needs to know how much to attend to each individual output.

### Attention

Attention does 2 things

1. It projects each token's representation into a $d_dim$ embedding
2. It calculates a weightage for every token's new representation for each token position. The final hidden state out of any attention block is therefore a new weighted sum of these projected vectors.
3. We then project this out to $d_output$

$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_k}})V
$$

We can see this here in a visualisation of the GPT-2 Transformer Block

![GPT-2 Attention](/attention_process.png)

Note that Q,K and V do not have the be the same matrix. Q and K will have the same dimensions `d_k` and V will have the dimensions of `d_v` . In the paper, the model has an embedding dimension `d_model` of `512` where `d_k = d_v = d_model`. Multi-Headed Attention is just simply going to be the same operation but performed on a smaller slice of the input dim.

![MHA Implementation](/MHA-attention-formula.png)

Multi-Head attention takes advantage of the Scaled Dot Product Attention mechanism. While the computation complexity is the same as that of a Single-head attention block, the authors find that it performs better.

This is because it allows the model to jointly attend to information from different representation subspaces at different positions. In the paper, they used `8` different heads.

![MHA-Attention](/mha-attention-visualisation.png)

Note that at the end when we derive our H matrix, it has dimensions $(n, d_v)$ but with the help of $w^o$, we’re able to scale our output back to $d_{model}$.

## Implementation

Intuitively, MOD layers are an example of learned routing mechanisms. [Deepseek MOE](/papers/deepseek-moe) employs a Token Choice Routing mechanism where each token chooses the expert it wants to be routed to. There have been other papers which have implemented Expert Choice routing mechanism where **each expert chooses the top-k tokens it wants to process**. This means that tokens that aren't chosen essentially just have the identity function applied upon them.

![Mixture Of Depths Routing](/mod-routing.png)

Each token is processed by a single router to produce a scalar weight. The top-k weights are then used to choose the token identies.

![Routing Mechanism](/mod-router.png)

in this specific equation

- $f_i$ represents the entire mod block
- $\tilde{X}$ represents the chosen set of tokens
- $X_i^{l+1}$ is the input for the $l+1$-th block
- $P_{\beta} (R_{l})$ represents the $\beta$ percentile of the router's outputs

Since the number of tokens chosen through this operation is less than the cardinality of the actual number of tokens.

## Results

Mixture-of-Depths transformers empirically demonstrate that one can improve on isoFLOP-optimal baseline performance with models that use fewer FLOPs per forward pass. This means that—for a given training FLOP budget—we can train models that are both faster and better performing than their baseline counterparts.

![Loss Curves](/loss-curves.png)

A few takeaways

- Learned Routing is Important : MoD transformers that use stochastic routing (implemented using a top-𝑘 operation on router weights sampled from a Gaussian distribution) perform drastically worse than both the baseline and normal MoD transformer
- aggressive capacity reduction was best (gradual improvements were observed when reducing the capacity down to 12.5% of the total sequence, corresponding to 87.5% of tokens routing around blocks
- MoD transformers had memory savings relative to equivalently sized baseline models at larger sizes, with some variants requiring fewer total device

![Routing Weights](/routing-weights.png)

- Some tokens appear to engage each block along the transformer’s depth, while others decide to route around blocks whenever possible. Preliminary analyses suggest that the tokens that engage with blocks more frequently are correlated with output predictions that have higher entropy

## Mixture Of Experts and Mixture of Depths

They implemented two variants

1. Staged MoDE : This routes tokens around or towards blocks prior to the self-attention step
2. Integrated MoDE : This implements MoD routing by integrating a "no-op" expert among the conventional MLP

## Problems

There are 3 main problems with a MOD based layer

1. Batching : It's difficult to batch operations where the routing of an input is going to be dynamically computed on the fly. This makes it difficult for us to batch operations in advance.
2. Causality : The causal attention mask in a transformer layer, specifically in the decoder block, plays a crucial role in maintaining the autoregressive property of the model. As a result, our transformer now uses future information to make a prediction on token level
3. Calculation of Attention: We assume that our MOD layer learns how to attend to the right token through its training set but this might not apply to future outputs.

### Causality

They try to fix casuality by two separate methods

1. **Change the distribution of the probs** They introduce a simple cross-entropy loss on the outputs of the router ( We use a binary cross-entropy loss wherein the router’s outputs provide the logits, and the top-𝑘 selections of these logits provide the targets (i.e. 1 if a token was among the top-𝑘, and 0 if not) )
2. **Add a separate predictor** The second method introduces a small auxiliary MLP predictor (akin to a second router) that receives the same inputs as the router

The goal here is to be able to then stick this in front of your router so we can sample autoregressively in front of a block ( before we apply the router ). Personally not super sure if this solved the issue but the classifier itself manages to achieve very high accuracy

![top-k-accuracy](/predictor-vs-top-k.png)

Learned routing mechanisms are sometimes non-causal; that is, information about the future is used to determine a given token’s routing decision. This is generally true for top-k routing mechanisms, which are useful because they forego the need for auxiliary balancing losses.

## Useful Resources

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Gabriel Mongaras's Walkthrough on the Paper](https://www.youtube.com/watch?v=M8QkiuSto6I)
- [Medium Article's tl;dr on the paper](https://dev.to/mikeyoung44/mixture-of-depths-dynamically-allocating-compute-in-transformer-based-language-models-moe)
- [Hugging Face Implementation of a MOD Paper](https://huggingface.co/blog/joey00072/mixture-of-depth-is-vibe)
