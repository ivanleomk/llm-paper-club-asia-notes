# Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach

[Link](https://arxiv.org/abs/2404.15993)

Authors: Linyu Liu, Yu Pan, Xiaocheng Li, Guanting Chen

Notes by: [@gabchuayz](www.twitter.com/gabchuayz)

# tldr

- Train a regression model (e.g. random forest) to estimate the uncertainty of a LLM's response
- **Input:** LLM's hidden-layer activations of the last token **OR** entropy- or probability-related outputs
- **Output:** Task-specific score between 0 and 1 about the certainty of the answer

# Existing methods

- Based directly on the outputof the LLM: Multiple sampling / add pertubations
- Unsupervised
- Applied to transformers, but not LLMs

# Why do this?

1. UI/UX
2. Improved performance
3. Hallucination detection
4. Auto-Eval (?)

#1 and #4 are my own reflections

# Expressing the problem mathematically

1. An LLM is given an input prompt and randomly generates a response

$$\text{prompt: }\mathbf{x}=(x_1,x_2, ... , x_k) \in \mathbf{\chi}$$
$$\text{response: }\mathbf{y}=(y_1, y_2, ... y_m) \in \mathbf{Y}$$

$$y_j \sim p_\theta(\cdot \mid \mathbf{x}, y_1, y_2, \ldots, y_{j-1})$$

2. We typically use the generated $\mathbf{y}$ for a downstream task (e.g. Q&A, MCQ, translation), and these task have their own scoring function (e.g. Rouge, BLEU):

$$s(\cdot, \cdot) : Y \times Y \rightarrow [0, 1]$$

3. The task of uncertainty estimation for LLMs is learning a function $g$ that predicts the score

$$g(x, y) \approx \mathbb{E}[s(\mathbf{y}, \mathbf{y}_{\text{true}}) \mid \mathbf{x}, \mathbf{y}]$$

# 1. Whitebox LLMs

1. Use the LLM to generate responses for the sample prompts, and construct the raw dataset:

$$D_{raw} = [ x_i, y_i, y_{i, true}, s(y_i, y_{i,true} )] \text{ for } i \in 1...n$$

Note how multiple prompts with different responses would count as different training instances

2. For each sample, extract the features to construct the uncertainty dataset:

$$D_{un} = [ v_{i}, s(y_i, y_{i,true} ) ] \text{ for } i \in 1...n$$

$v_{i}$ is the vector of selected features.

For whitebox LLMs, it is the hidden-layer activations. For the experiments in this paper, they use the activations from the middle and last layer.

> we note that another direct feature for predicting zi is to ask the LLM “how certain it is about the response” and incorporate its response to this question as a feature

3. Train a supervised learning model $\hat{g}$ to predict the score based on the features with the uncertainty dataset.

4. At inference time, generate the responses with the LLM, extract the features, and use the learnt $\hat{g}$ to predict the uncertainty score.

## Feature Selection

For this paper, they use 320 features. 20 from the greybox LLMs - see below.

300 from:

- 100 by LASSO
- 100 by top mutual information
- 100 by top absolute Pearson correlation coefficient

They then train these 320 features with a random forest regressor

# 2. Greybox LLMs

- The following 20 features are used
  ![20 features used for greybox LLMs](/llm_uncertainty_Table_5.png)

# 3. Blackbox LLMs

- Feed the prompt into the whitebox model to get the hidden-layer activations, and extract $v_{i}$

- In this paper, they treat Llama 7B and Gemma 7B as a black box, and use the other as the whitebox for uncertainty estimation.

# Numerical Results

3 Tasks:

1. Q&A - Rouge-1
2. MCQ - Yes/No accuracy
3. Translation - BLEU

Here are the results for Q&A and Translation
![Results](/llm_uncertainty_table_1.png)

![Example](/llm_uncertainty_figure_11.png)

# Choice of Layer for Hidden Activations

![Results](/llm_uncertainty_figure_1.png)

> This may come from the fact that the last layer focuses more on the generation of the next token instead of summarizing information of the whole sentence, as has been discussed by Azaria and Mitchell (2023).

# Preferences and Evals

![competition](/llm_uncertainty_competition.png)
