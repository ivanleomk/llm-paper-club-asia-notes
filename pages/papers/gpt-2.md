# Language Models are Unsupervised Multitask Learners

## Motivation

The problem that GPT-2 aims to solve is to demonstrate that language models,
given **_large_** enough capacity in terms of parameters, and **_large_** enough
**_unlabeled and high-quality_** text data, can solve specialized natural
language processing tasks such as question answering, translation, and
summarization, in a
[**_zero-shot_**](https://en.wikipedia.org/wiki/Zero-shot_learning) manner -
without the need for task-specific architectures or supervised fine-tuning.

The emphasis on the _large and high-quality_ text data cannot be understated as
the authors are hinging on the fact that the dataset is so **_diverse_**, and
therefore _bound_ to have examples of the _specialized_ tasks that the model can
learn from.

For example, if we are looking at translation tasks, then the data is bound to
have somewhat **sequential** and **natural occuring translation text** such as:

```python
The translation of the french sentence 'As-tu aller au cine ́ma?' to english is 'Did you go to the cinema?'.
```

The model can learn from such examples and generalize to perform well on the
translation task via the
[**_autoregressive_**](https://en.wikipedia.org/wiki/Autoregressive_model),
[**_self-supervised_**](https://en.wikipedia.org/wiki/Self-supervised_learning)
learning paradigm without the need for supervised fine-tuning.

## From GPT-1 to GPT-2

In
[**Natural Language Understanding**](https://en.wikipedia.org/wiki/Natural-language_understanding)
(NLU), there are a wide range of tasks, such as textual entailment, question
answering, semantic similarity assessment, and document classification. These
tasks are inherently labeled, but given the scarcity of such data, it makes
[discriminative](https://en.wikipedia.org/wiki/Discriminative_model) models such
as Bidirectional Long Short-Term Memory (Bi-LSTM) underperform[^9], leading to poor performance on these tasks.

In the GPT-1 paper
[_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
the authors demonstrated that _generative pre-training_ of a language model on a
diverse corpus of _unlabeled_ text, followed by _discriminative fine-tuning_ on
each specific task, can overcome the constraints of the small amount of
annotated data for these specific tasks. The process is collectively termed as
[semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)
and the goal is to learn an **_universal representation_** of the natural
language space that can be used across a wide range of tasks.

The pretraining objective is to predict the next token in a sequence, in an
**_autoregressive_** manner, given the previous tokens. The pretrained model,
often known as the **_foundational model_** (or _backbone_), serves as a base
from which specialized capabilities can be added through _fine-tuning_ on
specific tasks. In the fine-tuning phase, task-specific adaptations are
necessary: the input format must be adjusted to align with the particular
requirements of the task at hand, and the model's final layer—or "head"—needs to
be replaced to accommodate the task's specific class structure. The author
showed that this approach yielded state-of-the-art results on a wide range of
NLU tasks.

Notwithstanding the success of this approach, the same set of authors came up
with a new paper in the following year, titled
[_Language Models are Unsupervised Multitask Learners_](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
where they introduced a new model, _GPT-2_, that was larger in model capacity,
and trained on a much larger unlabeled corpus, **WebText**. However, the key
innovation was to void the supervised fine-tuning step, and instead, they
demonstrated that GPT-2 could be used directly on a wide range of NLU tasks
directly, with what they termed as the _zero-shot transfer_. The motivation is
that the authors think that foundational language models should be competent
generalists, rather than narrowly experts[^10]. They call
for the need to shift the language model paradigm to one that is generic enough
to handle NLU tasks without the need to curate specific training data for each
specific task.

## GPT-2 Paper Key Ideas

In this section, we would review the key ideas from the GPT-2 paper.

### Abstract Overview

Below are the key ideas from the abstract of the GPT-2 paper:

- All **previous pretrained language models** necessitated a secondary stage
  of **_supervised fine-tuning_** to tailor them to specific downstream tasks.
- The authors showcased that, given sufficient **_model capacity_** and
  **_data_**, language models can be adeptly adjusted to a broad spectrum of
  tasks **_without the need for task-specific architectural modifications_**.
- When tasked with a question-answering challenge, specifically conditioned on
  a document and questions using the
  [CoQA dataset](https://huggingface.co/datasets/stanfordnlp/coqa) — comprised
  of over 127,700 training examples — the model demonstrates the capability to
  **_match or surpass the performance of three baseline models_**.
- An emphasis is placed on the **_model's capacity_** as being integral to the
  success of **_zero-shot transfer_**. It's highlighted that the model's
  performance escalates in a **_log-linear fashion_** relative to the number
  of parameters, signifying that as the model's capacity increases
  _logarithmically_, its **performance** improves _linearly_.

### Introduction

In this section, we would discuss the key ideas from the introduction of the
GPT-2 paper.

#### Key 1. Competent Generalists over Narrow Experts (1)

- The authors cited other works that have demonstrated significant success of
  machine learning systems through a **_combination_** of **_large-scale
  data_**, **_high model capacity_**, along with **_supervised fine-tuning_**.
- However, such systems, termed as "**_narrow experts_**," are fragile, as
  they are highly dependent on the specific training regime and task. A slight
  **_perturbation_** to the input distribution can cause the model to perform
  poorly.
- The authors then expressed the desire for "**_competent generalists_**" that
  can perform well across a wide range of tasks **_without_** the need for
  task-specific architectures or supervised fine-tuning.

#### Key 2. IID Assumption Fails in Real World (2, 3)

- The overarching goal in machine learning is to **_generalize to unseen data
  points_**. To streamline the modeling of machine learning objectives, it's
  commonly assumed that the training and test data are drawn from the same
  distribution, a concept known as the
  [**_Independent and Identically Distributed (i.i.d.)_**](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
  assumption.
  - As an aside, the i.i.d. assumption is foundational in statistical
    modeling because it simplifies the process significantly. For example,
    it allows us to
    [**_express joint probability distributions_**](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
    as the product of marginal distributions.
  - Furthermore, evaluation techniques such as **_resampling_** and
    **_cross-validation_** with a holdout set rely on the assumption that
    the training and test data are drawn from the same distribution.
- However, as the authors highlighted, the i.i.d. assumption fails in the real
  world. The distribution of the test data is often different from the
  training data, and the model's performance degrades significantly when the
  test data distribution is different from the training data distribution.
- They attribute this to the prevalence of **single** task training on
  **single** domain datasets, which limits the model's ability to generalize
  across diverse conditions and tasks.

**Further Readings:**

- [On the importance of the i.i.d. assumption in statistical learning](https://stats.stackexchange.com/questions/213464/on-the-importance-of-the-i-i-d-assumption-in-statistical-learning)
- [Independent and identically distributed random variables - Wikipedia](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
- [Independence and Identically Distributed (IID) - GAO Hongnan](https://gao-hongnan.github.io/gaohn-galaxy/probability_theory/08_estimation_theory/maximum_likelihood_estimation/concept.html#independence-and-identically-distributed-iid)

#### Key 3. Multi-Task Learning is Nacent (4)

- The author then underscored that **_multi-task learning_** represents a
  **_promising framework_**. By training a single model on **_multiple tasks
  simultaneously_**, the model is enabled to leverage **_generalizable latent
  space embeddings and representations_** to excel across various tasks.
- It was further pointed out that recent work in the field utilizes, for
  example, **_10 (dataset, objective) pairs_**[^6] to
  train a singular model (an approach known as
  [**_meta-learning_**](<https://en.wikipedia.org/wiki/Meta-learning_(computer_science)>)).
  This implies that:
  - Each dataset and its corresponding objective are unique.
  - For instance, one dataset might focus on **_sentiment data_**, with the
    goal of **_predicting sentence sentiment_**, whereas another dataset
    might concentrate on **_named entity recognition_**, aiming to
    **_identify named entities within a sentence_**.
- The **_challenge_** then circles back to the **_compilation, curation, and
  annotation_** of these datasets and objectives to ensure the model's
  generalizability. Essentially, this dilemma mirrors the initial issue of
  **_single-task training on single-domain datasets_**. The implication is
  that training a **_multi-task model_** might require an equivalent volume of
  curated data as training several **_single-task models_**. Furthermore,
  scalability becomes a concern when the focus is limited to merely **_10
  (dataset, objective) pairs_**.

#### Key 4. From Word Embeddings to Contextual Embeddings (5,6)

- Initially, **_word embeddings_** such as **Word2Vec** and **GloVe**
  revolutionized the representation of words by mapping them into dense,
  fixed-dimensional vectors within a continuous $D$ dimensional space, hinging
  on the fact that words occuring in similar contexts/documents are similar
  semantically. These vectors were then used as input to a model to perform a
  specific task.
- The next advancement is capturing more _contextual information_ by using
  **_contextual embeddings_**, where the word embeddings are **conditioned**
  on the entire context of the sentence.
  [**Recurrent Neural Networks**](https://en.wikipedia.org/wiki/Recurrent_neural_network)
  (RNNs) is one example and the context embeddings can be "transferred" to
  other downstream tasks.

  Specifically, **unidirectional RNNs** are adept at assimilating context from
  preceding elements, whereas **bidirectional RNNs** excel in integrating
  context from both preceding and succeeding elements. Nonetheless, both
  strategies grapple with challenges in encoding long-range dependencies.

  Moreover, RNNs are notoriously plagued by the
  [**_gradient vanishing problem_**](https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen),
  which means that the model is **biased** by the most _recent_ tokens in the
  sequence, and the model’s performance **degrades** as the _sequence length_
  **increases**.

- **_Self-attention mechanisms_**, foundational to the **Transformer
  architecture**, mark a paradigm shift by enabling each token to "attend" to
  every other token within a sequence concurrently.

  - This allows the model to capture long-range dependencies and is the
    basis for the Transformer architecture. Consequently, self-attention is
    non-sequential by design and operates over a _set_ of tokens, and not a
    _sequence_ of tokens. This calls for the need to introduce positional
    encodings to the input embeddings to capture the sequential nature of
    the tokens.

  - This advancement transcends the limitations of static word embeddings.
    Now, given two sentences, _I went to the river bank_ versus _i went to
    the bank to withdraw money_, the word "bank" in the first sentence is
    semantically different from the word "bank" in the second sentence. The
    contextual embeddings can capture this difference.

- The authors then went on to mention that the above methods would still
  require supervised fine-tuning to adapt to a specific task.

  If there are minimal or no supervised data is available, there are other
  lines of work using language model to handle it - commonsense reasoning
  (Schwartz et al., 2017) and sentiment analysis (Radford et al., 2017).

**Further Readings:**

- [Why does the transformer do better than RNN and LSTM in long-range context dependencies?](https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen)
- [How Transformer is Bidirectional - Machine Learning](https://stackoverflow.com/questions/55158554/how-transformer-is-bidirectional-machine-learning)

#### Key 5. Zero Shot Learning and Zero Shot Transfer (7)

- Building upon the foundational concepts introduced previously, the authors
  explore the utilization of **_general methods of transfer_** to illustrate
  how language models can adeptly execute downstream tasks in a **_zero-shot
  manner_**, without necessitating any modifications to parameters or
  architecture.

- **_Zero-shot learning (ZSL)_** is characterized by a model's capability to
  accurately execute tasks or recognize categories that it was not explicitly
  trained to handle. The crux of ZSL lies in its ability to **_generalize from
  known to unknown_** classes or tasks by harnessing side information or
  semantic relationships.

  - For example, a model trained to recognize on a set of animals (including
    horses) but not on zebra, should be able to recognize a zebra as
    something close to horse, given the semantic relationship between the
    two animals.

- **_Zero-shot transfer_**, often discussed within the context of **transfer
  learning**, involves applying a model trained on one set of tasks or domains
  to a completely new task or domain without any additional training. Here,
  the focus is on the transferability of learned features or knowledge across
  different but related tasks or domains. Zero-shot transfer extends the
  concept of transfer learning by not requiring any examples from the target
  domain during training, relying instead on the model's ability to generalize
  across different contexts based on its pre-existing knowledge.

**Further Readings:**

- [Zero-shot learning - Wikipedia](https://en.wikipedia.org/wiki/Zero-shot_learning)
- [What is the difference between one-shot learning, transfer learning, and fine-tuning? - AI Stack Exchange](https://ai.stackexchange.com/questions/21719/what-is-the-difference-between-one-shot-learning-transfer-learning-and-fine-tun)
- [Zero-Shot Learning in Modern NLP - Joe Davison](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
- [Zero-Shot Learning Through Cross-Modal Transfer - arXiv](https://arxiv.org/abs/1301.3666)
- [Zero shot learning available labels in testing set - AI Stack Exchange](https://ai.stackexchange.com/questions/23527/zero-shot-learning-available-labels-in-testing-set)
- [Zero-Shot Learning: Can You Classify an Object Without Seeing It Before?](https://www.theaidream.com/post/zero-shot-learning-can-you-classify-an-object-without-seeing-it-before)
- [A Survey of Zero-Shot Learning: Settings, Methods, and Applications](https://dl.acm.org/doi/10.1145/3293318)

### Section 2. Approach

In this section, we would discuss the key ideas from the approach section of the
GPT-2 paper.

#### Key 1. Modeling Language Models over Joint Probability Distributions (1)

Language models strive to approximate the complex and inherently unknown
distribution of the natural language space, denoted as $\mathcal{D}$. In
contrast to supervised learning, which explicitly separates inputs
($\mathcal{X}$) from labels ($\mathcal{Y}$), unsupervised learning —
particularly when employing self-supervision as seen in language modeling —
blurs this distinction. Here, $\mathcal{Y}$ is conceptually a shifted
counterpart of $\mathcal{X}$, facilitating a unified approach where
$\mathcal{D}$ can be modeled exclusively over the space of $\mathcal{X}$. This
scenario allows us to frame $\mathcal{D}$ as a probability distribution across
sequences of tokens within $\mathcal{X}$, parameterized by
$\boldsymbol{\Theta}$.

In this context, the essence of language modeling is to characterize the
**_joint probability distribution_** of sequences
$\mathbf{x} = (x_1, x_2, \ldots, x_T)$ within $\mathcal{X}$. The goal is to
maximize the likelihood of observing these sequences in a corpus $\mathcal{S}$,
denoted as $\hat{\mathcal{L}}(\mathcal{S} ; \hat{\boldsymbol{\Theta}})$, where
$\hat{\boldsymbol{\Theta}}$ represents the estimated parameter space that
approximates the true parameter space $\boldsymbol{\Theta}$.

#### Key 2. Decompose Joint Distributions as Conditional Distributions via Chain Rule (2)

The joint probability of a sequence in natural language, **inherently ordered**[^10], can be factorized into the product of conditional
probabilities of each token in the sequence using the
[**chain rule of probability**](<https://en.wikipedia.org/wiki/Chain_rule_(probability)>).
This approach not only enables **_tractable sampling_** from and
**_estimation_** of the distribution
$\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta})$ but also facilitates modeling
conditionals in forms such as
$\mathbb{P}(x_{t-k} \ldots x_t \mid x_1 \ldots x_{t-k-1} ; \boldsymbol{\Theta})$
[^10]. Given a corpus $\mathcal{S}$ with $N$ sequences,
the likelihood function
$\hat{\mathcal{L}}(\mathcal{S} ; \hat{\boldsymbol{\Theta}})$ represents the
likelihood of observing these sequences. The ultimate objective is to maximize
this likelihood, effectively _approximating_ the joint probability distribution
through conditional probability distributions.

#### Key 3. Conditional on Task (3)

In the GPT-2 paper, _Language Models are Unsupervised Multitask Learners_, the
authors introduced the concept of _conditional on task_ where the GPT model
$\mathcal{G}$ theoretically should not only learn the conditional probability
distribution:

$$
\mathbb{P}(x_t \mid x_{\< t} ; \boldsymbol{\Theta})
$$

but also learn
the conditional probability distribution:

$$
\mathbb{P}(x_t \mid x_{\< t} ; \boldsymbol{\Theta}, \mathcal{T})
$$

where
$\mathcal{T}$ is the task that the model should implicitly learn[^10]. This is a powerful concept because if such a
hypothesis is correct, then the GPT model $\mathcal{G}$ can indeed be a
multi-task learner, and can be used directly on a wide range of NLU tasks
without the need for supervised fine-tuning for downstream domain-specific
tasks.

In practice, the authors mentioned that task conditioning is often implemented
at an architectural level, via task specific encoder and decoder in the paper
[_One Model To Learn Them All_](https://arxiv.org/abs/1706.05137)[^7], for instance, or at an algorithmic level, such as the
inner and outer loop optimization framework, as seen in the paper
[_Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_](https://arxiv.org/abs/1703.03400)[^8].

However, the authors further mentioned that without task-specific architectural
changes, one can leverage the sequential nature of the natural language space
where we can construct a tasks, inputs and outputs all as a sequence of symbols[^10]. For example, a translation task can be formulated
as a sequence of symbols via
`(translate to french, english sequence, french sequence)`, where the model can
now learn to also condition on the task `(translate to french)` in addition to
the sequence of tokens. The paper _The Natural Language Decathlon: Multitask
Learning as Question Answering_ exemplifies this concept with their model
**Multitask Question Answering Network (MQAN)**, where a single model is trained
to perform many diverse natural language processing tasks simultaneously.

#### Key 4. Optimizing Unsupervised is the same as Optimizing Supervised (4)

The GPT-2 paper _Language Models are Unsupervised Multitask Learners_
demonstrated that they want to do away with the supervised fine-tuning phase via
an interesting hypothesis, that **optimizing the unsupervised objective is the
same as optimizing the supervised objective** because the _global minimum_ of
the unsupervised objective is the same as the _global minimum_ of the supervised
objective[^10].

#### Key 5. Large Language Models has Capacity to Infer and Generalize (5)

In what follows, the author added that the internet contains a vast amount of
information that is passively available without the need for interactive
communication. The example that I provided on the french-to-english translation
would bound to exist naturally in the internet. They speculate that if the
language model is **large** enough in terms of **capacity**, then it should be
able to learn to perform the tasks demonstrated in natural language sequences in
order to better predict them, regardless of their method of procurement[^10].

In the figure below, we can see examples of naturally occurring demonstrations
of English to French and French to English translation found throughout the
WebText training set.

![Examples of naturally occurring demonstrations of English to French and French to English translation found throughout the WebText training set.](../../public/gpt-2-table-1.png)

### 2.1. Training Dataset

#### Key 1. Rejection of CommonCrawl (1,2)

- Prior research often focused on training language models on **_single-domain
  datasets_**, which relates to the concept of models becoming **_narrow
  experts_**.
- To cultivate **_competent generalists_**, the authors contend that models
  need exposure to a **_diverse array_** of tasks and domains.
- **_CommonCrawl_**, housing an expansive collection of web scrapes
  (essentially capturing the entirety of the internet), is recognized for its
  diversity.
- Nevertheless, CommonCrawl was ultimately **_rejected_** by the authors due
  to **_significant data quality issues_**.

#### Key 2. Construction of WebText Dataset

- The authors sought to compile a web scrape prioritizing **_document quality
  over quantity_**.
- To attain a certain level of document quality without the exorbitant costs
  of manual curation, the authors employed a strategy of **_indirect human
  curation_**. This involved scraping all **_outbound links from Reddit_**
  that garnered a minimum of **_3 karma_**. Karma, in this scenario, acts as a
  heuristic for content deemed interesting, educational, or entertaining by
  the Reddit community.
  - **_Outbound links_** refer to instances where a Reddit post links out to
    external websites; the authors included the content from these external
    sites in their dataset, contingent on the originating post receiving at
    least 3 karma.
- The resulting dataset, dubbed **_WebText_**, comprises text from
  approximately **_45 million links_**.
- Subsequent preprocessing efforts, including **_de-duplication,
  heuristic-based cleaning_**, and the **_exclusion of Wikipedia links_**,
  resulted in a dataset spanning about **_40GB of text (8 million
  documents)_**.
- The snapshot of the dataset is **_December 2017_**.
- Wikipedia's exclusion was deliberate, stemming from the authors' intention
  to minimize overlap with training sources prevalent in other studies. This
  decision aimed to facilitate more "authentic" **_evaluation/testing_**
  scenarios for their model by reducing data leakage.

### 2.2. Input Representation

#### Key 1. Byte Pair Encoding (BPE) (1,2,3)

- Traditional tokenization methods often involve steps such as
  **_lower-casing_**, **_punctuation stripping_**, and **_splitting on
  whitespace_**. Additionally, these methods might encode out-of-vocabulary
  words using a special token to enable the model to handle unseen words
  during evaluation or testing phases. For instance, language models (LMs) may
  struggle with interpreting emojis due to such constraints.
- These conventional approaches can inadvertently restrict the natural
  language input space $\mathcal{X}$, consequently limiting the model space
  $\mathcal{H}$. This limitation stems from the fact that the scope of
  $\mathcal{H}$ is inherently dependent on the comprehensiveness of
  $\mathcal{X}$ as we can see
  $\mathcal{H} = \mathcal{H}(\mathcal{X} ; \boldsymbol{\Theta})$, which means
  that the model space $\mathcal{H}$ is a function of the input space
  $\mathcal{X}$ and the parameter space $\boldsymbol{\Theta}$.
- To resolve this, the idea of **_byte-level encoding_** can be used - since
  you theoretically can encode any character in the world in **_UTF-8
  encoding_**.
- However, the limitation is current byte-level language models tend to
  perform poorly on word level tasks.
- The authors then introduced the BPE algorithm (is "byte-level" because it
  operates on UTF-8 encoded strings) where they striked a balance between
  character-level and word-level tokenization.
- So in summary, BPE is the **tokenizer** used to encode the input text into a
  sequence of tokens - which form the input representation to the model.

Further Readings:

- [minBPE GitHub repository by Andrej Karpathy](https://github.com/karpathy/minbpe)
- [Byte Pair Encoding on Hugging Face's NLP Course](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)

### 2.3. Model

The GPT-2 architecture is a **_transformer_**-based model, and as the name
suggests, it is a continuation of the GPT-1 model with some minor modifications.

#### Key 1. GPT-2 is a Continuation of GPT-1 with Self-Attention Mechanisms (1)

- GPT-2 utilizes a **Transformer** architecture[^1]
  as its backbone, which is distinguished by **_self-attention mechanisms_**.
  This architecture empowers the model to capture complex dependencies and
  relationships within the data.

#### Key 2. Modifications from GPT-1 and Model Stability (1)

- Modifications from GPT-1 include:

  - **Layer normalization** is repositioned to the **_input_** of each
    sub-block, mirroring a **_pre-activation residual network_**. This
    modification is believed to offer training stability and model
    performance. By normalizing the inputs to each sub-block, it is
    conjectured to alleviate issues tied to **_internal covariate shift_**,
    thus aiding in smoother and potentially faster training.
  - GPT-2 introduces an **_additional layer normalization step_** that is
    executed **_after the final self-attention block_** within the model.
    This additional normalization step can help ensure that the outputs of
    the transformer layers are normalized before being passed to subsequent
    layers or used in further processing, further contributing to model
    stability.
  - The GPT-2 paper introduces a modification to the standard weight
    initialization for the model's residual layers. Specifically, the
    weights are scaled by a factor of
    $\frac{1}{\sqrt{N_{\text{decoderblocks}}}}$, where
    $N_{\text{decoderblocks}}$ represents the number of blocks (or layers)
    in the Transformer's decoder.

    The rationale, as quoted from the paper: _"A modified initialization
    which accounts for the accumulation on the residual path with model
    depth is used"_[^10], is to ensure that the
    variance of the input to the block is the same as the variance of the
    block's output. This is to ensure that the signal is neither amplified
    nor diminished as it passes through the block. As the model depth
    increases, the activations get added/acculumated, and hence the scaling
    factor is $\frac{1}{\sqrt{N_{\text{decoderblocks}}}}$, to scale it
    down.

  - Clearly, we can see the empahsis on model stability. In training large
    language models, **numerical stability** is paramount; the cost of
    training is significantly high, with every loss and gradient spike that
    fails to recover necessitating a return to a previous checkpoint,
    resulting in substantial GPU hours and potentially tens of thousands of
    dollars wasted.
  - The model's **vocabulary** is expanded to 50,257 tokens.
  - The context window size is increased from 512 to 1024 tokens, enhancing
    the model's ability to maintain coherence over longer text spans.
  - A larger batch size of 512, GPT-2 benefits from more stable and
    effective gradient estimates during training, contributing to improved
    learning outcomes.

#### GPT-2 Variants

To this end, we encapsulate some key parameters in
the table below, which provides specifications for
several GPT-2 variants, distinguished by their scale.

| Parameters | Layers | d_model | H   | d_ff | Activation | Vocabulary Size | Context Window |
| ---------- | ------ | ------- | --- | ---- | ---------- | --------------- | -------------- |
| 117M       | 12     | 768     | 12  | 3072 | GELU       | 50,257          | 1024           |
| 345M       | 24     | 1024    | 16  | 4096 | GELU       | 50,257          | 1024           |
| 762M       | 36     | 1280    | 20  | 5120 | GELU       | 50,257          | 1024           |
| 1542M      | 48     | 1600    | 25  | 6400 | GELU       | 50,257          | 1024           |

See
[The Implementation of Generative Pre-trained Transformers (GPT)](https://www.gaohongnan.com/transformer/decoder/implementation.html)
for a more comprehensive walkthrough of the GPT-2 model architecture, annotated
with code.

[^1]:
    A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser, and I. Polosukhin.
    ["Attention is all you need"](https://arxiv.org/abs/1706.03762). In Advances
    in Neural Information Processing Systems, pp. 5998–6008, 2017.

[^2]:
    I. Loshchilov and F. Hutter,
    ["Decoupled weight decay regularization"](https://arxiv.org/abs/1711.05101),
    arXiv preprint arXiv:1711.05101, [Submitted on 14 Nov 2017 (v1), last
    revised 4 Jan 2019 (this version, v3)].

[^3]:
    D. P. Kingma and J. Ba,
    ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980),
    arXiv preprint arXiv:1412.6980, [Submitted on 22 Dec 2014 (v1), last revised
    30 Jan 2017 (this version, v9)].

[^4]:
    L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han,
    ["On the Variance of the Adaptive Learning Rate and Beyond"](https://arxiv.org/abs/1908.03265),
    arXiv preprint arXiv:1908.03265, [Submitted on 8 Aug 2019 (v1), last revised
    26 Oct 2021 (this version, v4)].

[^5]:
    A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola,
    ["Chapter 9. Recurrent Neural Networks"](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
    in Dive into Deep Learning, Cambridge University Press, 2023.

[^6]: Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. ["The natural language decathlon: multitask learning as question answering"](https://arxiv.org/abs/1806.08730). 2018. arXiv:1806.08730.
[^7]: Lukasz Kaiser, Aidan N. Gomez, Noam Shazeer, Ashish Vaswani, Niki Parmar, Llion Jones, and Jakob Uszkoreit. ["One model to learn them all"](https://arxiv.org/abs/1706.05137). 2017. arXiv:1706.05137.
[^8]: Chelsea Finn, Pieter Abbeel, and Sergey Levine. ["Model-agnostic meta-learning for fast adaptation of deep networks"](https://arxiv.org/abs/1703.03400). 2017.
[^9]:
    A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever,
    ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

[^10]:
    A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever,
    ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
