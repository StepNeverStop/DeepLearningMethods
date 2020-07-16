# Multi-Head Attention
![](./img/multi-head-attention_l1A3G7a.png)

**Multi-head Attention** is a module for attention mechanisms which runs through an attention mechanism several times in parallel. The independent attention outputs are then concatenated and linearly transformed into the expected dimension. Intuitively, multiple attention heads allows for attending to parts of the sequence differently (e.g. longer-term dependencies versus shorter-term dependencies). 

$$ \text{MultiHead}\left(\textbf{Q}, \textbf{K}, \textbf{V}\right) = \left[\text{head}_{1},\dots,\text{head}_{h}\right]\textbf{W}_{0}$$

$$\text{where} \text{ head}_{i} = \text{Attention} \left(\textbf{Q}\textbf{W}_{i}^{Q}, \textbf{K}\textbf{W}_{i}^{K}, \textbf{V}\textbf{W}_{i}^{V} \right) $$

Above $\textbf{W}$ are all learnable parameter matrices.

Note that scaled dot-product attention is most commonly used in this module, although in principle it can be swapped out for other types of attention mechanism.

Source: [Lilian Weng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms) 

# Scaled Dot-Product Attention
![](./img/SCALDE.png)

**Scaled dot-product attention** is an attention mechanism where the dot products are scaled down by $\sqrt{d_k}$. Formally:

$$ {\text{Attention}}(Q, K, V) = \text{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V $$

If we assume that $q$ and $k$ are $d_k$-dimensional vectors whose components are independent random variables with mean $0$ and variance $1$, then their dot product, $q \cdot k = \sum_{i=1}^{d_k} u_iv_i$, has mean $0$ and variance $d_k$.  Since we would prefer these values to have variance $1$, we divide by $\sqrt{d_k}$.

# Additive Attention
![](./img/Screen_Shot_2020-05-24_at_7.58.36_PM.png)

**Additive Attention** uses a one-hidden layer feed-forward network to calculate the attention alignment:

$$f_{att}\left(\textbf{h}_{i}, \textbf{s}_{j}\right) = v_{a}^{T}\tanh\left(\textbf{W}_{a}\left[\textbf{h}_{i};\textbf{s}_{j}\right]\right)$$

where $\textbf{v}_{a}$ and $\textbf{W}_{a}$ are learned attention parameters.

# Dot-Product Attention
![](./img/Screen_Shot_2020-05-25_at_12.32.09_PM_yYfmHYZ.png)

**Dot-Product Attention** is an attention mechanism where the alignment score function is calculated as: 

$$f_{att}\left(\textbf{h}_{i}, \textbf{s}_{j}\right) = h_{i}^{T}s_{j}$$

It is equivalent to [multiplicative attention](https://paperswithcode.com/method/multiplicative-attention) (without a trainable weight matrix, assuming this is instead an identity matrix).

# SAGAN Self-Attention Module
![](./img/Screen_Shot_2020-05-25_at_1.36.58_PM_79d4mU6.png)

The **SAGAN Self-Attention Module** is a self-attention module used in the Self-Attention GAN architecture for image synthesis.

The image features from the previous hidden layer $\textbf{x} \in \mathbb{R}^{C\text{x}N}$ are first transformed into two feature spaces $\textbf{f}$, $\textbf{g}$ to calculate the attention, where $\textbf{f(x) = W}_{\textbf{f}}{\textbf{x}}$, $\textbf{g}(\textbf{x})=\textbf{W}_{\textbf{g}}\textbf{x}$:

$$\beta_{j, i} = \frac{\exp\left(s_{ij}\right)}{\sum^{N}_{i=1}\exp\left(s_{ij}\right)} $$

$$ \text{where } s_{ij} = \textbf{f}(\textbf{x}_{i})^{T}\textbf{g}(\textbf{x}_{i}) $$

and $\beta_{j, i}$ indicates the extent to which the model attends to the $i$th location when synthesizing the $j$th region. Here, $C$ is the number of channels and $N$ is the number of feature
locations of features from the previous hidden layer. The output of the attention layer is $\textbf{o} = \left(\textbf{o}_{\textbf{1}}, \textbf{o}_{\textbf{2}}, \ldots, \textbf{o}_{\textbf{j}} , \ldots, \textbf{o}_{\textbf{N}}\right) \in \mathbb{R}^{C\text{x}N}$ , where,

$$ \textbf{o}_{\textbf{j}} = \textbf{v}\left(\sum^{N}_{i=1}\beta_{j, i}\textbf{h}\left(\textbf{x}_{\textbf{i}}\right)\right) $$

$$ \textbf{h}\left(\textbf{x}_{\textbf{i}}\right) = \textbf{W}_{\textbf{h}}\textbf{x}_{\textbf{i}} $$

$$ \textbf{v}\left(\textbf{x}_{\textbf{i}}\right) = \textbf{W}_{\textbf{v}}\textbf{x}_{\textbf{i}} $$

In the above formulation, $\textbf{W}_{\textbf{g}} \in \mathbb{R}^{\bar{C}\text{x}C}$, $W_{f} \in \mathbb{R}^{\bar{C}\text{x}C}$, $\textbf{W}_{\textbf{h}} \in \mathbb{R}^{\bar{C}\text{x}C}$ and $\textbf{W}_{\textbf{v}} \in \mathbb{R}^{C\text{x}\bar{C}}$ are the learned weight matrices, which are implemented as $1$×$1$ convolutions. 

In addition, the module further multiplies the output of the attention layer by a scale parameter and add back the input feature map. Therefore, the final output is given by,

$$\textbf{y}_{\textbf{i}} = \gamma\textbf{o}_{\textbf{i}} + \textbf{x}_{\textbf{i}}$$

where $\gamma$ is a learnable scalar and it is initialized as 0. Introducing $\gamma$ allows the network to first rely on the cues in the local neighborhood – since this is easier – and then gradually learn to assign more weight to the non-local evidence.

# Location-based Attention
![](./img/Screen_Shot_2020-05-25_at_12.27.45_PM.png)

**Location-based Attention** is an attention mechanism in which the alignment scores are computed from solely the target hidden state $s_{t}$ as follows:

$$ a_{t, i} = \text{softmax}(W_{a}s_{t}) $$

# Content-based Attention
![](./img/Screen_Shot_2020-05-24_at_8.40.59_PM.png)

**Content-based attention** is an attention mechanism based on cosine similarity:

$$f_{att}\left(\textbf{h}_{i}, \textbf{s}_{j}\right) = \cos\left[\textbf{h}_{i};\textbf{s}_{j}\right] $$

It was utilised in [Neural Turing Machines](https://paperswithcode.com/method/neural-turing-machine) as part of the Addressing Mechanism.

# Spatial Attention Module
![](./img/Screen_Shot_2020-06-25_at_1.27.27_PM_CjrAZaI.png)

A **Spatial Attention Module** is a module for spatial attention in convolutional neural networks. It generates a spatial attention map by utilizing the inter-spatial relationship of features. Different from the channel attention, the spatial attention focuses on where is an informative part, which is complementary to the channel attention. To compute the spatial attention, we first apply average-pooling and max-pooling operations along the channel axis and concatenate them to generate an efficient feature descriptor. On the concatenated feature descriptor, we apply a convolution layer to generate a spatial attention map $\textbf{M}_{s}\left(F\right) \in \mathcal{R}^{H×W}$ which encodes where to emphasize or suppress. We describe the detailed operation below.

We aggregate channel information of a feature map by using two pooling operations, generating two 2D maps: $\mathbf{F}^{s}_{avg} \in \mathbb{R}^{1\times{H}\times{W}}$ and $\mathbf{F}^{s}_{max} \in \mathbb{R}^{1\times{H}\times{W}}$. Each denotes average-pooled features and max-pooled features across the channel. Those are then concatenated and convolved by a standard convolution layer, producing the 2D spatial attention map. In short, the spatial attention is computed as:

$$ \textbf{M}_{s}\left(F\right) = \sigma\left(f^{7x7}\left(\left[\text{AvgPool}\left(F\right);\text{MaxPool}\left(F\right)\right]\right)\right) $$

$$ \textbf{M}_{s}\left(F\right) = \sigma\left(f^{7x7}\left(\left[\mathbf{F}^{s}_{avg};\mathbf{F}^{s}_{max} \right]\right)\right) $$

where $\sigma$ denotes the sigmoid function and $f^{7×7}$ represents a convolution operation with the filter size of 7 × 7.

# Channel Attention Module
![](./img/Screen_Shot_2020-06-25_at_1.27.21_PM_YDoPGUi.png)

A **Channel Attention Module** is a module for channel-based attention in convolutional neural networks. We produce a channel attention map by exploiting the inter-channel relationship of features. As each channel of a feature map is considered as a feature detector, channel attention focuses on ‘what’ is meaningful given an input image. To compute the channel attention efficiently, we squeeze the spatial dimension of the input feature map. 

We first aggregate spatial information of a feature map by using both average-pooling and max-pooling operations, generating two different spatial context descriptors: $\mathbf{F}^{c}_{avg}$ and $\mathbf{F}^{c}_{max}$, which denote average-pooled features and max-pooled features respectively. 

Both descriptors are then forwarded to a shared network to produce our channel attention map $\mathbf{M}_{c} \in \mathbb{R}^{C\times{1}\times{1}}$. The shared network is composed of multi-layer perceptron (MLP) with one hidden layer. To reduce parameter overhead, the hidden activation size is set to $\mathbb{R}^{C/r×1×1}$, where $r$ is the reduction ratio. After the shared network is applied to each descriptor, we merge the output feature vectors using element-wise summation. In short, the channel attention is computed as:

$$  \mathbf{M_{c}}\left(\mathbf{F}\right) = \sigma\left(\text{MLP}\left(\text{AvgPool}\left(\mathbf{F}\right)\right)+\text{MLP}\left(\text{MaxPool}\left(\mathbf{F}\right)\right)\right) $$

$$  \mathbf{M_{c}}\left(\mathbf{F}\right) = \sigma\left(\mathbf{W_{1}}\left(\mathbf{W_{0}}\left(\mathbf{F}^{c}_{avg}\right)\right) +\mathbf{W_{1}}\left(\mathbf{W_{0}}\left(\mathbf{F}^{c}_{max}\right)\right)\right) $$

where $\sigma$ denotes the sigmoid function, $\mathbf{W}_{0} \in \mathbb{R}^{C/r\times{C}}$, and $\mathbf{W}_{1} \in \mathbb{R}^{C\times{C/r}}$. Note that the MLP weights, $\mathbf{W}_{0}$ and $\mathbf{W}_{1}$, are shared for both inputs and the ReLU activation function is followed by $\mathbf{W}_{0}$.

Note that the channel attention module with just average pooling is the same as the SE module.

# Spatial Attention-Guided Mask
![](./img/Screen_Shot_2020-06-23_at_3.06.19_PM.png)

**A Spatial Attention-Guided Mask** predicts a segmentation mask on each detected box with a spatial attention map that helps to focus on informative pixels and suppress noise. The goal is to guide the mask head for spotlighting meaningful pixels and repressing uninformative ones. 

Once features inside the predicted RoIs are extracted by RoI Align with 14×14 resolution, those features are fed into four conv layers and the spatial attention module (SAM) sequentially. To exploit the spatial attention map $A_{sag}\left(X_{i}\right) \in \mathcal{R}^{1\times{W}\times{H}}$ as a feature descriptor given input feature map $X_{i} \in \mathcal{R}^{C×W×H}$, the SAM first generates pooled features $P_{avg}, P_{max} \in \mathcal{R}^{1\times{W}\times{H}}$ by both average and max pooling operations respectively along the channel axis and aggregates them via concatenation. Then it is followed by a 3 × 3 conv layer and normalized by the sigmoid function. The computation process
is summarized as follow:

$$
A_{sag}\left(X_{i}\right) = \sigma\left(F_{3\times{3}}(P_{max} \cdot P_{avg})\right)
$$

where $\sigma$ denotes the sigmoid function, $F_{3\times{3}}$ is 3 × 3 conv layer and $\cdot$ represents concatenate operation. Finally, the attention guided feature map $X_{sag} ∈ \mathcal{R}^{C\times{W}\times{H}}$ is computed as:

$$
X_{sag} = A_{sag}\left(X_{i}\right) \otimes X_{i}
$$

where ⊗ denotes element-wise multiplication. After then, a 2 × 2 deconv upsamples the spatially attended feature map to 28 × 28 resolution. Lastly, a 1 × 1 conv is applied for predicting class-specific masks.

# Location Sensitive Attention
![](./img/Screen_Shot_2020-07-08_at_4.22.26_PM_t2uXfvN.png)

**Location Sensitive Attention** is an attention mechanism that extends the [additive attention mechanism](https://paperswithcode.com/method/additive-attention) to use cumulative attention weights from previous decoder time steps as an additional feature. This encourages the model to move forward consistently through the input, mitigating potential failure modes where some subsequences are repeated or ignored by the decoder.

Starting with additive attention where:

$$ e_{i, j} = w^{T}\tanh\left(W{s}_{i-1} + Vh_{j} + b\right) $$

where $w$ and $b$ are vectors, $W$ and $V$ are matrices. We extend this to be location-aware by making it take into account the alignment produced at the previous step. First, we extract $k$ vectors
$f_{i,j} \in \mathbb{R}^{k}$ for every position $j$ of the previous alignment $\alpha_{i−1}$ by convolving it with a matrix $F \in R^{k\times{r}}$:

$$ f_{i} = F ∗ \alpha_{i−1} $$

These additional vectors $f_{i,j}$ are then used by the scoring mechanism $e_{i,j}$:

$$ e_{i,j} = w^{T}\tanh\left(Ws_{i−1} + Vh_{j} + Uf_{i,j} + b\right) $$

# LAMA
![](./img/default.gif)

Please enter a description about the method here

# Channel-wise Soft Attention
![](./img/Screen_Shot_2020-06-06_at_12.05.50_PM_vXIb7cY.png)

**Chanel-wise Soft Attention** is an attention mechanism in computer vision where alignment weights are learned each feature-map channel is produced using a weighted combination over splits in the input. For example the he $c$-th channel is calculated as:
$$
    V^k_c=\sum_{i=1}^R a^k_i(c) U_{R(k-1)+i} ,
$$
where $U$ is the representation input, and where $a_i^k(c)$ denotes a (soft) assignment weight given by:

$$
a_i^k(c) =
\begin{cases}
  \frac{exp(\mathcal{G}^c_i(s^k))}{\sum_{j=0}^R exp(\mathcal{G}^c_j(s^k))} &amp; \quad\textrm{if } R&gt;1, \
   \frac{1}{1+exp(-\mathcal{G}^c_i(s^k))} &amp; \quad\textrm{if } R=1,\
\end{cases}
$$

and mapping $\mathcal{G}_i^c$ determines the weight of each split for the $c$-th channel based on the global context representation $s^k$.

# Global and Sliding Window Attention
![](./img/Screen_Shot_2020-05-31_at_7.27.43_PM.png)

**Global and Sliding Window Attention** is an attention pattern for attention-based models. It is motivated by the fact that non-sparse attention in the original [Transformer](https://paperswithcode.com/method/transformer) formulation has a self-attention component with $O\left(n^{2}\right)$ time and memory complexity where $n$ is the input sequence length and thus, is not efficient to scale to long inputs. 

Since windowed and dilated attention patterns are not flexible enough to learn task-specific representations, the authors of the [Longformer](https://paperswithcode.com/method/longformer) add “global attention” on few pre-selected input locations. This attention is operation symmetric: that is, a token with a global attention attends to all tokens across the sequence, and all tokens in the sequence attend to it. The Figure to the right shows an example of a sliding window attention with global attention at a few tokens at custom locations. For example for classification, global attention is used for the [CLS] token while in QA global attention is provided on all question tokens.

# Strided Attention
![](./img/Screen_Shot_2020-05-30_at_3.19.11_PM.png)

**Strided Attention** is a factorized attention pattern that has one head attend to the previous
$l$ locations, and the other head attend to every $l$th location, where $l$ is the stride and chosen to be close to $\sqrt{n}$. It was proposed as part of the [Sparse Transformer](https://paperswithcode.com/method/sparse-transformer) architecture.

Formally, $A^{(1)}_{i} = ${$t, t + 1, ..., i$} for $t = \max\left(0, i − l\right)$, and $A^{(2)}_{i} = ${$j : (i − j) \mod l = 0$}. The $i$-th output vector of the attention head attends to all input vectors either from $A^{(1)}_{i}$ or $A^{(2)}_{i}$. This pattern can be visualized in the figure to the right.

This formulation is convenient if the data naturally has a structure that aligns with the stride, like images or some types of music. For data without a periodic structure, like text, however, we find that the network can fail to properly route information with the strided pattern, as spatial coordinates for an element do not necessarily correlate with the positions where the element may be most relevant in the future.

# Sliding Window Attention
![](./img/Screen_Shot_2020-05-31_at_7.27.29_PM.png)

**Sliding Window Attention** is an attention pattern for attention-based models. It was proposed as part of the [Longformer](https://paperswithcode.com/method/longformer) architecture. It is motivated by the fact that non-sparse attention in the original [Transformer](https://paperswithcode.com/method/transformer) formulation has a self-attention component with $O\left(n^{2}\right)$ time and memory complexity where $n$ is the input sequence length and thus, is not efficient to scale to long inputs. Given the importance of local context, this attention pattern employs a fixed-size window attention surrounding each token. Using multiple stacked layers of such windowed attention results in a large receptive field, where top layers have access to all input locations and have the capacity to build representations that incorporate information across the entire input. 

More formally, in this attention pattern, given a fixed window size $w$, each token attends to $\frac{1}{2}w$ tokens on each side. The computation complexity of this pattern is $O\left(n×w\right)$,
which scales linearly with input sequence length $n$. To make this attention pattern efficient, $w$ should be small compared with $n$. But a model with typical multiple stacked transformers will have a large receptive field. This is analogous to CNNs where stacking layers of small kernels leads to high level features that are built from a large portion of the input (receptive field)

In this case, with a transformer of $l$ layers, the receptive field size is $l × w$ (assuming
$w$ is fixed for all layers). Depending on the application, it might be helpful to use different values of $w$ for each layer to balance between efficiency and model representation capacity.

# Single-Headed Attention
![](./img/Screen_Shot_2020-07-02_at_10.01.26_PM_ZjRpjQr.png)

**Single-Headed Attention** is a single-headed attention module used in the SHA-RNN language model. The principle design reasons for single-headedness were simplicity (avoiding running out of memory) and scepticism about the benefits of using multiple heads.

# Dilated Sliding Window Attention
![](./img/Screen_Shot_2020-05-31_at_7.27.36_PM.png)

**Dilated Sliding Window Attention** is an attention pattern for attention-based models. It was proposed as part of the [Longformer](https://paperswithcode.com/method/longformer) architecture. It is motivated by the fact that non-sparse attention in the original [Transformer](https://paperswithcode.com/method/transformer) formulation has a self-attention component with $O\left(n^{2}\right)$ time and memory complexity where $n$ is the input sequence length and thus, is not efficient to scale to long inputs. 

Compared to a Sliding Window Attention pattern, we can further increase the receptive field without increasing computation by making the sliding window "dilated". This is analogous to [dilated CNNs](https://paperswithcode.com/method/dilated-convolution) where the window has gaps of size dilation $d$. Assuming a fixed $d$ and $w$ for all layers, the receptive field is $l × d × w$, which can reach tens of thousands of tokens even for small values of $d$. In multi-headed attention, each attention head computes a different attention score.

# LSH Attention
![](./img/Screen_Shot_2020-06-01_at_6.28.07_PM.png)

**LSH Attention**, or **Locality Sensitive Hashing Attention** is a replacement for dot-product attention with one that uses locality-sensitive hashing, changing its complexity from O($L^2$) to O($L\log L$), where $L$ is the length of the sequence. LSH refers to a family of functions (known as LSH families) to hash data points into buckets so that data points near each other are located in the same buckets with high probability, while data points far from each other are likely to be in different buckets. It was proposed as part of the [Reformer](https://paperswithcode.com/method/reformer) architecture.

# Fixed Factorized Attention
![](./img/Screen_Shot_2020-05-30_at_5.19.41_PM.png)

**Fixed Factorized Attention** is a factorized attention pattern where specific cells summarize previous locations and propagate that information to all future cells. It was proposed as part of the [Sparse Transformer](https://paperswithcode.com/method/sparse-transformer) architecture.

Formally, $A^{(1)}_{i} = ${$j : \left(\lfloor{j/l\rfloor}=\lfloor{i/l\rfloor}\right)$}, where the brackets denote the floor operation, and $A^{(2)}_{i} = ${$j : j \mod l \in ${$t, t+1, \ldots, l$}}, where $t=l-c$ and $c$ is a hyperparameter. The $i$-th output vector of the attention head attends to all input vectors either from $A^{(1)}_{i}$ or $A^{(2)}_{i}$. This pattern can be visualized in the figure to the right.

If the stride is 128 and $c = 8$, then all future positions greater than 128 can attend to positions 120-128, all positions greater than 256 can attend to 248-256, and so forth. 

A fixed-attention pattern with $c = 1$ limits the expressivity of the network significantly, as many representations in the network are only used for one block whereas a small number of locations are used by all blocks. The authors found choosing $c \in ${$8, 16, 32$} for typical values of $l \in
{128, 256}$ performs well, although this increases the computational cost of this method by $c$ in comparison to the strided attention.

Additionally, the authors found that when using multiple heads, having them attend to distinct subblocks of length $c$ within the block of size $l$ was preferable to having them attend to the same subblock.

# SortCut Sinkhorn Attention
![](./img/Screen_Shot_2020-07-07_at_11.55.12_PM_tC3hpi6.png)

**SortCut Sinkhorn Attention** is a variant of [Sparse Sinkhorn Attention](https://paperswithcode.com/method/sparse-sinkhorn-attention) where a post-sorting truncation of the input sequence is performed, essentially performing a hard top-k operation on the input sequence blocks within the computational graph. While most attention models mainly re-weight or assign near-zero weights during training, this allows for explicitly and dynamically truncate the input sequence. Specifically:

$$ Y = \text{Softmax}\left(Q{\psi_{S}}\left(K\right)^{T}_{\left[:n\right]}\right)\psi_{S}\left(V\right)_{\left[:n\right]} $$

where $n$ is the Sortfut budget hyperparameter.

# Dense Synthesized Attention
![](./img/Screen_Shot_2020-06-01_at_11.54.21_PM.png)

**Dense Synthesized Attention**, introduced with the [Synthesizer](https://paperswithcode.com/method/synthesizer) architecture, is a type of synthetic attention mechanism that replaces the notion of query-key-values in the self-attention module and directly synthesizes the alignment matrix instead. Dense attention is conditioned on each input token. The method accepts an input $X \in \mathbb{R}^{l\text{ x }d}$ and produces an output of $Y \in \mathbb{R}^{l\text{ x }d}$. Here $l$ refers to the sequence length and $d$ refers to the dimensionality of the model. We first adopt $F\left(.\right)$, a parameterized function, for projecting input $X_{i}$ from $d$ dimensions to $l$ dimensions.

$$B_{i} = F\left(X_{i}\right)$$

where $F\left(.\right)$ is a parameterized function that maps $\mathbb{R}^{d}$ to $\mathbb{R}^{l}$ and $i$ is the $i$-th token of $X$. Intuitively, this can be interpreted as learning a token-wise projection to the sequence length $l$. Essentially, with this model, each token predicts weights for each token in the input sequence. In practice, a simple two layered feed-forward layer with ReLU activations for $F\left(.\right)$ is adopted:

$$ F\left(X\right) = W\left(\sigma_{R}\left(W(X) + b\right)\right) + b$$

where $\sigma_{R}$ is the ReLU activation function. Hence, $B$ is now of $\mathbb{R}^{l\text{ x }d}$. Given $B$, we now compute:

$$ Y = \text{Softmax}\left(B\right)G\left(X\right) $$

where $G\left(.\right)$ is another parameterized function of $X$ that is analogous to $V$ (value) in the standard Transformer model. This approach eliminates the dot product altogether by replacing $QK^{T}$ in standard Transformers with the synthesizing function $F\left(.\right)$.

# Graph Self-Attention
![](./img/Screen_Shot_2020-07-08_at_5.38.28_PM_3e00cof.png)

**Graph Self-Attention (GSA)** is a self-attention module used in the [BP-Transformer](https://paperswithcode.com/method/bp-transformer) architecture, and is based on the [graph attentional layer](https://paperswithcode.com/method/graph-attentional-layer).

For a given node $u$, we update its representation according to its neighbour nodes, formulated as $\mathbf{h}_{u} \leftarrow \text{GSA}\left(\mathcal{G}, \mathbf{h}^{u}\right)$.

Let $\mathbf{A}\left(u\right)$ denote the set of the neighbour nodes of $u$ in $\mathcal{G}$, $\text{GSA}\left(\mathcal{G}, \mathbf{h}^{u}\right)$ is detailed as follows:

$$ \mathbf{A}^{u} = \text{concat}\left({\mathbf{h}_{v} | v \in \mathcal{A}\left(u\right)}\right) $$

$$ \mathbf{Q}^{u}_{i} = \mathbf{H}_{k}\mathbf{W}^{Q}_{i},\mathbf{K}_{i}^{u} = \mathbf{A}^{u}\mathbf{W}^{K}_{i},\mathbf{V}^{u}_{i} = \mathbf{A}^{u}\mathbf{W}_{i}^{V} $$

$$ \text{head}^{u}_{i} = \text{softmax}\left(\frac{\mathbf{Q}^{u}_{i}\mathbf{K}_{i}^{uT}}{\sqrt{d}}\right)\mathbf{V}_{i}^{u} $$

$$ \text{GSA}\left(\mathcal{G}, \mathbf{h}^{u}\right) = \left[\text{head}^{u}_{1}, \dots, \text{head}^{u}_{h}\right]\mathbf{W}^{O}$$

where d is the dimension of h, and $\mathbf{W}^{Q}_{i}$, $\mathbf{W}^{K}_{i}$ and $\mathbf{W}^{V}_{i}$ are trainable parameters of the $i$-th attention head.

# Multi-Head Linear Attention
![](./img/Screen_Shot_2020-07-08_at_12.17.05_AM_st5S0XV.png)

**Multi-Head Linear Attention** is a type of linear multi-head self-attention module, proposed with the [Linformer](https://paperswithcode.com/method/linformer) architecture. The main idea is to add two linear projection matrices $E_{i}, F_{i} \in \mathbb{R}^{n\times{k}}$ when computing key and value. We first project the original $\left(n \times d\right)$-dimensional key and value layers $KW_{i}^{K}$ and $VW_{i}^{V}$ into $\left(k\times{d}\right)$-dimensional projected key and value layers. We then compute a $\left(n\times{k}\right)$ dimensional context mapping $\bar{P}$ using scaled-dot product attention:

$$ \bar{\text{head}_{i}} = \text{Attention}\left(QW^{Q}_{i}, E_{i}KW_{i}^{K}, F_{i}VW_{i}^{V}\right) $$

$$ \bar{\text{head}_{i}} = \text{softmax}\left(\frac{QW^{Q}_{i}\left(E_{i}KW_{i}^{K}\right)^{T}}{\sqrt{d_{k}}}\right) \cdot F_{i}VW_{i}^{V} $$

Finally, we compute context embeddings for each head using $\bar{P} \cdot \left(F_{i}{V}W_{i}^{V}\right)$.

# Adaptive Masking
![](./img/Screen_Shot_2020-07-07_at_10.59.09_PM_mJpZE9X.png)

**Adaptive Masking** is a type of attention mechanism that allows a model to learn its own context size to attend over. For each head in [Multi-Head Attention](https://paperswithcode.com/method/multi-head-attention), we add a masking function to control for the span of the attention. A masking function is a non-increasing function that maps a
distance to a value in $\left[0, 1\right]$. We take the following soft masking function $m_{z}$ parametrized by a real value $z$ in $\left[0, S\right]$:

$$ m_{z}\left(x\right) = \min\left[\max\left[\frac{1}{R}\left(R+z-x\right), 0\right], 1\right] $$

where $R$ is a hyper-parameter that controls its softness. The shape of this piecewise function as a function of the distance. This soft masking function is inspired by [Jernite et al. (2017)](https://arxiv.org/abs/1611.06188). The attention weights from are then computed on the masked span:

$$ a_{tr} = \frac{m_{z}\left(t-r\right)\exp\left(s_{tr}\right)}{\sum^{t-1}_{q=t-S}m_{z}\left(t-q\right)\exp\left(s_{tq}\right)}$$

We add a $\mathcal{l}_{1}$ penalization on the parameters $z_{i}$ for each attention head $i$ of the model to the loss function:

$$ L = - \log{P}\left(w_{1}, \dots, w_{T}\right) + \frac{\lambda}{M}\sum_{i}z_{i} $$

where $\lambda &gt; 0$ is the regularization hyperparameter, and $M$ is the number of heads in each
layer. This formulation is differentiable in the parameters $z_{i}$, and learnt jointly with the rest of the model.

# Sparse Sinkhorn Attention
![](./img/Screen_Shot_2020-07-07_at_11.19.11_PM_7OGg1iN.png)

**Sparse Sinkhorn Attention** is an attention mechanism that reduces the memory complexity of the dot-product attention mechanism and is capable of learning sparse attention outputs. It is based on the idea of differentiable sorting of internal representations within the self-attention module. SSA incorporates a meta sorting network that learns to rearrange and sort input sequences. Sinkhorn normalization is used to normalize the rows and columns of the sorting matrix. The actual SSA attention mechanism then acts on the block sorted sequences.

# Factorized Random Synthesized Attention
![](./img/Screen_Shot_2020-06-02_at_12.06.20_AM_PkacRfG.png)

**Factorized Random Synthesized Attention**, introduced with the [Synthesizer](https://paperswithcode.com/method/synthesizer) architecture, is similar to factorized dense synthesized attention but for random synthesizers. We factorize $R$ into low rank matrices $R_{1}, R_{2} \in \mathbb{R}^{l\text{ x}k}$:

$$ Y = \text{Softmax}\left(R_{1}R_{2}^{T}\right)G\left(X\right) . $$

Therefore for each head, this reduces the parameter costs from $l^{2}$ to $2\left(lk\right)$ where
$k &lt;&lt; l$ and hence helps prevent overfitting. In practice, we use a small value of $k = 8$.

# Global Context Block
![](./img/Screen_Shot_2020-06-13_at_7.21.56_PM.png)

A **Global Context Block** is an image model block for global context modeling; this has the benefits of the simplified non-local (SNL) block with effective modeling on long-range dependency, and the squeeze-excitation (SE) block with lightweight computation. In this framework, we have (a) global attention pooling, which adopts a 1x1 convolution $W_{k}$ and softmax function to obtain the attention weights, and then performs the attention pooling to obtain the global context features, (b) feature transform via a 1x1 convolution $W_{v}$; (c) feature aggregation, which employs addition to aggregate the global context features to the features of each position. The GC block is proposed as a lightweight way to achieve global context modeling.

# Factorized Dense Synthesized Attention
![](./img/Screen_Shot_2020-06-01_at_11.54.21_PM_52J3Q9s.png)

**Factorized Dense Synthesized Attention** is a synthesized attention mechanism, similar to dense synthesized attention, but we factorize the outputs to reduce parameters and prevent overfitting. It was proposed as part of the [Synthesizer](https://paperswithcode.com/method/synthesizer) architecture. The factorized variant of the dense synthesizer can be expressed as follows:

$$A, B = F_{A}\left(X_{i}\right), F_{B}\left(X_{i}\right)$$

where $F_{A}\left(.\right)$ projects input $X_{i}$ into $a$ dimensions, $F_B\left(.\right)$ projects $X_{i}$ to $b$ dimensions, and $a \text{ x } b = l$. The output of the factorized module is now written as:

$$ Y = \text{Softmax}\left(C\right)G\left(X\right) $$

where $C = H_{A}\left(A\right) * H_{B}\left(B\right)$, where $H_{A}$, $H_{B}$ are tiling functions and $C \in \mathbb{R}^{l \text{ x } l}$. The tiling function simply duplicates the vector $k$ times, i.e., $\mathbb{R}^{l} \rightarrow \mathbb{R}^{lk}$. In this case, $H_{A}\left(\right)$ is a projection of $\mathbb{R}^{a} \rightarrow \mathbb{R}^{ab}$ and $H_{B}\left(\right)$ is a projection of $\mathbb{R}^{b} \rightarrow \mathbb{R}^{ba}$. To avoid having similar values within the same block, we compose the outputs of $H_{A}$ and $H_{B}$.

# CBAM
![](./img/new_Overview.jpg)

**Convolutional Block Attention Module (CBAM)** is an attention module for convolutional neural networks. Given an intermediate feature map, the module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement.

Given an intermediate feature map $\mathbf{F} \in \mathbb{R}^{C×H×W}$ as input, CBAM sequentially infers a 1D channel attention map $\mathbf{M}_{c} \in \mathbb{R}^{C×1×1}$ and a 2D spatial attention map $\mathbf{M}_{s} \in \mathbb{R}^{1×H×W}$. The overall attention process can be summarized as:

$$ \mathbf{F}' = \mathbf{M}_{c}\left(\mathbf{F}\right) \otimes \mathbf{F} $$

$$ \mathbf{F}'' = \mathbf{M}_{s}\left(\mathbf{F'}\right) \otimes \mathbf{F'} $$

During multiplication, the attention values are broadcasted (copied) accordingly: channel attention values are broadcasted along the spatial dimension, and vice versa. $\mathbf{F}''$ is the final refined
output.

# Routing Attention
![](./img/Screen_Shot_2020-07-08_at_12.03.21_AM_rLg04u7.png)

**Routed Attention** is an attention pattern proposed as part of the [Routing Transformer](https://paperswithcode.com/method/routing-transformer) architecture.  Each attention module
considers a clustering of the space: the current timestep only attends to context belonging to the same cluster. In other word, the current time-step query is routed to a limited number of context through its cluster assignment. This can be contrasted with [strided](https://paperswithcode.com/method/strided-attention) attention patterns and those proposed with the [Sparse Transformer](https://paperswithcode.com/method/sparse-transformer).

In the image to the right, the rows represent the outputs while the columns represent the inputs. The different colors represent cluster memberships for the output token.

# Random Synthesized Attention
![](./img/Screen_Shot_2020-06-02_at_12.06.20_AM.png)

Dense Synthesized Attention, introduced with the [Synthesizer](https://paperswithcode.com/method/synthesizer) architecture, learns synthetic attention by conditioning on each input of $X$ and projecting to $l$ dimensions. Hence, the Dense Synthesizer conditions on each token independently, as opposed to pairwise token interactions in the vanilla Transformer model. In contrast, **Random Synthesized Attention** is where the attention weights are not conditioned on any input tokens. Instead, the attention weights are initialized to random values. These values can then either be trainable or kept fixed. Let $R$ be a randomly initialized matrix. Random Synthesized Attention is defined as:

$$Y = \text{Softmax}\left(R\right)G\left(X\right) $$

where $R \in \mathbb{R}^{l \text{ x } l}$. Notably, each head adds 2 parameters to the overall network. The basic idea of the Random Synthesizer is to not rely on pairwise token interactions or any information from individual token but rather to learn a task-specific alignment that works well globally across many samples. This is a direct generalization of the recently proposed fixed self-attention patterns of Raganato et al (2020).

# Attention-augmented Convolution
![](./img/Screen_Shot_2020-06-12_at_9.00.46_PM.png)

**Attention-augmented Convolution** is a type of convolution with a two-dimensional relative self-attention mechanism that can replace convolutions as a stand-alone computational primitive for image classification. It employs scaled-dot product attention and multi-headed attention as with Transformers.

Concatenating convolutional and attentional feature maps: Formally, consider an original convolution operator with kernel size $k$, $F_{in}$ input filters and $F_{out}$ output filters. The corresponding attention augmented convolution can be written as"

$$\text{AAConv}\left(X\right) = \text{Concat}\left[\text{Conv}(X), \text{MHA}(X)\right] $$

We denote $v = \frac{d_{v}}{F_{out}}$, the ratio of attentional channels to number of original output filters and $\kappa = \frac{d_{k}}{F_{out}}$ the ratio of key depth to number of original output filters. Similarly to the convolution, the attention augmented convolution 1) is equivariant to translation and 2) can readily operate on inputs of different spatial dimensions.

# Multiplicative Attention
![](./img/Screen_Shot_2020-05-25_at_12.32.09_PM.png)

**Multiplicative Attention** is an attention mechanism where the alignment score function is calculated as:

$$f_{att}\left(\textbf{h}_{i}, \textbf{s}_{j}\right) = h_{i}^{T}\textbf{W}_{a}s_{j}$$

Additive and multiplicative attention are similar in complexity, although multiplicative attention is faster and more space-efficient in practice as it can be implemented more efficiently using matrix multiplication. Both variants perform similar for small dimensionality $d_{h}$ of the decoder states, but additive attention performs better for larger dimensions. One way to mitigate this is to scale $f_{att}\left(\textbf{h}_{i}, \textbf{s}_{j}\right)$ by $1/\sqrt{d_{h}}$ as with [scaled dot-product atttention](https://paperswithcode.com/method/scaled).

