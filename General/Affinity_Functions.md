# Embedded Gaussian Affinity
![](./img/default.gif)

**Embedded Gaussian Affinity** is a type of affinity or self-similarity function between two points $\mathbb{x_{i}}$ and $\mathbb{x_{j}}$ that uses a Gaussian function in an embedding space:

$$ f\left(\mathbb{x_{i}}, \mathbb{x_{j}}\right) = e^{\theta\left(\mathbb{x_{i}}\right)^{T}\phi\left(\mathbb{x_{j}}\right)} $$

Here $\theta\left(x_{i}\right) = W_{θ}x_{i}$ and $\phi\left(x_{j}\right) = W_{φ}x_{j}$ are two embeddings.

Note that the self-attention module used in the original Transformer model is a special case of non-local operations in the embedded Gaussian version. This can be seen from the fact that for a given $i$, $\frac{1}{\mathcal{C}\left(\mathbb{x}\right)}\sum_{\forall{j}}f\left(\mathbb{x}_{i}, \mathbb{x}_{j}\right)g\left(\mathbb{x}_{j}\right)$ becomes the softmax computation along the dimension $j$. So we have $\mathbb{y} = \text{softmax}\left(\mathbb{x}^{T}W^{T}_{theta}W_{\phi}\mathbb{x}\right)g\left(\mathbb{x}\right)$, which is the self-attention form in the Transformer model. This shows how we can relate this recent self-attention model to the classic computer vision method of non-local means.

# Embedded Dot Product Affinity
![](./img/default.gif)

**Embedded Dot Product Affinity** is a type of affinity or self-similarity function between two points $\mathbb{x_{i}}$ and $\mathbb{x_{j}}$ that uses a dot product function in an embedding space:

$$ f\left(\mathbb{x_{i}}, \mathbb{x_{j}}\right) = \theta\left(\mathbb{x_{i}}\right)^{T}\phi\left(\mathbb{x_{j}}\right) $$

Here $\theta\left(x_{i}\right) = W_{θ}x_{i}$ and $\phi\left(x_{j}\right) = W_{φ}x_{j}$ are two embeddings.

The main difference between the dot product and embedded Gaussian affinity functions is the presence of softmax, which plays the role of an activation function.

# Concatenation Affinity
![](./img/default.gif)

**Concatenation Affinity** is a type of affinity or self-similarity function between two points $\mathbb{x_{i}}$ and $\mathbb{x_{j}}$ that uses a concatenation function:

$$ f\left(\mathbb{x_{i}}, \mathbb{x_{j}}\right) = \text{ReLU}\left(\mathbb{w}^{T}_{f}\left[\theta\left(\mathbb{x}_{i}\right), \phi\left(\mathbb{x}_{j}\right)\right]\right)$$

Here $\left[·, ·\right]$ denotes concatenation and $\mathbb{w}_{f}$ is a weight vector that projects the concatenated vector to a scalar.

# Gaussian Affinity
![](./img/default.gif)

**Gaussian Affinity** is a type of affinity or self-similarity function between two points $\mathbb{x_{i}}$ and $\mathbb{x_{j}}$ that uses a Gaussian function:

$$ f\left(\mathbb{x_{i}}, \mathbb{x_{j}}\right) = e^{\mathbb{x^{T}_{i}}\mathbb{x_{j}}} $$

Here $\mathbb{x^{T}_{i}}\mathbb{x_{j}}$ is dot-product similarity.

