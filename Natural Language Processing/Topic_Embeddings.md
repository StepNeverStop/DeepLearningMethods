# [lda2vec](https://paperswithcode.com/method/lda2vec)
![](./img/Screen_Shot_2020-05-26_at_11.18.34_PM.png)

**lda2vec** builds representations over both words and documents by mixing word2vec’s skipgram architecture with Dirichlet-optimized sparse topic mixtures. 

The Skipgram Negative-Sampling (SGNS) objective of word2vec is modified to utilize document-wide feature vectors while simultaneously learning continuous document weights loading onto topic vectors. The total loss term $L$ is the sum of the Skipgram Negative Sampling Loss (SGNS) $L^{neg}_{ij}$ with the addition of a Dirichlet-likelihood term over document weights, $L_{d}$. The loss is conducted using a context vector, $\overrightarrow{c_{j}}$ , pivot word vector $\overrightarrow{w_{j}}$, target word vector $\overrightarrow{w_{i}}$, and negatively-sampled word vector $\overrightarrow{w_{l}}$:

$$ L = L^{d} + \Sigma_{ij}L^{neg}_{ij} $$

$$L^{neg}_{ij} = \log\sigma\left(c_{j}\cdot\overrightarrow{w_{i}}\right) + \sum^{n}_{l=0}\sigma\left(-\overrightarrow{c_{j}}\cdot\overrightarrow{w_{l}}\right)$$

source: [source](http://arxiv.org/abs/1605.02019v1)
