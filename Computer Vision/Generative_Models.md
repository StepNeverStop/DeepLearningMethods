# [AutoEncoder](https://paperswithcode.com/method/autoencoder)
![](./img/Autoencoder_schema.png)

An **Autoencoder** is a bottleneck architecture that turns a high-dimensional input into a latent low-dimensional code (encoder), and then performs a reconstruction of the input with this latent code (the decoder).

Image: [Michael Massi](https://en.wikipedia.org/wiki/Autoencoder#/media/File:Autoencoder_schema.png)

source: [source](https://science.sciencemag.org/content/313/5786/504)
# [GAN](https://paperswithcode.com/method/gan)
![](./img/gan.jpeg)

A **GAN**, or **Generative Adversarial Network**, is a generative model that simultaneously trains
two models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the
probability that a sample came from the training data rather than $G$.

The training procedure for $G$ is to maximize the probability of $D$ making
a mistake. This framework corresponds to a minimax two-player game. In the
space of arbitrary functions $G$ and $D$, a unique solution exists, with $G$
recovering the training data distribution and $D$ equal to $\frac{1}{2}$
everywhere. In the case where $G$ and $D$ are defined by multilayer perceptrons,
the entire system can be trained with backpropagation. 

(Image Source: [here](http://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html))

source: [source](https://arxiv.org/abs/1406.2661v1)
# [VAE](https://paperswithcode.com/method/vae)
![](./img/Screen_Shot_2020-07-07_at_4.47.56_PM_Y06uCVO.png)

A **Variational Autoencoder** is a type of likelihood-based generative model. It consists of an encoder, that takes in data $x$ as input and transforms this into a latent representation $z$,  and a decoder, that takes a latent representation $z$ and returns a reconstruction $\hat{x}$. Inference is performed via variational inference to approximate the posterior of the model.

source: [source](http://arxiv.org/abs/1312.6114v10)
# [CycleGAN](https://paperswithcode.com/method/cyclegan)
![](./img/Screen_Shot_2020-07-05_at_3.54.24_PM_aoT8JRU.png)

**CycleGAN**, or **Cycle-Consistent GAN**, is a type of generative adversarial network for unpaired image-to-image translation. For two domains $X$ and $Y$, CycleGAN learns a mapping $G : X \rightarrow Y$ and $F: Y \rightarrow X$. The novelty lies in trying to enforce the intuition that these mappings should be reverses of each other and that both mappings should be bijections. This is achieved through a cycle consistency loss that encourages $F\left(G\left(x\right)\right) \approx x$ and $G\left(Y\left(y\right)\right) \approx y$. Combining this loss with the adversarial losses on $X$ and $Y$ yields the full objective for unpaired image-to-image translation.

For the mapping $G : X \rightarrow Y$ and its discriminator $D_{Y}$ we have the objective:

$$ \mathcal{L}_{GAN}\left(G, D_{Y}, X, Y\right) =\mathbb{E}_{y \sim p_{data}\left(y\right)}\left[\log D_{Y}\left(y\right)\right] + \mathbb{E}_{x \sim p_{data}\left(x\right)}\left[log(1 − D_{Y}\left(G\left(x\right)\right)\right] $$

where $G$ tries to generate images $G\left(x\right)$ that look similar to images from domain $Y$, while $D_{Y}$ tries to discriminate between translated samples $G\left(x\right)$ and real samples $y$. A similar loss is postulated for the mapping $F: Y \rightarrow X$ and its discriminator $D_{X}$.

The Cycle Consistency Loss reduces the space of possible mapping functions by enforcing forward and backwards consistency:

$$ \mathcal{L}_{cyc}\left(G, F\right) = \mathbb{E}_{x \sim p_{data}\left(x\right)}\left[||F\left(G\left(x\right)\right) - x||_{1}\right] + \mathbb{E}_{y \sim p_{data}\left(y\right)}\left[||G\left(F\left(y\right)\right) - y||_{1}\right] $$

The full objective is:

$$ \mathcal{L}_{GAN}\left(G, F, D_{X}, D_{Y}\right) = \mathcal{L}_{GAN}\left(G, D_{Y}, X, Y\right) + \mathcal{L}_{GAN}\left(F, D_{X}, X, Y\right) + \lambda\mathcal{L}_{cyc}\left(G, F\right) $$

Where we aim to solve:

$$ G^{*}, F^{*} = \arg \min_{G, F} \min_{D_{X}, D_{Y}} \mathcal{L}_{GAN}\left(G, F, D_{X}, D_{Y}\right) $$

For the original architecture the authors use:

- two stride-2 convolutions, several residual blocks, and two fractionally strided convolutions with stride $\frac{1}{2}$.
- instance normalization
- PatchGANs for the discriminator
- Least Square Loss for the GAN objectives.

source: [source](http://arxiv.org/abs/1703.10593v6)
# [Denoising Autoencoder](https://paperswithcode.com/method/denoising-autoencoder)
![](./img/Denoising-Autoencoder_qm5AOQM.png)

A **Denoising Autoencoder** is a modification on the [autoencoder](https://paperswithcode.com/method/autoencoder) to prevent the network learning the identity function. Specifically, if the autoencoder is too big, then it can just learn the data, so the output equals the input, and does not perform any useful representation learning or dimensionality reduction. Denoising autoencoders solve this problem by corrupting the input data on purpose, adding noise or masking some of the input values.

Image Credit: [Kumar et al](https://www.semanticscholar.org/paper/Static-hand-gesture-recognition-using-stacked-Kumar-Nandi/5191ddf3f0841c89ba9ee592a2f6c33e4a40d4bf)

# [Restricted Boltzmann Machine](https://paperswithcode.com/method/restricted-boltzmann-machine)
![](./img/1_Z-uEtQkFPk7MtbolOSUvrA_qoiHKUX.png)

**Restricted Boltzmann Machines**, or **RBMs**, are two-layer generative neural networks that learn a probability distribution over the inputs. They are a special class of Boltzmann Machine in that they have a restricted number of connections between visible and hidden units. Every node in the visible layer is connected to every node in the hidden layer, but no nodes in the same group are connected. RBMs are usually trained using the contrastive divergence learning procedure.

Image Source: [here](https://medium.com/datatype/restricted-boltzmann-machine-a-complete-analysis-part-1-introduction-model-formulation-1a4404873b3)

# [Deep Belief Network](https://paperswithcode.com/method/deep-belief-network)
![](./img/Screen_Shot_2020-05-28_at_2.54.43_PM_EwgIrIu.png)

A **Deep Belief Network (DBN)** is a multi-layer generative graphical model. DBNs have bi-directional connections ([RBM](https://paperswithcode.com/method/restricted-boltzmann-machine)-type connections) on the top layer while the bottom layers only have top-down connections. They are trained using layerwise pre-training. Pre-training occurs by training the network component by component bottom up: treating the first two layers as an RBM and training, then treating the second layer and third layer as another RBM and training for those parameters.

Source: [Origins of Deep Learning](https://arxiv.org/pdf/1702.07800.pdf)

Image Source: [Wikipedia](https://en.wikipedia.org/wiki/Deep_belief_network)

# [WGAN](https://paperswithcode.com/method/wgan)
![](./img/Screen_Shot_2020-05-25_at_2.53.08_PM.png)

**Wasserstein GAN**, or **WGAN**, is a type of generative adversarial network that minimizes an approximation of the Earth-Mover's distance (EM) rather than the Jensen-Shannon divergence as in the original GAN formulation. It leads to more stable training than original GANs with less evidence of mode collapse, as well as meaningful curves that can be used for debugging and searching hyperparameters.

source: [source](http://arxiv.org/abs/1701.07875v3)
# [DCGAN](https://paperswithcode.com/method/dcgan)
![](./img/Screen_Shot_2020-07-01_at_11.27.51_PM_IoGbo1i.png)

**DCGAN**, or **Deep Convolutional GAN**, is a generative adversarial network architecture. It uses a couple of guidelines, in particular:

- Replacing any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Using batchnorm in both the generator and the discriminator.
- Removing fully connected hidden layers for deeper architectures.
- Using ReLU activation in generator for all layers except for the output, which uses tanh.
- Using LeakyReLU activation in the discriminator for all layer.

source: [source](http://arxiv.org/abs/1511.06434v2)
# [SAGAN](https://paperswithcode.com/method/sagan)
![](./img/Screen_Shot_2020-05-25_at_1.36.58_PM.png)

The **Self-Attention Generative Adversarial Network**, or **SAGAN**, allows for attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other.

source: [source](https://arxiv.org/abs/1805.08318v2)
# [StyleGAN](https://paperswithcode.com/method/stylegan)
![](./img/Screen_Shot_2020-06-28_at_9.15.44_PM.png)

**StyleGAN** is a type of generative adversarial network. It uses an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature; in particular, the use of adaptive instance normalization. Otherwise it follows Progressive GAN in using a progressively growing training regime. Other quirks include the fact it generates from a fixed value tensor not stochastically generated latent variables as in regular GANs. The stochastically generated latent variables are used as style vectors in the adaptive instance normalization at each resolution after being transformed by an 8-layer feedforward network. Lastly, it employs a form of regularization called mixing regularization, which mixes two style latent variables during training.

source: [source](http://arxiv.org/abs/1812.04948v3)
# [Pix2Pix](https://paperswithcode.com/method/pix2pix)
![](./img/Screen_Shot_2020-07-05_at_12.37.47_PM_dZrgNzj.png)

**Pix2Pix** is a conditional image-to-image translation architecture that uses a conditional GAN objective combined with a reconstruction loss. The conditional GAN objective for observed images $x$, output images $y$ and the random noise vector $z$ is:

$$ \mathcal{L}_{cGAN}\left(G, D\right) =\mathbb{E}_{x,y}\left[\log D\left(x, y\right)\right]+
\mathbb{E}_{x,z}\left[log(1 − D\left(x, G\left(x, z\right)\right)\right] $$

We augment this with a reconstruction term:

$$ \mathcal{L}_{L1}\left(G\right) = \mathbb{E}_{x,y,z}\left[||y - G\left(x, z\right)||_{1}\right] $$

and we get the final objective as:

$$ G^{*} = \arg\min_{G}\max_{D}\mathcal{L}_{cGAN}\left(G, D\right) + \lambda\mathcal{L}_{L1}\left(G\right) $$

The architectures employed for the generator and discriminator closely follow [DCGAN](https://paperswithcode.com/method/dcgan), with a few modifications:

- Concatenated skip connections are used to "shuttle" low-level information between the input and output, similar to a [U-Net](https://paperswithcode.com/method/u-net).
- The use of a PatchGAN discriminator that only penalizes structure at the scale of patches.

source: [source](http://arxiv.org/abs/1611.07004v3)
# [BigGAN](https://paperswithcode.com/method/biggan)
![](./img/Screen_Shot_2020-07-04_at_4.41.08_PM_cKfKr80.png)

**BigGAN** is a type of generative adversarial network that was designed for scaling generation to high-resolution, high-fidelity images. It includes a number of incremental changes and innovations. The baseline and incremental changes are:

- Using SAGAN as a baseline with spectral norm. for G and D, and using TTUR.
- Using a Hinge Loss GAN objective
- Using class-conditional batch normalization to provide class information to G (but with linear projection not MLP.
- Using a projection discriminator for D to provide class information to D.
- Evaluating with EWMA of G's weights, similar to ProGANs.

The innovations are:

- Increasing batch sizes, which has a big effect on the Inception Score of the model.
- Increasing the width in each layer leads to a further Inception Score improvement.
- Adding skip connections from the latent variable $z$ to further layers helps performance.
- Truncation trick: sampling the latent from a truncated normal.
- A new variant of Orthogonal Regularization.

source: [source](http://arxiv.org/abs/1809.11096v2)
# [PixelCNN](https://paperswithcode.com/method/pixelcnn)
![](./img/Screen_Shot_2020-05-16_at_7.27.51_PM_tpsd8Td.png)

A **PixelCNN** is a generative model that uses autoregressive connections to model images pixel by pixel, decomposing the joint image distribution as a product of conditionals. PixelCNNs are much faster to train than [PixelRNNs](https://paperswithcode.com/method/pixelrnn) because convolutions are inherently easier to parallelize; given the vast number of pixels present in large image datasets this is an important advantage.

source: [source](http://arxiv.org/abs/1606.05328v2)
# [VQ-VAE](https://paperswithcode.com/method/vq-vae)
![](./img/Screen_Shot_2020-06-28_at_4.26.40_PM.png)

**VQ-VAE** is a type of variational autoencoder that uses vector quantisation to obtain a discrete latent representation. It differs from [VAEs](https://paperswithcode.com/method/vae) in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. In order to learn a discrete latent representation, ideas from vector quantisation (VQ) are incorporated. Using the VQ method allows the model to circumvent issues of posterior collapse - where the latents are ignored when they are paired with a powerful autoregressive decoder - typically observed in the VAE framework. Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes.

source: [source](http://arxiv.org/abs/1711.00937v2)
# [InfoGAN](https://paperswithcode.com/method/infogan)
![](./img/Screen_Shot_2020-07-04_at_8.58.26_PM_7gbCeqC.png)

**InfoGAN** is a type of generative adversarial network that modifies the GAN objective to
encourage it to learn interpretable and meaningful representations. This is done by maximizing the
mutual information between a fixed small subset of the GAN’s noise variables and the observations.

Formally, InfoGAN is defined as a minimax game with a variational regularization of mutual information and the hyperparameter $\lambda$:

$$ \min_{G, Q}\max_{D}V_{INFOGAN}\left(D, G, Q\right) = V\left(D, G\right) - \lambda{L}_{I}\left(G, Q\right) $$

Where $Q$ is an auxiliary distribution that approximates the posterior $P\left(c\mid{x}\right)$ - the probability of the latent code $c$ given the data $x$ - and $L_{I}$ is the variational lower bound of the mutual information between the latent code and the observations.

In the practical implementation, there is another fully-connected layer to output parameters for the conditional distribution $Q$ (negligible computation ontop of regular GAN structures). Q is represented with a softmax non-linearity for a categorical latent code. For a continuous latent code, the authors assume a factored Gaussian.

source: [source](http://arxiv.org/abs/1606.03657v1)
# [Sparse Autoencoder](https://paperswithcode.com/method/sparse-autoencoder)
![](./img/Screen_Shot_2020-06-28_at_3.36.11_PM_wfLA8dB.png)

A **Sparse Autoencoder** is a type of autoencoder that employs sparsity to achieve an information bottleneck. Specifically the loss function is constructed so that activations are penalized within a layer. The sparsity constraint can be imposed with L1 regularization or a KL divergence between expected average neuron activation to an ideal distribution $p$.

Image: [Jeff Jordan](https://www.jeremyjordan.me/autoencoders/). Read his blog post (click) for a detailed summary of autoencoders.

# [GLOW](https://paperswithcode.com/method/glow)
![](./img/Screen_Shot_2020-06-28_at_8.43.24_PM_tNckkOB.png)

**GLOW** is a type of flow-based generative model that is based on an invertible $1 \times 1$ convolution. This builds on the flows introduced by [NICE](https://paperswithcode.com/method/nice) and [RealNVP](https://paperswithcode.com/method/realnvp). It consists of a series of steps of flow, combined in a multi-scale architecture; see the Figure to the right. Each step of flow consists of Act Normalization followed by an **invertible $1 \times 1$ convolution** followed by an affine coupling layer.

source: [source](http://arxiv.org/abs/1807.03039v2)
# [Deep Boltzmann Machine](https://paperswithcode.com/method/deep-boltzmann-machine)
![](./img/Screen_Shot_2020-05-28_at_3.03.43_PM_3zdwn5r.png)

A **Deep Boltzmann Machine (DBM)** is a three-layer generative model. It is similar to a Deep Belief Network, but instead allows bidirectional connections in the bottom layers. Its energy function is  as an extension of the energy function of the RBM:

$$ E\left(v, h\right) = -\sum^{i}_{i}v_{i}b_{i} - \sum^{N}_{n=1}\sum_{k}h_{n,k}b_{n,k}-\sum_{i, k}v_{i}w_{ik}h_{k} - \sum^{N-1}_{n=1}\sum_{k,l}h_{n,k}w_{n, k, l}h_{n+1, l}$$

for a DBM with $N$ hidden layers.

Source: [On the Origin of Deep Learning](https://arxiv.org/pdf/1702.07800.pdf)

# [Beta-VAE](https://paperswithcode.com/method/beta-vae)
![](./img/Screen_Shot_2020-06-28_at_4.00.13_PM.png)

**Beta-VAE** is a type of variational autoencoder that seeks to discovered disentangled latent factors. It modifies [VAEs](https://paperswithcode.com/method/vae) with an adjustable hyperparameter $\beta$ that balances latent channel capacity and independence constraints with reconstruction accuracy. The idea is to maximize the probability of generating the real data while keeping the distance between the real and estimated distributions small, under a threshold $\epsilon$. We can use the Kuhn-Tucker conditions to write this as a single equation:

$$ \mathcal{F}\left(\theta, \phi, \beta; \mathbf{x}, \mathbf{z}\right) = \mathbb{E}_{q_{\phi}\left(\mathbf{z}|\mathbf{x}\right)}\left[\log{p}_{\theta}\left(\mathbf{x}\mid\mathbf{z}\right)\right] - \beta\left[D_{KL}\left(\log{q}_{\theta}\left(\mathbf{z}\mid\mathbf{x}\right)||p\left(\mathbf{z}\right)\right) - \epsilon\right]$$

where the KKT multiplier $\beta$ is the regularization coefficient that constrains the capacity of the latent channel $\mathbf{z}$ and puts implicit independence pressure on the learnt posterior due to the isotropic nature of the Gaussian prior $p\left(\mathbf{z}\right)$.

We write this again using the complementary slackness assumption to get the Beta-VAE formulation:

$$ \mathcal{F}\left(\theta, \phi, \beta; \mathbf{x}, \mathbf{z}\right) \geq  \mathcal{L}\left(\theta, \phi, \beta; \mathbf{x}, \mathbf{z}\right) = \mathbb{E}_{q_{\phi}\left(\mathbf{z}|\mathbf{x}\right)}\left[\log{p}_{\theta}\left(\mathbf{x}\mid\mathbf{z}\right)\right] - \beta{D}_{KL}\left(\log{q}_{\theta}\left(\mathbf{z}\mid\mathbf{x}\right)||p\left(\mathbf{z}\right)\right)$$

source: [source](https://openreview.net/forum?id=Sy2fzU9gl)
# [LSGAN](https://paperswithcode.com/method/lsgan)
![](./img/Screen_Shot_2020-07-05_at_4.25.10_PM_xR8HCAz.png)

**LSGAN**, or **Least Squares GAN**, is a type of generative adversarial network that adopts the least squares loss function for the discriminator. Minimizing the objective function of LSGAN yields minimizing the Pearson $\chi^{2}$ divergence. The objective function can be defined as:

$$ \min_{D}V_{LSGAN}\left(D\right) = \frac{1}{2}\mathbb{E}_{\mathbf{x} \sim p_{data}\left(\mathbf{x}\right)}\left[\left(D\left(\mathbf{x}\right) - b\right)^{2}\right] + \frac{1}{2}\mathbb{E}_{\mathbf{z}\sim p_{data}\left(\mathbf{z}\right)}\left[\left(D\left(G\left(\mathbf{z}\right)\right) - a\right)^{2}\right] $$

$$ \min_{G}V_{LSGAN}\left(G\right) = \frac{1}{2}\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}\left(\mathbf{z}\right)}\left[\left(D\left(G\left(\mathbf{z}\right)\right) - c\right)^{2}\right] $$

where $a$ and $b$ are the labels for fake data and real data and $c$ denotes the value that $G$ wants $D$ to believe for fake data.

source: [source](http://arxiv.org/abs/1611.04076v3)
# [BiGAN](https://paperswithcode.com/method/bigan)
![](./img/Screen_Shot_2020-07-05_at_12.05.23_PM_vNHSXna.png)

A **BiGAN**, or **Bidirectional GAN**, is a type of generative adversarial network where the generator  not only maps latent samples to generated data, but also has an inverse mapping from data to the latent representation. The motivation is to make a type of GAN that can learn rich representations for us in applications like unsupervised learning.

In addition to the generator $G$ from the standard [GAN](https://paperswithcode.com/method/gan) framework, BiGAN includes an encoder $E$ which maps data $\mathbf{x}$ to latent representations $\mathbf{z}$. The BiGAN discriminator $D$ discriminates not only in data space ($\mathbf{x}$ versus $G\left(\mathbf{z}\right)$), but jointly in data and latent space (tuples $\left(\mathbf{x}, E\left(\mathbf{x}\right)\right)$ versus $\left(G\left(z\right), z\right)$), where the latent component is either an encoder output $E\left(\mathbf{x}\right)$ or a generator input $\mathbf{z}$.

source: [source](http://arxiv.org/abs/1605.09782v7)
# [RealNVP](https://paperswithcode.com/method/realnvp)
![](./img/Screen_Shot_2020-06-28_at_7.14.22_PM.png)

**RealNVP** is a generative model that utilises real-valued non-volume preserving (real NVP) transformations for density estimation. The model can perform efficient and exact inference, sampling and log-density estimation of data points.

source: [source](http://arxiv.org/abs/1605.08803v3)
# [StyleGAN2](https://paperswithcode.com/method/stylegan2)
![](./img/Screen_Shot_2020-06-29_at_9.55.21_PM.png)

**StyleGAN2** is a generative adversarial network that builds on [StyleGAN](https://paperswithcode.com/method/stylegan) with several improvements. First, adaptive instance normalization is redesigned and replaced with a normalization technique called weight demodulation. Secondly, an improved training scheme upon progressively growing is introduced, which achieves the same goal - training starts by focusing on low-resolution images and then progressively shifts focus to higher and higher resolutions - without changing the network topology during training. Additionally, new types of regularization like lazy regularization and path length regularization are proposed.

source: [source](https://arxiv.org/abs/1912.04958v2)
# [WGAN GP](https://paperswithcode.com/method/wgan-gp)
![](./img/Screen_Shot_2020-05-25_at_3.01.45_PM.png)

**Wasserstein GAN + Gradient Penalty**, or **WGAN-GP**, is a generative adversarial network that uses the Wasserstein loss formulation plus a gradient norm penalty to achieve Lipschitz continuity.

The original [WGAN](https://paperswithcode.com/method/wgan) uses weight clipping to achieve 1-Lipschitz functions, but this can lead to undesirable behaviour by creating pathological value surfaces and capacity underuse, as well as gradient explosion/vanishing without careful tuning of the weight clipping parameter $c$.

A Gradient Penalty is a soft version of the Lipschitz constraint, which follows from the fact that functions are 1-Lipschitz iff the gradients are of norm at most 1 everywhere. The squared difference from norm 1 is used as the gradient penalty.

source: [source](http://arxiv.org/abs/1704.00028v3)
# [SNGAN](https://paperswithcode.com/method/sngan)
![](./img/Screen_Shot_2020-07-03_at_11.31.16_AM_npFG865.png)

**SNGAN**, or **Spectrally Normalised GAN**, is a type of generative adversarial network that uses spectral normalization, a type of weight normalization, to stabilise the training of the discriminator.

source: [source](http://arxiv.org/abs/1802.05957v1)
# [LAPGAN](https://paperswithcode.com/method/lapgan)
![](./img/Screen_Shot_2020-07-03_at_11.13.02_AM_kegkITL.png)

A **LAPGAN**, or **Laplacian Generative Adversarial Network**, is a type of generative adversarial network that has a Laplacian pyramid representation. In the sampling procedure following training, we have a set of generative convnet models {$G_{0}, \dots , G_{K}$}, each of which captures the distribution of coefficients $h_{k}$ for natural images at a different level of the Laplacian pyramid. Sampling an image is akin to a reconstruction procedure, except that the generative
models are used to produce the $h_{k}$’s:

$$ \tilde{I}_{k} = u\left(\tilde{I}_{k+1}\right) + \tilde{h}_{k} = u\left(\tilde{I}_{k+1}\right) + G_{k}\left(z_{k}, u\left(\tilde{I}_{k+1}\right)\right)$$

The recurrence starts by setting $\tilde{I}_{K+1} = 0$ and using the model at the final level $G_{K}$ to generate a residual image $\tilde{I}_{K}$ using noise vector $z_{K}$: $\tilde{I}_{K} = G_{K}\left(z_{K}\right)$. Models at all levels except the final are conditional generative models that take an upsampled version of the current image $\tilde{I}_{k+1}$ as a conditioning variable, in addition to the noise vector $z_{k}$.

The generative models {$G_{0}, \dots, G_{K}$} are trained using the CGAN approach at each level of the pyramid. Specifically, we construct a Laplacian pyramid from each training image $I$. At each level we make a stochastic choice (with equal probability) to either (i) construct the coefficients $h_{k}$ either using the standard Laplacian pyramid coefficient generation procedure or (ii) generate them using $G_{k}:

$$ \tilde{h}_{k} = G_{k}\left(z_{k}, u\left(I_{k+1}\right)\right) $$

Here $G_{k}$ is a convnet which uses a coarse scale version of the image $l_{k} = u\left(I_{k+1}\right)$ as an input, as well as noise vector $z_{k}$. $D_{k}$ takes as input $h_{k}$ or $\tilde{h}_{k}$, along with the low-pass image $l_{k}$ (which is explicitly added to $h_{k}$ or $\tilde{h}_{k}$ before the first convolution layer), and predicts if the image was real or
generated. At the final scale of the pyramid, the low frequency residual is sufficiently small that it
can be directly modeled with a standard GAN: $\tilde{h}_{K} = G_{K}\left(z_{K}\right)$ and $D_{K}$ only has $h_{K}$ or $\tilde{h}_{K}$ as input.

Breaking the generation into successive refinements is the key idea. We give up any “global” notion of fidelity; an attempt is never made to train a network to discriminate between the output of a cascade and a real image and instead the focus is on making each step plausible.

source: [source](http://arxiv.org/abs/1506.05751v1)
# [ProGAN](https://paperswithcode.com/method/progan)
![](./img/Screen_Shot_2020-06-28_at_11.57.45_PM_OvA9EvH.png)

**ProGAN**, or **Progressively Growing GAN**, is a generative adversarial network that utilises a progressively growing training approach. The idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses.

source: [source](http://arxiv.org/abs/1710.10196v3)
# [BigBiGAN](https://paperswithcode.com/method/bigbigan)
![](./img/Screen_Shot_2020-07-05_at_12.16.34_PM_gglXYpk.png)

**BigBiGAN** is a type of [BiGAN](https://paperswithcode.com/method/bigan) with a [BigGAN](https://paperswithcode.com/method/biggan) image generator. The authors initially used ResNet as a baseline for the encoder $\mathcal{E}$ followed by a 4-layer MLP with skip connections, but they experimented with RevNets and found they outperformed with increased network width, so opted for this type of encoder for the final architecture.

source: [source](https://arxiv.org/abs/1907.02544v2)
# [Contractive Autoencoder](https://paperswithcode.com/method/contractive-autoencoder)
![](./img/Screen_Shot_2020-06-28_at_3.55.45_PM_Au3MiXB.png)

A **Contractive Autoencoder** is an autoencoder that adds a penalty term to the classical reconstruction cost function. This penalty term corresponds to the Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input. This penalty term results in a localized space contraction which in turn yields robust features on the activation layer. The penalty helps to carve a representation that better captures the local directions of variation dictated by the data, corresponding to a lower-dimensional non-linear manifold, while being more invariant to the vast majority of directions orthogonal to the manifold.

# [ALI](https://paperswithcode.com/method/ali)
![](./img/Screen_Shot_2020-07-04_at_10.55.33_PM_YvewMxz.png)

**Adversarially Learned Inference (ALI)** is a generative modelling approach that casts the learning of both an inference machine (or encoder) and a deep directed generative model (or decoder) in an GAN-like adversarial framework. A discriminator is trained to discriminate joint samples of the data and the corresponding latent variable from the encoder (or approximate posterior) from joint samples from the decoder while in opposition, the encoder and the decoder are trained together to fool the discriminator. Not is the discriminator asked to distinguish synthetic samples from real data, but it is required it to distinguish between two joint distributions over the data space and the latent variables.

An ALI differs from a [GAN](https://paperswithcode.com/method/gan) in two ways:

- The generator has two components: the encoder, $G_{z}\left(\mathbf{x}\right)$, which maps data samples $x$ to $z$-space, and the decoder $G_{x}\left(\mathbf{z}\right)$, which maps samples from the prior $p\left(\mathbf{z}\right)$ (a source of noise) to the input space.
<li>The discriminator is trained to distinguish between joint pairs $\left(\mathbf{x}, \tilde{\mathbf{z}} = G_{\mathbf{x}}\left(\mathbf{x}\right)\right)$ and $\left(\tilde{\mathbf{x}} =
G_{x}\left(\mathbf{z}\right), \mathbf{z}\right)$, as opposed to marginal samples $\mathbf{x} \sim q\left(\mathbf{x}\right)$ and $\tilde{\mathbf{x}} ∼ p\left(\mathbf{x}\right)$.</li>

source: [source](http://arxiv.org/abs/1606.00704v3)
# [PixelRNN](https://paperswithcode.com/method/pixelrnn)
![](./img/Screen_Shot_2020-05-24_at_12.06.56_AM.png)

**PixelRNNs** are generative neural networks that sequentially predicts the pixels in an image along the two spatial dimensions. They model the discrete probability of the raw pixel values and encode the complete set of dependencies in the image. Variants include the Row LSTM and the Diagonal BiLSTM, that scale more easily to larger datasets. Pixel values are treated as discrete random variables by using a softmax layer in the conditional distributions. Masked convolutions are employed to allow PixelRNNs to model full dependencies between the color channels.

source: [source](http://arxiv.org/abs/1601.06759v3)
# [CS-GAN](https://paperswithcode.com/method/cs-gan)
![](./img/Screen_Shot_2020-07-04_at_8.23.42_PM_iVNJt1z.png)

**CS-GAN** is a type of generative adversarial network that uses a form of deep compressed sensing, and latent optimisation, to improve the quality of generated samples.

source: [source](https://arxiv.org/abs/1905.06723v2)
# [BigGAN-deep](https://paperswithcode.com/method/biggan-deep)
![](./img/Screen_Shot_2020-07-04_at_4.41.35_PM_ERT4XZ3.png)

**BigGAN-deep** is a deeper version (4x) of [BigGAN](https://beta.paperswithcode.com/method/biggan).  The main difference is a slightly differently designed residual block. Here the $z$ vector is concatenated with the conditional vector without splitting it into chunks.  It is also based on residual blocks with bottlenecks. BigGAN-deep uses a different strategy than BigGAN aimed at preserving identity throughout the skip connections. In G, where the number of channels needs to be reduced, BigGAN-deep simply retains the first group of channels and drop the rest to produce the required number of channels. In D, where the number of channels should be increased, BigGAN-deep passes the input channels unperturbed, and concatenates them with the remaining channels produced by a 1 × 1 convolution. As far as the
network configuration is concerned, the discriminator is an exact reflection of the generator. 

There are two blocks at each resolution (BigGAN uses one), and as a result BigGAN-deep is four times
deeper than BigGAN. Despite their increased depth, the BigGAN-deep models have significantly
fewer parameters mainly due to the bottleneck structure of their residual blocks.

source: [source](http://arxiv.org/abs/1809.11096v2)
# [IAN](https://paperswithcode.com/method/ian)
![](./img/Screen_Shot_2020-07-03_at_2.08.12_PM_kGZdxK0.png)

The **Introspective Adversarial Network (IAN)** is a hybridization of [GANs](https://paperswithcode.com/method/gan) and [VAEs](https://paperswithcode.com/method/vae) that leverages the power of the adversarial objective while maintaining the VAE’s efficient inference mechanism. It uses the discriminator of the GAN, $D$, as a feature extractor for an inference subnetwork, $E$, which is implemented as a fully-connected layer on top of the final convolutional layer of the discriminator. We infer latent values $Z \sim E\left(X\right) = q\left(Z\mid{X}\right)$ for reconstruction and sample random values $Z \sim p\left(Z\right)$ from a standard normal for random image generation using the generator network, $G$.

Three distinct loss functions are used:

- $\mathcal{L}_{img}$, the L1 pixel-wise reconstruction loss, which is preferred to the L2 reconstruction loss for its higher average gradient.
- $\mathcal{L_{feature}}$, the feature-wise reconstruction loss, evaluated as the L2 difference between the original and reconstruction in the space of the hidden layers of the discriminator.
<li>$\mathcal{L}_{adv}$, the ternary adversarial loss, a modification of the adversarial loss that forces the discriminator to label a sample as real, generated, or reconstructed (as opposed to a binary
real vs. generated label).</li>

Including the VAE’s KL divergence between the inferred latents $E\left(X\right)$ and the prior $p\left(Z\right)$, the loss function for the generator and encoder network is thus:

$$\mathcal{L}_{E, G} = \lambda_{adv}\mathcal{L}_{G_{adv}} + \lambda_{img}\mathcal{L}_{img}  + \lambda_{feature}\mathcal{L}_{feature}  + D_{KL}\left(E\left(X\right) || p\left(Z\right)\right) $$

Where the $\lambda$ terms weight the relative importance of each loss. We set $\lambda_{img}$ to 3 and leave the other terms at 1. The discriminator is updated solely using the ternary adversarial loss. During each training step, the generator produces reconstructions $G\left(E\left(X\right)\right)$ (using the standard VAE reparameterization trick) from data $X$ and random samples $G\left(Z\right)$, while the discriminator observes $X$ as well as the reconstructions and random samples, and both networks are simultaneously updated.

source: [source](http://arxiv.org/abs/1609.07093v3)
# [NVAE](https://paperswithcode.com/method/nvae)
![](./img/NVAE_VdRVsB1.png)

**NVAE**, or **Nouveau VAE**, is deep, hierarchical variational autoencoder. It can be trained with the original [VAE](https://paperswithcode.com/method/vae) objective, unlike alternatives such as [VQ-VAE-2](https://paperswithcode.com/method/vq-vae-2). NVAE’s design focuses on tackling two main challenges: (i) designing expressive neural
networks specifically for VAEs, and (ii) scaling up the training to a large number of hierarchical
groups and image sizes while maintaining training stability.

To tackle long-range correlations in the data, the model employs hierarchical multi-scale modelling. The generative model starts from a small spatially arranged latent variables as $\mathbf{z}_{1}$ and samples from the hierarchy group-by-group while gradually doubling the spatial dimensions. This multi-scale approach enables NVAE to capture global long-range correlations at the top of the hierarchy and local fine-grained dependencies at the lower groups.

Additional design choices include the use of residual cells for the generative models and the encoder, which employ a number of tricks and modules to achieve good performance, and the use of residual normal distributions to smooth optimization. See the components section for more details.

source: [source](https://arxiv.org/abs/2007.03898v1)
# [TGAN](https://paperswithcode.com/method/tgan)
![](./img/Screen_Shot_2020-07-05_at_8.41.09_PM_t1WUmee.png)

**TGAN** is a type of generative adversarial network that is capable of learning representation from an unlabeled video dataset and producing a new video. The generator consists of two sub networks
called a temporal generator and an image generator. Specifically, the temporal generator first yields a set of latent variables, each of which corresponds to a latent variable for the image generator. Then, the image generator transforms these latent variables into a video which has the same number of frames as the variables. The model comprised of the temporal and image generators can not only enable to efficiently capture the time series, but also be easily extended to frame interpolation. The authors opt for a WGAN as the basic GAN structure and objective, but use singular value clipping to enforce the Lipschitz constraint.

source: [source](http://arxiv.org/abs/1611.06624v3)
# [TrIVD-GAN](https://paperswithcode.com/method/trivd-gan)
![](./img/Screen_Shot_2020-07-05_at_11.18.41_PM_dWv48xR.png)

**TrIVD-GAN**, or **Transformation-based &amp; TrIple Video Discriminator GAN**, is a type of generative adversarial network for video generation that builds upon [DVD-GAN](https://paperswithcode.com/method/dvd-gan). Improvements include a novel transformation-based recurrent unit (the TSRU) that makes the generator more expressive, and an improved discriminator architecture. 

In contrast with DVD-GAN, TrIVD-GAN has an alternative split for the roles of the discriminators, with $\mathcal{D}_{S}$ judging per-frame global structure, while $\mathcal{D}_{T}$ critiques local spatiotemporal structure. This is achieved by downsampling the $k$ randomly sampled frames fed to $\mathcal{D}_{S}$ by a factor $s$, and cropping $T \times H/s \times W/s$ clips inside the high resolution video fed to $\mathcal{D}_{T}$, where $T, H, W, C$ correspond to time, height, width and channel dimension of the input. This further reduces the number of pixels to process per video,
from $k \times H \times W + T \times H/s \times W/s$ to $\left(k + T\right) \times H/s \times W/s$.

source: [source](https://arxiv.org/abs/2003.04035v1)
# [k-Sparse Autoencoder](https://paperswithcode.com/method/k-sparse-autoencoder)
![](./img/Screen_Shot_2020-06-28_at_3.46.28_PM.png)

**k-Sparse Autoencoders** are autoencoders with linear activation function, where in hidden layers only the $k$ highest activities are kept. This achieves exact sparsity in the hidden representation. Backpropagation only goes through the the top $k$ activated units. This can be achieved with a ReLU layer with an adjustable threshold.

source: [source](http://arxiv.org/abs/1312.5663v2)
# [LOGAN](https://paperswithcode.com/method/logan)
![](./img/Screen_Shot_2020-07-04_at_8.28.02_PM_XgI4ng6.png)

**LOGAN** is a generative adversarial network that uses a latent optimization approach using natural gradient descent (NGD). For the Fisher matrix in NGD, the authors use the empirical Fisher $F'$ with Tikhonov damping:

$$ F' = g \cdot g^{T} + \beta{I} $$

They also use Euclidian Norm regularization for the optimization step.

For LOGAN's base architecture, BigGAN-deep is used with a few modifications: increasing the size of the latent source from $186$ to $256$, to compensate the randomness of the source lost
when optimising $z$. 2, using the uniform distribution $U\left(−1, 1\right)$ instead of the standard normal distribution $N\left(0, 1\right)$ for $p\left(z\right)$ to be consistent with the clipping operation, using  leaky ReLU (with the slope of 0.2 for the negative part) instead of ReLU as the non-linearity for smoother gradient flow for $\frac{\delta{f}\left(z\right)}{\delta{z}}$ .

source: [source](https://arxiv.org/abs/1912.00953v2)
# [NICE](https://paperswithcode.com/method/nice)
![](./img/Screen_Shot_2020-06-28_at_8.31.53_PM.png)

**NICE**, or **Non-Linear Independent Components Estimation** is a framework for modeling complex high-dimensional densities. It is based on the idea that a good representation is one in which the data has a distribution that is easy to model. For this purpose, a non-linear deterministic transformation of the data is learned that maps it to a latent space so as to make the transformed data conform to a factorized distribution, i.e., resulting in independent latent variables.  The transformation is parameterised so that computing the determinant of the Jacobian and inverse Jacobian is trivial, yet it maintains the ability to learn complex non-linear transformations, via a composition of simple building blocks, each based on a deep neural network. The training criterion is simply the exact log-likelihood. The transformation used in NICE is the affine coupling layer without the scale term, known as additive coupling layer:

$$ y_{I_{2}} = x_{I_{2}} + m\left(x_{I_{1}}\right) $$

$$ x_{I_{2}} = y_{I_{2}} + m\left(y_{I_{1}}\right) $$

source: [source](http://arxiv.org/abs/1410.8516v6)
# [PresGAN](https://paperswithcode.com/method/presgan)
![](./img/Screen_Shot_2020-06-29_at_10.44.07_PM.png)

**Prescribed GANs** add noise to the output of a density network and optimize an entropy-regularized adversarial loss. The added noise renders tractable approximations of the predictive log-likelihood and stabilizes the training procedure. The entropy regularizer encourages PresGANs to capture all the modes of the data distribution. Fitting PresGANs involves computing the intractable gradients of the entropy regularization term; PresGANs sidestep this intractability using
unbiased stochastic estimates.

source: [source](https://arxiv.org/abs/1910.04302v1)
# [DVD-GAN](https://paperswithcode.com/method/dvd-gan)
![](./img/Screen_Shot_2020-07-05_at_9.04.09_PM_lIg8LSX.png)

**DVD-GAN** is a generative adversarial network for video generation built upon the [BigGAN](https://paperswithcode.com/method/biggan) architecture.

DVD-GAN uses two discriminators: a Spatial Discriminator $\mathcal{D}_{S}$ and a
Temporal Discriminator $\mathcal{D}_{T}$. $\mathcal{D}_{S}$ critiques single frame content and structure by randomly sampling $k$ full-resolution frames and judging them individually.  The temporal discriminator $\mathcal{D}_{T}$ must provide $G$ with the learning signal to generate movement (not evaluated by $\mathcal{D}_{S}$).

The input to $G$ consists of a Gaussian latent noise $z \sim N\left(0, I\right)$ and a learned linear embedding $e\left(y\right)$ of the desired class $y$. Both inputs are 120-dimensional vectors. $G$ starts by computing an affine transformation of $\left[z; e\left(y\right)\right]$ to a $\left[4, 4, ch_{0}\right]$-shaped tensor. $\left[z; e\left(y\right)\right]$ is used as the input to all class-conditional Batch Normalization layers
throughout $G$. This is then treated as the input (at each frame we would like to generate) to a Convolutional GRU.

This RNN is unrolled once per frame. The output of this RNN is processed by two residual blocks. The time dimension is combined with the batch dimension here, so each frame proceeds through the blocks independently. The output of these blocks has width and height dimensions which
are doubled (we skip upsampling in the first block). This is repeated a number of times, with the
output of one RNN + residual group fed as the input to the next group, until the output tensors have
the desired spatial dimensions. 

The spatial discriminator $\mathcal{D}_{S}$ functions almost identically to BigGAN’s discriminator. A score is calculated for each of the uniformly sampled $k$ frames (default $k = 8$) and the $\mathcal{D}_{S}$ output is the sum over per-frame scores. The temporal discriminator $\mathcal{D}_{T}$ has a similar architecture, but pre-processes the real or generated video with a $2 \times 2$ average-pooling downsampling function $\phi$. Furthermore, the first two residual blocks of $\mathcal{D}_{T}$ are 3-D, where every convolution is replaced with a 3-D convolution with a kernel size of $3 \times 3 \times 3$. The rest of the architecture follows BigGAN.

source: [source](https://arxiv.org/abs/1907.06571v2)
# [HDCGAN](https://paperswithcode.com/method/hdcgan)
![](./img/hdcgan_hCkSA8e.png)

In order to boost network convergence of DCGAN and achieve good-looking high-resolution results we propose a new layered network, HDCGAN, that incorporates current state-of-the-art techniques for this effect.

source: [source](http://arxiv.org/abs/1711.06491v12)
# [VQ-VAE-2](https://paperswithcode.com/method/vq-vae-2)
![](./img/Screen_Shot_2020-06-28_at_4.56.19_PM.png)

**VQ-VAE-2** is a type of variational autoencoder that combines a a two-level hierarchical VQ-VAE with a self-attention autoregressive model ([PixelCNN](https://paperswithcode.com/method/pixelcnn)) as a prior. The encoder and decoder architectures are kept simple and light-weight as in the original [VQ-VAE](https://paperswithcode.com/method/vq-vae), with the only difference that hierarchical multi-scale latent maps are used for increased resolution.

source: [source](https://arxiv.org/abs/1906.00446v1)
