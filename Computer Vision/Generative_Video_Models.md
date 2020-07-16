# [TGAN](https://paperswithcode.com/method/tgan)
![](./img/Screen_Shot_2020-07-05_at_8.41.09_PM_t1WUmee.png)

**TGAN** is a type of generative adversarial network that is capable of learning representation from an unlabeled video dataset and producing a new video. The generator consists of two sub networks
called a temporal generator and an image generator. Specifically, the temporal generator first yields a set of latent variables, each of which corresponds to a latent variable for the image generator. Then, the image generator transforms these latent variables into a video which has the same number of frames as the variables. The model comprised of the temporal and image generators can not only enable to efficiently capture the time series, but also be easily extended to frame interpolation. The authors opt for a WGAN as the basic GAN structure and objective, but use singular value clipping to enforce the Lipschitz constraint.

source: [source](http://arxiv.org/abs/1611.06624v3)
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
# [TrIVD-GAN](https://paperswithcode.com/method/trivd-gan)
![](./img/Screen_Shot_2020-07-05_at_11.18.41_PM_dWv48xR.png)

**TrIVD-GAN**, or **Transformation-based &amp; TrIple Video Discriminator GAN**, is a type of generative adversarial network for video generation that builds upon [DVD-GAN](https://paperswithcode.com/method/dvd-gan). Improvements include a novel transformation-based recurrent unit (the TSRU) that makes the generator more expressive, and an improved discriminator architecture. 

In contrast with DVD-GAN, TrIVD-GAN has an alternative split for the roles of the discriminators, with $\mathcal{D}_{S}$ judging per-frame global structure, while $\mathcal{D}_{T}$ critiques local spatiotemporal structure. This is achieved by downsampling the $k$ randomly sampled frames fed to $\mathcal{D}_{S}$ by a factor $s$, and cropping $T \times H/s \times W/s$ clips inside the high resolution video fed to $\mathcal{D}_{T}$, where $T, H, W, C$ correspond to time, height, width and channel dimension of the input. This further reduces the number of pixels to process per video,
from $k \times H \times W + T \times H/s \times W/s$ to $\left(k + T\right) \times H/s \times W/s$.

source: [source](https://arxiv.org/abs/2003.04035v1)
