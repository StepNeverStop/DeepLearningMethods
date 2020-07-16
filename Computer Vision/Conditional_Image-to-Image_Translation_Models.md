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
