# [OHEM](https://paperswithcode.com/method/ohem)
![](./img/Screen_Shot_2020-06-08_at_11.34.39_AM_RvmJwmo.png)

Some object detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more
effective and efficient. **OHEM**, or **Online Hard Example Mining**, is a bootstrapping technique that modifies SGD to sample from examples in a non-uniform way depending on the current loss of each example under consideration. The method takes advantage of detection-specific problem structure in which each SGD mini-batch consists of only one or two images, but thousands of candidate examples. The candidate examples are subsampled according to a distribution
that favors diverse, high loss instances.

source: [source]http://arxiv.org/abs/1604.03540v1
# [ATSS](https://paperswithcode.com/method/atss)
![](./img/Screen_Shot_2020-06-13_at_3.26.16_PM.png)

**Adaptive Training Sample Selection**, or **ATSS**, is a method to automatically select positive and negative samples according to statistical characteristics of object. It bridges the gap between anchor-based and anchor-free detectors. 

For each ground-truth box $g$ on the image, we first find out its candidate positive samples. As described in Line $3$ to $6$, on each pyramid level, we select $k$ anchor boxes whose center are closest to the center of $g$ based on L2 distance. Supposing there are $\mathcal{L}$ feature pyramid levels, the ground-truth box $g$ will have $k\times\mathcal{L}$ candidate positive samples. After that, we compute the IoU between these candidates and the ground-truth $g$ as $\mathcal{D}_g$ in Line $7$, whose mean and standard deviation are computed as $m_g$ and $v_g$ in Line $8$ and Line $9$. With these statistics, the IoU threshold for this ground-truth $g$ is obtained as $t_g=m_g+v_g$ in Line $10$. Finally, we select these candidates whose IoU are greater than or equal to the threshold $t_g$ as final positive samples in Line $11$ to $15$. 

Notably ATSS also limits the positive samples' center to the ground-truth box as shown in Line $12$. Besides, if an anchor box is assigned to multiple ground-truth boxes, the one with the highest IoU will be selected. The rest are negative samples.

source: [source]https://arxiv.org/abs/1912.02424v4
# [IoU-Balanced Sampling](https://paperswithcode.com/method/iou-balanced-sampling)
![](./img/Screen_Shot_2020-06-24_at_9.42.43_PM_DwR5Ggy.png)

**IoU-Balanced Sampling** is hard mining method for object detection. Suppose we need to sample $N$ negative samples from $M$ corresponding candidates. The selected probability for each sample under random sampling is:

$$ p = \frac{N}{M} $$

To raise the selected probability of hard negatives, we evenly split the sampling interval into $K$ bins according to IoU. $N$ demanded negative samples are equally distributed to each bin. Then we select samples from them uniformly. Therefore, we get the selected probability under IoU-balanced sampling:

$$ p_{k} = \frac{N}{K}*\frac{1}{M_{k}}\text{ , } k\in\left[0, K\right)$$

where $M_{k}$ is the number of sampling candidates in the corresponding interval denoted by $k$. $K$ is set to 3 by default in our experiments.

The sampled histogram with IoU-balanced sampling is shown by green color in the Figure to the right. The IoU-balanced sampling can guide the distribution of training samples close to the one of hard negatives.

source: [source]http://arxiv.org/abs/1904.02701v1
