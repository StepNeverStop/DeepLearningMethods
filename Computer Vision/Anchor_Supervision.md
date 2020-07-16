# [FreeAnchor](https://paperswithcode.com/method/freeanchor)
![](./img/new_3Dpipeline.jpg)

**FreeAnchor** is an anchor supervision method for object detection. Many CNN-based object detectors assign anchors for ground-truth objects under the restriction of object-anchor Intersection-over-Unit (IoU). In contrast, FreeAnchor is a learning-to-match approach that breaks the IoU restriction, allowing objects to match anchors in a flexible manner. It updates hand-crafted anchor assignment to free anchor matching by formulating detector training as a maximum likelihood estimation (MLE) procedure. FreeAnchor targets at learning features which best explain a class of objects in terms of both classification and localization.

source: [source](https://arxiv.org/abs/1909.02466v2)
