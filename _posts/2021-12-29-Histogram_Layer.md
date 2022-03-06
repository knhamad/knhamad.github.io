---
layout: single
title: "Histogram Layers for Texture Analysis"
date: 2022-01-25
tags: [deep learning, histograms, image classification, texture analysis]
---

## Problem Statement: Shortcomings of Convolutional Neural Networks
Convolutional neural networks (CNNs) have shown tremendous ability for a variety of applications.  One reason for the success of CNNs are that they excel at representing and detecting **structural textures**. However, they are **not** as effective at **statistical textures**. To illustrate this point, below is an example of different structural and statistical textures:
![Texture](/images/Textures_v2.jpg)
<br/>We can visually see the distinct differences between the different texture combinations. The structural textures are a checkboard, cross, and stripe. The statistical textures are shown through foreground pixels values sampled from multinomial, binomial, and constant distributions. A CNN could easily distinguish the structural textures, but would struggle with the statistical texures. 

### Why would a CNN struggle with statistical textures?
Structural texture approaches consist of defining a set of texture examples and an order of spatial positions for each exemplar [(Materka et al., 1998)](https://www.researchgate.net/profile/Andrzej-Materka/publication/249723259_Texture_Analysis_Methods_-_A_Review/links/02e7e51ef8d539a9da000000/Texture-Analysis-Methods-A-Review.pdf). On the other hand, statistical texture methods represent the data through parameters that characterize the distributions and correlation between the intensity and/or feature values in an image as opposed to understanding the structure of each texture [(Humeau-Heurtier, 2019)](https://ieeexplore.ieee.org/abstract/document/8600329). To understand differences between these two texture types, we created an example using the three distributions discussed above to generate statistical textures: multinomial, binomial, and constant classes. The first statistical class was sampled from a multinomial distribution with the following three intensity value choices which were of equal probability (p = 1/3): 64, 128, and 192. For the second statistical class, a binomial distribution (p = .5) was used where either an intensity value of 64 or 192 was selected for each foreground pixel. The last statistical class only contained foreground pixels with an intensity values of 128. Below we show the distribution of foreground pixel intensities after several trials (n=1800) and observe the means are the same. 
![Distribution Images](/images/Distributions.JPG)

<br/>A convolution is a weighted sum operator that uses spatial information to learn local relationships between pixels. Given enough samples from each distribution, the mean values are approximately the same and outputs from a convolution will be similar. We show example images convolved with a random convolution and observe the outputs are close in value. After many images are generated, the distribution of convolution outputs are similar as shown:
![Example convolution outputs](/images/Sampling_v2.gif)
<br/>The average operation is a special case of convolution where the all of the weights are equal to 1/number of data points. As a result, the CNN will struggle to capture a linear combination of pixels that learns the statistical information of the data (*i.e.*, cannot easily learn weights to discriminate statistical exemplars). Here is an example where if a 3x3 convolution is used, the model can easily learn weights to tell the cross and checkboard apart. However, if we sample from a different distribution and retain the same shape, a convolution operation cannot learn weights to distinguish this change as the convolution is unable to account for individual pixel intensity changes.
![CNN_Failure](/images/CNN_Failure_v2.jpg) 

<br/>In summary, CNNs are great at structural texture changes. However, for statistical textures, CNNs fail to characterize the spatial distribution of features and require more layers to capture the distributions of values in a region. 

## Method: Histogram Layer
### Standard Histogram Operation
The proposed solution is a **local** histogram layer. Instead of computing global histograms as done previously, the proposed histogram layer directly computes the local, spatial distribution of features for texture analysis, and parameters for the layer are estimated during backpropagation. Histograms perform a counting operation for values that fall within a certain range. Below is an example where we are counting the number of 1s, 2s, and 3s in local windows of the image:
<br/>![Local_Hist](/images/Stand_Hist.gif)

### Radial Basis Function Alternative 
<br/> The standard histogram operation is not differentiable and is brittle (*i.e.*, sensitive to parameter selection). However, a smooth approximation (*i.e.*, radial basis function or RBF) can be used and the parameters (*i.e.*, bin centers and widths) are updated via backpropagation. RBFs have a maximum value of 1 when the feature value is equal to the bin center and the minimum value approaches 0 as the feature value moves further from the bin center. Also, RBFs are more robust to small changes in bin centers and widths than the standard histogram operation because there is some allowance of error due to the soft binning assignments and the smoothness of the RBF. The means of the RBFs (&mu;<sub>bk</sub>) serve as the location of each bin (*i.e.*, bin centers), while the bandwidth (&gamma;<sub>bk</sub>) controls the spread of each bin (*i.e.*, bin widths), where *b* is the number of histogram bins and *k* is the feature channel in the input feature tensor. The normalized frequency count, *Y*<sub>rcbk</sub>, is computed with a sliding window of size *S*x*T*, and the binning operation for a histogram value in the *k*<sup>th</sup> channel of the input **X** is defined as:
<br/>![RBF_equation](/images/RBF_annotated.png)

We show an example of the local operation of the RBF with a 3x3 window (*S*,*T* = 3), 1 input channel (*K* = 1), and three bins (*b* = 3) below:
<br/>![Local_RBF](/images/RBF_Hist.gif)

## Implementation of Histogram Layer
<br/> Another advantage of the proposed method is that the histogram layer is easily implemented using pre-exisiting layers! Any deep learning framework (*e.g.*, Pytorch, TensorFlow) can be used to integerate the histogram layer into deep learning models. We show the configuration of the histogram layer using pre-existing layers and psuedocode below:
![Implementation](/images/Implementation_v2.png)
![Algorithm](/images/algorithm.jpg)

## Applications of Histogram Layer
There are several real-world applications for the histogram layer! Statistical texture approaches are vital because there are important properties these methods inherit. For the histogram layer, the proposed method is globally permutation and rotationally **invariant**. If applied to smaller spatial regions, the operation is permutation and rotationally **equivariant**. As a result, the local operation is not sensitive to the order of the data and is robust to image transformations such as rotation. Additionally, the histogram layer is also translationally-equivarient (just like CNNs). Examples of usefulness of these properties include automatic target recognition [(Frigui and Gader, 2009)](https://ieeexplore.ieee.org/document/4610973), remote sensing image classification [(Zhang, et. al., 2017)](https://www.mdpi.com/1424-8220/17/7/1474/htm), medical image diagnosis [(Bourouis, et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9324838), and crop quality management [(Yuan, et. al., 2019)](https://www.nature.com/articles/s41598-019-50480-x).

<br/>
[![Plants][9]][10]
<br/> This is an example image of grass from [GTOS-mobile](https://openaccess.thecvf.com/content_cvpr_2018/html/Xue_Deep_Texture_Manifold_CVPR_2018_paper.html). The image contains other textures and not only grass. Local histograms can distinguish portions of the image containing pure grass (top two histograms) or a mixture of other textures (bottom histogram) despite structual similarities. The histograms shown here are the distribution of intensity values from the red, green, and blue channels. Each histogram contains the aggregated intensity values (over the three color channels) in the corresponding image portion.


## Check Out the Code and Paper!
This [work](https://ieeexplore.ieee.org/document/9652037) was accepted to the **IEEE Transactions on Artificial Intelligence**! Our [code](https://github.com/GatorSense/Histogram_Layer) and [paper](https://arxiv.org/abs/2001.00215) are available! 

## Citation

### Plain Text:

J. Peeples and W. Xu and A. Zare, "Histogram Layers for Texture Analysis," 
in IEEE Transactions on Artificial Intelligence, DOI 10.1109/TAI.2021.3135804, 2021.

### BibTex:

@Article{Peeples2021Histogram,<br>
Title = {Histogram Layers for Texture Analysis},<br>
Author = {Peeples, Joshua and Xu, Weihuang  and Zare, Alina},<br>
Journal = {IEEE Transactions on Artificial Intelligence},<br>
Volume = {},<br>
Year = {2021},<br>
number={}<br>
pages={1-1}<br>
doi={10.1109/TAI.2021.3135804}}


## Links
<!-- [![alt text](image link)](web link) -->
[![ArXiv Paper][1]][2][![Github Repositor][3]][4][![IEEE Paper][5]][6][![Lab][7]][8]

[1]: /images/arxiv_25.jpg
[2]: https://arxiv.org/abs/2001.00215
[3]: /images/code_25.png
[4]: https://github.com/GatorSense/Histogram_Layer
[5]: /images/ieee_50.jpg
[6]: https://ieeexplore.ieee.org/document/9652037
[7]: /images/logo_50.png
[8]: https://faculty.eng.ufl.edu/machine-learning
[9]: /images/HistTextures_2.PNG
[10]: https://arxiv.org/pdf/2001.00215.pdf

<!-- [![ArXiv Paper](/images/arxiv.jpg"ArXiv Paper")](https://arxiv.org/abs/2001.00215)
[![Github Repository](/images/code.png"Code")](https://github.com/GatorSense/Histogram_Layer)
[![IEEE Paper](/images/ieee.jpg"IEEE Transactions on AI Paper")](https://ieeexplore.ieee.org/document/9652037)
[![Lab](/images/logo.png"GatorSense Lab Website")](https://faculty.eng.ufl.edu/machine-learning) -->



