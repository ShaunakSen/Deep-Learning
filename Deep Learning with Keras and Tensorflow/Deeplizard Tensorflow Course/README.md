# Understanding MobileNet

Based on the explanation video: https://www.youtube.com/watch?v=HD9FnjVwU8g
Original paper link: https://arxiv.org/pdf/1704.04861.pdf

TODO: Shufflenet v2 paper discussed mobilenet vs automated nw search models: read and add points https://www.youtube.com/watch?v=g2vlqhefADk

## Introduction

2 key ideas:

- Uses depthwise separable convolutions to build light weight deep neural networks
- We introduce two simple global hyperparameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem

The paper starts off with acknowledging the fact that CNNs have been a game-changer since the days of AlexNet. However the major progress so far has been attributed only to make the networks deeper and hence, more complicated. This increases training and prediction times, which is often unfeasible for real-time applications like robotics and self-driving cars

## Convolution operations across multiple input channels

Credits: https://d2l.ai/chapter_convolutional-neural-networks/channels.html

We understand that when we have a filter, that convolves across an image with a single ip channel, it creates a feature map and the number of op channels is equal to the number of filters applied

![https://deeplizard.com/images/convolution-animation-1.gif](https://deeplizard.com/images/convolution-animation-1.gif)

Credits: [https://deeplizard.com/learn/video/YRhxdVk_sIs](https://deeplizard.com/learn/video/YRhxdVk_sIs)

But what happens when we have multiple input channels, for example RGB images have 3 ip channels

When the input data contain multiple channels, we need to construct a convolution kernel with the same number of input channels as the input data, so that it can perform cross-correlation with the input data. So, for example, if number of input channels in the image is 2, the filter also needs to have 2 channels

![https://d2l.ai/_images/conv-multi-in.svg](https://d2l.ai/_images/conv-multi-in.svg)

Input: 3x3x2, kernel(filter): 2x2x2. Here we show the convolution across the first group of 2x2 pixels. Note that the feature map, has number of channels == no of filters (here 1). Basically, first channel of filter is applied to first channel of image and so on.. And the results are added

## Depth-wise separable convolutions

Credits: https://www.youtube.com/watch?v=T7o3xvJLuHk

Consider an image of shape: $D_{F} \times D_{F} \times M$, in the above example this was 3x3x2

We apply a convolution filter of size $D_{K} \times D_{K} \times M$, in the above example, this was 2x2x2

The op of this single filter will be $D_{G} \times D_{G} \times 1$ (a single feature map)

Now, if we apply N such filters we will have N feature maps, i.e the shape will be $D_{G} \times D_{G} \times N$

![https://i.imgur.com/SYPVmRv.png](https://i.imgur.com/SYPVmRv.png)

Let us explore the cost of this convolution operation:

This can be approximated by counting the number of multiplication operations required

Lets look at the previous diagram. For one convolution operation shown (0.0 + 1.1 + 3.2 + 4.3), we require 4, i.e 2^2 multiplications where 2 is the dimension of the filter. So for one filter, number of operations for 1 dimension = D_k^2, for M dimensions : D_k^2 x M. But this is just for one convolution. We perform 4 (D_G^2) such convolutions as we slide the filter over the input image.

So number of multiplications required: D_G^2 x D_K^2 x M (for 1 filter/kernel)

For N filters: N x D_G^2 x D_K^2 x M (this is the cost for the traditional convolution operation)

---

In depth wise convolution there are 2 steps:

1. Depth-wise convolution : Filtering stage
2. Point-wise convolution : Combining stage

### Depth-wise convolution: Filtering stage

In the standard convolution, we studied, the convolution was applied to add channels and we took the sum as output

Let us imagine an image with dimensions: D_F x D_F x M

In depth-wise convolution we have filters which have only one channel (we have M such filters). Assume each filter is of shape: D_K x D_K x 1

The filters have one channels as each of the filters are only applied to a single channel of the input

![https://i.imgur.com/pz2DloA.png](https://i.imgur.com/pz2DloA.png)

 

For each of these filters we apply convolution and get M channel outputs (remember number of op channels == no of filters) of shape D_G x D_G x 1

Stacking these M channels together, we obtain an op volume of D_G x D_G x M

![https://i.imgur.com/DLu2fzL.png](https://i.imgur.com/DLu2fzL.png)

### Point-wise convolutions: Combining stage

Here the input is the output of the previous step, i.e the volume of shape D_G x D_G x M

Here the filter shape used is 1x1xM and we have N such filters

So this filter is applied in the same manner as the traditional filters and each filter yields an op of shape D_G x D_G x 1

As we use N filters, final shape: D_G x D_G x N

![https://i.imgur.com/bQGexsP.png](https://i.imgur.com/bQGexsP.png)

Note that after depth-wise and point-wise we get the same N (D_G x _G ) feature maps

### Basic difference between standard and Depth-wise separable convolutions

![https://i.imgur.com/XLxNQR0.png](https://i.imgur.com/XLxNQR0.png)

In standard convolution, as we say, each filter has M channels where M = no of channels in the input image

As we convolve this filter across the image the filter is applied to each channel, we take the product and finally take the sum 

But in Depth-wise separable convolutions, we first take filters of 1 channel (M of these). Each of these goes through one channels at a time and creates separate feature maps for that channel

**So we independently find features in each channel**

After this, we stack these feature maps into a block: D_G x D_G x M

Next we take filter of shape 1x1xM. This 1x1xM fits 1 pixel at a time and computes how this one pixel is related to every other pixel in the M channels 

Lets reinforce the differences here:

https://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/

A [regular convolutional layer](https://machinethink.net/blog/convolutional-neural-networks-on-the-iphone-with-vggnet/) applies a convolution kernel (or “filter”) to all of the channels of the input image. It slides this kernel across the image and at each step performs a weighted sum of the input pixels covered by the kernel across all input channels.

The important thing is that the convolution operation *combines* the values of all the input channels. If the image has 3 input channels, then running a single convolution kernel across this image results in an output image with only 1 channel per pixel.

So for each input pixel, no matter how many channels it has, the convolution writes a new output pixel with only a single channel. (In practice we run many convolution kernels across the input image. Each kernel gets its own channel in the output.)

![https://machinethink.net/images/mobilenets/RegularConvolution.png](https://machinethink.net/images/mobilenets/RegularConvolution.png)

For an image with 3 channels, a depthwise convolution creates an output image that also has 3 channels. Each channel gets its own set of weights.

The purpose of the depthwise convolution is to filter the input channels. Think edge detection, color filtering, and so on.

![https://machinethink.net/images/mobilenets/DepthwiseConvolution.png](https://machinethink.net/images/mobilenets/DepthwiseConvolution.png)

The depthwise convolution is followed by a pointwise convolution. This really is the same as a regular convolution but with a 1×1 kernel:

![https://machinethink.net/images/mobilenets/PointwiseConvolution.png](https://machinethink.net/images/mobilenets/PointwiseConvolution.png)

In other words, this simply adds up all the channels (as a weighted sum). As with a regular convolution, we usually stack together many of these pointwise kernels to create an output image with many channels.

The purpose of this pointwise convolution is to *combine* the output channels of the depthwise convolution to create new features.

When we put these two things together — a depthwise convolution followed by a pointwise convolution — the result is called a depthwise separable convolution. A regular convolution does both filtering and combining in a single go, but with a depthwise separable convolution these two operations are done as separate steps.

### Depth-wise convolution: Complexity

Kernels have shape: D_K x D_K x 1

For 1 single convolution, number of multiplications is D_K^2, for 1 channel this filter slides over D_G to right and D_G to bottom, performing a convolution at each step

No of multiplications for 1 filter for 1 channel = D_G^2 x D_K^2

There are M such filters : M x D_G^2 x D_K^2

![https://i.imgur.com/0JL2IKS.png](https://i.imgur.com/0JL2IKS.png)

### Point-wise convolution: Complexity

Here shape of kernel = 1 x 1 x M

each kernel is applied to each channel of the input 

1 filter applied to one part of image results in 1 multiplication. This then slides over D_G to right and D_G to bottom, performing a convolution at each step, so for 1 filter channel , multiplications = D_G^2

For M such channels: multiplications = D_G^2 x M

Now for N filters: D_G^2 x M x N

Total multiplications: 

![https://i.imgur.com/sSwa2Fj.png](https://i.imgur.com/sSwa2Fj.png)

### Comparison

![https://i.imgur.com/u170b2w.png](https://i.imgur.com/u170b2w.png)

The above image tells us that if we want our output to have N = 1024 feature maps and a kernel size of 3, then standard convolution has 9 x more multiplications than depth-wise convolutions!

### Comparison: Number of parameters

The parameters here are the number of weights in the filters which have to be learned via backpropagation

![https://i.imgur.com/UQkSoqi.png](https://i.imgur.com/UQkSoqi.png)

## Some applications of Depth-wise separable convolutions

1. Multi-modal networks
2. Xception CNN network architecture 
3. MobileNets

![https://i.imgur.com/dBYkqKN.png](https://i.imgur.com/dBYkqKN.png)

Mult-adds : no of multiplications + additions