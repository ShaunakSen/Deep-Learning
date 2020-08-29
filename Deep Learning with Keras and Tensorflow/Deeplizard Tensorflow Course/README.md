# Understanding MobileNet

Based on the explanation video: https://www.youtube.com/watch?v=HD9FnjVwU8g
Original paper link: https://arxiv.org/pdf/1704.04861.pdf

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