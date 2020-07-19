# GAN Notes

Based on the classic paper tutorial by **Yannic Kilcher**: https://www.youtube.com/watch?v=eyxmSmjmNS0
Original paper link: https://arxiv.org/abs/1406.2661

## Introduction:

Some background: Before GANs came into the picture, we did not really have that much breakthrough in generating new data.

We have 2 models:

1. A discriminative model (D): This decides whether or not a given data pt comes from the original data or from the fake distribution
2. A generative model (G): Creates this fake data (X)

So basically we sample either from the original data or G and D is supposed to decide if it comes from the data or from G

So we train D as a simple image classifier

This came at a time when people already knew about CNNs. T*he main idea here is: since AlexNet came about 2 years ago, we know how to train really good image classifiers. Can we harness this power to build new data?*

What differentiates this from Auto Encoders is that here there is no connection bw the data and the Generator

## Basic Approach

- the Generator (G) is not trained from the data directly
- The philosophy here is to use the power of discriminative models, which we know how to build in order to train the generator
- So the task of G is not to match any data pt. *Its task is to produce data pts that D classifies as real data.*  We can do that by backpropagating through D to G.
- So the training procedure for G is to **maximize the probability of D making a mistake, kind of like a minimax 2 player game**

![https://i.imgur.com/hHNhEu8.png](https://i.imgur.com/hHNhEu8.png)

## Loss function - getting an intuition

![https://i.imgur.com/z0Vzt49.png](https://i.imgur.com/z0Vzt49.png)

There is a lot to this equation. Lets understand this slowly at a high level

- log(D(x)) is the log probability of data
- log(D(G(z)) is the log probability of the generated samples, where z is the input to the G network, which has a prior probability distribution.
- So basically the nw G takes an ip: z and turns it into something that resembles the original data to a certain extent. Then D takes that output and finally we compute D(G(z))).
- Now note that G(z) is a fake data pt. So ideally we want D(G(z)) to be low (from the perspective of D). So from the perspective of D we want to minimize D(G(z)) → min (log(D(G(z)) → max  negative (log(D(G(z)).
- log(1-D(G(z)) → log(1) - log(D(G(z))
- Ok so for the 2nd part of the eqn, from the point of view of D, it is trying to maximize it. Now similarly G tries to fool D. So it wants D to believe that D(G(z))  is high. So it will try to minimize the 2nd part of the equation
- Now lets come to the first part of the eqn. Here x is simply real data sampled from a prior distribution. D(x) should be high from the perspective of D. There is no G in this term.
- Now if we look at the entire eqn, for D its better to maximize both first and second parts of the eqn and it makes sense for G to minimize the 2nd part. So the eqn as a whole should be maximized from the perspective of D and minimized from the perspective of G
- Now lets look more closely at the LHS. G is trying to minimize whatever is the max D. So G is essentially  trying to minimize against the best possible D. So this is like a min(max) game and not a max(min) game
- Now for the second part even if we have -log(D) instead of log(1-D(..)) the underlying objective remains same. The key takeaway here is that we can play around with this to get the best formulation that achieves the best practical results and that is why today we have so many applications of GANs

    ![https://i.imgur.com/PPVyjTX.png](https://i.imgur.com/PPVyjTX.png)