# GAN - Understanding the Basics

Based on the classic paper tutorial by **Yannic Kilcher**: https://www.youtube.com/watch?v=eyxmSmjmNS0
Original paper link: https://arxiv.org/abs/1406.2661

## Introduction:

Some background: Before GANs came into the picture, we did not really have that much breakthrough in generating new data.

We have 2 models:

1. A discriminative model (D): This decides whether or not a given data pt comes from the original data or from the fake distribution
2. A generative model (G): Creates this fake data (X)

So basically we sample either from the original data or G and D is supposed to decide if it comes from the data or from G

So we train D as a simple image classifier

This came at a time when people already knew about CNNs. *The main idea here is: since AlexNet came about 2 years ago, we know how to train really good image classifiers. Can we harness this power to build new data?*

What differentiates this from Auto Encoders is that here there is no connection bw the data and the Generator

## Basic Approach

- the Generator (G) is not trained from the data directly
- The philosophy here is to use the power of discriminative models, which we know how to build in order to train the generator
- So the task of G is not to match any data pt. *Its task is to produce data pts that D classifies as real data.*  We can do that by backpropagating through D to G.
- So the training procedure for G is to **maximize the probability of D making a mistake, kind of like a minimax 2 player game**
- We reach a solution when G learns the training data totally and D can make nothing but a random guess whether it is real or fake (1/2 probability of being correct)

![https://i.imgur.com/hHNhEu8.png](https://i.imgur.com/hHNhEu8.png)

## Loss function - getting an intuition

![https://i.imgur.com/z0Vzt49.png](https://i.imgur.com/z0Vzt49.png)

There is a lot to this equation. Lets understand this slowly at a high level

- log(D(x)) is the log probability of data
- log(D(G(z)) is the log probability of the generated samples, where z is the input to the G network, which has a prior probability distribution.  z is the noise distribution, so p(z) is prior on the noise distribution and we sample a data from that noise distribution (NOTE: we are not sampling from the original data)
- So basically the network G takes an ip: z and turns it into something that resembles the original data to a certain extent. Then D takes that output and finally we compute D(G(z))).
- Now note that G(z) is a fake data pt. So ideally we want D(G(z)) to be low (from the perspective of D). So from the perspective of D we want to minimize D(G(z)) → min (log(D(G(z)) → max  negative (log(D(G(z)).
    - log(1-D(G(z)) → log(1) - log(D(G(z)) → -log(D(G(z))
- Ok so for the 2nd part of the eqn, from the point of view of D, it is trying to maximize it. Now similarly G tries to fool D. So it wants D to believe that D(G(z))  is high. So it will try to minimize the 2nd part of the equation
- Now lets come to the first part of the eqn. Here x is simply real data sampled from a prior distribution. D(x) should be high from the perspective of D. There is no G in this term. So it does not seem as if G ever sees real data (x)
- Now if we look at the entire eqn, for D its better to maximize both first and second parts of the eqn and it makes sense for G to minimize the 2nd part. So the eqn as a whole should be maximized from the perspective of D and minimized from the perspective of G
- Now lets look more closely at the LHS. G is trying to minimize whatever is the max D. So G is essentially  trying to minimize against the best possible D. So this is like a min(max) game and not a max(min) game
- Now for the second part even if we have -log(D) instead of log(1-D(..)) the underlying objective remains same. The key takeaway here is that we can play around with this to get the best formulation that achieves the best practical results and that is why today we have so many applications of GANs

    ![https://i.imgur.com/PPVyjTX.png](https://i.imgur.com/PPVyjTX.png)

### Intuition behind the loss function

We have the z-space which is sampled uniformly

From z→x is G

- The blue curve is the fake data distribution
- The black curve is the real data distribution
- D is supposed to tell us where there is real data and where there is fake data
- Not initially D is half-trained so it is not very good (step a)

     

    ![https://i.imgur.com/NYQiksE.png](https://i.imgur.com/NYQiksE.png)

- But as we max D as based on the eqn above we come to step b where D has become very good at basically classifying real data from fake data (e.g: real img vs fake img of dog using a CNN). Note, we already know how to do this
- Now we train G according to max D. So the grad of D is up the slope and we shift our green curve in that direction:

    ![https://i.imgur.com/dSWRKpE.png](https://i.imgur.com/dSWRKpE.png)

- So in first step we trained D (max D), next we trained G (min G)
- So now G curve has been shifted along grad of D (blue curve)

    ![https://i.imgur.com/emd5Z0f.png](https://i.imgur.com/emd5Z0f.png)

- Note that at any stage the green curve (G) has not seen the black curve (real data). The G simply sees the blue curve and goes along the grad of the blue curve and that is how the data generated by G starts to closely match the real data
- As we train more and more the op of G matches the real data so closely that D can no longer identify bw the two

    ![https://i.imgur.com/dCwit35.png](https://i.imgur.com/dCwit35.png)

- This can happen if G simply remembers the training data, but this is prevented by a few things. Note that G generates continuous data, and as we can see from the dots of the black curve, the real data is a bunch of discrete data pts. So there will be gaps as represented by the circle above.
    - Maybe since the G generates continuous data, we can obtain smooth transitions bw fake images as in applications of GANs
- Also G never really sees the training data directly. So it does not remember it exactly

### GAN Algorithm

![https://i.imgur.com/XZROMBT.png](https://i.imgur.com/XZROMBT.png)

So now image D and G are two NNs

We sample a bunch of noise samples and give it to G, it gives an op. We feed this op to D and get D(G(z))

Also we feed a bunch of real data  samples to D and obtain D(x)

We calculate the loss as shown and find gradient wrt theta_d where probably theta_d means the wts of D

We ascend the gradient i.e try to maximize the objective in this case

Thus in this way we update (train) the discriminator

After this is done, we sample a bunch of noise samples and pass it through G. Now we have to minimize the objective function wrt the weights in G, so the term of LHS equals 0 as there is no G there. So we are only left with the term:

![https://i.imgur.com/xzxddVC.png](https://i.imgur.com/xzxddVC.png)

We have to minimize this so we descend the gradient. Thus now G will be trained to create samples that D will fail to understand

Also in the G network they input the noise only at the lowest layer. So if we imagine G to be a DNN which ops an image. We just put it in the first layer, but we could have put it at any/all layer(s). but here we input noise at the very beginning as a vector and let the NN produce the image from that

Also there is a bunch of theoretical analysis in the paper which basically concludes that there exists a global optimum point which occurs when the generator captures the data distribution perfectly. This can and will be achieved if we can optimize the prob distributions with reasonable degree of freedom and NNs do give us good hope that they will be achieved in practical scenarios