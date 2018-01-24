---
layout: post
title: Real World Applications
---

Probabilistic graphical models have numerous real-world applications, of which we give a brief overview here.

# Probabilistic Models of Images

Consider a distribution $$p(x)$$ over images (a matrix of pixels) that assigns high probability to images that look realistic and low probability to everything else. Given such model, we can solve a wide array of interesting tasks.

## Image Generation

[Radford et al.](https://arxiv.org/abs/1710.10196) train a probabilistic model $$ p(x) $$ that assigns high probability to images that look like bedrooms (based on some training data):

**Training Data**<br /> 
![bedroom1](bedroominpainting1.png)<br /> 

If we sample $$x \sim p(x)$$, we are **generating** new (realistic) images. 

**Generated Data**<br /> 
![bedroom2](bedroominpainting2.png)

Similiarly, we can learn a model for faces or objects:

![faces1](progressiveGAN.png)
![faces1](pnpgan.png) 

Note that the images are not perfect and may need to be refined; however, sampling generates images that are very similiar to what one might expect. 

## In-Painting

Using the same $$p(x)$$, we can also 'fill in the rest of the image'. For example, given $$p(x)$$ and a patch of an existing image (e.g., a piece of a photograph), we can sample from $$p(Image \mid patch)$$ and generate different possible ways of completing the image:

![inpainting2](inpainting3.png)

Note the importance of a probabilistic model that captures uncertainty: there could be multiple ways to complete the image!

## Image Denoising

Similarly, given an image corrupted by noise (e.g., an old photograph), we can attempt to restore it based on our probabilistic model of what images look like:

![Image Denoising](imageDenoising4.png)

# Language Models

Knowing the probability distribution can also help us model natural langauge utterances. In this case, we want to construct a probability distribution $$p(x)$$ over sequences of words or characters $$x$$ that assigns high probability to proper (English) sentences. This distribution can be gathered by using articles from Wikipedia. 

## Generation

We can sample from the model and generate new Wikipedia-like articles like the following one:

---
Naturalism and decision for the majority of Arab countries' capitalide was grounded
by the Irish language by [[John Clair]], [[An Imperial Japanese Revolt]], associated 
with Guangzham's sovereignty. His generals were the powerful ruler of the Portugal 
in the [[Protestant Immineners]], which could be said to be directly in Cantonese 
Communication, which followed a ceremony and set inspired prison, training. The 
emperor travelled back to [[Antioch, Perth, October 25|21]] to note, the Kingdom 
of Costa Rica, unsuccessful fashioned the [[Thrales]], [[Cynth's Dajoard]], known 
in western [[Scotland]], near Italy to the conquest of India with the conflict. 
Copyright was the succession of independence in the slop of Syrian influence that 
was a famous German movement based on a more popular servicious, non-doctrinal 
and sexual power post. Many governments recognize the military housing of the 
[[Civil Liberalization and Infantry Resolution 265 National Party in Hungary]], 
that is sympathetic to be to the [[Punjab Resolution]]
(PJS)[http://www.humah.yahoo.com/guardian.
cfm/7754800786d17551963s89.htm Official economics Adjoint for the Nazism, Montgomery 
was swear to advance to the resources for those Socialism's rule, 
was starting to signing a major tripad of aid exile.]]

---

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Translation

Suppose that we have gathered a training set of paragraphs that were transcribed in both English and Chinese. We can build a probabilistic mode $$p(y|x)$$ to generate an English sentence $$y$$ conditioned on the corresponding Chinese sentence $$x$$; this is an instance of *machine translation*.

![Neural Machine Translation](nmt-model-fast.gif)

# Audio Models

We can also use probabilitic graphical models for audio applications. Suppose we construct a probability distribution $$p(x)$$ over audio signals that assigns high probability to ones that sound like human speech.

## Upsampling or super-resolution

Given a low resolution version of an audio signal, we can attempt to increase its resolution. 

[Super resolution of audio signals demo](https://kuleshov.github.io/audio-super-res/)

## Speech synthesis

As we did in image processing, we can also sample the model and generate (synthesize) speech signals.

[Super resolution of audio signals demo](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Speech recognition
Given a (joint) model of speech signals and language (text), we can attempt to infer spoken words from audio signals.
![Speech](speech.png)

# Applications in Science Today
## Error Correcting Codes
In the non-theoretical world, probabilistic models are often used to model communication channels (e.g., Ethernet or Wifi). i.e., if you send a message over a channel, you might get something different on the other end due to noise. Error correcting codes and techniques based on graphical models are used to detect and correct communication errors.
![codes](Picture1.png)


## Computational Biology

Graphical models are also widely used in computational biology. For example, given a model of how DNA sequences evolve over time, it is possible to reconstruct a phylogenetic tree from DNA sequences of a given set of species.
![philo](philo.png)

## Ecology
Graphical models are used to study phenomena that evolve over space and time, capturing spatial and temporal dependencies. For example, they can be used to study bird migrations.

![birds](bird_new.gif)

## Economics

Graphical models can be used to model spatial distributions of quantities of interests (e.g., assets or expenditures based measures of wealth).
![birds](uganda.png.jpg)


The last two applications are what are known as spatio-temporal models. They depend on data that is collected across time as well as space.

<br/>

|[Index](../../) | [Previous](../probabilityreview/) |  [Next](../../representation/directed/)|
