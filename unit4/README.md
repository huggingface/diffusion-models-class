# Unit 4: Going Further with Diffusion Models

Welcome to Unit 4 of the Hugging Face Diffusion Models Course! In this unit we will look at some of the many improvements and extensions to diffusion models appearing in the latest research. It will be less code-heavy than previous units have been, and is designed to give you a jumping off point for further research and set up for possible additional units in the future.

## Start this Unit :rocket:

Here are the steps for this unit:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when additional units are added to the course
- Read through the material below for an overview of the different topics covered in this unit
- Dive deeper into any specific topics with the linked videos and resources
- Complete the [TODO some sort of exercise/capstone project]

:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.
 
## Introduction

By the end of Unit 3...

## Faster Sampling via Distillation

Introduce the idea... The core mechanism is illustrated in this diagram from the [paper that introduced the idea](http://arxiv.org/abs/2202.00512):

![image](https://user-images.githubusercontent.com/6575163/211016659-7dac24a5-37e2-45f9-aba8-0c573937e7fb.png)

The idea of using an existing model to 'teach' a new model can be extended to create guided models where the classifier-free guidance technique is used by the teacher model and the student model must learn to produce an equivalent output in a single step based on an additional input specifying the targeted guidance scale. This further reduces the number of model evaluations required to produce high-quality samples.

Key papers:
- [PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS](http://arxiv.org/abs/2202.00512)
- [ON DISTILLATION OF GUIDED DIFFUSION MODELS](http://arxiv.org/abs/2210.03142)

## Training Improvements

![image](https://user-images.githubusercontent.com/6575163/211021220-e87ca296-cf15-4262-9359-7aeffeecbaae.png)
_Figure 2 from the [ERNIE-ViLG 2.0 paper](http://arxiv.org/abs/2210.15257)_

There have been a number of additional tricks developed to improve training. A few key ones are
- Tuning the noise schedule, loss weighting and sampling trajectories (Karras et al)
- Training on diverse aspect rations [TODO link patrick's talk from the launch event]
- Cascading diffusion models, training one model at low resolution and then one or more super-res models (D2, Imagen, eDiffi)
- Rich text embeddings (Imagen) or multiple types of conditioning (eDiffi)
- Incorporating pre-trained image captioning and object detection models into the training process to create more informative captions and produce better performance in a process known as 'knowledge enhancement' (ERNIE-ViLG 2.0)
- MoE training different variants of the model ('experts') for different noise levels...

Key Papers:
- [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364)
- [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](http://arxiv.org/abs/2211.01324)
- [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts](http://arxiv.org/abs/2210.15257)
- [Imagen][TODO]

## Inference Improvements

- eDiffi paint with words,
- Image editing with diffusion models video
- Textual inversion, null text inversion
- ??

## Video

Video diffusion, Imagen Video

Key Papers:
- [Video Diffusion Models](https://video-diffusion.github.io/)
- [IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS](https://imagen.research.google/video/paper.pdf)

## Audio

- Riffusion (and possibly notebook on the idea)
- Non-spectrogram paper

## New Architectures and Approaches

Transformer in place of UNet (DiT)

Recurrent Interface Networks (https://arxiv.org/pdf/2212.11972.pdf)

MUSE/MaskGIT and Paella


## Project Time

TODO
