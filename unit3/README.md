# Unit 3: Stable Diffusion

Welcome to Unit 3 of the Hugging Face Diffusion Models Course! In this unit you will meet a powerful diffusion model called Stable Diffusion (SD) and explore what it can do.

## Start this Unit :rocket:

Here are the steps for this unit:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when new material is released
- Read through the material below for an overview of the key ideas of this unit
- Check out the _**Stable Diffusion Introduction**_ notebook to see SD applied in practice to some common use-cases
- Use the _**Annotated Dreambooth**_ notebook to fine-tune your own custom Stable Diffusion model and share it with the community for a chance to win some prizes and swag
- (Optional) Check out the [_**Stable Diffusion Deep Dive video**_](https://www.youtube.com/watch?app=desktop&v=0_BBRNYInx8) and the accompanying [_**notebook**_](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) for a deeper exploration of the different componets and how they can be adapted for different effects. This material was created for the new FastAI course, 'Stable Diffusion from the Foundations' - the first few lessons are already available and the rest will be released in the next few months, making this a great supplement to this class for anyone curious about building these kinds of models completely from scratch. 


:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.

## Introduction

![SD example images](sd_demo_images.jpg)<br>
_Example images generated using Stable Diffusion_

Stable Diffusion is a powerful text-conditioned latent diffusion model. Don't worry, we'll explain those words shortly! Its ability to create amazing images from text descriptions has made it an internet sensation. In this unit we're going to explore how SD works and see what other tricks it can do.
 
## Latent Diffusion

As image size grows, so does the computational power required to work with those images. This is especially pronounced in an operation called self-attention, where the amount of operations grows quadratically with the number of inputs. A 128px square image has 4x as many pixels as a 64px square image, and so requires 16x (i.e. 4<sup>2</sup>) the memory and compute in a self-attention layer. This is a problem for anyone who'd like to generate high-resolution images!

![latent diffusion diagram](https://github.com/CompVis/latent-diffusion/raw/main/assets/modelfigure.png)<br>
_Diagram from the [Latent Diffusion paper](http://arxiv.org/abs/2112.10752)_

Latent diffusion helps to mitigate this issue by using a separate model called a Variational Auto-Encoder (VAE) to **compress** images to a smaller spatial dimension. The rationale behind this is that images tend to contain a large amount of reduntat information - given enough training data, a VAE can hopefully learn to produce a much smaller representation of an input image and then reconstruct the image based on this small **latent** representation with a high degree of fidelity. The VAE using in SD takes in 3-channel images and produces a 4-channel latent representation with a reduction factor of 8 for each spatial dimension. That is, a 512px square input image will be compressed down to a 4x64x64 latent.

By applying the diffusion process on these **latent representations** rather than on full-resolution images, we can get many of the benefits that would come from using smaller images (lower memory usage, fewer layers needed in the unet, faster generation times...) and still decode the result back to a high-resolution image once we're ready to view the final result. This innovation dramatically lowers the cost to train and run these models. 

## Text Conditioning

In Unit 2 we showed how feeding additional information to the unet allows us to have some additional control over the types of images generated. We call this conditioning. Given a noisy version of an image, the model is tasked with predicting the denoised version **based on additional clues** such as a class label or, in the case of Stable Diffusion, a text description of the image. At inference time, we can feed in the description of an image we'd like to see and some pure noise as a starting point, and the model do its best to 'denoise' the random input into something that matches the caption. 

![text encoder diagram](text_encoder_noborder.png)<br>
_Diagram showing the text encoding process which transforms the input prompt into a set of text embeddings (the encoder_hidden_states) which can then be fed in as conditioning to the unet._

For this to work, we need to create a numeric representation of the text that captures relevant information about what it describes. To do this, SD leverages a pre-trained transformer model based on something called CLIP. CLIP's text encoder was designed to process image captions into a form that could be used to compare images and text, and so it is well suited to the task of creating useful representations from image descriptions. An input prompt is first tokenized (based on a large vocabulary where each word or sub-word is assigned a specific token) and then fed through the CLIP text encoder, producing a 768-dimensional (in the case of SD 1.X) or 1024-dimensional (SD 2.X) vector for each token. To keep things consistent, prompts are always padded/truncated to be 77 tokens long, and so the final representation which we use as conditioning is a tensor of shape 77x1024 per prompt.

![conditioning diagram](sd_unet_color.png)


OK, so how do we actually feed this conditioning information into the unet for it to use as it makes predictions? The answer is something called cross-attention. Scattered throughout the unet are cross-attention layers. Each spatial location in the unet can 'attend' to different tokens in the text conditioning, bringing in relevant information from the prompt. The diagram above shows how this text conditioning (as well as a timestep-based conditioning) is fed in at different points. As you can see, at every level the unet has ample opportunity to make use of this conditioning!

## Classifier-free Guidance

Explain CFG

## Super-Resolution

Include?


## Hands-On Notebook

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fine-tuning and Guidance                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              |
| Class-conditioned Diffusion Model Example                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              |

At this point, you know enough to get started with the accompanying notebooks! Open them in your platform of choice using the links above. Dreambooth requires quite a lot of compute power, so if you're using Kaggle or Google Colab make sure you set the runtime type to 'GPU' for best results. 

In intro notebook ...

In Annotated Dreambooth ...


## Project Time

Follow the instructions in [todo link dreambooth notebook] to train your own model for one of the specified categories. Make sure you include the example outputs in your submission so that we can choose the best models in each category! Prizes will be awarded [TODO details]. If you're short of GPU power, [TODO info on colab credits]

## Some Additional Resources

[High-Resolution Image Synthesis with Latent Diffusion Models](http://arxiv.org/abs/2112.10752) - The paper that introduced the approach behind Stable Diffusion

[CLIP](https://openai.com/blog/clip/) - CLIP learns to connect text with images and the CLIP text encoder is used to transform a text prompt into the rich numerical representation used by SD. See also, [this article on OpenCLIP](https://wandb.ai/johnowhitaker/openclip-benchmarking/reports/Exploring-OpenCLIP--VmlldzoyOTIzNzIz) for some background on recent open-source CLIP variants (one of which is used for SD version 2).

Found more great resources? Let us know and we'll add them to this list.
