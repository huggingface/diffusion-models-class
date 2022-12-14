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

TODO cool images here

Stable Diffusion is ...
 
## Latent Diffusion

![latent diffusion diagram](https://github.com/CompVis/latent-diffusion/raw/main/assets/modelfigure.png)
_Diagram from the [Latent Diffusion paper](http://arxiv.org/abs/2112.10752)

Explain latent diffusion

## Text Conditioning

![conditioning diagram](sd_unet_color.png)

Explain text conditioning

## Classifier-free Guidance

Explain CFG


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
