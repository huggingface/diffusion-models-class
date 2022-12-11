# Unit 2: Fine-Tuning, Guidance and Conditioning

Welcome to Unit 2 of the Hugging Face Diffusion Models Course! In this unit you will learn how to use and adapt pre-trained diffusion models in new ways. You will also see how we can create diffusion models that take additional inputs as **conditioning** to control the generation process.

## Start this Unit :rocket:

Here are the steps for this unit:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when new material is released
- Read through the material below for an overview of the key ideas of this unit
- Check out the _**Fine-Tuning and Guidance**_ notebook below to fine-tune an existing diffusion model on a new dataset using the 🤗 Diffusers library
- Create your own custom pipeline and share it as a Gradio demo

:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.
 
## Fine-Tuning

As you saw in Unit 1, training diffusion models from scratch can be time-consuming! Especially as we push to higher resolutions, the time and data required to train a model from scratch can become impractical. Fortunately, there is a solution: begin with a model that has already been trained! This way we start from a model that has already learnt to denoise images of some kind, and the hope is that this provides a better starting point than beginning from a randomly initialized model.

![Example images generated with a model trained on LSUN Bedrooms and fine-tuned for 500 steps on WikiArt](https://api.wandb.ai/files/johnowhitaker/dm_finetune/2upaa341/media/images/Sample%20generations_501_d980e7fe082aec0dfc49.png)

Fine-tuning typically works best if the new data somewhat resembles the base model's original training data (for example, beginning with a model trained on faces is probably a good idea if you're trying to generate cartoon faces) but suprisinggly the benefits persist even if the domain is changed quite drastically. The image above is generated from a [model](https://huggingface.co/johnowhitaker/sd-class-wikiart-from-bedrooms) trained on the LSUN Bedrooms dataset](https://huggingface.co/google/ddpm-bedroom-256) and fine-tuned for 500 steps on [the WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart). The training script is included for reference alongside the notebooks for this unit.

## Guidance

Unconditional models don't give much control over what is generated. We can train a conditional model (more on that in the next section) that takes additional inputs to help steer the generation process, but what if we already have a trained unconditional model we'd like to use? Enter **_guidance_**, a process by which the model predictions at each step in the generation process are evaluated against some guidance function and modified such that the final generated image is more to our liking. 

This guidance function can be almost anything, making this a powerful technique! In the notebook we build up from a simple example to one utilizing a powerful pre-trained model called CLIP which lets us guide generation based on a text description. 

## Hands-On Notebook

At this point, you know enough to get started with the accompanying notebooks!

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fine-Tuning and Guidance                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_finetuning_and_guidance.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_finetuning_and_guidance.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_finetuning_and_guidance.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_finetuning_and_guidance.ipynb)              |


## Project Time

The best way to learn is by doing! Fine-tune your own diffusion model on a new image dataset and create a Gradio demo that can generate images from it. Don't forget to share your work with us on Discord, Twitter or elsewhere!

## Some Additional Resources
 
* [GLIDE](https://www.youtube.com/watch?v=lvv4N2nf-HU) (text conditioned diffusion model)

Found more great resources? Let us know and we'll add them to this list.
