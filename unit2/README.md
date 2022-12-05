# Unit 2: Fine-Tuning, Guidance and Conditioning

Welcome to Unit 2 of the Hugging Face Diffusion Models Course! In this unit you will learn how to use and adapt pre-trained diffusion models in new ways. You will also see how we can create diffusion models that take additional inputs as **conditioning** to control the generation process.

## Start this Unit :rocket:

Here are the steps for this unit:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when new material is released
- Read through the material below for an overview of the key ideas of this unit
- Check out the _**FINE TUNING AND GUIDANCE NOTEBOOK TODO LINK**_ to fine-tune an existing diffusion model on a new dataset using the 🤗 Diffusers library
- Read through the **CONDITIONING NOTEBOOK** to see how we can add additional control to the generation process.
- Create your own custom pipeline and share it as a Gradio demo

:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.
 
## Fine-Tuning

As you may have seen in Unit 1, training diffusion models from scratch can be time-consuming! Especially as we push to higher resolutions, the time and data required to train a model from scratch can become impractical. Fortunately, there is a solution: begin with a model that has already been trained! This way we start from a model that has already learnt to denoise images of some kind, and the hope is that this provides a better starting point than beginning from a randomly initialized model. Fine-tuning typically works best if the new data somewhat resembles the base model's original training data (for example, beginning with a model trained on faces is probably a good idea if you're trying to generate cartoon faces) but suprisinggly the benefits persist even if the domain is changed quite drastically.

## Guidance

Unconditional models don't give much control over what is generated. We can train a conditional model (more on that in the next section) that takes additional inputs to help steer the generation process, but what if we already have a trained unconditional model we'd like to use? Enter guidance, a process by which the model predictions at each step in the generation process are evaluated against some guidance function and modified such that the final generated image is more to our liking. 

This guidance function can be almost anything, making this a powerful technique! In the notebook we build up from a simple example to one utilizing a powerful pre-trained model called CLIP that lets us guide generation based on a text description. 

## Conditioning

Guidance is a great way to get some additional mileage from an unconditional diffusion model, but if we have additional information (such as a class label or an image caption) available during training then we can also feed this to the model for it to use as it makes its predictions. In doing so, we create a **conditional** model, which we can control at inference time by controlling what is fed in as conditioning. The notebook shows an example of a class-conditioned model which learns to generate images according to a class label. TODO note about timestep conditioning?

## Hands-On Notebook

At this point, you know enough to get started with the accompanying notebooks!

TODO link table and descriptions

## Project Time

Create a custom pipeline using some or all of the ideas covered in this unit and share it with the community

## Some Additional Resources
 
GLIDE (text conditioned diffusion model) TODO link

Thomas' example TODO link



Found more great resources? Let us know and we'll add them to this list.
