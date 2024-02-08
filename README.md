# Image Colorization Project with GANs 🎨🖌️ 

## Project Overview

This project aims to provide practical experience in implementing image colorization using Generative Adversarial Networks (GANs) with a U-Net architecture. Participants will work with a curated dataset of grayscale images paired with their corresponding colored images and will explore the process of training a GAN to predict colors for grayscale inputs.

### Objectives
- Understanding the principles of image colorization.
- Exploring the architecture and capabilities of U-Net GANs.
- Training a U-Net GAN model on a custom dataset for colorization tasks.
- Evaluating the performance of the trained model.

## Repository Structure

- `src/`: Contains the source code for the project.
  - `data.py`: Custom data loader for handling the image colorization dataset.
  - `ganloss.py`: Implementation of GAN loss functions.
  - `models.py`: Implementation of the U-Net architecture for the generator and discriminator.
  - `weight_initializer.py`: Script for initializing model weights.
- `images/`: Directory containing the images for training and testing.

## Getting Started
You can upload the notebook and the code directly in Colab. Start by exploring the `main.ipynb` notebook and filling in the TODOs to train and evaluate the image colorization model using the provided dataset. Make sure to install the required dependencies listed in `requirements.txt`. Happy coding! 🎨🖌️
