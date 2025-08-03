# Variational Autoencoder for Face Image Generation using CelebA

This project explores the use of **Variational Autoencoders (VAEs)** for generating realistic human face images. A VAE is a generative deep learning model that learns the underlying distribution of data and can sample new data points similar to the training data. We apply this to the **CelebA** dataset, a large-scale face dataset with over 200,000 celebrity images.

---

## Objective

The goal is to train a VAE on face images so it can:
- **Reconstruct** input face images
- **Generate** entirely new face images by sampling from the learned latent space

---

## Why VAE?

Unlike regular autoencoders, VAEs add a probabilistic twist — they learn a **distribution over the latent space**, typically assumed to be Gaussian. This enables:
- Smooth interpolation between faces
- Sampling new faces from the distribution
- Better generalization due to the regularization imposed by KL divergence

---

## Dataset: CelebA

The **CelebA** dataset contains celebrity face images with:
- Large diversity in pose, background, and expression
- Rich attribute annotations (e.g., smiling, glasses, gender)

For this project, only face images are used (attributes can be used later for conditional generation).

---

## Model Summary

The model consists of:
- **Encoder**: Maps input image to latent mean and variance vectors
- **Reparameterization Trick**: Samples latent vector from these parameters
- **Decoder**: Reconstructs image from sampled latent vector

Loss = **Reconstruction Loss** + **KL Divergence**

---

## Results

- The VAE learns to **accurately reconstruct** input faces.
- By sampling random latent vectors, it can generate **realistic-looking new faces**.
- The learned latent space allows smooth **morphing between identities**.

---

## Insights

- VAEs work well on CelebA even at 64×64 resolution.
- Increasing latent dimension improves generation quality but adds complexity.
- CelebA's diversity helps the model generalize to various facial features.

---

## Output 
<img width="830" height="464" alt="image" src="https://github.com/user-attachments/assets/a81226b3-1609-4dc0-b723-f07759cc36cd" />


## Conclusion

This project demonstrates how VAEs can effectively learn a compressed, meaningful latent space for complex data like human faces. Trained on CelebA, the model is capable of generating novel, high-quality face images and opens the door to attribute-guided generation in future work.
