# GANS

In this project is will implement a DCGAN to see if I can generate handbag designs. 

To accomplish this I'm using 3,000 handbag images from Imagenet. These are sample images:
 
<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/sample%20image.PNG" width="300"></p>

# DCGAN

DCGANs were introduced by Alec Radford, Luke Metz and Soumith Chintala in 2016 (paper: https://arxiv.org/pdf/1511.06434.pdf). The following diagram explains the architecture of a DCGAN:

<p align="center"><img src="https://github.com/prodillo/GAN_project/blob/master/images/dcgan%20diagram.png" width="800"></p>

The role of the Generator is explained as follows in the paper: "A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions then convert this high level representation into a 64 × 64 pixel image."
