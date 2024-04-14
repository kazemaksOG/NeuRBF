# NeuRBF reproduction and ablation studies
Authors:
- Alexandra (5233194, A.I.Neagu@student.tudelft.nl)
- Davis (5300606, D.Kazemaks@student.tudelft.nl)
- Teo (5320992, T.Oprescu@student.tudelft.nl)
- Mim (5274559, M.vandenBos-1@student.tudelft.nl)

The code can be found [here](https://github.com/kazemaksOG/NeuRBF).

## Introduction
In this reproduction and ablation project, we will consider the "NeuRBF: A Neural Fields Representation with Adaptive Radial Basis Functions" paper[^1]. This paper claims an improvement over other Neural Fields methods by using Radial Basis Functions (RBFs) to represent an image, instead of the commonly used grid-based approaches.

The remainder of this blog post is structured as follows. First, we introduce some of the fundamental ideas used in the paper such as neural fields. Then, we outline the main contributions NeuRBF makes compared to other neural field approaches. Then, we show a reproduction of the results in the paper. Finally, we ablate the method by changing the sinusoidal composition component with a polynomial composition of degree 3, and experimenting with different radial basis functions (RBFs).

## Background

### Neural Fields
The main idea of a neural field is to train a neural network where the input is a coordinate in a field, and the output is a value at that point. This can be done by training the network on one or more images that show a certain perspective of the space.

The NeuRBF method supports many use cases such as signed distance fields of a 3D model, or neural radiance fields. In the remainder of this post, we mainly consider the most straightforward use case where the neural network is trained on a single input image, and the desired output value of the network is a color faithful to the original when queried at a certain point.

### Quality metrics
As part of this analysis, we employ two widely recognized image quality metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), as used in the ablation studies conducted in the paper.

**Peak Signal-to-Noise Ratio (PSNR)** is a traditional metric used to assess the quality of reconstructed or compressed images. It measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better reconstruction quality, with the maximum value typically being 100 dB for lossless compression.

**Structural Similarity Index (SSIM)** is a perceptual metric that evaluates the structural similarity between two images. Unlike PSNR, SSIM considers the perceived changes in structural information, luminance, and contrast. It provides a more nuanced assessment of image quality, particularly in scenarios where human perception plays a crucial role. Higher SSIM values indicate better similarity between the reconstructed image and the original.

### NeuRBF
NeuRBF introduces several techniques that optimize the neural network representation and learning speed.

**Radial Basis Functions** are functions that depend only on the radius to the center point. While grid-based functions are also RBFs, the paper claims their functions more closely fit the target distribution and can require fewer parameters.

**Sinusoidal composition** is a technique to allow a single RBF to influence different frequencies. A radial basis function is normally a scalar function, but there is more than one weight that is multiplied with the same RBF. By composing the RBF into a multi-channel function that is the RBF applied to various frequencies of a sinusoid (these frequencies are predetermined by log-linearly dividing the range of frequencies in the image, with a trainable globally shared bias), each weight can fit the same RBF, but at a different frequency. The paper claims since each parameterization of an RBF can now represent many different bases, the representation ability is increased.

## Reproduction

### Choice of 2D images
To successfully reproduce the paper, we chose to use the 8000 x 8000 x 3 Pluto image that was referenced and used by the paper. This will make it easy to see if our reproduction has been successful.

Later on in the ablation study, we will additionally use an image of a colorful scenery of size 2500 x 1563 x 3, to see if the color diversity has any effect on the reconstruction of the images. 

Both of these images can be seen below.

<figure>
  <img
  src="https://hackmd.io/_uploads/ryztrUuxC.jpg"
  alt="error_map_8k_run"
  width="250">
  <img
  src="https://hackmd.io/_uploads/B1COrUOeA.jpg"
  alt="error_map_8k_run"
  width="350">
 <figcaption>Image of Pluto and of a colorful scenery </figcaption>
</figure>


### Performance
One of the main differences between the claims the paper makes and our own reproduction is the amount of time the model takes to run. This can be attributed to the difference in hardware available for the experiments. The most powerful machine we had access to in our experiments has a GeForce RTX 2070 SUPER GPU. While this is not the least powerful GPU, it was not sufficient to run the experiments as intended.

For instance, reproducing the 8000 x 8000 x 3 Pluto image on our local machines was not possible without offloading memory to the CPU with `--ds_device cpu`. The GPU did not have enough memory to run the model and quickly failed with a `torch.cuda.OutOfMemoryError: CUDA out of memory.`, notifying us the 8GB of memory available on the GPU is not enough. We think this to be the main cause for the training time for a single image to be in the order of hours instead of the minutes claimed by the paper.

For this reason, in some parts of the blog, smaller versions of this image shall be used, to both reduce the memory usage and training time.

<!-- Even with the `--ds_device cpu` flag enabled, at the final step a large amount of memory is needed and the operating system kills the process due to going out of memory (>14 GB used by the process). The final PSNR is reported before this, which allows us to compare the results to the paper, but the error map generated below is for a scaled version of the image. -->

<!-- **NOTE from Davis:** My laptop has 32 GB, so I managed to run and save the image -->

### Results

<!-- **NOTE from Davis:** Changed this section, but left the original as a comment for comparison. Also, did someone run the MINER version? where did you get the 54.83 db result, since it is better than what was in the report (I assume it just downscaled NeuRBF) -->

For our reproduction, we have focused on Figure 4 of the original paper. In this figure, a comparison is made between methods by showing an error map, trainable parameters, and PSNR of an image. We have reproduced this for the Pluto image shown in the figure. 

We have run the method twice for the 8K version of the Pluto image, once with 3,500 steps once with 35,000 steps, and one time with a scaled-down image to 1000 x 1000 x 3. The results are listed in the table below.
| | PSNR | #Trainable parameters
|-|-|-|
|NeuRBF 8k run (10x reduced steps)| 48.34 dB | 57.46M
|NeuRBF 8K run | 53.55 dB | 57.46M
|NeuRBF 1K run (10x reduced steps)| 54.83 dB | -
|NeuRBF reported by paper| 53.86 dB | 57.46M
|Instant NGP reported by paper| 48.53 dB | 61.45M
|MINER reported by paper| 50.66 dB | 67.24M



Although our NeuRBF is slightly below the value reported in the paper, they are very close and can vary between runs due to random initialization. It can be seen that the 1K image gets a higher PSNR score even for a factor of ten reduction in steps. This is explained by the lowered complexity of the task since less distinct values need to be represented in the neural field. 

The comparison between our and the original paper's error maps can be seen below. Error map of the 1K version will be used as a baseline for the ablations that follow.

<figure>
  <img
  src="https://hackmd.io/_uploads/r1Db7Vwl0.png"
  alt="error_map_8k_run"
  width="350">
  <img
  src="https://hackmd.io/_uploads/SkRTH4wxA.png"
  alt="error_map_8k_run"
  width="250">
  <figcaption> Comparing error maps of 8K Pluto images. The left image is our run, right is taken from the original paper Figure 4</figcaption>
</figure>

<figure>
  <img
  src="https://hackmd.io/_uploads/Sy-UQ9SxR.png"
  alt="error_map_normal_run">
  <figcaption>2D image fitting error map of the 1K Pluto image using NeuRBF</figcaption>
</figure> 

Due to the paper not sharing specifics of how they produce error maps, the color encoding is slightly different, but the tone still can be used to deduce error patterns (darker tone means less error).

It can be seen that the error map's tones match in most areas of the image, indicating that the error patterns are very similar. Adding this on top of the fact that the number of parameters and PSNR values are very close, it strongly suggests that we were successful in reproducing the results of this paper.  

<!-- For our reproduction, we have focused on Figure 4 of the original paper. In this figure, a comparison is made between methods by showing an error map and PSNR of an image. We have reproduced this for the Pluto image shown in the figure.

We have run the method twice for the 8K version of the Pluto image (which is the same version as reported in the paper) and once for a 1K scaled version. The results are listed in the table below.
| | PSNR |
|-|-|
|NeuRBF our first 8K run| 53.57 dB |
|NeuRBF our second 8K run| 53.11 dB |
|NeuRBF reported by paper| 53.86 dB |
|Instant NGP reported by paper| 48.53 dB |
|MINER report 1K run (10x reduced steps)| 54.83 dB |

Although both 8K runs are slightly below the value reported in the paper, they are very close and can vary between runs due to random initialization. Both of the runs are still significantly better than the other two methods mentioned in the paper.

Unfortunately, we could not get the error map of the 8K version of the Pluto image due to the issues mentioned above. Instead, we have run the method on a lower-quality version of the same image and used that to create the error map. As shown in the table, the 1K image gets a higher PSNR score even for a factor of ten reduction in steps. This is explained by the lowered complexity of the task since less distinct values need to be represented in the neural field. As a result, this error map cannot be directly compared to the results in the paper, but we use this as a baseline for the ablations that follow.

<figure>
  <img
  src="https://hackmd.io/_uploads/Sy-UQ9SxR.png"
  alt="error_map_normal_run">
  <figcaption>2D image fitting error map of the 1K Pluto image using NeuRBF</figcaption>
</figure> -->

## Ablation
Due to the large training time required on the hardware available to us, as described in the reproduction section, we chose to reduce both the image size and the number of runs used in our ablation studies. This allows us to ablate a larger amount of parameters, at the cost of less interesting use cases (since it is not interesting to reduce the size of small images). We make this trade-off since our main aim is understanding the relative impact of each parameter, and less so showing the potential of the method.

Unless otherwise specified, all experiments in the ablation section are run on a 1000 x 1000 x 3 scaled version of the Pluto image (instead of 8000 x 8000 x 3), and for 3,500 steps (instead of 35,000). We chose this amount of steps in order to keep consistent with the paper's results, as they also conducted ablation studies in Section 5.4 using 3,500 steps for training. Furthermore, Table 1 in the original paper also supports that 3,500 step runs still maintain reasonable quality.


### Ablation of Sinusoidal Composition on Radial Basis

In this study, we examine the impact of replacing the sinusoidal composition with a polynomial composition of degree 3 within the RBF framework. Keep in mind that for this, the paper's implementation of RBF was studied, with a version of the Inverse Quadratic (IVQ) function used, where the scaling factor is applied per dimension pair. We will analyze the impact different functions make in the next study.

In order to achieve these results, we disabled the sinusoidal composition of the system by commenting out lines 546-551 in `network.py`, and we introduced the polynomial composition of degree 3 (`rbf_out = rbf_out ** 3`). We chose to study a polynomial composition of degree 3 instead of 2 because it allows for a more flexible representation of complex patterns and variations in the data. While a polynomial of degree 2 (quadratic) can capture some curvature in the data, a polynomial of degree 3 (cubic) offers additional degrees of freedom, enabling it to better model nonlinear relationships and finer details in the data. Below is a table with our results:

| | PSNR | SSIM | Error map MSE |
|-|-|-|-|
|RBF with sinusoidal composition| 54.83 dB | 0.9980 | 0.4414 |
|RBF with degree 3 polynomial composition|49.87 dB |0.9953 | 0.8620|

The results indicate a noticeable decrease in PSNR and SSIM when the sinusoidal composition is replaced with a polynomial composition of degree 3. Additionally, the MSE of the error map shows an increase in reconstruction error, suggesting a degradation in image quality compared to the normal run configuration. This can be further visualized by comparing the error maps of the two configurations:  

<figure>
  <img
  src="https://hackmd.io/_uploads/Sy-UQ9SxR.png"
  alt="error_map_normal_run">
  <figcaption>Error map of the normal run configuration</figcaption>
</figure>


<figure>
  <img
  src="https://hackmd.io/_uploads/ByeP7creC.png"
  alt="error_map_polynomial_composition">
  <figcaption>Error map of the polynomial composition</figcaption>
</figure>

The increased noise observed in the error map of the polynomial composition compared to the normal run suggests a degradation in the fidelity of image reconstruction, particularly in areas with a lot of fine detail. This could be attributed to the limitations of using a polynomial composition of degree 3, which may not effectively capture the underlying patterns in the data as well as the sinusoidal composition. The higher MSE observed in the polynomial composition error map further supports this conclusion. In other words, the polynomial composition introduces more errors in the reconstruction process compared to the sinusoidal composition.

These findings emphasize the importance of the choice of composition function within the RBF framework. While sinusoidal composition may provide smoother and more accurate reconstructions, polynomial composition may introduce more noise and errors, leading to a decrease in overall image quality.

### Ablation of the RBF

In this study, we investigate the impact of different radial basis functions on the performance of the model. The original function utilized in the paper was the Inverse Quadratic (IVQ) RBF, denoted as ivq_a. The *_a* variant signifies that the scaling factor is applied per dimension pair. To explore the effects of alternative RBFs, we conduct an ablation study by replacing ivq_a with two different types of RBFs: Exponential Quadratic (EQ) RBF (rbf_mqd_a_fb) and Exponential Sine (ES) RBF (rbf_expsin_a_fb), with similar scaling factor variant. First, we provide an overview and explanation of these ablated RBFs.

The IVQ RBF, also known as the Inverse Multiquadric RBF, is characterized by a kernel function that decreases inversely with the squared distance between the input data points. Mathematically, it can be represented as $IVQ$($r$)=$\frac{1}{\sqrt{1+r^2}}$, where r denotes the Euclidean distance between the input data point and the RBF center. The IVQ RBF exhibits a distinct behavior where its influence decreases rapidly with increasing distance from the center, emphasizing data points that are closer to the center while downplaying those farther away. This property makes the IVQ RBF suitable for capturing both local and global patterns in the data, offering flexibility in modeling complex relationships.

The EQ RBF, also known as the Multiquadric RBF, is defined by a kernel function that decays exponentially with the squared distance between the input data points. Mathematically, it is expressed as $EQ$($r$) = $\sqrt{1+r^2}$, where $r$ again represents the Euclidean distance between the input data point and the RBF center. This RBF exhibits a smooth and monotonic decay in influence with increasing distance from the center, making it suitable for capturing smooth variations in the data.

The ES RBF combines the sinusoidal properties of the sine function with exponential decay. It is defined by the kernel function: $ES$($r$)=$e^{-sin(r)}$, where $r$ again denotes the distance between the input data point and the RBF center. This RBF introduces periodicity through the sinusoidal term while also exhibiting exponential decay, enabling it to capture both smooth and oscillatory patterns in the data.

By substituting the original $IVQ$ RBF with these alternative RBFs, we aim to assess their efficacy in modeling the underlying data patterns and their influence on the overall performance of the model. Below is a table with our results:

| RBF | PSNR | SSIM | Error map MSE |
|-|-|-|-|
|IVQ (original)| 54.83 dB | 0.9980 | 0.4414 |
|EQ|54.81 dB |0.9981 | 0.4581|
|ES|53.67 dB |0.9974| 0.5050|

The PSNR values for all RBF variations are relatively high, indicating that the reconstructed image maintains high fidelity to the original image.
The SSIM values are also close to 1, indicating strong structural similarity between the reconstructed and original images. The original RBF (IVQ) and the EQ RBF achieve similar performance in terms of PSNR and SSIM. The difference in PSNR and SSIM between IVQ and EQ is marginal.
The ES RBF, however, exhibits slightly lower PSNR and SSIM compared to IVQ and EQ. This suggests that the ES RBF may introduce more distortion or artifacts in the reconstructed images. Let us further visualize the error maps of these functions:

<figure>
  <img
  src="https://hackmd.io/_uploads/Sy-UQ9SxR.png"
  alt="error_map_normal_run">
  <figcaption>Error map of the IVQ RBF configuration</figcaption>
</figure>

<figure>
  <img
  src="https://hackmd.io/_uploads/S1W4FiBlA.png"
  alt="error_map_eq">
  <figcaption>Error map of the EQ RBF configuration</figcaption>
</figure>

<figure>
  <img
  src="https://hackmd.io/_uploads/ByR8YoHxC.png"
  alt="error_map_es">
  <figcaption>Error map of the ES RBF configuration</figcaption>
</figure>

Based on the visual inspection of these error maps and their close MSE results, it can be seen that the EQ and the original IVQ RBFs perform comparably well in terms of reconstructing the input image and minimizing reconstruction errors. Both of them yield error maps that exhibit minimal differences and are visually similar, indicating their effectiveness in capturing the underlying patterns and structures of the image.

However, the ES RBF, while still providing a reasonable reconstruction, demonstrates slightly inferior performance compared to the EQ and IVQ RBFs. The presence of more noise in the error map of the ES RBF, particularly in finer detail areas, suggests that it may not preserve image details as effectively as EQ and IVQ RBFs. Additionally, the higher MSE error map value for the ES RBF further confirms its comparatively lower reconstruction quality. This slightly inferior performance may stem from the exponential sine's specific functional form, which may not be as well-suited to capture the intricate details present in the input image.

Overall, while all three RBFs are capable of reconstructing the input image to some degree, the EQ and IVQ RBFs exhibit more consistent and reliable performance, with the ES RBF showing slightly diminished accuracy, particularly in preserving fine image details. Therefore, for applications where precise reconstruction and preservation of image details are crucial, the EQ and IVQ RBFs may be preferred over the ES RBF.


### Ablation on hyperparameters
The paper's implementation provides us with a base hyperparameter configuration. To see how each of these parameters influences the quality of the output, multiple experiments were conducted.

These experiments were performed on 2 images: the 2000 x 2000 x 3 Pluto image and a colorful scenery. The second image was chosen to see if the color diversity has any effect on the learning rate and reconstruction quality while ablating these hyperparameters. 

Since many experiments were performed, the maximum step amount was lowered to 3,500 to finish in a reasonable time.

After running these experiments, it was noticed that most of the tunable hyperparameters had no significant effect (PSNR difference less than 0.5 db). The hyperparameters that did have a significant impact are `lr` (learning rate), `n_kernel` (kernel size), `lc_act` (activation function in every layer), `act` (activation function at the output layer).

<!-- ### Color diversity effects

# found out this is due to trainable parameter differences, so not really a valid comparison

Running the code on both images with the same configurations gives very unexpected results. In the original paper, the colorful image was able to achieve a higher PSNR value than the image with Pluto (60.52dB vs 53.86dB) with almost twice as less trainable parameters. In our experiments, it seems that the colorful image struggled to approach such high-quality reconstruction as can be seen below.

<figure>
  <img
  src="https://hackmd.io/_uploads/rkgWHPOlC.png"
  alt="error_map_es">
  <figcaption>Comparing different learning rates (lr) on the colorful image</figcaption>
</figure> -->



#### Learning rate

The learning rate affects the step size moving toward the minimum of the loss function. Having the learning rate too big may lead the step to oscillate over the local minima, meaning it never is able to converge to the target. Having the learning rate too small may make learning too slow, or not progress in learning at all.


<figure>
  <img
  src="https://hackmd.io/_uploads/SJQSMIue0.png"
  alt="error_map_es">
  <figcaption>Comparing different learning rates (lr) on the colorful image</figcaption>
</figure>


This is also what we see in this experiment when comparing small learning rates with big learning rates. Since this is a very well-documented problem in Machine Learning, we decided not to further ablate this hyperparameter, since we don't expect to find anything novel.




#### Kernel size

Kernel functions are used to extract useful features from images. Kernel size in this context references the total amount of inputs the kernel function will take. In convolutional neural networks, it is common to have kernel function sizes that range between 2x2 to 5x5[^2]. Still, during experimentation, we noticed that this implementation uses much larger kernel sizes (in ranges from 50K-200K) that are derived using the heuristics of the image. This could be explained by the fact that the model requires a large receptive field to obtain the necessary context for representing features that allow to reconstruction of the original image.

To ablate this hyperparameter, we decided to significantly increase and decrease the kernel size, and observe its impact on the number of trainable parameters and the PSNR value. The results are summarized in the table below.

| | PSNR | #Trainable parameters | Kernel size
|-|-|-|-|
|Pluto 2K default amount | 52.01 dB | 4.36M | 73K
|Pluto 2K smaller kernel | 41.51 dB | 2.24M | 7.3K
|Pluto 2K bigger kernel | 67.15 dB | 25.58M | 736K
|Pluto 2K extremely small kernel | 37.40 dB | 2.01M | 75
|Colorful default amount | 45.73 dB | 6.57M | 107K
|Colorful smaller kernel | 35.86 dB | 3.48M | 10.7K
|Colorful bigger kernel | 64.37 dB | 37.48M | 1073K
|Colorful extremely small kernel | 30.82 dB | 3.14M | 75

As can be seen in the table, increasing the kernel size increases both the trainable parameter size and PSNR value. It can also be derived that there are many trainable parameters that are not related to the kernel function, since the run with a kernel size that is about 5x5x3, there were still plenty of trainable parameters, and it was possible to reconstruct the image.

<figure>
  <img
  src="https://hackmd.io/_uploads/HkpiFIFx0.png"
  alt="error_map_es"
  width=180>
  <img
  src="https://hackmd.io/_uploads/HkOVqUYgR.png"
  alt="error_map_es"
  width=180>
  <img
  src="https://hackmd.io/_uploads/S16-5LYlA.png"
  alt="error_map_es"
  width=230>
  <figcaption> Colorful image errors map in asceding kernel size order</figcaption>
</figure>

<figure>
  <img
  src="https://hackmd.io/_uploads/HyrosLtxR.png"
  alt="error_map_es"
  width=180>
  <img
  src="https://hackmd.io/_uploads/SyQOs8KeC.png"
  alt="error_map_es"
  width=180>
  <img
  src="https://hackmd.io/_uploads/BJ-X2IKeC.png"
  alt="error_map_es"
  width=230>
  <figcaption> Pluto image errors map in asceding kernel size order</figcaption>
</figure>

Observing the error maps, it can be seen that at some point, by increasing the kernel size, the images have close to no error when reconstructed. Additionally, in spots where there are not many color changes (the middle of Pluto and the sky in the colorful picture), it can be seen that the error is much smaller than in places where there is color diversity.

These findings strongly suggest that kernel size is a major contributor to having more accurate reconstruction of images. Having it at too low of a value will impair the model's ability to accurately reconstruct the image, but having it too high may cause the neural network to become too big and too inefficient for representation.

#### Activation functions

Activation functions are used to introduce non-linearity in neural networks. This allows the network to learn complex relationships that are not linear. There are two common ways activation functions are used: using an activation function in every hidden layer or using it only in the final output layer. 

The provided implementation uses no activation function in the hidden layers and uses a ReLu at the output layer. To ablate this, we will use multiple different activation functions in both hidden layers and output layers. Since the base implementation only provides 3 activation functions, we added 4 more activation functions to the codebase to see how they influence image reconstruction. The table below summarizes the findings.

| Pluto | Colorful | Hidden layer activaiton function | Output activation function
|-|-|-|-|
|51.95 dB | 45.81 dB | none | ReLu
|52.08 dB | 46.01 dB| none | Leaky ReLu (0.1)
|51.18 dB | 45.19 dB | none | none
|51.88 dB | 46.01 dB | none | Sigmoid
|22.15 dB | 27.27 dB | none | Sine
|52.40 dB | 46.42 dB | none | Softmax
|51.73 dB | 45.53 dB | none | Tanh
|50.79 dB| 45.93 dB | Leaky Relu (0.1) | ReLu
|42.84 dB | 37.35 dB | Relu | ReLu
|46.34 dB | 41.21 dB | Sigmoid | ReLu
|52.65 dB | 46.68 dB | Sine | ReLu
|46.27 dB | 38.11 dB | Softmax | ReLu
|51.99 dB | 45.80 dB | Tanh | ReLu

It can be seen that most output layer activation functions do not have any significant impact on the PSNR value, with only the exception of sine which heavily underperformed in this experiment. This can be explained by the fact that sine values may introduce loss of detail since sine is a cyclic function, which may represent different input values with the same sine value.

For most hidden layer activation functions, they presented worse results than not using any activation function. This could be due to the fact that activation functions may introduce more loss of information if used in every layer of the network, contrary to only using it at the output layer. The only activation functions that were able to perform as well as the default configuration were sine and tanh activation functions.


Most error maps did not present any noticeable patterns, except for the case of using sine as the activation function for hidden layers. This can be observed below.


<figure>
  <img
  src="https://hackmd.io/_uploads/SykIzwteC.png"
  alt="error_map_es"
  width=290>
  <img
  src="https://hackmd.io/_uploads/H1H47vtlA.png"
  alt="error_map_es"
  width=340>
  <figcaption> Colorful image: Sine vs. no activation function used for hidden layer</figcaption>
</figure>

<figure>
  <img
  src="https://hackmd.io/_uploads/S163EwKeA.png"
  alt="error_map_es"
  width=265>
  <img
  src="https://hackmd.io/_uploads/H160NDYg0.png"
  alt="error_map_es"
  width=340>
  <figcaption> Pluto image: Sine vs. no activation function used for hidden layer</figcaption>
</figure>

It seems encoding the output of hidden layers as a sine value induces the model to perform better reconstruction in places where there is more color diversity (tree on the right side of colorful image), but contrarily, it has worse construction when the color in the region is more similar (middle of Pluto).

From these findings, we can conclude that activation functions may lose some information between layers if used inappropriately. Additionally, activation functions between hidden layers may allow certain patterns to be more recognizable than others, in this instance, the number of different colors within regions.


## Conclusion
In our reproduction of the NeuRBF paper, we have obtained similar results as claimed by the original paper for the image use case. 

Furthermore, we have ablated several of the parameters used in the paper to gain more insight into the workings of the method. One thing we can see from this is that the choice of composition function and radial basis significantly impacts the performance of the NeuRBF model. In the ablation study of the sinusoidal composition, we observed a notable decrease in both PSNR and SSIM metrics when switching from the original sinusoidal composition to a polynomial composition of degree 3. This suggests that the sinusoidal composition is better suited for capturing the underlying patterns in the data compared to polynomial compositions of a higher degree. 

Similarly, in the study on different RBFs, we found that the EQ RBF produced results similar to the original IVQ RBF in terms of PSNR and SSIM metrics, indicating comparable performance in capturing data patterns. However, the ES RBF exhibited slightly lower PSNR and SSIM scores, suggesting that it may be less effective in preserving image details. These findings underscore the importance of carefully selecting composition functions and RBFs to achieve optimal performance in NeuRBF-based applications.

Investigating the influence of hyperparameters on the quality of reconstruction, we have found 3 impactful parameters: _learning rate_, _kernel size_, and _activation functions_ for both hidden layers and output layer. 

Learning rate affects the model the same way it would affect any other machine learning model: high learning rate may cause oscillation that never reaches local minima, and low learning rates may cause the model to learn too slowly.

Kernel size has a positive correlation with both PSNR and the number of trainable parameters. One of the main propositions of the original paper was to minimize trainable parameters while maintaining a high PSNR value. This value presents more opportunity for fine tunning, to find the best balance of accurate reconstruction and model size.

Additionally, it was observed that reconstructing the colorful image was more difficult than the Pluto image when the kernel sizes were similar. This may indicate that having more color details in images requires more tunable parameters to be accurately reconstructed.

Activation functions can improve and diminish reconstruction quality depending if their use is appropriate. Interestingly, using sine as the activation function for hidden layers introduced different error patterns than what was usually observed.


<!-- **TODO - Add more conclusions other than the ablations?** -->

## Extra experiments

Numerous experimental runs with varying hyperparameter configurations inspired the selection of the most interesting hyperparameters discussed in this report. The results of these 68 runs (which conducted ablations on many hyperparameters such as kernel size, activation function, number of layers, learning rate, or RBF type) are compiled in a grid, accessible through the following link: 

https://drive.google.com/uc?export=download&id=1refyM30mxIBU0Hmab45fqVmBG77f5u6C

## Contributions


| Who | What |
| --------- | -------- |
| Alexandra | Ablations of RBFs, ablation of sinusoidal composition |
| Davis     | Error map visualization, exploration of ablation parameters |
| Teo       | Ablation automation, exploration of ablation parameters |
| Mim       | Figure 4 reproduction, environment setup |
| All       | Report writing, discussion, experiments |



[^1]: https://arxiv.org/abs/2309.15426
[^2]: https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363