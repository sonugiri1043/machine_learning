# Types of convolution
>
## Dilated Convolution
![Dialted convoltion](https://www.saama.com/wp-content/uploads/2017/12/1_SVkgHoFoiMZkjy54zM_SUw.gif)
> This convolution introduces one additional parameter into the convolution, called dialation rate (d). Normal convolution has a dialation rate of 1. A dialation rate of 2 means spacing of 1 between kernel weights. After introducing a dialation factor the normal convolution is done as shown in above picture. Dialated convolution with 3x3 kernel and d=2 can look into 5x5 input matrix. With d=3, the receptive field will increase to 7x7. This increases the receptive filed and makes colvolution computationally less expensive.
> This technique is used in real time segmentation, which requires us to perform downscaling using convolutional layer followed by upscaling using deconvolution.
![dialation](https://qph.fs.quoracdn.net/main-qimg-e3b2b1cfb271af5de919230f4c973801)
> 
# Transpose or Deconvnvolution
![deconvolution](https://www.saama.com/wp-content/uploads/2017/12/1_Lpn4nag_KRMfGkx1k6bV-g.gif )
> Deconvolution is not mathematical-inverse of convolution. In machine learning, deconvolution is another convolution layer that spaces out the pixels and performs an up-sampling. A normal convolution integrate input from nxn area ( nxn kernel ) and reduces it into a single value. A deconvolution aka transpose convolution takes a single input and spread information in an area of kernel size.
> This is used in image segmentation task. Normal convolution (with stride 1 and padding 1) reduces the spatial dimesntion of the input. Transpose convolution increases the spatial dimension of the input. Deconvolution is used after convolution to retain spatial dimension.
> Performing back propogation for dialted convolution is just transpose convolution. A series of convolution followed by series of deconvolution is encoder-decoder architecture.
> 
# Depthwise Convolution
> In depthwise convolution filter is applied seperately to each channel in input then a 1x1 convolution is applied to the resulting feature map.
>
# Seperable Convolution
> A convolution kernel is called seperable if it can be represented as a mutiplication of different kernels.
>  K = k1 * k2
> 
>  A 3x3 filter will have same effect as 1x3 filter followed by 3x1 filter. By doing so number of parameters gets reduced from 9 to 6. This reduces the computational cost thereby making it suitable for small devices.

# Grouped Convolution
> In group convolution several filter are used in parallel. This allows parallelizing across GPU.
>

# Pooling Layer
> This introduces non-linearity in the network. It's not a convolutionl layer. This is performed for downsampling which reduces the amount of computation that needs to be done. It also help reduce the no of parameters. A 2x2 pooling will reduce no of paramters by 75%. Pooling layer loses positional information about different feature inside image.
> ![pooling](https://blog.liang2.tw/2015Talk-DeepLearn-CNN/pics/external/cs231n_note_conv_pooling.png)
>

# Data Augmentation
> It alters the training data by moifying the input array while keeping the labels same. This include rotation, horizontal flip, vertical flip, resizing, grayscale, cropping, translatin. This artificially expand the available dataset. This overcomes the limitation of CNN that it doesn't pass the position information.
>  

# DropOut
> This is used to overcome overfitting effect. 'Overfitting' happens when the CNN model is so much tuned to training data that it's accuracy is more than 99% on training but significantly less with test data. In this technique, some values in the CNN layer are dropped to zero. This is only performed during training stage. 
>