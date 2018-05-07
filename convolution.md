#Session 1

## Convolution
> Convolution operation in context of Convolutional Neural Network ( CNN ) is element wise matrix multiplication and summation. It involves a kernel/filter and an input. The filter is element wise multipled with the input array and summation of multiplied values is done. The filter moves across the input and this operation is repeated at each step.
> 
> ![img](https://indoml.files.wordpress.com/2018/03/convolution-operation-14.png)
> 
> The kernel moves over the input array with a fixed stride. The output is known as feature map whose dimension depends on the padding and stride.
> 
> As shown in above image, for an input array of size 6x6 and filter of size 3x3 with zero padding and stride as 1 the feature map will have dimension 4x4. For stride 1, a feature map will have (k-1) less rows and columns where k is filter dimension.
> 
> Convolution makes feature detection in input spatially invariant since the kernel moves across image. 

## Filters/Kernels
> A filter is a feature identifier used on an image. It is a 2-D array with numbers, number are called weight. The filter moves across the input image and at each step an operation (for e.g: convolution ) is performed. This results into a feature map which serves an input for the next layer.

## 3x3 Convolution
> A 3x3 convolution involves a 3x3 filter. Filter moves across the input and at each step element wise mutiplication followed by summation is done. This operation when performed on entire image results in a feature map. If zero padding ( stride of size 1 ) is used then this will give a feature map with size (M-2)x(N-2) where MxN is the input size. The image in convolution section describe 3x3 convolution.

## 1x1 Convolution
> 1x1 convoultion is used to merge the channel in input. The operation involves multiplication of values at same position across channels. In the below image this is demonsrated on an input of 6x6 array with depth of 5, 2 filters with same depth as input is used. After 1x1 convolution 2 feature map with 6x6 dimension is obtained. 
> This is basically a NxD element wise multiplication where D is the depth of input and N is the no of filters.
![1x1 convoltion](https://indoml.files.wordpress.com/2018/03/1x1-convolution1.png)

## Feature Maps
A feature map is the result of convolution applied on an input with a filter. Feature map is aka activation map. A feature map from one layer serves as an input for the subsequent layer. Each values in feature map represent the chances that a feature, represented by filter, in present in input over an area of filter size. As we go higher in the CNN chain feature map represent more and more complex feature. A feature can be simple colour, edge, striaght line etc.
![feature map](https://qph.ec.quoracdn.net/main-qimg-134024e4e35d7c7cbc4ccbe3a62dc8b2.webp)

## Receptive Field
> Receptive filed refers to the region in the input the filter is looking at. A NxN filter will be looking at N^2 pixels which defines its receptive field.
> It can be local or global. A local receptive field refers to the values filter is looking at in immediate input layer. Global receptive field is the all the values a feature map has looked into in the original input. 
> For a input of 10x10 and filter of size 3x3 local receptive field has looked at 9 pixels where as global receptive field at layer 2 would have looked at 25 values.
> The below image describing a local and a global receptive field
> 
![Receptive Filed](https://cdn-images-1.medium.com/max/2000/1*mModSYik9cD9XJNemdTraw.png)
