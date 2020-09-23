# ConvolutionalNeuralnetwork
The automatic brain tumor classification is very challenging task in large spatial and structural variability of surrounding region of brain tumor. 
In this work, automatic brain tumor detection is proposed by using Convolutional Neural Networks (CNN) classification.

•	The image data that was used for this problem is Brain MRI Images of brain scans and come from various sources. 
•	The dataset contains the images in two groups, first one containing images that were diagnosed with a Tumor and the second group of images that weren’t diagnosed with tumor. 
•	The decision is given as:

YES, for tumor is present. Encoded as 1. COUNT: 155

NO for tumor is not present. Encoded as 0. COUNT: 98

In deep learning a CNN is a class of deep neural networks, most commonly applied to analyzing images by converting them to numerical values and creating decision trees based on the weights and values leaned from the training dataset. A Convolutional Neural Network has the following layers: 

•	Convolutional layer: The convolutional layer is the core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels), which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter.

•	Pooling layer: Another important concept of CNNs is pooling, which is a form of non-linear down-sampling. There are several non-linear functions to implement pooling among which max pooling is the most common. It partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum.

•	ReLU layer: ReLU is the abbreviation of rectified linear unit, which applies the non-saturating activation function. It effectively removes negative values from an activation map by setting them to zero. It increases the nonlinear properties of the decision function and of the overall network without affecting the receptive fields of the convolution layer.

•	Loss layer: The "loss layer" specifies how training penalizes the deviation between the predicted (output) and true labels and is normally the final layer of a neural network. Various loss functions appropriate for different tasks may be used.

