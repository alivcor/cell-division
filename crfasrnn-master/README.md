# CRF-RNN for Semantic Image Segmentation
![sample](sample.png)

[![License (3-Clause BSD)](https://img.shields.io/badge/license-BSD%203--Clause-brightgreen.svg?style=flat-square)](https://github.com/torrvision/crfasrnn/blob/master/LICENSE)

<b>Live demo:</b> [http://crfasrnn.torr.vision](http://crfasrnn.torr.vision)
<b>update:</b> This version of code is integrated with the latest caffe future version.

This package contains code for the "CRF-RNN" semantic image segmentation method, published in the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf). This paper was initially described in an [arXiv tech report](http://arxiv.org/abs/1502.03240). Our software is built on top of the [Caffe](http://caffe.berkeleyvision.org/) deep learning library. The current version was developed by:

[Sadeep Jayasumana](http://www.robots.ox.ac.uk/~sadeep/),
[Shuai Zheng](http://kylezheng.org/),
[Bernardino Romera Paredes](http://romera-paredes.com/), 
[Anurag Arnab](http://www.robots.ox.ac.uk/~aarnab/),
and
Zhizhong Su.

Supervisor: [Philip Torr](http://www.robots.ox.ac.uk/~tvg/)

Our work allows computers to recognize objects in images, what is distinctive about our work is that we also recover the 2D outline of the object.

Currently we have trained this model to recognize 20 classes. This software allows you to test our algorithm on your own images – have a try and see if you can fool it, if you get some good examples you can send them to us.

Why are we doing this? This work is part of a project to build augmented reality glasses for the partially sighted. Please read about it here: [smart-specs](http://www.va-st.com/smart-specs/). 

For demo and more information about CRF-RNN please visit the project website <http://crfasrnn.torr.vision>.

If you use this code/model for your research, please consider citing the following papers:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```
```
@inproceedings{higherordercrf_ECCV2016,
	author = {Anurag Arnab and Sadeep Jayasumana and Shuai Zheng and Philip H. S. Torr},
	title  = {Higher Order Conditional Random Fields in Deep Neural Networks},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year   = {2016}
}
```


#How to use the CRF-RNN layer
Copy and paste the layer into a new prototxt file, the usage of this layer is indicatd as below. Check example folder for more detailed examples.
```
# This is part of FCN, coarse is the variable coming from FCN
layer { type: 'Crop' name: 'crop' bottom: 'bigscore' bottom: 'data' top: 'coarse' }

# This layer is used to split the output of FCN into two, which is required by CRF-RNN
layer { type: 'Split' name: 'splitting'
  bottom: 'coarse' top: 'unary' top: 'Q0'
}

layer {
  name: "inference1"#if you set name "inference1", code will load parameters from caffemodel. Otherwise it will create a new layer with manually set parameters
  type: "MultiStageMeanfield" #type of this layer
  bottom: "unary" #input from FCN
  bottom: "Q0" #input from FCN
  bottom: "data" #input image
  top: "pred" #output of CRF-RNN
  param {
    lr_mult: 10000#learning rate for W_G
  }
  param {
  lr_mult: 10000#learning rate for W_B
  }
  param {
  lr_mult: 1000 #learning rate for compatiblity transform matrix
  }
  multi_stage_meanfield_param {
   num_iterations: 10 #Number of iterations for CRF-RNN
   compatibility_mode: POTTS#Initialize the compatilibity transform matrix with a matrix whose diagonal is -1.
   threshold: 2
   theta_alpha: 160
   theta_beta: 3
   theta_gamma: 3
   spatial_filter_weight: 3
   bilateral_filter_weight: 5
  }
}
```
#Installation Guide
First, you should clone the project by doing as below.
```
git clone --recursive https://github.com/torrvision/crfasrnn.git
```

You need to compile the modified Caffe library in this repository. Instructions for Ubuntu 14.04 are included below. You can also consult the generic [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html).


###1.1 Install dependencies
#####General dependencies
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
```

#####CUDA 
Install CUDA correct driver and its SDK. Download CUDA SDK from Nvidia website. 

In Ubuntu 14.04. You need to make sure the required tools are installed. You might need to blacklist the required modules so that they do not interfere with the driver installation. You also need to uninstall your default Nvidia Driver first.
```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
``` 
open /etc/modprobe.d/blacklist.conf and add:
```
blacklist amd76x_edac
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
```
```
sudo apt-get remove --purge nvidia*
```

When you restart your PC, before loging in, try "Ctrl+Alt+F1" switch to a text-based login. Try:
```
sudo service lightdm stop
chmod +x cuda*.run
sudo ./cuda*.run
```

#####BLAS
Install ATLAS or OpenBLAS or MKL.

#####Python 
Install Anaconda Python distribution or install the default Python distribution with numpy/scipy/...

#####MATLAB (optional)
Install MATLAB using a standard distribution.

###1.2 Build the custom Caffe version
Set the path correctly in the Makefile.config. You can copy the Makefile.config.example to Makefile.config, as most common parts are filled already. You need to change it according to your environment.

After this, in Ubuntu 14.04, try:
```
make
```

If there are no error messages, you can then compile and install the python and matlab wrappers:
```
make matcaffe
```

```
make pycaffe
```

That's it! Enjoy our software!


###1.3 Run the demo
Matlab and Python scripts for running the demo are available in the matlab-scripts and python-scripts directories, respectively. You can choose either of them. Note that you should change the paths in the scripts according your environment.

####For Python fans:
First you need to download the model. In Linux, this is:
```
sh download_trained_model.sh
```
Atlernatively, you can also get the model by directly clicking the link in python-scripts/README.md.

Get into the python-scripts folder, and then type:
```
python crfasrnn_demo.py
```
You will get an output.png image.

To use your own images, just replace "input.jpg" in the crfasrnn_demo.py file.

####For Matlab fans:
First you need to download the model. In Linux, this is:
```
sh download_trained_model.sh
```
Atlernatively, you can also get the model by directly clicking the link in matlab-scripts/README.md.

Get into the matlab-scripts folder, load your matlab, then run the crfrnn_demo.m.

To use your own images, just replace "input.jpg" in the crfrnn_demo.m file.

You can also find part of our model in [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/).

####Explanation about the CRF-RNN layer:
If you would like to try out the CRF-RNN model we trained, you should keep the layer name as it is "inference1", so that the code will load the parameters from caffemodel. Otherwise, it will use the parameters set by the users in the deploy.prototxt file.

You should find out that the end-to-end trained CRF-RNN model does better than the alternatives. If you set the CRF-RNN layer name to "inference2", you should observe lower performance since the parameters for both CNN and CRF are not jointly optimized.


####For training purpose:
If you would like to train the CRF-RNN model on other dataset, please follow the piecewise steps described in our paper. You should first train a strong pixel-wise CNN model. After this, you could plug in our CRF-RNN layer into those model by adding the layer to the prototxt. Then you should be able to train the CNN and CRF-RNN layer end-to-end.

Notice that the current deploy.prototxt file we provided is tailed for PASCAL VOC Challenge. This dataset contains 21 class labels including background. You should change the num_output in the corresponding layer if you would like to finetune our model for other dataset. Also, the deconvolution layer in current code does not allow initialize the parameters through prototxt. If you change the num_output there, you should manually re-initialize the parameters in caffemodel file.

See the examples/segmentationcrfasrnn.

####why predictions are all black?
This could because you set different names for classifier in prototxt, causing the weights are not properly load. This is could also because you change the number of outputs in deconvolution layer in prototxt but you didnot initialize the deconvolution layer properly. 

####MultiStageMeanfield seg fault?
This error message occurs when you didnot place the spatial.par and bilateral.par in the script path.

####Python training script from third parties
We would like to thank Martinkersner Masahiro Imai to provide python training scripts for crf-rnn. 

1. [martinkersner python scripts for Train-CRF-RNN](https://github.com/martinkersner/train-CRF-RNN)
2. [MasazI python scripts for crfasrnn-training](https://github.com/MasazI/crfasrnn-training)

####Merge with the upstream caffe
This is possible to integrate the crfrnn layer into upstream caffe. However, due to the change of crop layers, the caffemodel we provided might require extra training to provide the same accuracy. mtourne Kindly provided a version that merged the code with upstream caffe. 

1. [mtourne upstream version with CRFRNN](https://github.com/mtourne/crfasrnn)

####GPU version of CRF-RNN
hyenal kindly provided a purely GPU version of CRF-RNN. This would lead to considerable faster training and testing for CRF-RNN.

1. [hyenal's GPU crf-rnn](https://github.com/hyenal/crfasrnn)

####Latest Caffe with CPU/GPU CRF-RNN
[crfasrnn-caffe](https://github.com/bittnt/caffe/tree/crfrnn)

Let us know if we miss any other works from third parties.


For more information about CRF-RNN please vist the project website http://crfasrnn.torr.vision. Contact: <crfasrnn@gmail.com>
