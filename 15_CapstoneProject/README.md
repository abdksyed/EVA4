# Assignment 15 - Image Segmentation and Depth Estimation

Vision is one of the most important senses humans possess. The ability to capture light and make meaning out of it is a very convoluted task. For computers, these images are nothing but matrices and understanding the nuances behind these matrices has been a challenging task for researchers for many years until the rise of Deep Learning in Computer Vision. Many problems which were previously considered untouchable are now showing astounding results.

Two such problems are Image Segmentation and Depth Estimation.  
In Image Segmentation, the machine has to partition the image into different segments, each of them representing a different entity.  
In Depth Estimation, the machine has to extract the depth information of the foreground entities from a single image.

### Model Used : UNet with BatchNormalizatoin.

### Number of Parameters - 7M and 31M
* Two models were built to test, one with 7 million Parameters and other with 31 Million Parameters, and both performed almost equally well.


### Below are the Results for each

#### UNet - 7M Parameters.
![FinalResult_7M](https://github.com/abdksyed/EVA4/blob/master/15_CapstoneProject/results/FinalResults_7M.png)

#### UNet - 31M Parameters.
![FinalResult_31M](https://github.com/abdksyed/EVA4/blob/master/15_CapstoneProject/results/FinalResults_31M.png)
