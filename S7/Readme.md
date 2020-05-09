**Members:**
[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:[https://colab.research.google.com/drive/1GOzQzdFFGgejLamHnWxlHmKdGh26-ZnF#scrollTo=ur2bt7oBauPc)

To run the model, we need to upload all the necessary packagesto the colab directory. The packages can be found in the S7 folder of EVA4 repo.


### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- The code must utilize GPU
- The architecture must be C1C2C3C40 (basically 3 MPs).
- total RF must be more than 44.
- Atleast one of the layers must use Depthwise Separable Convolution
- Atleast one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target no. of classes (optional)
- achieve **80%** accuracy, as many epochs as you want. Total Params to be less than **1M.**

### **Model Statistics:**

- Batch Size: 128
- Number of Parameters: 256,440
- Dropout: 0.1
- GhostBatchNormalization
- Epochs: 50

### **Results**

Achieved accuracy of

**Test - 84.98%**
**Train - 86.66%**

Misclassified Images:

![MissClassifiedImages.png](hhttps://github.com/abksyed/EVA4/blob/master/S7/Images/MissClassifiedImages.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/S7/Images/Test-vs-Train%20Accuracy.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy.png]([https://github.com/abksyed/EVA4/blob/master/S7/Images/Test-vs-Train%20Accuracy.png](https://github.com/abksyed/EVA4/blob/master/S7/Images/Test-vs-Train%20Accuracy.png))

Class Wise Accuracies:

Accuracy of plane : 84 %

Accuracy of car : 92 %

Accuracy of bird : 77 %

Accuracy of cat : 67 %

Accuracy of deer : 84 %

Accuracy of dog : 80 %

Accuracy of frog : 91 %

Accuracy of horse : 86 %

Accuracy of ship : 92 %

Accuracy of truck : 92 %
