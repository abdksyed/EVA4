**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/github/abksyed/EVA4/blob/master/09_DataAugmentation/Assignment_S9.ipynb)

### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
- Please make sure that your test_transforms are simple and only using ToTensor and Normalize
- Implement GradCam function as a module. 
- Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
- Target Accuracy is 87%

### **Model Statistics:**

- Custom ResNet18 - BasicBlock - [2,2,2,2] - Last Layer Stride = 1 - So Final Output is 8 x 8!
- Batch Size: 128
- Number of Parameters: 11,173,962
- Epochs: 15

### **Results**

Achieved accuracy of

**Test - 86.95%**

**Train - 90.65%**

Misclassified Images:

![MissClassifiedImages.png](https://github.com/abksyed/EVA4/blob/master/09_DataAugmentation/Images/MisClassify.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/09_DataAugmentation/Images/TrainTestLossAcc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy.png](https://github.com/abksyed/EVA4/blob/master/09_DataAugmentation/Images/TestvTrain.png)

GradCAM HeatMap for Mis Classified Images:
![Mis_HeatMap.png](https://github.com/abksyed/EVA4/blob/master/09_DataAugmentation/Images/Mis_HeatMap.png)

GradCAM for Mis Classified Images:
![Mis_GradCAM.png](https://github.com/abksyed/EVA4/blob/master/09_DataAugmentation/Images/Mis_GradCAM.png)


Class Wise Accuracies:

Accuracy of plane : 85 %

Accuracy of   car : 97 %

Accuracy of  bird : 85 %

Accuracy of   cat : 79 %

Accuracy of  deer : 92 %

Accuracy of   dog : 80 %

Accuracy of  frog : 83 %

Accuracy of horse : 86 %

Accuracy of  ship : 88 %

Accuracy of truck : 91 %
