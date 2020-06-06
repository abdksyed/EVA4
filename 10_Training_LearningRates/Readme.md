**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/github/abksyed/EVA4/blob/master/09_DataAugmentation/Assignment_S9.ipynb)

To run the model, first we need to install the package 'pynoob' using !pip install pynoob.

Alternately, we can upload all the necessary packagesto the colab directory. The packages can be found in the 10_Training_LearningRates/packages folder of EVA4 repo.

Link to pynoob: (https://github.com/abksyed/pynoob)

Link to model used: (https://github.com/abksyed/EVA4/blob/master/models/S10_model_CusResNet.py)

### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
- Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
    -Move LR Finder code to your modules
    -Implement LR Finder (for SGD, not for ADAM)
    -Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
-Find best LR to train your model
-Use SDG with Momentum
-Train for 50 Epochs. 
-Show Training and Test Accuracy curves
-Target 88% Accuracy.
-Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.

### **Model Statistics:**

- Custom ResNet18 - BasicBlock - [2,2,2,2] - Last Layer Stride = 1 - So Final Output is 8 x 8!
- Batch Size: 128
- Number of Parameters: 11,173,962
- Epochs: 50

### **Results**

Achieved accuracy of

**Test - 92.95%**

**Train - 90.65%**

Learning Rate Finder:

![LRFinderPlot.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/LRFinderPlot.png)

Change in Learning Rate:

![ChangeLR.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/ChangeLR.png)

Misclassified Images:

![MissClassifiedImages.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/MisClassify.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/TrainTestLossAcc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/TestvTrain.png)

GradCAM for Mis Classified Images (w.r.t Predicted Class):
![Mis_HeatMap.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/Mis_GradCAM_Pred.png)

GradCAM for Mis Classified Images(w.r.t Actual Class):
![Mis_GradCAM.png](https://github.com/abksyed/EVA4/blob/master/10_Training_LearningRates/Images/Mis_GradCAM_Actual.png)


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
