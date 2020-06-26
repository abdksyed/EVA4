**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/drive/1j4tQxzskc4YI-G2i1nG6hja2NowSclqv?usp=sharing)


To run the model, first we need to install the package 'pynoob' using !pip install pynoob==0.1.0.7

Alternately, we can upload all the necessary packagesto the colab directory. The packages can be found in the 12_TinyImageNet_Classification/packages folder of EVA4 repo.

Link to pynoob: (https://github.com/abksyed/pynoob/tree/master/pynoob)

Link to model used: (https://github.com/abksyed/EVA4/blob/master/models/S12_ResNet_S12.py)

Link to all models: (https://github.com/abksyed/EVA4/blob/master/models/)

Link to Deep Learning Assistant Sara powered by Newton: (https://github.com/abksyed/pynoob/blob/master/pynoob/Newton.py)

### **Objective**

Assignment A:

- Download TINY IMAGENET (Links to an external site.) dataset. 
- Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
- Submit Results. Of course, you are using your own package for everything.


### **Model Statistics:**

- Custom Model 
- Batch Size: 256
- Number of Parameters: 11,271,432
- Epochs: 25

### **Results**

Achieved accuracy of

**Test - 53.82%**

**Train - 83.26%**

**Learning Rate Finder:**

![LRFinderPlot.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/LRFinderPlot.png)

**Change in Learning Rate:**

![ChangeLR.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/OneCycleLR.png)


**Train and Test Accuracies and Loss:**

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/TrainTestLossAcc.png)

**Train vs Test Accuracy:**

![Test-vs-Train Accuracy.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/TestvTrain.png)

**Misclassified Images:**

![MissClassifiedImages.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/MisClassify.png)

**For List of Classes in TinyImageNet**
(https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/classes.txt)

## **Entire GradCAM for Mis Classified Images (w.r.t Predicted Class):**
![Mis_GradCAM_Pred.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/Mis_GradCAM_Pred.png)

## **Entire GradCAM for Mis Classified Images(w.r.t Actual Class):**
![Mis_GradCAM_Actual.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/Mis_GradCAM_Actual.png)