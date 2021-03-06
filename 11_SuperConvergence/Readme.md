**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/drive/1j4tQxzskc4YI-G2i1nG6hja2NowSclqv?usp=sharing)


To run the model, first we need to install the package 'pynoob' using !pip install pynoob==0.1.0.6

Alternately, we can upload all the necessary packagesto the colab directory. The packages can be found in the 11_SuperConvergence/packages folder of EVA4 repo.

Link to pynoob: (https://github.com/abksyed/pynoob/tree/master/pynoob)

Link to model used: (https://github.com/abksyed/EVA4/blob/master/models/S11_model.py)

Link to all models: (https://github.com/abksyed/EVA4/blob/master/models/)

Link to Deep Learning Assistant Sara powered by Newton: (https://github.com/abksyed/pynoob/blob/master/pynoob/Newton.py)

### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that

![Assignment.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/Assignment.png)

- Write a code which

    - uses this new ResNet Architecture for Cifar10:
        - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        - Layer1 -
            - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
            - Add(X, R1)
        - Layer 2 -
            - Conv 3x3 [256k]
            - MaxPooling2D
            - BN
            - ReLU
        - Layer 3 -
            - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            - Add(X, R2)
        - MaxPooling with Kernel Size 4
        - FC Layer 
        - SoftMax
    - Uses One Cycle Policy such that:
        - Total Epochs = 24
        - Max at Epoch = 5
        - LRMIN = FIND
        - LRMAX = FIND
        - NO Annihilation
    - Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    - Batch size = 512
    - Target Accuracy: 90%. 
    - The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you'd get

### **Model Statistics:**

- Custom Model 
- Batch Size: 512
- Number of Parameters: 6,573,130
- Epochs: 21

### **Results**

Achieved accuracy of

**Test - 92.68%**

**Train - 97.5%**

**Learning Rate Finder:**

![LRFinderPlot.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/LRFinderPlot.png)

*Using LR 0.032 even if loss is diverged, to get regularization effect and create voltality in the descent. Since, it is constrained to reach max LR at 5th epoch so to get more diversity going from 0.001(max_lr/32) tp 0.032. This was decided after many empirical iterations.*


**Change in Learning Rate:**

![ChangeLR.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/OneCycleLR.png)


**Train and Test Accuracies and Loss:**

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/TrainTestLossAcc.png)

**Train vs Test Accuracy:**

![Test-vs-Train Accuracy.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/TestvTrain.png)

**Misclassified Images:**

![MissClassifiedImages.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/MisClassify.png)


## **Entire GradCAM for Mis Classified Images (w.r.t Predicted Class):**
![Mis_GradCAM_Pred.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/Mis_GradCAM_Pred.png)

## **Entire GradCAM for Mis Classified Images(w.r.t Actual Class):**
![Mis_GradCAM_Actual.png](https://github.com/abksyed/EVA4/blob/master/11_SuperConvergence/Images/Mis_GradCAM_Actual.png)


## **Class Wise Accuracies:**

Accuracy of plane : 95 %

Accuracy of   car : 97 %

Accuracy of  bird : 88 %

Accuracy of   cat : 84 %

Accuracy of  deer : 93 %

Accuracy of   dog : 88 %

Accuracy of  frog : 96 %

Accuracy of horse : 95 %

Accuracy of  ship : 94 %

Accuracy of truck : 91 %