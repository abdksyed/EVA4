**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/drive/116PGNm13iu26DlpDQseY2tGFZ-02hJOQ?usp=sharing)

To run the model, first we need to install the package 'pynoob' using !pip install pynoob.

Alternately, we can upload all the necessary packagesto the colab directory. The packages can be found in the 08_StandardArchitectures/packages folder of EVA4 repo.

Link to pynoob: (https://github.com/abksyed/pynoob)
Link to model used: (https://github.com/abksyed/EVA4/blob/master/models/S8-model_ResNet.py)

### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Extract the ResNet18 model from this repository and add it to your API/repo
- Use data loader, train, test, and utils code to train ResNet18 on Cifar10.
- Use default ResNet18 code (so params are fixed).
- achieve **85%** accuracy, as many epochs as required.

### **Model Statistics:**

- ResNet18 - BasicBlock - [2,2,2,2]
- Batch Size: 128
- Number of Parameters: 11,173,962
- Epochs: 25

### **Results**

Achieved accuracy of

**Test - 90.16%**

**Train - 95.30%**

Misclassified Images:

![MissClassifiedImages.png](https://github.com/abksyed/EVA4/blob/master/08_StandardArchitectures/Images/MissClassify.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss.png](https://github.com/abksyed/EVA4/blob/master/08_StandardArchitectures/Images/LossandAcc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy.png](https://github.com/abksyed/EVA4/blob/master/08_StandardArchitectures/Images/TestvTrainAcc.png)


Class Wise Accuracies:

Accuracy of plane : 96 %

Accuracy of   car : 97 %

Accuracy of  bird : 85 %

Accuracy of   cat : 74 %

Accuracy of  deer : 93 %

Accuracy of   dog : 86 %

Accuracy of  frog : 92 %

Accuracy of horse : 93 %

Accuracy of  ship : 90 %

Accuracy of truck : 93 %
