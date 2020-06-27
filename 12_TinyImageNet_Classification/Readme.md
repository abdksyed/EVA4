**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab Link for Assignment A:(https://colab.research.google.com/drive/1SHm8INvtwTBQk2xCugIi3y5xNb3ApT3U?usp=sharing)

GitHub Link for Assignment A: (https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Assignment_S12.ipynb)

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

**Lets check for predicted/actual classes for some of examples from above**

*For 5th row 1st Image and 5th row and 5th Image.*

![CheckClasses.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/CheckClasses.png)

**For List of Classes in TinyImageNet**

(https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/classes.txt)

### **Entire GradCAM for Mis Classified Images (w.r.t Predicted Class):**
![Mis_GradCAM_Pred.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/Mis_GradCAM_Pred.png)

### **Entire GradCAM for Mis Classified Images(w.r.t Actual Class):**
![Mis_GradCAM_Actual.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/Images/Mis_GradCAM_Actual.png)



## Assignment B:

- Download 50 images of dogs/cats. 
- Use VGG Annotation tool to annotate bounding boxes around the dogs/cats.
- Download JSON file. 
- Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
- Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub.

Colab Link for Assignment B: (https://colab.research.google.com/drive/1Aqo9rNg-DUd4KAonJpLq5BA_dsy4PCrM?usp=sharing)

GitHub Link for Assignment B: (https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/JerryBhau/Annotations/Assignment_S12_Annotate.ipynb)
### Exaplanation of the json file
The JSON file is a Key:Value pair in a text format, where the Keys are unique and mapped to respective data regarding it under values.

The main key is the Image Name and the value is characteristics such as,

- filename: the name of the file
- size: the size of the image.
- regions: this is an array consisting of the bounding boxes
  - shape_attributes: how the Bounding Box is defined, could be a circle, rectangle, etc. The x and y values are the top left corner of the Bounding Box. The width and height is the total width and height of the box.
  - region_attributes: this contained the label for the region, here 'Jerrry'.
  *If more than two bounding box are available, then the regions and region_attributes gets repreated for each bonding box*
- file_attributes: The other extra data like captions, public_domain and urls.

**Sample Bounding Box Image**
![Jerry_Annotated.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/JerryBhau/Annotations/Jerry_Annotated.png)

**K means Clustering**

*K=3*
![K=3cluster.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/JerryBhau/Annotations/K=3cluster.png)

*K=4*
![K=3cluster.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/JerryBhau/Annotations/K=4cluster.png)

**IOU**
![iou.png](https://github.com/abksyed/EVA4/blob/master/12_TinyImageNet_Classification/JerryBhau/Annotations/iou.png)
