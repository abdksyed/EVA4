**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

GitHub Link for Assignment A: (https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Yolo_OpenCV/yolo_object_detection.py)

Colab Link for Assignment B: (https://colab.research.google.com/github/abksyed/EVA4/blob/master/13_YoloV3/Assignment_S13.ipynb)

GitHub Link for Assignment B: (https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Assignment_S13.ipynb) 

### **Objective**

Assignment A - OpenCV Yolo:

- Run [this](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/) code on your laptop or Colab. 
- Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
- Run this image through the code above. 
- Upload the link to GitHub implementation of this

1) When your eBike gets discharged on a Highway!

![Moto_Col.png](https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Images/Moto_Col.png)

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
