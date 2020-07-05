**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

**Link for Assignments:**

### Assignment A - OpenCV Yolo:

- Run [this](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/) code on your laptop or Colab. 
- Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
- Run this image through the code above. 
- Upload the link to GitHub implementation of this

[GitHub Link](https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Yolo_OpenCV/yolo_object_detection.py) for Assignment A(OpenCV YOLO)

### Assignment B - Training Custom Dataset on Colab for YoloV3

- Refer to this Colab File: LINK (Links to an external site.)
- Refer to this GitHub Repo (Links to an external site.)
- Collect a dataset of 500 images and annotate them. Please select a class for which you can find a YouTube video as well. Steps are explained in the readme.md file on GitHub.
- Once done:
    - Download (Links to an external site.) a very small (~10-30sec) video from youtube which shows your class. 
    - Use ffmpeg (Links to an external site.) to extract frames from the video. 
    - Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
    - Inter on these images using detect.py file.
    - *'python detect.py --conf-thres 0.1 --output output_folder_name'*
    - Use ffmpeg (Links to an external site.) to convert the files in your output folder to video
    - Upload the video to YouTube. 

[Colab Link](https://colab.research.google.com/github/abksyed/EVA4/blob/master/13_YoloV3/Assignment_S13.ipynb) for Assignment B(Yolo Training on Guitar)

[GitHub Link](https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Assignment_S13.ipynb) for Assignment B(Yolo Training on Guitar)

### Annotating on OpenCV Yolo:

1) **When your eBike gets discharged on a Highway!**

Github Link for Code: [Object Detector](https://github.com/abksyed/EVA4/blob/master/13_YoloV3/Yolo_OpenCV/yolo_object_detection.py)
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
