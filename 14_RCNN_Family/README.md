
# Dataset Creation Assignment:
## Objective: 

The Data set must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images.

#### In total you MUST have:
* 400k fg_bg images
* 400k depth images
* 400k mask images

#### Which are generated from:
* 100 backgrounds
* 100 foregrounds, plus their flips
* 20 random placement on each background.

#### Add your data set statistics:

* Kinds of images (fg, bg, fg_bg, masks, depth)
* Total images of each kind
* The total size of the dataset
* Mean/STD values for your fg_bg, masks and depth images


#### Explain how you created your data set
* how were fg created with transparency
* how were masks created for fgs
* how did you overlay the fg over bg and created 20 variants
* how did you create your depth images?

##### Add the notebook file to your repo, one which you used to create this dataset  
##### Add the notebook file to your repo, one which you used to calculate statistics for this dataset

#### Things to remember while creating this dataset:

> *stick to square images to make your life easy.*

We would use these images in a network which would take an fg_bg image AND bg image, and predict your MASK and Depth image. So the input to the network is, say, 224x224xM and 224x224xN, and the output is 224x224xO and 224x224xP.

pick the resolution of your choice between 150 and 250 for ALL the image.
  

## 

### Dataset Google Drive link
The link contains **TWO** datasets, one of *192x192* size 1.2 million images and other *96x96* size 1.2 million images.
https://drive.google.com/drive/folders/1lChNmM_Cgtt_fCqeWpPELDV7KUqDySD2?usp=sharing


### Dataset Creation

Github Link: 
https://github.com/abdksyed/EVA4/blob/master/14_RCNN_Family/Creating_Dataset.ipynb

Colab Link: 
https://colab.research.google.com/drive/1J9_xAP6BM2i-nHUEceSj-ZGVzQsz5K31?usp=sharing

### Depth Map creation

Github Link:
https://github.com/abdksyed/EVA4/blob/master/14_RCNN_Family/Create_Depth.ipynb

Colab link:
https://colab.research.google.com/drive/1_GCNTBvty3eV5cxreOInyyPXL9-gszby?usp=sharing


### Finding Statistics

Github Link:
https://github.com/abdksyed/EVA4/blob/master/14_RCNN_Family/Datasets_Statistics.ipynb

Colab Link:
https://colab.research.google.com/drive/1YjU2SVP4utorndX01QI7fH_ox7adWhzv?usp=sharing

### Dataset Stats:
****************************** FINAL STATISTICS Of 192x192 Image Dataset ******************************
**Background**        : 
mean:[0.5810659527778625, 0.5633341670036316, 0.5403051972389221]  
std:[0.18558748066425323, 0.19134603440761566, 0.20390349626541138]
**FG_BG Overlay**     : 
mean:[0.56414794921875, 0.5432481169700623, 0.5214933753013611
std:[0.208669513463974, 0.213966503739357, 0.2230370044708252]
**Mask of FG**        :
 mean:[0.09639497846364975, 0.09639497846364975, 0.09639497846364975]
 std:[0.2823259234428406, 0.2823259234428406, 0.2823259234428406]
**Depth of FG_BG**    : 
mean:[0.40525129437446594, 0.40525129437446594, 0.40525129437446594]
std:[0.24363896250724792, 0.24363896250724792, 0.24363896250724792]


****************************** FINAL STATISTICS Of 96x96 Image Dataset ******************************
**Background**        :
 mean:[0.5058116316795349, 0.5046520829200745, 0.475577712059021
 std:[0.1984403431415558, 0.19773632287979126, 0.21839052438735962]
**FG_BG Overlay**     :
 mean:[0.49555137753486633, 0.49184027314186096, 0.4642339050769806]
 std:[0.2157932072877884, 0.21571052074432373, 0.23315846920013428]
**Mask of FG**        :
 mean:[0.0985172837972641, 0.0985172837972641, 0.0985172837972641]
 std:[0.27966567873954773, 0.27966567873954773, 0.27966567873954773]
**Depth of FG_BG**    :
 mean:[0.34641894698143005, 0.34641894698143005, 0.34641894698143005]
 std:[0.22371560335159302, 0.22371560335159302, 0.22371560335159302]
  

### Dataset Visualization