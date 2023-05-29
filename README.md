# OpenPifPaf

# Introduction

The goal of our project was to do 2D object detection using the OpenPifPaf model provided here. Indeed, OpenPifPaf already has an extension called *NuScenes 2D detection* that detects the 2D bounding boxes of objects from the NuScenes dataset. However, we decided to use the classic cifcaf model in order to detect and predict the boundign boxes around the objects. 
Thus, we decided to implement 2D object detection using the *cifcaf* module normally used for keypoints detection. 

# Methodology

To make our model able to take the coordinates of the bounding boxes as they were keypoints, in order to build a "skeleton" corresponding to it, we had to create a new plugin for that dataset. 

We created the *openpifpaf_coco_det* plugin that is very similar to the coco plugin. The biggest changes are in the *constants.py* file, where we defined how the keypoints of the bounding boxes are related to each other (to build the skeleton), their names, how they are located in the space and their precision of localization. In this way, our model will not detect the upper left corner of a box, but it will detect the upper left corner of a **human box**. 

Then, we had to change the input of the model during training: instead of taking the keypoints, it should take the coordinates of the angles of the bounding box. In a JSON file, the bbox is given as [x,y, width, height, visibility], whereas the keypoints are given as [..., x_i, y_i, visibility_i, ... ]. Therefore, we had to transform the bbox into the keypoints format. This was made at two places in the code:
1. In the *encoder.annrescaler.py* file, used to extract and rescale the annotations given to the model
2. In the *transforms.annotations.py* file, used to extract the data before transforming it (for example horizontal flip). 

# What is not implemented yet

For now, only the **person** category has been taken into account. The model predicts the keypoints of the bounding box and not the bounding boxes themselves.

We can thus improve the following points in the future:

- Multiple class detection: in a further improvement, we could add new classes to be detected (up to 80 classes if we use the COCO dataset) by changing the number of box keypoints taken into consideration. The network will then assign keypoints specific to each class. (example: *left_corner_human* if it is the left corner of a human box or *left_corner_cat* if it is the left corner of a cat box).
- Box keypoints to bounding boxes transformation: we can either change the decoder to do it automaticaly, or compute bonding box from the box keypoints outputs. Indeed, for now the "bounding boxes" output by the model are several thick lines creating a rectangle with all vertices linked to the center. A bounding box should display only a rectangle around the object with the name of the class corresponding to that object.

Furthermore, we trained the model with a *shufflenetv2k16* basenet. To improve the performance of our model, we could try other basenets, add a weight decay or fine tune the learning rate.
These are more model tuning improvements that can be studied and applied.  

Finally we could implement a validation function made for this application. For now we are using the openpifpaf "person" keypoint validation function, which is not adapted to bounding box prediction.

# Conclusion 

In conclusion we have a model that detects human bonding boxes. This model was trained by receiving the bounding box coordinates as keypoints during the training. The prediction of our model has a low accuracy for now, but we proved that it is possible to do detection from box keypoints using openpifpaf keypoints detection model. It could be interesting to dig more into the subject to see if it is possible to improve the accuracy of the model and open it to multiclass detection.
As explained above the model could be improve either by model tuning (hyperparameters tuning) or by acting on the dataset loader (to add other classes). 

# How to use it?

To use our model to predict the detection on your images, you can follow the following steps.

1. Download the project code from this git repository: 

2. Run the following command from the main folder for the prediction: 

To save the image and the json output:
```sh
  python3 -m openpifpaf.predict <image_path> --checkpoint outputs/shufflenetv2k16-230528-181950-cocoboxkp.pkl.epoch002 --image-output <image_path> --json-output <json_path>
```

To vizualized the image directly (you need to install matplotlib) and select the weights you want to use: 
```sh
  python3 -m openpifpaf.predict <image_path> --checkpoint outputs/shufflenetv2k16-230528-181950-cocoboxkp.pkl.epoch002 --show
```


## [Guide](https://vita-epfl.github.io/openpifpaf/intro.html)

The detailed documentation concerning the OpenPifPaf model can be found in the __[OpenPifPaf Guide](https://vita-epfl.github.io/openpifpaf/intro.html)__.
For developers, there is also the
__[DEV Guide](https://vita-epfl.github.io/openpifpaf/dev/intro.html)__
which is the same guide but based on the latest code in the `main` branch.

We based our work on this guide, please refer to it if the information provided below are not sufficient. Moreover, if you like to use the other plugins or datasets provided by openpifpaf, please use the corresponding code in the __[original GitHub repository](https://github.com/vita-epfl/openpifpaf)__.

## Examples

![Example image with predicted bbox](https://github.com/alechp13/openpifpaf_bbox/blob/main/images_readme/0018.jpg.predictions.jpeg)

Created with:
```sh
python3 -m openpifpaf.predict docs/coco/0018.jpg --image-output
```


# Commercial License

The open source license is in the [LICENSE](https://github.com/vita-epfl/openpifpaf/blob/main/LICENSE) file.
This software is also available for licensing via the EPFL Technology Transfer
Office (https://tto.epfl.ch/, info.tto@epfl.ch).


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/
