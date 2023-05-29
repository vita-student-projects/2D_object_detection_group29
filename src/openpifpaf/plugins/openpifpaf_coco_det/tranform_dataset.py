from collections import defaultdict
import copy
import logging
import os
import json
import time
import torch.utils.data
from PIL import Image
import openpifpaf
import numpy as np
from pycocotools.coco import COCO

from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_BOX_SKELETON,
    COCO_BOX_SIGMAS,
    COCO_BOX_SCORE_WEIGHTS,
    COCO_UPRIGHT_POSE,
    HFLIP,
)



class Json_Updating: 

    def __init__(self, input_file):
        print('#################################### Person to Box json file init ####################################')
        self.json_file = {}
        self.input_file = input_file

        # Keep only the folder and not the file name
        file_name = os.path.basename(os.path.normpath(input_file))
        self.output_path = input_file.replace(file_name,"")
        
        os.path.basename(os.path.normpath('input_file'))


    def initiate_json(self):
        """
        Initiate json file: one for training phase and another one for validation.
        """
        self.json_file["info"] = dict(url="https://github.com/vita-epfl/openpifpaf",
                                    date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                                    description="Conversion of ApolloCar3D dataset into MS-COCO format")
        
        self.json_file["categories"] = [] # Empty for initialization
        self.json_file["images"] = []  # Empty for initialization
        self.json_file["annotations"] = []  # Empty for initialization

    def process_category(self, name, id, skeleton, supercategory, keypoints):
        """
        Update image field in json file
        """
        self.json_file["categories"].append({
            'name' : name,  # Category name
            'id' : id,  # Id of category
            'skeleton' : skeleton,  # Skeleton connections (check constants.py)
            'supercategory' : supercategory,  # Same as category if no supercategory
            'keypoints' : keypoints})


    def process_image(self, images_anns):
        """
        Update image field in json file
        """
        self.json_file["images"].append(images_anns)


    def process_annotation(self, image_id, category_id, iscrowd, id, area, bbox, num_keypoints, keypoints, segmentation):
        """
        Process and include in the json file a single annotation (instance) from a given image
        """
        self.json_file["annotations"].append({
            'image_id': image_id,  # Image id
            'category_id': category_id,  # Id of the category (like car or person)
            'iscrowd': iscrowd,  # 1 to mask crowd regions, 0 if the annotation is not a crowd annotation
            'id': id,  # Id of the annotations
            'area': area,  # Bounding box area of the annotation (width*height)
            'bbox': bbox,  # Bounding box  coordinates (x0, y0, width, heigth), where x0, y0 are the left corner
            'num_keypoints': num_keypoints,  # number of keypoints
            'keypoints': keypoints,  # Flattened list of keypoints [x, y, visibility, x, y, visibility, .. ]
            'segmentation': segmentation})  # To add a segmentation of the annotation, empty otherwise


    def save_json_files(self):

        file_name = 'bbox_keypoints_2017.json'
        folder = self.output_path
        path_json = os.path.join(folder, file_name)

        with open(path_json, 'w') as outfile:
                json.dump(self.json_file, outfile)

        return path_json

    

    def transform_bbox2keypoints(self):
        
        coco_transform = COCO(self.input_file)

        category_ids = coco_transform.getCatIds()
        #print(len(category_ids))

        self.initiate_json()

        cat_ids = -1
        for cat in category_ids:
            cat_ids = cat_ids + 1
            categories = images = coco_transform.loadCats(cat)

            name = [cats.get('name') for cats in categories]
            id = [cats.get('id') for cats in categories]
            supercategory = [cats.get('supercategory') for cats in categories]

            keypoints = [f'{name[0]}_box_center',f'{name[0]}_box_left_up_corner',f'{name[0]}_box_right_up_corner',f'{name[0]}_box_left_down_corner',f'{name[0]}_box_right_down_corner']
            cat_mult = 5*(cat_ids)
            skeleton = [[1 + cat_mult, 2 + cat_mult], [1 + cat_mult, 3 + cat_mult], [1 + cat_mult, 4 + cat_mult], [1 + cat_mult, 5 + cat_mult], [2 + cat_mult, 3 + cat_mult], [2 + cat_mult, 4 + cat_mult], [3 + cat_mult, 5 + cat_mult], [5 + cat_mult, 4 + cat_mult]]

            self.process_category(name[0], id[0], skeleton, supercategory[0], keypoints)
            
            ids = coco_transform.getImgIds(catIds=cat)
            #print(ids)

            # nb of cats = 80
            box_80 = [0 for i in range(80*15)]

            for image_id in ids:
                # Get ids from annotation and categories 
                ann_ids = coco_transform.getAnnIds(imgIds=image_id, catIds=cat)
                #cat_ids = coco.getCatIds()

                ########### Images extraction ###########

                # Extract the data of annotation, images and categories
                images = coco_transform.loadImgs(image_id)
                
                # Write the images anns in the json
                self.process_image(images[0])
                
                ########### annotation extraction ###########

                # load anns from the json file
                anns = coco_transform.loadAnns(ann_ids) # load segemntation data
                #anns = [ann for ann in anns if not ann.get('iscrowd')]
                
                # extract kps from the anns
                #kp_anns = [ann for ann in anns if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])] # Check si keypoint présents et > 0
                #p_anns = [ann.get('keypoints') for ann in kp_anns] #Load the kp

                # extract boxes from the anns
                box_anns = [ann for ann in anns if 'bbox' in ann and any(v > 0.0 for v in ann['bbox'])] # Check si il y a au moins 1 bbox > 0
                box_anns = [ann.get('bbox') for ann in box_anns]

                # Create new boxes
                if len(box_anns) > 0:
                    new_box_anns = []
                    box = box_anns[0]
                    b0, b1, w, h = box[0], box[1], box[2], box[3]
                    box_vals = [round((b0 + w)/2.0, 2), round((b1+h)/2.0, 2), 2., round(b0,2), round(b1,2), 2., round(b0+w,2), round(b1,2), 2., round(b0,2), round(b1+h,2), 2., round(b0+w,2), round(b1+h,2), 2.]
                    for i in range(len(box_vals)): new_box_anns.append(box_vals[i])
                else : 
                    new_box_anns = [0 for i in range(15)] # Put new boxes to 0 if there is no bounding boxes
                    box_anns = [[0,0,0,0]]

                # put box in the form of 80 categories boxes
                for i in range(15): 
                    box_80[i + (cat_ids)*15] = new_box_anns[i]

                # extract other non changing annotations to copy inside the json file from the anns
                segm_anns = [ann.get('segmentation') for ann in anns]
                num_keypoints_anns = len(new_box_anns)/3
                area_anns = [ann.get('area') for ann in anns]
                iscrowd_anns = [ann.get('iscrowd') for ann in anns]
                id_anns = [ann.get('id') for ann in anns]
                category_id_anns = [ann.get('category_id') for ann in anns]


                self.process_annotation(image_id, category_id_anns[0], iscrowd_anns[0], id_anns[0], area_anns[0], box_anns[0], num_keypoints_anns, box_80, segm_anns[0])

        output_path = self.save_json_files()

        #Once the new file created, it return its path
        return output_path


