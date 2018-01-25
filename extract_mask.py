import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class extract_mask():
	def init(self, root_dir, model_log, model_path):
		self.ROOT_DIR = root_dir # Root directory of the project
		self.MODEL_DIR = model_log # Directory to save logs and trained model
		self.COCO_MODEL_PATH = model_path # Local path to trained weights file

		if not os.path.exists(self.COCO_MODEL_PATH):
    		utils.download_trained_weights(self.COCO_MODEL_PATH)

    	self.config = InferenceConfig()
		self.config.display()

		# COCO Class names
		# Index of the class in the list is its ID. For example, to get ID of
		# the teddy bear class, use: class_names.index('teddy bear')
		self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
		               'bus', 'train', 'truck', 'boat', 'traffic light',
		               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
		               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
		               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
		               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		               'kite', 'baseball bat', 'baseball glove', 'skateboard',
		               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
		               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
		               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
		               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
		               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
		               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
		               'teddy bear', 'hair drier', 'toothbrush']




   	def inference(self, image_path):
		# Create model object in inference mode.
		model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

		# Load weights trained on MS-COCO
		model.load_weights(self.COCO_MODEL_PATH, by_name=True)

		# path = '/home/sazadi/projects/objCompose/exp/person_umbrella'
		# image_path = '%s/images/00021840-COCO_train2014_000000522464-fake_B.png'%path
		name = image_path.split('/')[-1]
		image = skimage.io.imread(image_path)

		output_path = '%s/masks'%path
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		# Run detection
		results = model.detect([image], verbose=1)

		# Visualize results
		r = results[0]

		pred_classes = ([class_names[r['class_ids'][i]] for i in range(len(r['class_ids']))])
		pred_scores = r['scores']

		visualize.display_instances(output_path, name, image, r['rois'], r['masks'], r['class_ids'], 
		                            class_names, r['scores'])

		return pred_classes, pred_scores



