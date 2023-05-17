import os
import logging
import numpy as np
from ultralytics import YOLO

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'


class DetectorV2:

    dds_v2_classes = {
        "actions": [0, 1, 2, 3, 4, 5, 6, 7],
        "persons": [8, 9],
        "roadside-objects": [10, 11]
    }

    def __init__(self, model_path='best.pt'):
        self.logger = logging.getLogger("object_detector v2")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.model_path = model_path
        self.model = YOLO(self.model_path)

        self.logger.info("Object detector initialized")

    # def run_inference_for_single_image(self, image, graph):
    def run_inference_for_single_image(self, image):

        output_dict = {}

        # YOLOv8 Implementation
        result = self.model(image)[0]

        # result = self.model.predict(source=image, show=False, save=True, save_txt=True, save_conf=True)[0]
        # time.sleep(1)

        len_results = len(result.boxes)
        boxes = result.boxes.xyxy
        list_of_boxes = [t.cpu().numpy() for t in boxes]
        categories = result.boxes.cls

        ###########################################

        classes = np.ones([100], dtype=int)
        for i in range(0, 100):
            if len(boxes) > i:
                classes[i] = categories[i].item()

        ###########################################

        boxes_array = []
        for i in range(100):
            if len(boxes) > i:
                xMin = list_of_boxes[i][0]
                yMin = list_of_boxes[i][1]
                xMax = list_of_boxes[i][2]
                yMax = list_of_boxes[i][3]
                # boxes_array.append(list_of_boxes[i].tolist())
                boxes_array.append(
                    [yMin / image.shape[0], xMin / image.shape[1], yMax / image.shape[0], xMax / image.shape[1]])
            else:
                boxes_array.append([float(0), float(0), float(0), float(0)])

        detection_boxes_array = np.vstack(boxes_array)

        ###########################################

        scores = result.boxes.conf
        confidence_array = np.zeros([100], dtype=float)
        for i in range(0, 100):
            if len(boxes) > i:
                confidence_array[i] = scores[i].item()

        ###########################################

        output_dict['num_detections'] = len_results

        output_dict['detection_classes'] = classes

        output_dict['detection_boxes'] = detection_boxes_array

        output_dict['detection_scores'] = confidence_array

        return output_dict

    def infer(self, image_np):
        imgae_crops = image_np

        # this output_dict contains both final layer results and RPN results
        output_dict = self.run_inference_for_single_image(imgae_crops)

        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        for i in range(len(output_dict['detection_boxes'])):
            object_class = output_dict['detection_classes'][i]
            relevant_class = False
            for k in DetectorV2.dds_v2_classes.keys():
                if object_class in DetectorV2.dds_v2_classes[k]:
                    object_class = k
                    relevant_class = True
                    break
            if not relevant_class:
                continue

            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            confidence = output_dict['detection_scores'][i]
            box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
            results.append((object_class, confidence, box_tuple))

        # Get RPN regions along with classification results
        # rpn results array will have (class, (xmin, xmax, ymin, ymax)) typles
        results_rpn = []
        # results_rpn code base removed

        return results, results_rpn
