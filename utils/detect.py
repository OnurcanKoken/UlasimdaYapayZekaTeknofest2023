import time
from pathlib import Path

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized, TracedModel


class Yolo(object):
    def __init__(self):
        # PARAMETERS
        self.trace = True
        self.save_txt = False
        self.save_conf = False
        self.save_img = False   # save inference images
        self.opt_augment = False
        self.printable = False
        self.plot = False       # draw detected bounding boxes on images
        self.imgsz = 640
        self.conf_thres = 0.25  # object confidence threshold
        self.iou_thres = 0.45

        # Directories
        self.name = 'exp'
        self.project = 'runs/detect'
        self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=False))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load Model
        self._load_model()

    def _load_model(self):
        # Initialize
        self.opt_device = ''
        set_logging()
        self.device = select_device(self.opt_device)
        print("select device: ", self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.weights = ['weights/best.pt']
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, 640)

        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once

    def __compute_intersection_landing(self, bbx1_iou_land, bbx2_iou_land):
        x_left = max(bbx1_iou_land[0], bbx2_iou_land[0])
        y_top = max(bbx1_iou_land[1], bbx2_iou_land[1])
        x_right = min(bbx1_iou_land[2], bbx2_iou_land[2])
        y_bottom = min(bbx1_iou_land[3], bbx2_iou_land[3])
        if x_right < x_left or y_bottom < y_top:
            return 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            # print("\nintersection area: ", intersection_area)
            if intersection_area > 0:
                return 1
            else:
                return 0

    def check_landing(self, np_detections):
        landing_list = []
        bounding_boxes = []
        for j in range(0, len(np_detections)):
            check_intersection = 0
            bbx1, bby1, bbx2, bby2, conf, cls = np_detections[j]
            bounding_boxes.append([bbx1, bby1, bbx2, bby2, cls])
            if cls == 0 or cls == 1:
                landing_list.append(-1)
            elif cls == 2 or cls == 3:
                check_intersection = 0
                for k in range(0, len(np_detections)):
                    bbx1_2, bby1_2, bbx2_2, bby2_2, conf_2, cls_2 = np_detections[k]
                    if cls_2 == 2 or cls_2 == 3:
                        continue
                    check_intersection = self.__compute_intersection_landing([bbx1, bby1, bbx2, bby2],
                                                                            [bbx1_2, bby1_2, bbx2_2, bby2_2])
                    if check_intersection == 1:  # break, its not available, no need to check further
                        print("INTERSECTION")
                        break
                if check_intersection == 1:
                    landing_list.append(0)  # not available for landing
                else:
                    landing_list.append(1)  # available for landing
            else:
                landing_list.append(-1)
        return landing_list

    def plot_one_box(self, x, img, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = [100, 0, 100]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def detect_model(self, source_path):
        # Default detect
        det_numpy = np.empty([1, 1])
        landing_list = np.empty([1, 1])

        # Set Dataloader
        dataset = LoadImages(source_path, img_size=self.imgsz, stride=self.stride)

        # Get names
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.opt_augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.opt_augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                if len(det):  # det is tensor type
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    if self.printable:
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Add bbox to image
                    if self.plot:
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            self.plot_one_box(xyxy, im0, label=label, line_thickness=1)

                # Print time (inference + NMS)
                if self.printable:
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")

                # xmin, ymin, xmax, ymax, conf, cls
                # detect numpy:
                # [[        853         515         945         580     0.97341           0]
                # [        419         398         498         455     0.95355           0]
                # [        585         637         677         705     0.93958           0]]
                # landing:
                # [-1, -1, -1]
                det_numpy = det.numpy()
                landing_list = self.check_landing(det_numpy)

        return det_numpy, landing_list
