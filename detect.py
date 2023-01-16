import time
from pathlib import Path

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized, TracedModel


def compute_intersection_landing(bbx1_iou_land, bbx2_iou_land):
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


def check_landing(np_detections):
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
                check_intersection = compute_intersection_landing([bbx1, bby1, bbx2, bby2],
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


# convert bounding box format from yolo to voc
def yolo_to_voc(x_center, y_center, width, height, img_width, img_height):
    box_height = height * img_height
    box_width = width * img_width
    ymin = round(float((y_center * img_height) - (box_height / 2.0)), 2)  # top_left_y
    xmin = round(float((x_center * img_width) - (box_width / 2.0)), 2)  # top_left_x
    ymax = round(float(box_height + ymin), 2)  # bottom_right_x
    xmax = round(float(box_width + xmin), 2)  # bottom_right_y
    return [xmin, ymin, xmax, ymax]


def detect_model(source_path):
    # Default detect
    det_numpy = np.empty([1, 1])
    landing_list = np.empty([1, 1])

    # PARAMETERS
    source = source_path  # can be 1 image path or a folder, both fine, not for a video stream
    weights = ['C:\\Users\\YavoBalo\\PycharmProjects\\UlasimdaYapayZekaTeknofest2023\\yolov7\\weights']
    # weights = ['yolov7.pt']
    trace = True
    save_txt = True
    save_conf = False
    save_img = False  # save inference images
    imgsz = 640
    conf_thres = 0.25  # object confidence threshold
    opt_augment = False
    opt_device = ''

    # Directories
    name = 'exp'
    project = 'runs/detect'
    iou_thres = 0.45
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt_device)
    print("select device: ", device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt_augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt_augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):  # det is tensor type
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")

            det_numpy = det.numpy()
            # xmin, ymin, xmax, ymax, conf, cls
            # detect numpy: 
            # [[        853         515         945         580     0.97341           0]
            # [        419         398         498         455     0.95355           0]
            # [        585         637         677         705     0.93958           0]]
            # landing:  
            # [-1, -1, -1]
            print("detect numpy: ", det_numpy)
            landing_list = check_landing(det_numpy)
            print("\nlanding: ", landing_list, "\n")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    return det_numpy, landing_list


if __name__ == '__main__':
    source_path = "teknofest-images\\frame_000040.jpg"
    # source_path = "inference\images\horse_22.jpg"
    with torch.no_grad():
        detect_model(source_path)
