import os
import sys
import time
import warnings
import numpy as np
import cv2
import torch
import torchvision.ops
import xdg.BaseDirectory
from torchvision import transforms
from retinaface.pre_trained_models import get_model
from align_faces import warp_and_crop_face
import align_faces

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if device == 'cpu':
        model = get_model("resnet50_2020-07-20", max_size=720, device=device)
    else:
        model = get_model("resnet50_2020-07-20", max_size=1400, device=device)
model.eval()

face_model = None
face_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def load_arcface():
    global face_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(os.path.join(xdg.BaseDirectory.xdg_cache_home, "InsightFace-v2", "BEST_checkpoint_r101.tar"), map_location=torch.device(device))
        face_model = checkpoint['model'].module.to(device)
        face_model.device = device
        face_model.eval()

# image needs to be RGB not BGR
def embed_faces(annotations, filename=None, image=None):
    global face_model
    if len(annotations) < 1:
        return True
    with torch.no_grad():
        if image is None and filename is not None:
            image = cv2.imread(filename)
            if image is None:
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            return False
        images = torch.zeros(len(annotations), 3, 112, 112).to(device)
        for i, annotation in enumerate(annotations):
            face = warp_and_crop_face(image, annotation['landmarks'], reference_pts=align_faces.REFERENCE_FACIAL_POINTS_112, crop_size=(112,112))
            images[i] = face_transforms(face).to(device)
        embedding = face_model(images)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()
        for i, annotation in enumerate(annotations):
            annotation['embedding'] = embedding[i]
        return True

def annotate(annotations, filename=None, image=None, scale=1.0, face_id=None):
    if image is None and filename is not None:
        image = cv2.imread(filename)
    if image is None:
        return None
    for i, annotation in enumerate(annotations):
        color = (0, 0, 255)
        if face_id is not None and face_id == i:
            color = (0, 255, 0)
        bbox = (int(annotation['bbox'][0] * scale + 0.5), int(annotation['bbox'][1] * scale + 0.5), int(annotation['bbox'][2] * scale + 0.5), int(annotation['bbox'][3] * scale + 0.5))
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        for landmark in annotation['landmarks']:
            image = cv2.circle(image, (int(landmark[0] * scale + 0.5), int(landmark[1] * scale + 0.5)), 1, (0, 0, 255), 2)

        y = bbox[1] - 4
        if y - 24 < 0:
            y = bbox[3] + 28

        tag = ""
        if 'tag' in annotation:
            tag = annotation['tag']

        image = cv2.putText(image, str(i) + tag, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        image = cv2.putText(image, str(i) + tag, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return image

# image needs to be RGB not BGR
def get_faces(filename=None, image=None):
    with torch.no_grad():
        if image is None and filename is not None:
            image = cv2.imread(filename)
            if image is None:
                return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            return []
        annotations = model.predict_jsons(image, nms_threshold=0.25, confidence_threshold=0.85)
        if len(annotations) < 1 or len(annotations[0]['bbox']) != 4 or len(annotations) > 500:
            return []

        boxes = torch.zeros(len(annotations), 4)
        for i, annotation in enumerate(annotations):
            boxes[i, :] = torch.tensor((annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2], annotation['bbox'][3])).float()

        valid_boxes = torchvision.ops.remove_small_boxes(boxes, 40)

        valid_annotations = []
        for box_id in valid_boxes:
            annotation = annotations[box_id]
            valid_annotations.append(annotation)
        embed_faces(valid_annotations, image=image)

        return valid_annotations

if __name__ == "__main__":
    load_arcface()
    for filename in sys.argv[1:]:
        process_start = time.perf_counter()
        annotations = get_faces(filename)
        process_time = time.perf_counter() - process_start
        print(f"Processing time: {process_time:.4f}s")
        image = annotate(annotations, filename)
        try:
            cv2.imshow('Image', image)
            cv2.waitKey(0)
        except:
            pass
