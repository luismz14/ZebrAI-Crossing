import torch
import os
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    detections = results.pandas().xyxy[0]
    results.render()

    G_list = []
    R_list = []
    zebra_crops = []

    for _, row in detections.iterrows():
        class_name = row['name']
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        coords = (xmin, ymin, xmax, ymax)
        confidence = round(float(row['confidence']), 2)

        if class_name == 'Zebra_Cross':
            cropped = img[ymin:ymax, xmin:xmax]
            zebra_crops.append((cropped, confidence))
        elif class_name == 'G_Signal':
            G_list.append({'coords': coords, 'confidence': confidence})
        elif class_name == 'R_Signal':
            R_list.append({'coords': coords, 'confidence': confidence})

    return results, G_list, R_list, zebra_crops

def display_results(image_path, zebra_crops, G_list, R_list):
    if zebra_crops:
        for crop, _ in zebra_crops:
            cv2.imshow("", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
    
    if G_list:
        print(f'{os.path.basename(image_path)} - G_Signal:', G_list)
    if R_list:
        print(f'{os.path.basename(image_path)} - R_Signal:', R_list)

if __name__ == "__main__":
    model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
    carpeta = '../data/'

    for filename in os.listdir(carpeta):
        image_path = os.path.join(carpeta, filename)
        results, G_list, R_list, zebra_crops = process_image(image_path, model)
        display_results(image_path, zebra_crops, G_list, R_list)
