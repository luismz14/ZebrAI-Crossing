import torch
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    detections = results.pandas().xyxy[0]
    results.render()

    ptl_list = []
    zebra_crops = []

    for _, row in detections.iterrows():
        class_name = row['name']
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        coords = (xmin, ymin, xmax, ymax)
        confidence = round(float(row['confidence']), 2)

        if class_name == 'Zebra_Cross':
            zebra_crops.append((coords, confidence))
        elif class_name == 'G_Signal':
            ptl_list.append({'coords': coords, 'confidence': confidence, 'color': 'green'})
        elif class_name == 'R_Signal':
            ptl_list.append({'coords': coords, 'confidence': confidence, 'color': 'red'})

    return results, ptl_list, zebra_crops

def get_results(row, ptl_list, zebra_crops, zebra, red, green):
    if row['zebra'] == 1:  # Ground truth: Zebra crossing present
        if zebra_crops:  # Prediction: Zebra crossing detected
            zebra[0] += 1  # TP
        else:  # Prediction: Zebra crossing not detected
            zebra[2] += 1  # FN
    else:  # Ground truth: No zebra crossing
        if zebra_crops:  # Prediction: Zebra crossing detected
            zebra[1] += 1  # FP
        else:  # Prediction: Zebra crossing not detected
            zebra[3] += 1  # TN

    if ptl_list:  # Traffic light(s) detected
        if row['mode'] in [0, 3]:  # Ground truth: Red light should be present
            if ptl_list[0]['color'] == 'red':
                red[0] += 1  # TP
            else:
                red[2] += 1  # FN
        else:  # Ground truth: Red light should NOT be present
            if ptl_list[0]['color'] == 'red':
                red[1] += 1  # FP
            else:
                red[3] += 1  # TN

        if row['mode'] in [1, 2]:  # Ground truth: Green light should be present
            if ptl_list[0]['color'] == 'green':
                green[0] += 1  # TP
            else:
                green[2] += 1  # FN
        else:  # Ground truth: Green light should NOT be present
            if ptl_list[0]['color'] == 'green':
                green[1] += 1  # FP
            else:
                green[3] += 1  # TN
    else:  # No traffic lights detected
        if row['mode'] in [0, 3]:  # Ground truth: Red light should be present
            red[1] += 1  # FP (Model didn't detect it)
        else:  # Ground truth: Red light should NOT be present
            red[3] += 1  # TN (Model correctly didn't detect it)

        if row['mode'] in [1, 2]:  # Ground truth: Green light should be present
            green[1] += 1  # FP (Model didn't detect it)
        else:  # Ground truth: Green light should NOT be present
            green[3] += 1  # TN (Model correctly didn't detect it)

    return zebra, red, green

def calculate_metrics(type):
    total = sum(type)
    tp = type[0]
    fp = type[1]
    fn = type[2]
    tn = type[3]

    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1_score, 2)
    }

def plot_confusion_matrices_with_metrics(zebra, red, green, zebra_metrics, red_metrics, green_metrics, filename="Matriu de confusió.png"):
    clases = ['Pas de zebra', 'Semàfor vermell', 'Semàfor verd']
    datos = [zebra, red, green]
    all_metrics = [zebra_metrics, red_metrics, green_metrics]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, clase, (tp, fp, fn, tn), metrics in zip(axs, clases, datos, all_metrics):
        matrix = np.array([[tp, fn],
                           [fp, tn]])

        sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax, xticklabels=['Positiu', 'Negatiu'], yticklabels=['Positiu', 'Negatiu'], annot_kws={"size": 15})

        ax.set_title(f'{clase}')
        ax.set_xlabel('Predicció')
        ax.set_ylabel('Real')

        metrics_text = f"Accuracy: {metrics['accuracy']}\nPrecision: {metrics['precision']}\nRecall: {metrics['recall']}\nF1: {metrics['f1_score']}"
        ax.text(0.05, -0.3, metrics_text, size=18, ha='left', va='top', transform=ax.transAxes) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    model = torch.hub.load('../yolov5', 'custom', path='../best.pt', source='local')
    carpeta = '../data'

    df = pd.read_csv('../data/dataset.csv')
    zebra = [0, 0, 0, 0]  # TP, FP, FN, TN
    red = [0, 0, 0, 0]
    green = [0, 0, 0, 0]

    for index, row in df.iterrows():     
        filename = row['file']
        image_path = carpeta + filename

        results, ptl_list, zebra_crops = process_image(image_path, model)
        zebra, red, green = get_results(row, ptl_list, ptl_list, zebra, red, green)

    zebra_metrics = calculate_metrics(zebra)
    red_metrics = calculate_metrics(red)
    green_metrics = calculate_metrics(green)

    plot_confusion_matrices_with_metrics(zebra, red, green, zebra_metrics, red_metrics, green_metrics)