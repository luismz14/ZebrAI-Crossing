import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'none'
import numpy as np
import cv2
import math
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
import itertools
import random
from sklearn.linear_model import RANSACRegressor, LinearRegression
import torch
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import seaborn as sns
random.seed(42)


# -----------------------------------------------------------------------------------
#  YOLO TRAFFIC LIGHT DETECTION
# -----------------------------------------------------------------------------------

def process_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    detections = results.pandas().xyxy[0]

    traffic_lights = []
    zebra_crops = []

    for _, row in detections.iterrows():
        class_name = row['name']
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        coords = (xmin, ymin, xmax, ymax)
        confidence = round(float(row['confidence']), 2)

        if class_name == 'Zebra_Cross':
            zebra_crops.append((coords, confidence))
        elif class_name == 'G_Signal':
            traffic_lights.append({'coords': coords, 'confidence': confidence, 'color': 'green'})
        elif class_name == 'R_Signal':
            traffic_lights.append({'coords': coords, 'confidence': confidence, 'color': 'red'})

    return traffic_lights, zebra_crops

# -----------------------------------------------------------------------------------
#  UTILS
# -----------------------------------------------------------------------------------

def display_images(imgs, titles, width=5):
    n = len(imgs)
    rows = math.ceil(n / 3)
    cols = n // rows
    h, w = imgs[0].shape[:2]
    fig, axs = plt.subplots(rows, cols, figsize=(width * cols, width * h / w * rows))
    axs = axs.flatten()
    for i in range(rows * cols):
        if i < n:
            img = imgs[i]
            axs[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()


def to_float(img):
    img = img.astype(np.float64)
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def to_uint8(img):
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.round(norm * 255).astype(np.uint8)

def is_empty(x):
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        return False

# -----------------------------------------------------------------------------------
#  NOISE FILTER  &  THRESHOLD BINARY MASK  &  EDGE DETECTION
# -----------------------------------------------------------------------------------

def noise_filter(src, noise_option='median', med_k=5, gauss_sigma=1.5, bilat_d=6, bilat_sigmaColor=75, bilat_sigmaSpace=75, morph_kernel=5, **kwargs):
    try:
        if noise_option == 'median':
            return cv2.medianBlur(src, med_k)
        elif noise_option == 'gaussian':
            ksize = round(6 * gauss_sigma + 1)
            ksize += 1 - (ksize % 2)
            return cv2.GaussianBlur(src, (ksize, ksize), gauss_sigma)
        elif noise_option == 'bilateral':
            return cv2.bilateralFilter(src, d=bilat_d, sigmaColor=bilat_sigmaColor, sigmaSpace=bilat_sigmaSpace)
        elif noise_option == 'morph_max':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
            return cv2.morphologyEx(src, cv2.MORPH_DILATE, kernel)
        elif noise_option == 'top_hat':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
            return cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    except:
        return src
    
def find_threshold(src, thr_option='mixture', **kwargs):
    try:
        if thr_option == 'otsu':
            thr, _ = cv2.threshold(src, 0, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thr
        elif thr_option == 'mixture':
            hist, bins = np.histogram(src, bins=256, range=(0, 256))
            sample = cv2.resize(src, (14, 14), interpolation=cv2.INTER_CUBIC).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(sample)
            means = gmm.means_.flatten()
            m1, m2 = np.sort(means)
            hist_smooth = gaussian_filter1d(hist, sigma=2)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mask = (bin_centers > m1) & (bin_centers < m2)
            thr = bin_centers[mask][np.argmin(hist_smooth[mask])]
            return thr
    except:
        return 128
    
def get_edges(src, edges_option="laplacian", canny_thr1=50, canny_thr2=150, **kwargs):
    if edges_option == "canny":
        return cv2.Canny(src, canny_thr1, canny_thr2, apertureSize=3)
    elif edges_option == "laplacian":
        src_blur = cv2.GaussianBlur(src, (5,5), 0)
        lap = cv2.Laplacian(src_blur, ddepth=cv2.CV_64F)
        z1 = np.roll(lap, 1, axis=0) * np.roll(lap, -1, axis=0)
        z2 = np.roll(lap, 1, axis=1) * np.roll(lap, -1, axis=1)
        return np.logical_or(z1 < 0, z2 < 0).astype(np.uint8) * 255
    else:
        raise ValueError(f"Invalid edges option: {edges_option}")
    
# -----------------------------------------------------------------------------------
#  LINE DETECTION (HOUGH + VALIDATION)
# -----------------------------------------------------------------------------------

def get_intersection(l1, l2):
    rho1, theta1 = l1
    rho2, theta2 = l2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.abs(np.linalg.det(A)) < 1e-6:
        return None
    intersection = np.linalg.solve(A, b)
    return intersection.flatten()

def point_line_distance(x0, y0, rho, theta):
    return abs(x0 * np.cos(theta) + y0 * np.sin(theta) - rho)

def get_filtered_lines(edges, high_thr=240, low_thr=150, centroids_sep=100, line_dist_thr=30, **kwargs):
    few_raw = cv2.HoughLines(edges, 1, np.pi/180, threshold=high_thr)
    raw = cv2.HoughLines(edges, 1, np.pi/180, threshold=low_thr)
    if is_empty(few_raw) or len(few_raw) <= 1:
        raise ValueError("Hough restrictiu no detecta línies")
    
    few_lines = few_raw[:, 0, :]
    lines = raw[:, 0, :]
    intersections = []
    for l1, l2 in itertools.combinations(few_lines, 2):
        pt = get_intersection(l1, l2)
        if pt is not None and np.all(np.isfinite(pt)):
            intersections.append(pt)
    if not intersections:
        return few_lines
    intersections = np.array(intersections)

    clustering = DBSCAN(eps=40, min_samples=5).fit(intersections)
    labels = clustering.labels_
    if np.sum(labels != -1) == 0:
        return few_lines

    unique_labels = np.unique(labels[labels != -1])
    cluster_sizes = np.array([np.sum(labels == k) for k in unique_labels])
    centroids = np.stack([intersections[labels == k].mean(axis=0) for k in unique_labels])
    centroids = centroids[np.argsort(cluster_sizes)[::-1]]

    points = []
    for centroid in centroids:
        if all(centroid[0]-p[0] > centroids_sep and centroid[1]-p[1] > centroids_sep for p in points):
            points.append(centroid)
    points = np.array(points)

    filtered_lines = []
    for rho, theta in lines:
        for x, y in points:
            if point_line_distance(x, y, rho, theta) < line_dist_thr:
                filtered_lines.append([rho, theta])
                break
    filtered_lines = np.array(filtered_lines)

    return filtered_lines

def remove_duplicate_lines(lines, shape):
    if is_empty(lines):
        raise ValueError("Filtratge de punt de fuga elimina totes les línies")

    lines = np.array(lines)
    h, w = shape[:2]
    d = np.hypot(h, w)
    theta_mean = np.mean(lines[:, 1])
    keep = np.ones(len(lines), dtype=bool)

    def get_line_points(rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = np.array([x0 + d * -b, y0 + d * a])
        pt2 = np.array([x0 - d * -b, y0 - d * a])
        return pt1, pt2

    def intersection(p1, p2, q1, q2):
        A = np.array([[p2[0] - p1[0], q1[0] - q2[0]],
                      [p2[1] - p1[1], q1[1] - q2[1]]])
        b = np.array([q1[0] - p1[0], q1[1] - p1[1]])
        if np.linalg.matrix_rank(A) < 2:
            return None
        t = np.linalg.solve(A, b)
        return p1 + t[0] * (p2 - p1)

    for i in range(len(lines)):
        if not keep[i]:
            continue
        rho1, theta1 = lines[i]
        p1, p2 = get_line_points(rho1, theta1)
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            rho2, theta2 = lines[j]
            q1, q2 = get_line_points(rho2, theta2)
            inter = intersection(p1, p2, q1, q2)
            if inter is not None:
                x, y = inter
                if 0 <= x < w and 0 <= y < h:
                    dist_i = abs(theta1 - theta_mean)
                    dist_j = abs(theta2 - theta_mean)
                    if dist_i > dist_j:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False
    return lines[keep]

# -----------------------------------------------------------------------------------
#  VERTICAL (LATERAL) LINES DETECTION (USING RANSAC) 
# -----------------------------------------------------------------------------------

def segment_from_rho_theta(rho, theta, edge_img, epsilon=1.5):

    ys, xs = np.nonzero(edge_img) 
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    n = np.array([np.cos(theta), np.sin(theta)]) 
    v = np.array([-np.sin(theta),  np.cos(theta)])
    d = np.abs(pts @ n - rho)
    on_line = pts[d < epsilon]

    if len(on_line) < 2:
        return None
    
    t = on_line @ v
    p_min = on_line[np.argmin(t)]
    p_max = on_line[np.argmax(t)]
    
    return tuple(p_min.astype(int)), tuple(p_max.astype(int))

def count_near_points(p1, p2, pts, dist):
    pts = np.asarray(pts, dtype=float)
    p1  = np.asarray(p1,  dtype=float)
    p2  = np.asarray(p2,  dtype=float)

    v = p2 - p1
    norm_v = np.linalg.norm(v)

    if norm_v == 0:
        raise ValueError("p1 y p2 no poden coincidir.")
    dif   = p1 - pts
    cross = v[0] * dif[:, 1] - v[1] * dif[:, 0]

    dists = np.abs(cross) / norm_v
    return np.sum(dists < dist)

def get_line_ransac(points, ransac_dist=20, ransac_n_iter=100, **kwargs):
    n_lines = len(points)
    best_p=(0,0)
    best_k=0
    if points is not None:
        for i in range(ransac_n_iter):
            p1 = points[random.randint(0, n_lines-1)]
            p2 = points[random.randint(0, n_lines-1)]
            while np.all(p1 == p2):
                p2 = points[random.randint(0, n_lines-1)]
            k = count_near_points(p1,p2, points, ransac_dist)
            if k > best_k:
                best_k = k
                best_p = (p1, p2)
    return best_p

# -----------------------------------------------------------------------------------
#  MID POINTS  +  MID LINE (ANGLE & POSITION)
# -----------------------------------------------------------------------------------

def intersect_polar_with_segment(rho, theta, segment, eps=1e-6):
    (x1, y1), (x2, y2) = segment
    dx, dy = x2 - x1, y2 - y1
    denom = dx*np.cos(theta) + dy*np.sin(theta)
    if abs(denom) < eps:
        return None
    t = (rho - (x1*np.cos(theta) + y1*np.sin(theta))) / denom
    x = x1 + t*dx
    y = y1 + t*dy
    return (int(round(x)), int(round(y)))

def get_limits(representative_lines, edges, ransac_dist=20, ransac_n_iter=100, **kwargs):
    lista_extremos = []
    for (rho, theta) in representative_lines:
        extremos = segment_from_rho_theta(rho, theta, edges, epsilon=1.5)
        if extremos is not None:
            lista_extremos.append(extremos)
    lista_extremos = np.array(lista_extremos)

    # Retornem els límits de la imatge en cas de no trobar múltiples extrems
    if len(lista_extremos) <= 1:
        h, w = edges.shape
        return ((0,0), (0,h)), ((w,0), (w,h))

    izq = lista_extremos[:, 0]
    der = lista_extremos[:, 1]
    p_izq=get_line_ransac(izq, ransac_dist, ransac_n_iter)
    p_der=get_line_ransac(der, ransac_dist, ransac_n_iter)

    return p_izq, p_der

def get_mid_points(lines, p_izq, p_der, shape):

    lista_midpoints = []
    h, w = shape

    for rho, theta in lines:
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        if abs(sin_t) < 1e-6:
            continue
        pi = intersect_polar_with_segment(rho, theta, p_izq)
        pd = intersect_polar_with_segment(rho, theta, p_der)

        # Calcul i filtrat de interseccions
        border = []
        y0 = (rho - 0*cos_t)/sin_t
        yw = (rho - w*cos_t)/sin_t
        if 0 <= y0 <= h: border.append((0, int(round(y0))))
        if 0 <= yw <= h: border.append((w, int(round(yw))))
        x0 = (rho - 0*sin_t)/cos_t if abs(cos_t)>1e-6 else None
        xh = (rho - h*sin_t)/cos_t if abs(cos_t)>1e-6 else None
        if x0 is not None and 0 <= x0 <= w: border.append((int(round(x0)), 0))
        if xh is not None and 0 <= xh <= w: border.append((int(round(xh)), h))
        if len(border) < 2:
            continue
        
        border = sorted(border, key=lambda p: p[0])
        start_border, end_border = border[0], border[-1]

        # Seleccio de punts: interseccions si existeixen, si no, els extrems de la imatge
        start = pi if pi is not None else start_border
        end   = pd if pd is not None else end_border

        # Calcul de punt mitjà
        mx = int(round((start[0] + end[0]) / 2))
        my = int(round((start[1] + end[1]) / 2))
        lista_midpoints.append((mx, my))

    return lista_midpoints

def get_angle(midpoints, shape):
    h, w = shape
    pts = np.array(midpoints)
    xs, ys = pts[:,0], pts[:,1]
    X = xs.reshape(-1,1)
    y = ys

    ransac = RANSACRegressor(LinearRegression(), 
                            residual_threshold = w * 0.005,
                            max_trials=1000, random_state=42)
    ransac.fit(X, y)

    m = 1/ransac.estimator_.coef_[0]
    angle_rad = np.atan(m)
    angle_deg = angle_rad * 180/np.pi
    
    return angle_rad, angle_deg

def get_init_point(midpoints):
    return max(midpoints, key=lambda p: p[1])

# -----------------------------------------------------------------------------------
#  PREDICTION
# -----------------------------------------------------------------------------------

def predict_img(img_path, model, **kwargs):
    path = '../data'+img_path
    columns = ["zebra", "mode", "blocked", "x", "y", "theta_rad", "theta_deg"]
    if not os.path.exists(path):
        print("Fitxer no trobat:", path)
        return pd.Series([0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=columns)

    # Zebra Crossing ROI & Traffic Light Detection
    traffic_lights, zebra_crops = process_image(path, model)
    if is_empty(zebra_crops):
        return pd.Series([0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=columns)
    blocked = not is_empty(traffic_lights)
    mode = int(traffic_lights[0]["color"] == "green") if blocked else 0

    # Image ROI Crop
    img = plt.imread(path)
    (xmin, ymin, xmax, ymax) = zebra_crops[0][0]
    if xmax - xmin <= 0 or ymax - ymin <= 0:
        raise ValueError(f"ROI invàlida: {(xmin, ymin, xmax, ymax)}")
    img = img[ymin:ymax, xmin:xmax]
    if img.size == 0:
        raise ValueError(f"ROI buida: {(xmin, ymin, xmax, ymax)}")
    gray = to_uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    try:
        # Zebra Crossing Pos + Angle Search
        src = noise_filter(gray, **kwargs)
        thr = find_threshold(src, **kwargs)
        if not (0 <= thr <= 255):
            raise ValueError(f"Threshold invàlid: {thr}")
        bw = 255*(src > thr).astype(np.uint8)
        edges = get_edges(bw, **kwargs)
        filtered_lines = get_filtered_lines(edges, **kwargs)
        filtered_lines = remove_duplicate_lines(filtered_lines, gray.shape)
        if is_empty(filtered_lines):
             raise Exception("No s'han trobat línies")

        p_izq, p_der = get_limits(filtered_lines, edges, **kwargs)
        midpoints = get_mid_points(filtered_lines, p_izq, p_der, gray.shape)
        angle_rad, angle_deg = get_angle(midpoints, gray.shape)
        x, y = get_init_point(midpoints)

        return pd.Series([1, mode, int(blocked), x, y, angle_rad, angle_deg], index=columns)
    
    except ValueError:
        return pd.Series([1, mode, int(blocked), xmax//2, ymax//2, 0, 0], index=columns)

    except Exception as e:
        print("Error: ", e, " Imatge: ", path)
        return pd.Series([1, mode, int(blocked), xmax//2, ymax//2, 0, 0], index=columns)
        # raise e

# -----------------------------------------------------------------------------------
#  EVALUATION
# -----------------------------------------------------------------------------------

def show_metrics(y_test, y_pred):

    # Simulem que y_test i y_pred ja estan definits
    columns_classification = ['zebra', 'mode', 'blocked']
    columns_regression = ['x', 'y', 'theta_rad', 'theta_deg']

    # Inicialitzem diccionaris per guardar resultats
    classification_metrics = {}
    mae_metrics = {}

    # Mètriques de classificació
    for col in columns_classification:
        if col in ['mode', 'blocked']:
            # Filtrar només els casos on zebra == 1
            mask = (y_test['zebra'] == 1) & (y_pred['zebra'] == 1)
        else:
            # Per zebra, usem totes les files
            mask = ~(y_test[col].isna() | y_pred[col].isna())

        y_true = y_test.loc[mask, col]
        y_hat = y_pred.loc[mask, col]

        if y_true.empty:
            continue

        # Convertim a enters
        y_true = y_true.astype(int)
        y_hat = y_hat.astype(int)

        classification_metrics[col] = {
            'accuracy': accuracy_score(y_true, y_hat),
            'precision': precision_score(y_true, y_hat, zero_division=0),
            'recall': recall_score(y_true, y_hat, zero_division=0),
            'f1': f1_score(y_true, y_hat, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_hat)
        }

    # Mètriques de regressió
    for col in columns_regression:
        y_true = y_test[col]
        y_hat = y_pred[col]

        # Filtrar NaNs
        valid = ~(y_true.isna() | y_hat.isna())
        y_true_valid = y_true[valid]
        y_hat_valid = y_hat[valid]

        if len(y_true_valid) == 0:
            continue

        mae_metrics[col] = mean_absolute_error(y_true_valid, y_hat_valid)

    mae_df = pd.Series(mae_metrics, name='MAE').to_frame()
    print("\n{:>12} | {:>20}".format("Variable", "MAE"))
    print("-" * 35)
    for var, val in mae_df["MAE"].items():
        print(f"{var:>12} | {val:>20.5f}")

    for col in classification_metrics:
        cm = classification_metrics[col]['confusion_matrix']
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {col}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        output_path = os.path.join("./results", f'conf_matrix_{col}.png')
        plt.savefig(output_path)


def get_metrics(y_test, y_pred):
    columns_regression = ['x', 'y', 'theta_rad', 'theta_deg']
    mae_metrics = {}

    for col in columns_regression:
        y_true = y_test[col]
        y_hat = y_pred[col]
        valid = ~(y_true.isna() | y_hat.isna())
        y_true_valid = y_true[valid]
        y_hat_valid = y_hat[valid]
        if len(y_true_valid) == 0:
            continue
        mae_metrics[col] = mean_absolute_error(y_true_valid, y_hat_valid)

    return pd.Series(mae_metrics, name='MAE').sort_index().to_frame()


def random_search(X_val, y_val, model, n_iter=10):
    samples = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    results = []

    for i, params in enumerate(tqdm(samples)):
        try:
            y_hat = X_val.apply(lambda path: predict_img(path, model, **params))
            y_hat = y_hat.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)
            mae_df = get_metrics(y_val, y_hat)
            mae_mean = sum(weights[k] * mae_df.loc[k, 'MAE'] for k in weights if k in mae_df.index)
            results.append((params, mae_mean))
        except Exception as e:
            print(f"Error amb combinació {i}: {e}")
            continue

    results.sort(key=lambda x: x[1])
    return results

weights = {
    'x': 0.15,
    'y': 0.15,
    'theta_rad': 0.4,
    'theta_deg': 0.3
}

param_grid = {
    'noise_option': ['median', 'gaussian', 'bilateral'],
    'med_k': [3, 5, 7],
    'gauss_sigma': [1.0, 1.5, 2.0],
    'bilat_d': [5, 7],
    'bilat_sigmaColor': [50, 75, 100],
    'bilat_sigmaSpace': [50, 75, 100],
    'morph_kernel': [3, 5, 7],
    'thr_option': ['otsu', 'mixture'],
    'edges_option': ['laplacian', 'canny'],
    'canny_thr1': [25, 50, 75, 100],
    'canny_thr2': [125, 150, 175, 200],
    'high_thr': [180, 200, 220, 240, 260],
    'low_thr': [100, 120, 140, 160, 180],
    'centroids_sep': [60, 80, 100, 120],
    'line_dist_thr': [20, 30, 40, 50, 60],
    'ransac_dist': [10, 15, 20, 25, 30],
    'ransac_n_iter': [100, 200, 500, 1000]
}

if __name__ == "__main__":
    model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
    tqdm.pandas()
    df = pd.read_csv('../data/dataset.csv')
    X, y = df["file"], df.drop(["file"], axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # y_pred = X_test.progress_apply(lambda path: predict_img(path, model))
    # y_pred = y_pred.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)

    # df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    # df_test.to_csv("./results/test.csv", index=False)
    # y_pred.to_csv("./results/pred.csv", index=False)
    # show_metrics(y_test, y_pred)
    
    results = random_search(X_test, y_test, model)
    print(results)
