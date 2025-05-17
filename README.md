# ZebrAI Crossing: *Crosswalk and traffic light detection system for urban navigation* 🚦

Aquest repositori ha estat desenvolupat com a pràctica per a l’assignatura de Visió per Computador del Grau en Enginyeria Informàtica de la Universitat Autònoma de Barcelona (UAB).

| Nom de l’alumne  | NIU     |
| ---------------- | ------- |
| Albert Capdevila | 1587933 |
| Levon Kesoyan    | 1668018 |
| Luis Martínez    | 1668180 |

## Objectiu
ZebrAI Crossing és un sistema de visió per computador que, donada una imatge d’un entorn urbà, detecta:
- La presència d’un pas de zebra.
- La seva orientació respecte la càmera.
- La posició de l’inici del pas de zebra.
- La presència de semàfors de vianants.
- L’estat del semàfor (verd o vermell).

Aquest sistema forma part del projecte **OrionWay**, un robot autònom destinat a guiar persones amb discapacitat visual.

## Estructura del repositori

```
├── data/                     # Dades d’entrada (ignorat per .gitignore)
├── experiments/              # Proves i prototips
│   ├── cube.png              # Imatge de prova
│   ├── data.ipynb            # Tractament de dades
│   ├── edges.ipynb           # Detecció de vores
│   ├── hough.ipynb           # Transformada de Hough
│   └── PTLdetecion.py        # Detecció de semàfors
├── latex/                    # Informe i figures en LaTeX
├── src/
│   └── results/
│   └── ZebraAI.py            # Script principal del projecte
├── yolov5/                   # Carpeta amb el model YOLOv5 (ignorada per .gitignore)
├── best.pt                   # Ponderacions del model YOLO (ignorat per .gitignore)
├── .gitignore
└── README.md
```

### Fitxers importants
- **Script principal:** [`ZebraAI.py`](./src/ZebraAI.py)
- **Informe:** [`1587933_1668180_1668018_ZebrAI.pdf`](./latex/1587933_1668180_1668018_ZebrAI.pdf)
- **Notebook del procés:** [`edges.ipynb`](./experiments/edges.ipynb)
- **Hough implementat des de zero:** [`hough.ipynb`](./experiments/hough.ipynb)