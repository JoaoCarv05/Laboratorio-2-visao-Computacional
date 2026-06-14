# Laboratório 2 – Extração de Características (Features)

**ESZA019 – Visão Computacional — 2026.2**

## Descrição

Este repositório contém o relatório e os códigos do Laboratório 2 da disciplina de Visão Computacional, focado em **extração de características (features)** e **feature matching** utilizando OpenCV.

## Conteúdo

- `Lab2_Features.ipynb` — Relatório completo em formato Jupyter Notebook
- `sift_webcam.py` — Script standalone para SIFT matching em tempo real com duas webcams
- `imagens/` — Imagens utilizadas nos experimentos e resultados gerados

## Tópicos Abordados

1. **Fundamentação Teórica**
   - Detector de Harris (Harris Corner Detector)
   - Detector de Shi-Tomasi (Good Features to Track)
   - SIFT (Scale-Invariant Feature Transform)

2. **Experimentos Práticos**
   - Detecção de cantos com Harris e Shi-Tomasi
   - Detecção de keypoints SIFT
   - Feature Matching + Homografia para detecção de objetos
   - SIFT matching em tempo real com webcams

## Requisitos

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib jupyter
```

## Execução

### Notebook
```bash
jupyter notebook Lab2_Features.ipynb
```

### Script Webcam (requer duas webcams)
```bash
python sift_webcam.py
```

## Autor

- João Victor de Castro Carvalho

## Referências

- [OpenCV Feature Detection and Description](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [Feature Matching + Homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)
- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
