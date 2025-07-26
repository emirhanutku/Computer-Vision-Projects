# Computer Vision  Projects

This repository contains three projects completed for the Computer Vision course at Hacettepe University. Each project is self‑contained in its own folder and includes code, results, and links to the datasets used.

## Table of Contents

1. [Project 1: Image Classification with Convolutional Neural Networks](#project-1-image-classification-with-convolutional-neural-networks)  
2. [Project 2: Object Detection and Counting](#project-2-object-detection-and-counting)  
3. [Project 3: Perspective Correction using Edge and Line Fitting](#project-3-perspective-correction-using-edge-and-line-fitting)  

---

## Prerequisites

- **Python 3.x**  
- **Dependencies** (install via `pip`):  
  - `numpy`  
  - `opencv-python`  
  - `matplotlib`  
  - `scikit-learn`  
  - `scikit-image`  
  - **Project 1**: `torch`, `torchvision`  
  - **Project 2**: `ultralytics` (YOLOv8)  

---

## Project 1: Image Classification with Convolutional Neural Networks

**Description**  
Implementation of a convolutional neural network to classify images into their respective categories. Includes data loading, preprocessing, model definition, training loop, evaluation metrics, and visualization of performance (accuracy, loss curves, confusion matrix).

**Folder:** `Project1_Image_Classification/`

**Dataset:**  
[Image Classification Dataset (Google Drive)](https://drive.google.com/file/d/1a0uuiylWnyGAr0JZVfhw2EPZ6XzcPtyb/view?usp=sharing)

---

## Project 2: Object Detection and Counting

**Description**  
Training and inference using a YOLOv8 model to detect and count objects in images. Includes scripts for training, validation, and a simple interface to run inference on new images. Output predictions and labeled images are saved in the included `runs/` folder.

**Folder:** `Project2_Object_Detection_Counting/`

**Dataset:**  
[Object Detection Dataset (Google Drive)](https://drive.google.com/file/d/1RwB_X8SxQnqoVT7IiTQD9GsX9-FRd8Ib/view?usp=sharing)

**Notes:**  
- The `runs/` directory contains model checkpoints, training logs, and example output images.  

---

## Project 3: Perspective Correction using Edge and Line Fitting

**Description**  
A document dewarping pipeline that detects and corrects perspective distortions using classical computer vision methods. Steps include:  
1. Edge detection (Canny) with adaptive Gaussian blur  
2. Line detection via custom Hough Transform  
3. RANSAC‑based line refinement  
4. Quadrilateral corner detection  
5. Homography estimation and bilinear warping  
6. Evaluation using Structural Similarity Index (SSIM)

**Folder:** `Project3_Perspective_Correction/`

**Dataset:**  
[Perspective Correction Dataset (Google Drive)](https://drive.google.com/file/d/1aPfzmYxLazyj15_zgCD96ImYdN9IZCPw/view?usp=sharing)

---

## How to Use

Each project folder contains:

1. A Jupyter notebook (`.ipynb`) with step‑by‑step explanations and code.  
2. A Python script (`.py`) with the core implementation.  
3. (Project 3) A PDF report summarizing methodology, results, and discussion.

To run any project:

```bash
cd <ProjectFolder>
# install dependencies, e.g.
pip install -r requirements.txt
# launch notebook
jupyter notebook
