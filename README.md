# Computer Vision  Projects

This repository contains three self-contained projects completed for the Computer Vision course at Hacettepe University. Each project folder includes:

- An **Introduction** notebook or document that motivates the problem, describes the approach, and outlines the pipeline.  
- A **Report** (PDF or Markdown) detailing methodology, experiments, results and discussion.  
- The core code in both Jupyter notebook (`.ipynb`) and script (`.py`) form.  

## Table of Contents

1. [Project 1: Image Classification with Convolutional Neural Networks](#project-1-image-classification-with-convolutional-neural-networks)  
2. [Project 2: Object Detection and Counting](#project-2-object-detection-and-counting)  
3. [Project 3: Perspective Correction using Edge and Line Fitting](#project-3-perspective-correction-using-edge-and-line-fitting)  

---

## Prerequisites

- **Python 3.x**  
- **Dependencies** (install via `pip install -r requirements.txt` in each project folder):  
  - `numpy`  
  - `opencv-python`  
  - `matplotlib`  
  - `scikit-learn`  
  - `scikit-image`  
  - **Project 1**: `torch`, `torchvision`  
  - **Project 2**: `ultralytics` (YOLOv8)  

---

## Project 1: Image Classification with Convolutional Neural Networks

**General Description**  
This project builds and trains a Convolutional Neural Network from scratch to classify images into predefined categories. You’ll see data loading and augmentation, model architecture definition, training and validation loops, and visualization of performance metrics (accuracy and loss curves), as well as a confusion matrix to analyze misclassifications.

**Includes:**  
- **Introduction**: overview of image classification, dataset characteristics, and problem motivation.  
- **Report**: detailed write-up of experiments, hyperparameter choices, and analysis of results.  

**Dataset:**  
[Image Classification Dataset (Google Drive)](https://drive.google.com/file/d/1a0uuiylWnyGAr0JZVfhw2EPZ6XzcPtyb/view?usp=sharing)

---

## Project 2: Object Detection and Counting

**General Description**  
This project applies the YOLOv8 framework to detect objects in scenes and count their occurrences. It covers training the detector on a custom dataset, running inference on test images, and saving annotated outputs. The counting script reads prediction results and generates summary statistics.

**Includes:**  
- **Introduction**: background on object detection, the YOLO family of models, and dataset particulars.  
- **Report**: performance evaluation (precision, recall, mAP), sample detection outputs, and discussion of failure modes.  

**Dataset:**  
[Object Detection Dataset (Google Drive)](https://drive.google.com/file/d/1RwB_X8SxQnqoVT7IiTQD9GsX9-FRd8Ib/view?usp=sharing)

**Notes:**  
- The `runs/` folder contains model checkpoints, training logs, and example output images.

---

## Project 3: Perspective Correction using Edge and Line Fitting

**General Description**  
This project implements a classical computer-vision pipeline to correct perspective distortion in images of documents or planar scenes. It performs edge detection, custom Hough-based line fitting, RANSAC line refinement, corner extraction, homography estimation, and warping. Final quality is evaluated using SSIM against ground-truth scans.

**Includes:**  
- **Introduction**: explanation of geometric distortions, homography fundamentals, and the edge‐and‐line approach.  
- **Report**: step-by-step methodology, qualitative and quantitative results, and potential improvements.  

**Dataset:**  
[Perspective Correction Dataset (Google Drive)](https://drive.google.com/file/d/1aPfzmYxLazyj15_zgCD96ImYdN9IZCPw/view?usp=sharing)

---

## How to Run

Each project folder contains:

```bash
cd <ProjectFolder>
pip install -r requirements.txt
jupyter notebook  # or open the Introduction notebook directly
