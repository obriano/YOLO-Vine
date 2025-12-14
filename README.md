# YOLO-Vine
Final Project for ECE5990 - Introduction to Deep Learning

This project presents YOLO-Vine, a YOLOv12-derived detector that performs a selective module transplant inspired by LEAF-YOLO to reduce model complexity for UAV edge deployment. Compared to the baseline YOLOv12-n model, YOLO-Vine replaces several backbone and neck building blocks with lightweight multi-scale feature extractors and receptive-field enhancement modules (RFM), targeting improved feature diversity under a reduced parameter budget. 

On VisDrone, YOLO-Vine achieves an $\mathrm{mAP}_{50}$ of 0.314 versus 0.315 for the baseline YOLOv12-n while reducing parameters from 2.51M to 1.92M and compute from 5.8 to 5.3 GFLOPs. Additional evaluation is reported on a unified UAV dataset for cross-domain comparison. These results indicate that carefully chosen architectural substitutions can preserve near-baseline UAV detection performance while improving suitability for resource-constrained onboard deployment.

# Dataset Downloads
Visdrone - https://drive.google.com/drive/folders/1ZYzVUexGLwUEFDSjDAZgj9XnqyDl7yZC?usp=sharing

Custom Dataset - https://drive.google.com/drive/folders/1MtLPExTmecOF8lzznN7I3LAv97jCfw8j?usp=sharing

# Pretrained Model Weights
Individual Weights (Visdrone only / Custom) https://drive.google.com/drive/folders/1NLJ9wWGsitzCAf_oJXz8xn476nttvcYJ?usp=sharing
