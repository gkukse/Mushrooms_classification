# Mushrooms_classification
## Overview
Annually, around [7,500 mushroom poisoning cases occur in the US](https://www.tandfonline.com/doi/full/10.1080/00275514.2018.1479561), primarily due to misidentification of edible species. This preventable issue can be addressed through better education and awareness. 

This project aims to develop a machine learning model to accurately identify mushroom types. 


## Dataset
Dataset is from [Kaggle datasets](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images).

## Python Libraries
Code was done on Python 3.11.9. Packages can be found in requirements.txt


## Findings
* Exploratory Data Analysis (EDA): Dataset is made of 9 classses.
    - Cortinarius has toxic species, but the rest 8 classes, also has toxic species. Thus toxic/non-toxic is not an option in this project.
    - Original dataset has 6714 images, but after removing duplicates (color histogram), greyscale encoding and non-mushroom objects (ResNet80), 6169 images were left.

* Models: ResNet18 was used as pretrained Transfer Learning backbone model. AdamW and Stochastic Gradient Descent (SGD) optimizers were tested. Best model was Fine Tuned model with AdamW, which had 79% accuracy, 75% recall and Inference mean time was 3.24 +/- 0.19 [ms].


## Future Work
- Adding a new layer betweem pretrained backbone and outer layer. 
