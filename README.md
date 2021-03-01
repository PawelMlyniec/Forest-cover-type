# Forest cover type prediction
## Task
Goal was to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables.

## Data 
Data are described on https://www.kaggle.com/c/forest-cover-type-prediction

## Solution
Solution consists of:
1. Calculating skewness
2. Corellation matrix
3. Box plots
4. Deleting variables with lasso regression
5. PCA and MCA
6. Naive Bayes
7. K Nearest Neighbours
8. C5.0 Tree
9. eXtreme Gradient Boosting

## Best results
| Model | Train time | prediction time | AUC |
|-------|------|-----|-----|
| XGBoost | 1.38h | 0.07s | 0.96 |
| C5.0 Tree | 49.9s | 0.53s | 0.93 |

## Co-authors 
Hubert Borkowski

## Documentation in Polish 
1. forest-cover-type.pdf

## Info
Project is done as a part of course "Advanced Machine Learning" at Warsaw University of Technology.

Mark 35/40
