# Data Preparation Techniques in Machine Learning

## Description

In this project, we focus on implementing multiple data preparation techniques on the selected data to improve the prediction results of a machine learning model.

> Due to the nature of the selected dataset for this project, I formulate the problem as a classification task since the target variable is a nominal variable.

## Experiment Summary

### Evaluation Method

- I calculate the improvement of each method based on different baseline methods. The increase in performance is shown in red color while the decrease in performance is shown in blue color.

- For each data preparation stage, I choose the best method in that stage as the baseline respectively.

- The experiment results of the next stage are compared with the baseline method of the previous stage.

### Experiment Results

| Experiment | Description | AUC | Accuracy | Improvement | Comment |
| --- | --- | --- | --- | --- | --- |
| <i>Deletion</i> ||||||
| Exp 1             | Listwise Deletion                           | 0.7825 | 0.7481 | - | Baseline 1 |
| Exp 2             | Variable Deletion                           | 0.7014 | 0.6680 | <font color='blue'>&#8595; -10.3642%</font> | - |
| <i>Imputation</i> ||||||
| Exp 3             | Mean Substitution                           | 0.8165 | 0.7578 | <font color='red'>&#8593; +4.3450%</font> | - |
| Exp 4             | Median Substitution                         | 0.8064 | 0.7539 | <font color='red'>&#8593; +3.0543%</font> | - |
| Exp 5             | Frequency Substitution                      | 0.8233 | 0.7578 | <font color='red'>&#8593; +5.2141%</font> | Baseline 2 |
| Exp 6             | Multiple Imputation                         | 0.8243 | 0.7422 | <font color='red'>&#8593; +5.3419%</font> | - |
| Exp 7             | KNN Imputer                                 | 0.8069 | 0.7500 | <font color='red'>&#8593; +3.1182%</font> | - |
| <i>Data Correction</i> ||||||
| Exp 8             | Neareset Neighbors Outlier Detection        | 0.8962 | 0.8303 | <font color='red'>&#8593; +8.8546%</font> | Baseline 3 |
| Exp 9             | Isolation Forest Outlier Detection          | 0.8948 | 0.8582 | <font color='red'>&#8593; +8.6846%</font> | - |
| <i>Feature Scaling</i> ||||||
| Exp 10            | Min-Max Normalization on Columns            | 0.9132 | 0.8257 | <font color='red'>&#8593; +1.8969%</font> | - |
| Exp 11            | Min-Max Normalization on Instances          | 0.8292 | 0.6835 | <font color='blue'>&#8595; -7.4760%</font> | - |
| Exp 12            | Min-Max Scaling                             | 0.9099 | 0.8395 | <font color='red'>&#8593; +1.5287%</font> | - |
| Exp 13            | Maximum Absolute Scaling on Columns         | 0.9125 | 0.8211 | <font color='red'>&#8593; +1.8188%</font> | - |
| Exp 14            | Maximum Absolute Scaling on Instances       | 0.8303 | 0.6835 | <font color='blue'>&#8595; -7.3533%</font> | - |
| Exp 15            | Standardize                                 | 0.9095 | 0.8395 | <font color='red'>&#8593; +1.4840%</font> | - |
| Exp 16            | Robust Scaling                              | 0.9094 | 0.8395 | <font color='red'>&#8593; +1.4729%</font> | - |
| Exp 17            | Interval-based Discretization               | 0.9168 | 0.8395 | <font color='red'>&#8593; +2.2986%</font> | Baseline 4 |
| Exp 18            | Frequency-based Discretization              | 0.9137 | 0.8211 | <font color='red'>&#8593; +1.9527%</font> | - |
| Exp 19            | Clustering-based Discretization             | 0.9133 | 0.8395 | <font color='red'>&#8593; +1.9081%</font> | - |
| <i>Feature Transformation</i> ||||||
| Exp 20            | Polynomial Expansion                        | 0.7709 | 0.7706 | <font color='blue'>&#8595; -15.9140%</font> | - |
| Exp 21            | Power Transformation using Box-Cox          | 0.9125 | 0.8303 | <font color='blue'>&#8595; -0.4690%</font> | - |
| Exp 22            | Power Transformation using Yeo-Johnson      | 0.9084 | 0.8257 | <font color='blue'>&#8595; -0.9162%</font> | - |
| Exp 23            | Log Transformation                          | 0.9234 | 0.8303 | <font color='red'>&#8593; +0.7199%</font> | Baseline 5 |
| <i>Feature Expansion</i> ||||||
| Exp 24            | Kernel-induced Feature Expansion            | 0.9133 | 0.8395 | <font color='blue'>&#8595; -1.0938%</font> | - |
| <i>Feature Contraction</i> ||||||
| Exp 25            | Principal Component Analysis                | 0.8832 | 0.8303 | <font color='blue'>&#8595; -4.3535%</font> | - |
| Exp 26            | T-distributed Stochastic Neighbor Embedding | 0.8894 | 0.8257 | <font color='blue'>&#8595; -3.6820%</font> | - |
| Exp 27            | Isometric Mapping                           | 0.8823 | 0.8303 | <font color='blue'>&#8595; -4.4509%</font> | - |
| Exp 28            | Multi-Dimensional Scaling                   | 0.8825 | 0.8303 | <font color='blue'>&#8595; -4.4293%</font> | - |
| <i>Feature Selection</i> ||||||
| Exp 29            | Exhaustive Feature Selection                | 0.9234 | 0.8303 | <font color='red'>&#8593; 0</font> | Best Model |
| <i>Instance Generation</i> ||||||
| Exp 30            | Synthetic Minority Over-sampling            | 0.9115 | 0.8071 | <font color='blue'>&#8595; -1.2887%</font> | - |

### Experiment Analysis

- As we can see, Exp No.29 produces the best prediction results.

- In Exp No.29, I use __frequency substitution__, __nearest neighbors outlier detection__, __log transformation__, and __exhaustive feature selection__ to process the selected dataset at each data preparation stage.

- Compared to the first baseline method, _our final model increases the prediction accuracy on the selected dataset by 18.01%_.

## Reference

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988, November). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the annual symposium on computer application in medical care (p. 261). American Medical Informatics Association.
