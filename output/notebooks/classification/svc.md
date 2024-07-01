---
title: Linear SVC for Resume Classification
category: NLP
---


### Linear SVC

For this first model, I train a baseline Linear SVC{% sidenote 'Linear Support Vector Classifier (SVC) is a classification algorithm that seeks to find the *maximum-margin hyperplane*, that is, the hyperplane that most clearly classifies observations' '' %} using the TF-IDF vectors. I then performance Latent Semantic Analysis (LSA) by applying Truncated Singular Value Decomposition (SVD){% sidenote 'Truncated Singular Value Decomposition (SVD) is a dimensionality reduction technique that decomposes a matrix into three smaller matrices, retaining only the most significant features of the original matrix.' '' %} to the TF-IDF matrix, to reduce the matrix size to a lower dimensional space. I evaluate the performance of both models, which will serve as the baseline to compare with more complex models in the upcoming sections.

> #### Model Takeaways
>
> -   Linear SVC achieved an accuracy of 84% on the test set.
> -   Linear SVC with Truncated SVD achieved an accuracy of 81% on the test set with only 5% of the original number of features.
> -   The baseline model achieved a high level of accuracy and is more cost-effective when reducing the dimensionality of the feature matrix.

#### Import Packages and Data

Aside from the standard data processing libraries, I import two custom functions: classifier_report to generate a classification report and confusion matrix, and save_performance saves the performance metrics of the model to a JSON file for future analysis. I also load the pre-trained Label Encoder to label the encoded categories in upcoming plots.
{: #open}

``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.modules.ml import classifier_report
from src.modules.utils import save_performance

df = pd.read_parquet('./data/3-processed/resume-features.parquet')
# Label model to label categories
le = joblib.load('./models/le-resumes.gz')
```

#### Baseline LinearSVC

I split the dataset into 70% training and 30% testing sets using the tfidf_vectors column as the feature matrix and the Category column as the target variable. To demonstrate the split, I print the shape of the training and testing sets.
{: #open}

``` python
X = np.vstack(df['tfidf_vectors'].values)
y = df['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")
```

    Training set: (2016, 11618), (2016,)
    Testing set: (864, 11618), (864,)

Next, I train a Linear SVC model with default hyperparameters. The classifier_report function generates a classification report and confusion matrix to evaluate the model’s performance.

``` python
svc = LinearSVC(dual="auto")
accuracy = classifier_report([X_train, X_test, y_train, y_test], svc, le.classes_, True)
```

    SVC accruacy score 84.84%

The model achieved an accuracy of 84% on the test set, which is already very good for a baseline model. However, TF-IDF vectors tend to be sparse, resulting in a high-dimensional representation with many redundant features. To address this issue, I applied Truncated Singular Value Decomposition (SVD) to reduce the dimensionality of the feature matrix, aiming to improve the model’s performance and efficiency.

#### LinearSVC with Truncated SVD

I use TruncatedSVD to reduce the number of components to 500, which surprisingly is less than 5% of the original number of features. Below I apply the transformation and split the transformed feature matrix into training and testing sets before training the new model.
{: #open}

``` python
t_svd = TruncatedSVD(n_components=500, algorithm='arpack')
X_svd = t_svd.fit_transform(X)


X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(X_svd, 
                                                                    y, 
                                                                    test_size=0.30, 
                                                                    random_state=42)


print(f"Training set: {X_train_svd.shape}, {y_train_svd.shape}")
print(f"Testing set: {X_test_svd.shape}, {y_test_svd.shape}")
```

    Training set: (2016, 500), (2016,)
    Testing set: (864, 500), (864,)

I now train a new Linear SVC model using the SVD-transformed feature matrix and generate a classification report and confusion matrix.

``` python
svc_svd = LinearSVC(dual="auto")
accuracy = classifier_report([X_train_svd, X_test_svd, y_train_svd, y_test_svd], svc_svd, le.classes_, True)
```

    SVC accruacy score 81.94%

The model achieved an accuracy of 81% on the test set, which is slightly lower than the baseline model. However, these results come from a model trained on less than 5% of the original number of features while still retaining a high level of accuracy. The results demonstrate the sparcity of the TF-IDF vectors, while also showing that the model can still achieve an excellent level of accuracy with a substantially reduced feature matrix.
In training Linear SVC for resume classification, I achieved an accuracy of 84% on the test set using the baseline model and 81% using the model with SVD-transformed features. The results demonstrate that the baseline model achieves a high level of accuracy and is particularly more cost-effective when reducing the dimensionality of the feature matrix.
To conclude, I will save the performance metrics for the baseline model for later comparison.
{: #open}

``` python
save_performance(model_name='LinearSVC',
                 architecture='default',
                 embed_size='n/a',
                 learning_rate='n/a',
                 epochs='n/a',
                 optimizer='n/a',
                 criterion='n/a',
                 accuracy=87.15
                 )
```
