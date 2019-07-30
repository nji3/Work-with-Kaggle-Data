# Credit Card Fraud Data

The [credit card fraud data](https://www.kaggle.com/mlg-ulb/creditcardfraud) is a very popular dataset on kaggle. This dataset contains about 290000 records of data with 31 variables. Among all the variables, we only have three meaningful variables which are the time, amount and the class. The other 28 variables are the top 28 principle components after the PCA on the original dataset. Because of the privacy of the original data, we could not know more information. However, this prePCA work just reduce the difficulty to use this dataset for prediction purpose.

## How could this dataset help us?

This dataset is unique because it is a very imbalanced dataset. Though it only has two classes, the ratio between the majority group and the minority group is very huge. Below are several plots show some basic information of the distributions.

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/basic_distribution.png" width="600px"</img> 
</div>

We would notice two things:

1. Almost 99.9% of the records are normal and only 0.17% records are fruad. This is an extremely inbalanced dataset.

2. Many Fraud records only have a very small amount of money. There is no simple and direct relation between time, amount and the fraud.

This dataset gives us a good chance to gain experience of handling the imbalanced dataset during the model fitting.

## Check out with other pre-PCA variables

Because of the data privacy, the information of the real variables are not provided. The only provided variables are the PCA transformation with the reduced dimension. We could start with the visualization of the first two components. Additionally, a t-sne visualization could be used to display the condensed information of all principle components.

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/pca_distributions.png" width="600px"</img> 
</div>

Visualize the first three principle components.

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/first_3_pcs.png" width="600px"</img> 
</div>

There are examples that people do a further PCA on the principle components. However, I don't think it would give more senses because the principle components have already been orthonormaled as much as possible. However, we could try LDA to do a further projections of the data and check if the two classes are more seperable.

## Simple Seperation of Two groups by LDA and TSNE

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/lda_res.png" width="400px"</img> 
</div>

Here I did a very rough LDA on the whole dataset, which means that I did not seperate the training and testing dataset. I just used the whole dataset (definitely bring the overfitting problem). One point I want to make is that the super imbalanced dataset is definitely hard for traditional classification methods. Here the class 0 with lots of data gather very close to each other, but the Fraud class data spreads a lot. Especially at the boundary, it is very difficult to seperate the two classes.

Visualization by T-sne

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/tsne_res.png" width="400px"</img> 
</div>

Using all the data for tsne will cost too much. Here a random sample (5000 obs) of non-fraud records is selected and combined with all the fraud records (492 obs). As a result, 5492 records in total are used in the tsne. We could see that two classes are seperated in general. However, still many fraud records cannot be seperated from the non-fraud records. In real life, the fraud happens rarely but if it happens, it will cost a great loss. We should make the classification much more precisely. So now, we should start with some model fitting. We need to deal with the imbalanced data and try to have a more legit prediction model.

## Model Fitting

The trainning set and testing set split here would be a bit different from we usually do. Because of the highly imbalanced ratio, the test set we want should still maintain the ratio of the data in two classes. To do that, we could use the stratified split to seperate the data.

To evaluate the imbalanced data in the classification task, the accuracy would not make much sense because if our model simply classify every thing to non-fraud, the accuracy could go up to 99%. The evaluation metrics we want should contain:

**Confusion Matrix**: a table showing correct predictions and types of incorrect predictions.

**Precision**: the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier’s exactness. Low precision indicates a high number of false positives.

**Recall**: the number of true positives divided by the number of positive values in the test data. Recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier’s completeness. Low recall indicates a high number of false negatives.

**F1 Score**: the weighted average of precision and recall.

And also the **ROC scores and plot**.

### Applying Resampling Techniques for Classification

In general, there are two resampling techniques dealing with the imbalanced dataset, **Oversampling** the minority class and **Undersampling** the majority class. These two techniques help the proportion of all groups in the training data closer to each other.

**Oversampling** the minority class can be done by simple repetition, bootstrap or even more advanced design like SMOTE (synthetic minority over-sampling technique). I will try the repetition sampling with a small random noise add into the data.

**Undersampling** the majority class can be done by just sampling a subset of the original larger group. However, this would definitely cause the infromation loss. Directly using one sample of the majority class will not be proper, we could use the random forest idea that keep sampling and fit the model and average the result at the end. This could be an idea for those who want to be innovative when doing the model fitting but I will not do it here.

**Oversampling + Logistic Regression**:

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/over_logistic.png" width="400px"</img> 
</div>

**Oversampling + Random Forest**:

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/over_randomforest.png" width="400px"</img> 
</div>

Though Above results show that generally random forest model could give a better classification result for the minority group than the logistic regression. However, simpel oversampling and adding some random noise still not fix the problem well. The precision of minority group is just too small.

**SMOTE (synthetic minority over-sampling technique)**

SMOTE uses a nearest neighbors algorithm to generate new and synthetic data we can use for training our model.

**SMOTE + Logistic Regression**:

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/smote_logistic.png" width="400px"</img> 
</div>

**SMOTE + Random Forest**:

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/smote_randomforest.png" width="400px"</img> 
</div>

With the SMOTE and Random Forest, we have some pretty legit result! We could give a relatively high precision in the minority class even when we have just 5 trees in the model. It's a pretty good start with the model tuning.

As a result, the combination of good resampling technique and classification model can help a lot for the classification in the imbalanced data. In the next part, we are going to try another idea that during the model fitting, we can add more penalty to the wrongly classified individules so that it is more costable if the minority group members are classified as the majority goup members. The boosting machines are methods that could help us.

### Boosting Machines

**Adaboost**

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/ada.png" width="400px"</img> 
</div>

**Gradient Boost**

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/gb.png" width="400px"</img> 
</div>

**XGBoost**

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/xgb.png" width="400px"</img> 
</div>

XGBoost Feature importance plot

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/xgb_importance.png" width="400px"</img> 
</div>

XGBoost ROC plot

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/xgb_roc.png" width="400px"</img> 
</div>

Without any model tunning, both adaboost and xgboost a model we would want. Xgboost model just gave a model with very high precision in both classes and xgboost model also trains very fast. Considering about all the time and efforts we needed in the data preparation work for sampling techniques, the bagging techniques like boosting machines could be a really effective way for us to solve imbalanced dataset classification task.

### Neural Networks

We could use neural networks to help us with the classification. Here I would only construct a very simple NN model for the classification and an Autoencoder model. The autoencoder model would also compress the information in to a lower dimensional space. We could see if we can have a better seperation of the two classes at there.

**Standard Dense NN**

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/nn.png" width="400px"</img> 
</div>

**Autoencoder**

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/auto.png" width="400px"</img> 
</div>

<div align="center">
        <img src="https://github.com/nji3/Work-with-Kaggle-Data/blob/master/Credit Card Fraud Detection/readme_image/auto_latent.png" width="400px"</img> 
</div>