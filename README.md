---
title: "Titanic Kaggle"
author: "Sergio Rubio"
date: "February 16, 2018"
output: html_document
---

<!---
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
--->

# Table of contents #
1. [Introduction](#introduction)
    1. [Context](#context)
    2. [Challenge](#challenge)
    3. [Evaluation](#evaluation)
    4. [Data](#data)
    
2. [Data understanding](#data_understanding)

3. [Modeling](#modeling)

4. [Evaluation](#evaluation)

5. [Conclusions](#conclusions)


# 1. Introduction <a name="introduction"></a> #

[**Titanic: Machine Learning from Disaster - Kaggle**](https://www.kaggle.com/c/titanic)

## 1.1. Context <a name="context"></a> ##

![](./images/RMS_Titanic.png "RMS Titanic")

_The sinking of the **RMS Titanic** is one of the most infamous shipwrecks in history.  On **April 15, 1912**, during her maiden voyage, the Titanic sank after colliding with an iceberg, **killing 1.502 out of 2.224 passengers and crew**. This sensational tragedy shocked the international community and led to better safety regulations for ships._

_One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, **some groups of people were more likely to survive than others**, such as women, children, and the upper-class._

[Sinking of the RMS _Titanic_ - Wikipedia](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic)

[Titanic sinking - National Geographic](https://www.youtube.com/watch?v=9xoqXVjBEF8)

## 1.2. Challenge <a name="challenge"></a> ##

The goal of this competition is to complete the analysis of what sorts of people were likely to survive the sinking. In particular, to apply the tools of machine learning to **predict which passengers survived the tragedy**.

This is a **classification** problem, specifically, a [binary classification](https://en.wikipedia.org/wiki/Binary_classification) problem, because there are only two different states we are classifying.

## 1.3. Evaluation <a name="evaluation"></a> ##

### Goal ###

The goal is to **predict if a passenger survived the sinking of the Titanic or not**.
For each PassengerId in the test set, the model must predict a 0 or 1 value for the Survived variable.

### Metric ###

The score is the **percentage of passengers correctly predicted**. This is known simply as "accuracy”.

### Submission File Format ###

The submission file should be a **csv** file with exactly **418 entries** plus a header row. Submission will show an error if the file has extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:

* **PassengerId** (sorted in any order)
* **Survived** (contains your binary predictions: 1 for survived, 0 for deceased)


```
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
```

## 1.4. Data <a name="data"></a> ##

### Overview ###

The data has been split into two groups:

* **Training set** (train.csv)
* **Test set** (test.csv)

The **training set** should be used to build the machine learning models. For the training set, each passenger has an outcome (also known as the “ground truth”). The model will be based on “features” like passengers’ gender and class. It is possible to create **new features** using [feature engineering](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/).

![](./images/Training_set.png "Training set")

The **test set** should be used to see how well the model performs on unseen data. For the test set, the ground truth for each passenger is not provided. The goal is to **predict**, for each passenger in the test set, whether or not they survived the sinking of the Titanic.

![](./images/Testing_set.png "Testing set")

The **gender_submission.csv** file is a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

### Data dictionary ###

**Variable**: Definition - Key

* **survival**: Survival - (0 = No, 1 = Yes)
* **pclass**: Ticket class - (1 = 1st, 2 = 2nd, 3 = 3rd)
* **sex**: Sex 	
* **age**: Age in years 	
* **sibsp**: # of siblings / spouses aboard the Titanic 	
* **parch**: # of parents / children aboard the Titanic 	
* **ticket**: Ticket number 	
* **fare**: Passenger fare 	
* **cabin**: Cabin number 	
* **embarked**: Port of Embarkation - (C = Cherbourg, Q = Queenstown, S = Southampton)

### Variable notes ###

* **pclass**: A proxy for socio-economic status (SES)
    * 1st = Upper
    * 2nd = Middle
    * 3rd = Lower
* **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
* **sibsp**: The dataset defines family relations in this way...
    * Sibling = brother, sister, stepbrother, stepsister
    * Spouse = husband, wife (mistresses and fiancés were ignored)
* **parch**: The dataset defines family relations in this way...
    * Parent = mother, father
    * Child = daughter, son, stepdaughter, stepson (Some children travelled only with a nanny, therefore parch=0 for them).
    
    
# 2. Data understanding <a name="data_understanding"></a> #

# 3. Data preparation <a name="data_preparation"></a> #

# 4. Modeling <a name="modeling"></a> #

# 5. Evaluation <a name="evaluation"></a> #

# 6. Conclusions <a name="conclusions"></a> #
