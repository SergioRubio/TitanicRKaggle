---
title: "Titanic Kaggle"
author: "Sergio Rubio"
date: "February 16, 2018"
output: "html_document"
---

# Table of contents #

1. [Introduction](#introduction)
    1. [Context](#context)
    2. [Challenge](#challenge)
    3. [Evaluation](#evaluation)
    4. [Data](#data)
    5. [Packages](#packages)
    
2. [Data understanding](#data_understanding)
    1. [Importing data](#importing_data)
    2. [Checking data](#checking_data)
    3. [Describing data](#describing_data)
    
3. [Data preparation](#data_preparation)
    1. [Feature engineering](#feature_engineering)
    2. [Correlation](#correlation)

4. [Modeling](#modeling)

5. [Evaluation](#evaluation)

6. [Conclusions](#conclusions)


# 1. Introduction <a name="introduction"></a> #

[**Titanic: Machine Learning from Disaster - Kaggle**](https://www.kaggle.com/c/titanic)

The first step of every data project is to have a **clear understanding of the problem** we are trying to solve. In this section we describe the **goal** of the competition and its **rules** and briefly describe the available data.

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
    
## 1.5. Packages <a name="packages"></a> ##

We need to use several **R packages** for data manipulation, visualization, modeling, etc.
    
```{r message = FALSE, warning = FALSE}
# Load packages

library('dplyr') # Data manipulation
library('ggplot2') # Visualization
library('mice') # Missing values
library('forcats') # Working with factors
library('corrplot') # Correlation matrix
```
    
# 2. Data understanding <a name="data_understanding"></a> #

The second step is to **know the details of the data** we have available to solve the problem. In this section we **load** the datasets, we **check consistency** and we **describe** them in detail. 

## 2.1. Importing data <a name="importing_data"></a> ##

```{r}
# Load train and test set preventing R to cast strings as factors

train <- read.csv('./data/train.csv', stringsAsFactors=FALSE)
test <- read.csv('./data/test.csv', stringsAsFactors=FALSE)

# Add type of set as a column

train$Set <- "train"
test$Set <- "test"

# Combine train and test sets. "Survived" is filled with NA for the test set

test$Survived <- NA
full <- rbind(train, test)
```

We combine the data from both the train and test set because, when performing **feature engineering**, it's useful to know the **full range of possible values** and the **distributions of all known values**. We added column `Set` to keep track of the training and test data.

## 2.2. Checking data <a name="checking_data"></a> ##

### Dimensions ###

```{r results = 'hide', message = FALSE, warning = FALSE}
# Check dimensions of imported sets

dim(train)
dim(test)
dim(full)
```

We confirm that the train set has **891 observations** and the test set has **418 observation** for a total of **1309 observations** as expected.

### Factors ###

We have to check if all the **categorical variables** are correctly set to [factors](https://www.stat.berkeley.edu/classes/s133/factors.html). 

```{r}
# Structure of the full set

str(full)
```

Because we imported the datasets with the `stringsAsFactors` parameter set to `FALSE`, we need to convert `Sex`, `Embarked` and `Set` variables **from string to factor**. We also convert `Survived` and `Pclass` variables **from int to factor**.

```{r}
# Cast variables to factor

factorize <- function(set, columns_to_factor) {
    set_factor <- set
    set_factor[columns_to_factor] <- lapply(select(set, one_of(columns_to_factor)), factor)
    return(set_factor)
}

columns_to_factor <- c("Survived", "Pclass", "Sex", "Embarked", "Set")

full <- factorize(full, columns_to_factor)
train <- factorize(train, columns_to_factor)
test <- factorize(test, columns_to_factor)
```

### Missing values ###

```{r}
# Summary of the full set

summary(full)
```

From the summary of the dataset we can see that there are several observations with **missing variables** `Age`, `Fare` and/or `Embarked`. Note that the 418 NA's of the `Survived` variable are from our test set.

Character variables (`Name`, `Ticket` and `Cabin`) might be missing as well and we wouldn't see it from the summary, because the character representation of a missing value is the **empty character** ("").

```{r}
# Count missing values by variable

colSums(is.na(full) | full=='')
```

We can plot missing values to show the **missing percentage** over the total number of observations.

```{r}
# Bar plot showing % of missing values by variable

data.frame(Perc = (colSums(is.na(full) | full=='')/dim(full)[1])*100, Var = names(full)) %>%
    filter(Perc > 0) %>%
    ggplot(aes(x=reorder(Var, -Perc), y = Perc)) +
    theme_classic() +
    geom_bar(stat='identity', width=0.75, fill="#F8766D") + 
    geom_text(aes(label=paste(round(Perc, 2), "%")), vjust=-0.5) + 
    labs(x='Variable', y='% missing', title='Missing values by variable')
```

When dealing with missing data it is important to **understand why they are missing**, when possible. Let's see each case in detail.

* **Cabin**: Titanic cabins were divided by passenger's class. Cabins **A-D** were assigned for the **1st class**, **E** for the **2nd class** and **F** for the **3rd class**. Cabin **T** was also a **1st class** cabin.

![](./images/Titanic_cabins.png "Titanic cabins")

Because of the high percentage of observations without an informed cabin, we can't just assign the most common cabins to missing values or throw them put. One solution is to **assign a value that indicates a missing**, because the fact that the value is missing can be **useful information** in and of itself.

```{r}
# Assign "Unknown" value to missing cabins

full$Cabin[full$Cabin == ""] <- "Unknown"
```

* **Embarked**: There were three possible ports of embarkation, **Cherbourg**, **Queenstown**, and **Southampton**.

![](./images/Titanic_embark.png "Titanic embark")

Because there are only 2 observations with a missing `Embarked`, we could **impute the most common embarkation point** to them.

```{r}
# Get observations with missing Embarked variable

filter(full, Embarked=="")

# Check most common embarkation point

summary(full$Embarked)
```

Both observations are 1st class passengers with the **same ticket**. From the summary we know that the most common embarkation point is **Southampton** (**~70%** of passengers), so we impute it to the two observations.

```{r}
# Impute most common embarked point to observations with missing Embarked variable

full$Embarked[full$Embarked == ""] <- "S" 

# Drop empty levels

full$Embarked <- factor(full$Embarked)
```

* **Fare**: With only one observation with missing `Fare`, we can follow the same approach as the `Embarked` case, but this time we will consider more variables for the imputation. Instead of directly impute the average fare, we will use the **median fare for the tickets with the same class and embarked point** than the one we are dealing with. This is because probably the fare of a ticket is mostly influenced by the class and the embarked point.

```{r}
# Get observations with missing Fare variable

filter(full, is.na(Fare))
```

The passenger has a **3rd class** ticket from **Southampton**. We get the median fare for 3rd class tickets from Southampton and assign it to the observation.

```{r}
# Median fare by class and embarked point

full %>% 
    group_by(Pclass, Embarked) %>%
    summarise(count = n(), median_fare = median(Fare, na.rm=TRUE))
```

```{r results = 'hide', message = FALSE, warning = FALSE}
# Assign median fare by class and embarked point to observations with missing fare

full <- full %>% 
            group_by(Pclass, Embarked) %>%
            mutate(Fare = ifelse(is.na(Fare), round(median(Fare, na.rm = TRUE), 4), Fare))
```

* **Age**: Variable `Age` has a relative high percentage of missings. One common way to deal with missing nominal variables is to **asign the average value to the missings**. This is a simple approach, but it's only recommended for variables that aren't really important, and probably in our case `Age` is one of the most important variables. 

A more complex approach is to **use a regression or another model to predict the missings**. We can apply this approach by using the [MICE package](https://datascienceplus.com/imputing-missing-data-with-r-mice-package/).

```{r results = 'hide', message = FALSE, warning = FALSE}
# Get Predictor-Matrix
init = mice(full, maxit=0, print=FALSE) 
predM = init$predictorMatrix

# Filter what columns not to use to impute the values
predM[, c("PassengerId", "Name","Ticket","Cabin")]=0    
imp <- mice(full, m=5, predictorMatrix = predM, print=FALSE)

# Get set with imputed age values filled
full_imp <- complete(imp)
```

```{r}
# Keep original Survived column and remove temporal full_imp set

full_imp$Survived <- full$Survived
full <- full_imp
rm(full_imp)
```

## 2.3. Describing data <a name="describing_data"></a> ##

In this section we will **look at the data** in as many different ways as possible. This is called [Exploratory Data Analysis (EDA)](https://en.wikipedia.org/wiki/Exploratory_data_analysis), and it's an approach to **summarize the main characteristics of the dataset**, often with visual methods.

```{r}
# Subset of full set containing only train data

full_tr <- filter(full, Set=="train")
```

We use the subset of train data from our full set because we need `Survived` informed and we want to **keep the imputations of missings** from the previous section.

```{r message = FALSE, warning = FALSE}
# Bar plot of the general survival rate

ggplot(full_tr, aes(x = Survived)) +
    theme_classic() +
    geom_bar(width = 0.75) +
    geom_text(stat='count', aes(label=paste(..count..," (",(round((..count../sum(..count..))*100, 2)), "%)"),vjust=-0.5)) + 
    labs(x='Survived', y='Passengers', title='Titanic survival rate')
```

Only **342 passengers of the 891** from the trainning set survived, a **38.38%** rate.

### Hypothesis ###

Before we start looking for relationships in the data, we should **formulate some hypothesis**.

[_Women and children first_](https://en.wikipedia.org/wiki/Women_and_children_first) is a code of conduct famously associated with the sinking of Titanic, where  survival resources such as **lifeboats were limited**.

![](./images/Titanic_lifeboat.png "Titanic lifeboat")

Another common hypothesis is that **rich passengers survival rate is higher than poor passengers**. A possible explanation for this is the **cabins location**; 1st class cabins were at the top level while 3rd class were at the bottom, making it hard to reach the lifeboats.

We can start then with the following hypothesis:

* **Women survival rate is higher than men**.
* **Children and young passengers survival rate is higher than adults**.
* **Rich passengers survival rate is higher than poor passengers**.

### Relationships ###

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by sex

surv_sex <- ggplot(full_tr, aes(x=Sex, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Sex', y='Survival rate', title='Titanic survival rate by sex/class') +
    guides(fill=FALSE)

# Bar plot survival rate by passenger class

surv_class <- ggplot(full_tr, aes(x=Pclass, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Class', y='', title='')

Rmisc::multiplot(surv_sex, surv_class, cols=2)
```

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by sex and passenger class

ggplot(full_tr, aes(x=Sex, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    facet_wrap(~Pclass) +
    labs(x='Sex', y='Survival rate', title='Titanic survival rate by sex and class')
```

```{r message = FALSE, warning = FALSE}
ggplot(full_tr) +
    theme_classic() +
    geom_freqpoly(aes(x = Age, color = Survived), binwidth = 1) +
    labs(x='Age', y='Passengers', title='Titanic survival rate by age')
```

```{r message = FALSE, warning = FALSE}
ggplot(full_tr) +
    theme_classic() +
    geom_freqpoly(aes(x = Fare, color = Survived), binwidth = 0.05) +
    scale_x_log10() +
    labs(x='Fare', y='Passengers', title='Titanic survival rate by fare (log10)')
```

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by SibSp

surv_sibsp <- ggplot(full_tr, aes(x=SibSp, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='SibSp', y='Survival rate', title='Titanic survival rate by SibSp/Parch') +
    guides(fill=FALSE)

# Bar plot survival rate by Parch

surv_parch <- ggplot(full_tr, aes(x=Parch, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Parch', y='', title='')

Rmisc::multiplot(surv_sibsp, surv_parch, cols=2)
```

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by embarked point

ggplot(full_tr, aes(x=Embarked, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Embarked', y='Survival rate', title='Titanic survival rate by embarked point')
```

# 3. Data preparation <a name="data_preparation"></a> #

In this section we prepare our dataset for the modelling step, **adding or removing features** (variables).

## 3.1. Feature engineering <a name="feature_engineering"></a> ##

One way to improve the results of a machine learning algorithm is to **create additional relevant features** from the existing features in the data. This process is called [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering).

### Title ###

If we take a look to the name variable we can see that it contains the **title of the passenger** (Mr., Mrs, etc.).

```{r}
# Top observations of Name variable

head(full$Name)
```

That could be relevant information, we can create the `Title` feature from the name.

```{r}
# Create Title feature from the Name variable

full <- full %>%
    mutate(Title = sub("^.*, (.*?)\\..*$", "\\1", Name))


# Count passengers by Title

full %>% 
    group_by(Title) %>% 
    summarise(Count = n()) %>% 
    mutate(Freq = round(Count/sum(Count),2)) %>%
    arrange(desc(Count))
```

As we can see, Mr, Miss and Mrs are taking more than **93%** of the values, Master the **5%** and the rest the remaining **2%**. We group this last remaining group to other titles.

```{r}
# Replace less frequent titles with more frequent ones

full <- full %>%
    mutate(Title = factor(case_when(
        Title %in% c("Mlle","Ms","Lady","Dona") ~ "Miss",
        Title == 'Mme' ~ 'Mrs',
        Title %in% c('Capt','Col','Major','Dr','Rev','Don','Sir','the Countess','Jonkheer') ~ 'Officer',
        TRUE ~ Title
    )))
```

Let's check the survival rate by our new `Title` feature.

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by passenger title

ggplot(filter(full, Set=='train'), aes(x=Title, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Title', y='Survival rate', title='Titanic survival rate by passenger title')
```


### Age ###

From the age variable we can create a **group age feature** that could be more useful than the raw age value.

```{r}
# Add age group feature

full <- full %>%
    mutate(Age_group = factor(case_when(
        Age < 5 ~ 'Infant',
        Age >= 5 & Age < 12 ~ 'Child',
        Age >= 12 & Age < 18 ~ 'Teenager',
        Age >= 18 & Age < 35 ~ 'Young adult',
        Age >= 35 & Age < 60 ~ 'Adult',
        Age >= 60 ~ 'Senior'
    )))
```

We check the survival rate by age group.

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by passenger age group

ggplot(filter(full, Set=='train'), aes(x=Age_group, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Age group', y='Survival rate', title='Titanic survival rate by passenger age group')
```

### Family ###

From `SibSp` and `Parch` variables we can derive the **number of family members** for each passenger.

```{r}
# Add family size and family type features

full$Family_size <- full$SibSp + full$Parch + 1

full <- full %>%
    mutate(Family_type = factor(case_when(
        Family_size == 1 ~ 'Single',
        Family_size >= 2 & Family_size <= 4 ~ 'Small',
        Family_size >= 5 ~ 'Big'
    )))
```

We can now check the survival rate by family size.

```{r message = FALSE, warning = FALSE}
# Bar plot survival rate by passenger family size

ggplot(filter(full, Set=='train'), aes(x=Family_type, fill=Survived)) +
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Family size', y='Survival rate', title='Titanic survival rate by family size')
```

### Ticket ###

Checking the ticket variable we can see that some passengers **shared the same ticket number**.

```{r}
# Summary of tickets by number of passengers

full %>%
    group_by(Ticket) %>%
    summarise(Count = n()) %>%
    group_by(Count) %>%
    summarise(Count_size = n()) %>%
    mutate(Freq_size = round(Count_size/sum(Count_size),2)) %>%
    arrange(desc(Count_size))
```

This summary shows that only **77% of the tickets belonged to one single passenger**, 14% were shared by two passengers, 5% by 3, etc.

We will replicate the approach of the `Family_size` variable adding the `Ticket_size` feature.

```{r}
# Add ticket size and ticket type features

full <- full %>%
    group_by(Ticket) %>%
    summarise(Ticket_size = n()) %>%
    inner_join(full, by='Ticket') %>%
    mutate(Ticket_type = factor(case_when(
        Ticket_size == 1 ~ 'Single',
        Ticket_size >= 2 & Ticket_size <= 4 ~ 'Small',
        Ticket_size >= 5 ~ 'Big'
    )))

```

Let's check the survival rate by the new `Ticket_type` feature.

```{r}
# Bar plot survival rate by passenger family size
ggplot(filter(full, Set=='train'), aes(x=Ticket_type, fill=Survived)) + 
    theme_classic() +
    geom_bar(stat='count', width=0.75, position='fill') +
    labs(x='Ticket size', y='Survival rate', title='Titanic survival rate by ticket size')
```


## 3.2. Correlation <a name="correlation"></a> ##

It is common that in a dataset some of the variables are related. Sometimes, the connection between some variables is **strongly enough so that one of them is superflous**. 

The best way to check if our variables are correlated among each other is visualising the **correlation matrix**.

```{r}
# Correlation matrix of the variables

filter(full, Set=='train') %>%
    select(-PassengerId, -Name, -Cabin, -Ticket, -Set, -Family_size) %>%
    mutate(Sex = fct_recode(Sex,'0'='male', '1'='female')) %>%
    mutate(Sex = as.integer(Sex),
           Pclass = as.integer(Pclass),
           Survived = as.integer(Survived),
           Title = as.integer(Title),
           Age_group = as.integer(Age_group),
           Family_type = as.integer(Family_type),
           Ticket_type = as.integer(Ticket_type),
           Embarked = as.integer(Embarked)) %>%
    cor(use="complete.obs") %>%
    corrplot(type="lower", diag=FALSE)
```

A correlation plot shows the correlation between each variable with colour circles. The **size and shading depends on the absolute values of the coefficients** and the **colour depends on direction (positive or negative correlation)**. 

From our correlation matrix we can see, among others, the following relations:

* `Survived` relates to `Pclass`, `Sex`, `Fare`, `Family_type` and `Ticket_type`.
* `Pclass` relates to `Age`, `Fare` and `Ticket_type`.
* `Age` relates to `Title` and `Age_group`.
* `Family_type` relates to `Ticket_type`.


# 4. Modeling <a name="modeling"></a> #

# 5. Evaluation <a name="evaluation"></a> #

# 6. Conclusions <a name="conclusions"></a> #
