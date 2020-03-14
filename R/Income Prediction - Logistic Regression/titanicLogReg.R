library(ggplot2)
library(caTools)
library(dplyr)
library(corrplot)
library(corrgram)
library(countrycode)
library(Amelia)
library(caret)
library(caTools)

adult <- read.csv('~/R/Machine Learning Projects/R/Titanic Survival - Logistic Regression/adult_sal.csv')

head(adult)

adult <- adult %>% select(-X) # X(index) is not neeeded

head(adult)

# Structure of the data

str(adult)

# Summary

str(adult)



# Reducing factor levels

# For type_employer attribute

# Frequency of each factor for type_employer attribute
table(adult$type_employer)

adult$type_employer <- as.character(adult$type_employer)

adult[adult$type_employer == 'Never-worked' | adult$type_employer == 'Without-pay',]$type_employer <- 'Unemployed'
adult[adult$type_employer == 'Local-gov' | adult$type_employer == 'State-gov',]$type_employer <- 'SL-gov'
adult[adult$type_employer == 'Self-emp-inc' | adult$type_employer == 'Self-emp-not-inc',]$type_employer <- 'self-emp'

adult$type_employer <- as.factor(adult$type_employer)

table(adult$type_employer)


# For married attribute

table(adult$marital)

adult$marital <- as.character(adult$marital)

adult[adult$marital == 'Married-AF-spouse' | adult$marital == 'Married-civ-spouse' | adult$marital == 'Married-spouse-absent',]$marital <- 'Married'
adult[adult$marital == 'Divorced' | adult$marital == 'Widowed' | adult$marital == 'Separated',]$marital <- 'Not-Married'

adult$marital <- as.factor(adult$marital)
table(adult$marital)


str(adult)


# For country attribute

table(adult$country)

adult$country <- as.character(adult$country)

adult[adult$country == '?' | adult$country == 'South' ,]$country = 'Other'

adult[adult$country == 'Columbia',]$country = 'Americas'

adult[adult$country == 'England' | adult$country == 'Scotland' | adult$country == 'Yugoslavia',]$country = 'Europe'

adult[adult$country == 'Hong',]$country = 'Asia'


adult[adult$country != 'Other' & adult$country != 'Americas' & adult$country != 'Europe' & adult$country != 'Asia', ]$country <- 
  countrycode(adult[adult$country != 'Other' & adult$country != 'Americas' & adult$country != 'Europe' & adult$country != 'Asia', ]$country, 
               origin = "country.name", destination = "continent")

table(adult$country)

adult$country <- as.factor(adult$country)


# Replacing "?" values with NA

adult[adult=='?'] = NA

# Checking for missing values
missmap(adult,y.at=c(1),y.labels = c(''),col=c('Red','Black')) # About 1% values are missing which can be ignored

# Omitting NA values

adult <- na.omit(adult)

adult <- droplevels(adult)

# Verify that no values are missing anymore

missmap(adult,y.at=c(1),y.labels = c(''),col=c('Red','Black')) # About 1% values are missing which can be ignored

# Structure of data frame after EDA

str(adult)


# Histogram of age

pl <- ggplot(adult, aes(x = age)) +
      geom_histogram(aes(fill = income), color = 'Black',bins = 50)

pl


# Histogram of hours worked per week

pl2 <- ggplot(adult, aes(x = hr_per_week)) +
  geom_histogram()

pl2


# Renaming the country attribute to region

adult <- adult %>% rename(region = country)

# Barplot of region
pl3 <- ggplot(adult, aes(x= region)) +
        geom_bar(aes(fill=income), color = 'black')

pl3

set.seed(17)

# Splitting into test and train set

sample <- sample.split(adult, 0.75)
  
train <- subset(adult, sample == TRUE)
test <- subset(adult, sample == FALSE)


# Model

formula <- income ~ .

log.model <- glm(formula, family = binomial(link = "logit"), data = train)


# Removing attributes that don't contribute significantly to the model

log.model2 <- step(log.model)

# Prediction
fit.prob <- predict(log.model2,newdata=test,type='response')

# Evaluation
fit.results <- ifelse(fit.prob > 0.5, 1,0 )

testRes <- ifelse(test$income == ">50K", 1, 0)

misClasificError <- mean(fit.results != testRes)
print(paste('Accuracy',1-misClasificError))


table(testRes, fit.prob > 0.5)

fit.results <- as.factor(fit.results)
testRes <- as.factor(testRes)

precision <- posPredValue(fit.results, testRes, positive="1")
recall <- sensitivity(fit.results, testRes, positive="1")

F1 <- (2 * precision * recall) / (precision + recall)
