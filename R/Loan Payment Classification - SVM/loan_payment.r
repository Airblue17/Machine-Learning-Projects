library(dplyr)
library(e1071)
library(ggplot2)

# Load the data
loans <- read.csv ('R/Machine Learning Projects/R/Loan Payment Classification - SVM/loan_data.csv')

# Summary and Structure
summary(loans)

str(loans)

# converting certain attributes to Factor

loans$inq.last.6mths  <- factor(loans$inq.last.6mths)
loans$delinq.2yrs <- factor(loans$delinq.2yrs)
loans$pub.rec  <- factor(loans$pub.rec)
loans$not.fully.paid <- factor(loans$not.fully.paid)
loans$credit.policy <- factor(loans$credit.policy)

# Exploratory Data Analysis

path = 'R/Machine Learning Projects/R/Loan Payment Classification - SVM/plots/'

# Histogram of fico scores colored by not.fully.paid

pl <- ggplot(data = loans, aes(x=fico)) + geom_histogram(binwidth = 6, aes(fill = not.fully.paid), color = 'Black')

print(pl)
ggsave(paste(path, "hist_fico.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Bar Plots of purpose counts colored by not.fully.paid

pl <- ggplot(data = loans, aes(x=purpose)) + geom_bar(position = 'dodge', aes(fill = not.fully.paid))

print(pl)

ggsave(paste(path, "bar_purposecounts.jpg", sep = ""), plot = last_plot(), dpi = 300)

# Scatter plot fico score vs int.rate 

pl <- ggplot(data = loans, aes(x = int.rate, y = fico)) + geom_point(aes(color = not.fully.paid))

print(pl)

ggsave(paste(path, "ficoVsintrare.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Model

library(caTools)

sample <- sample.split(loans$not.fully.paid, SplitRatio = 0.7)

train <- subset(loans, sample = T)
test <- subset(loans, sample = F)
formula <- not.fully.paid ~.

svm.model <- svm(formula, train[1:14])

summary(svm.model)

# Prediction

pred.results <- predict(svm.model, test[1:13])

table(pred.results, test$not.fully.paid)

# Grid Search

tune.results <- tune(svm, train.x = formula, data = train, kernel = 'radial',
                     ranges = list(cost = seq(0.1, 0.3, 0.1), gamma <- seq(0.1, 0.3, 0.1)))
