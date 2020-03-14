library(randomForest)
library(rpart)
library(ISLR)
library(ggplot2)
library(dplyr)

head(College)

df <- College

#### 
# EDA
###

path = 'R/Machine Learning Projects/R/School Classification - Tree Methods/plots/'

# Scatter Plot of Graduation rate against Room+Board Cost colored by the school classification
pl <- ggplot(data = df, aes(x=Room.Board,y=Grad.Rate)) + geom_point(aes(color = Private))

print(pl)

ggsave(paste(path, "scatter_gratevscost.jpg", sep = ""), plot = last_plot(), dpi = 300)

# Histogram of full-time undergrad students colored by the school classsification

pl <- ggplot(data= df, aes(x=F.Undergrad)) + geom_histogram(binwidth = 500, aes(fill = Private), color = 'Black')

print(pl)

ggsave(paste(path, "hist_ugrad.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Histogram of graduation rate colored by the school classsification

pl <- ggplot(data= df, aes(x=Grad.Rate)) + geom_histogram(binwidth = 2, aes(fill = Private), color = 'Black')

print(pl) # One of the colleges has graduation rate > 100%!

ggsave(paste(path, "hist_grate.jpg", sep = ""), plot = last_plot(), dpi = 300)

# College that has graduation rate > 100%

print(filter(df, Grad.Rate > 100))

#Changing it to 100
df$Grad.Rate[df$Grad.Rate>100] = 100

###
# Train and Test Split
###

library(caTools)

set.seed(101)

sample <- sample.split(df$Private, SplitRatio = 0.7)

train <- subset(df, sample == T)
test <- subset(df, sample == F)

###
# Decision Tree
###

formula <- Private ~.
dtree <- rpart(formula, method = 'class', data=train)

pred.prob <- predict(dtree, test)

head(pred.prob)

pred.prob <- as.data.frame(pred.prob)

pred.prob$Private <- 'No'

pred.prob[pred.prob['Yes'] >= 0.5,]$Private = 'Yes'

pred.prob$Private <- as.factor(pred.prob$Private)

str(pred.prob)

# Confustion Matrix
table(pred.prob$Private,test$Private)

# Visualization of the model
library(rpart.plot)

jpeg(paste(path, "dtree.jpg", sep = ""), res = 100)
print(prp(dtree))
title("Private School Classification - Decision Tree", line = 2.5)
dev.off()


###
# Random Forest
###

rf <- randomForest(formula, train, importance = T)

# confusion matrix

print(rf$confusion)

print(rf$importance)


pred.prob_rf <- predict(rf, test)

head(pred.prob_rf)

# Confustion Matrix
table(pred.prob_rf,test$Private)
