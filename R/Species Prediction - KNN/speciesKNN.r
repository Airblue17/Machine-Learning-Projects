library(ISLR)
library(dplyr)
library(caTools)
library(class) # for knn
library(ggplot2)

head(iris)
summary(iris)
str(iris)

# For k-nn, attributes need to be standarized

std_iris <- scale(iris[1:4])
str(std_iris)
var(std_iris)

std_iris <- cbind(std_iris, iris[5])

# Train-Test Split


set.seed(101)

sample <- sample.split(std_iris$Species, SplitRatio = 0.7)

train <- subset(std_iris, sample == TRUE)
test <- subset(std_iris, sample == FALSE)

pred.species <- knn(train[1:4], test[1:4], train$Species, k=1)

error <- mean(pred.species != test$Species)

print(paste("Misclassification Rate when k = 1: ",error))


# Choosing a k-value
error_rates <- c()
for (i in seq(1,10,1)){
  set.seed(101)
  pred.species <- knn(train[1:4], test[1:4], train$Species, k=i)
  
  error_rates[i] <- mean(pred.species != test$Species)
  
  print(paste("Misclassification Rate when k = ",i,": ",error_rates[i]))
}

# Visualizing error_rate vs k


path = 'R/Machine Learning Projects/R/Species Prediction - KNN/plots/'

eRate_k <- data.frame(eRate = error_rates, k = seq(1,10,1))

print(eRate_k)

pl <- ggplot(data = eRate_k, aes(x = k, y = eRate))

pl <- pl + geom_point() + geom_line(color = 'Red')

print(pl)

ggsave(paste(path, "err_rate.jpg", sep = ""), plot = last_plot(), dpi = 300)
