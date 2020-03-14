library(neuralnet)
library(dplyr)
library(ggplot2)

bnotes <-  read.csv('R/Machine Learning Projects/R/Authentic Bank Note Classification - Neural Nets/bank_note_data.csv')

head(bnotes)

str(bnotes)

summary(bnotes)

# EDA

path = 'R/Machine Learning Projects/R/Authentic Bank Note Classification - Neural Nets/plots/'


# Histogram, of variance of wavelet transformed image colored by class

pl <- ggplot(bnotes, aes(x = Image.Var)) + geom_histogram(binwidth = 0.5, aes(fill = as.factor(Class)), color = 'Black')

print(pl)

ggsave(paste(path, "hist_varImage.jpg", sep = ""), plot = last_plot(), dpi = 300)

# Histogram, of skewnness of wavelet transformed image colored by class

pl <- ggplot(bnotes, aes(x = Image.Skew)) + geom_histogram(binwidth = 0.8, aes(fill = as.factor(Class)), color = 'Black')

print(pl)

ggsave(paste(path, "hist_skewImage.jpg", sep = ""), plot = last_plot(), dpi = 300)

# Model

# Train Test Split

library(caTools)

sample <- sample.split(bnotes$Class, SplitRatio = 0.7)

train <- subset(bnotes, sample = T)

test <- subset(bnotes, sample = F)

# Formula generation

n <- names(train)

f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

# Training
nn.model <- neuralnet(f, train, linear.output = F, hidden = 8)

# Visualization

#jpeg(paste(path, "n_net.jpg", sep = ""))
plot(nn.model)
#dev.off()

# Predictions

nn.predictions <- neuralnet::compute(nn.model, test[1:4])

pred.values <- nn.predictions$net.result

head(pred.values)

pred.values <- round(pred.values)

head(pred.values)

table(test$Class, pred.values)


# Comparing with Random Forest

library(randomForest)
rf <- randomForest(f, train, importance = T)

# confusion matrix

print(rf$confusion)

print(rf$importance)


pred.prob_rf <- predict(rf, test)

head(pred.prob_rf)

pred.prob_rf <- round(pred.prob_rf)

# Confustion Matrix
table(pred.prob_rf,test$Class)
