library(ggplot2)
library(caTools)
library(dplyr)
library(corrplot)
library(corrgram)


df <- read.csv('bikeshare.csv')

head(df)

#str(df)

df$datetime <- as.POSIXct(df$datetime, format = "%Y-%m-%d %H:%M:%S", tz = "EST5EDT")

#str(df)


#Plotting temp against count(Response variable)
plot1 <- ggplot(df, aes(x = temp, y = count)) + geom_point(aes(color = temp), alpha = 0.2)

ggsave(path="plots",filename="countVtemp.jpg",plot = plot1)

#plotting datetime against count with color filled by corresponding temperature values

plot2 <- ggplot(df, aes(x = datetime, y = count)) + geom_point(aes(color = temp)) + 
         scale_color_gradient( low = 'cyan', high = 'orange')

ggsave(path="plots",filename="countVdatetime.jpg",plot = plot2)


#Correlation between temp and count
corr.data <- cor(df[,c("temp","count")])

#box plot between season and count

plot3 <- ggplot(df, aes(x = factor(season), y = count)) + geom_boxplot(aes(color = factor(season)))

ggsave(path="plots",filename="countVseason.jpg",plot = plot3)

#generating hour attribute
df$hour <- format(df$datetime, "%H")

#scatter plot between count and hour for working days
plot4 <- ggplot(subset(df, workingday == 1), aes(x = hour, y = count)) + 
      geom_point(aes(color = temp), position=position_jitter(w=1, h=0) ) +     
         scale_color_gradientn(colors = c("cyan","green","Orange"))

ggsave(path="plots",filename="countVhour(1).jpg",plot = plot4)

#non working days
plot4b <- ggplot(subset(df, workingday == 0), aes(x = hour, y = count)) + 
  geom_point(aes(color = temp), position=position_jitter(w=1, h=0) ) +     
  scale_color_gradientn(colors = c("cyan","green","Orange")) 

ggsave(path="plots",filename="countVhour(2).jpg",plot = plot4b)

#model based on just temperature
temp.model <- lm(count ~ temp, data = df)

summary(temp.model)

#for predicting count when temp = 25c
test <- data.frame(temp = c(25))

pred <- predict(temp.model, test)

df$hour <- sapply(df$hour,as.numeric)

numeric.cols <- sapply(df, is.numeric)
corr.data2 <- cor(df[,numeric.cols])
print(corrplot(corr.data2,method = "color"))

#predicting using other columns
formula = as.formula(count ~ season + holiday + workingday + weather + temp + humidity + windspeed + hour)

bikeShare.model <- lm(formula, data = df)

summary(bikeShare.model)
