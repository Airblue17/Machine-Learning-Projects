library(ggplot2)
library(dplyr)

df_w <-  read.csv('R/Machine Learning Projects/R/Wine Color Clustering - K-Means/winequality-white.csv', sep = ";")

df_r <- read.csv('R/Machine Learning Projects/R/Wine Color Clustering - K-Means/winequality-red.csv', sep =";")

df_w$color = 'White' 

df_r$color = 'Red'

str(df_w)

head(df_r)

wine <- rbind(df_w, df_r)

str(wine)

# EDA

path = 'R/Machine Learning Projects/R/Wine Color Clustering - K-Means/plots/'

# Histogram of residual sugar colored by wine color

pl <- ggplot(data = wine, aes(x = residual.sugar)) + geom_histogram(binwidth = 1.5, aes(fill = color), color = 'Black')

print(pl)

ggsave(paste(path, "hist_residualsugar.jpg", sep = ""), plot = last_plot(), dpi = 300)


#  Histogram of citric.acid colored by wine color

pl <- ggplot(data = wine, aes(x = citric.acid)) + geom_histogram(binwidth = 0.03, aes(fill = color), color = 'Black')

print(pl)

ggsave(paste(path, "hist_citricacid.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Histogram of alcohol % colored by wine color


pl <- ggplot(data = wine, aes(x = alcohol)) + geom_histogram(binwidth = 0.15, aes(fill = color), color = 'Black')

print(pl)


ggsave(paste(path, "hist_alcperc.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Scatterplot of residual.sugar versus citric.acid colored by wine color

pl <- ggplot(data = wine, aes(x = citric.acid, y = residual.sugar)) + geom_point(aes(color = color))

print(pl)

ggsave(paste(path, "ressugarVScitacid.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Scatterplot of residual.sugar versus volatile.acidity colored by wine color

pl <- ggplot(data = wine, aes(x = volatile.acidity, y = residual.sugar)) + geom_point(aes(color = color))

print(pl)

ggsave(paste(path, "ressugarVSvolacid.jpg", sep = ""), plot = last_plot(), dpi = 300)


# Model
train <- wine[1:12]

set.seed(101)

wine.clusters <- kmeans(train,centers = 2, nstart = 30)

print(wine.clusters$centers)


table(wine.clusters$cluster, wine$color)

# Visualization for top two components in terms of explaining the variability in the data

library(cluster)

jpeg(paste(path, "clusters.jpg", sep = ""))
print(clusplot(train, wine.clusters$cluster, color = T, shade = T, labels = 0, lines = 0))
dev.off()
