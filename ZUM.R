path ='C:\\Users\\ja\\Documents\\elkaPW\\ZUM2\\';
data <- read.csv(paste(path,"train.csv",sep=''), sep=",")

summary(data)
data <- data[,-1]
data.pca <- prcomp(data, center=TRUE)
install.packages("factoextra")
library(factoextra)
fviz_eig(data.pca)
print(data.pca)
summary(data.pca)
data.pca
