path ='C:\\Users\\ja\\Documents\\elkaPW\\ZUM2\\';
data <- read.csv(paste(path,"train.csv",sep=''), sep=",")

summary(data)
print(colnames(data))
data <- data[,-1]
summary(data)

#PCA
  #install.packages("factoextra")
  library(factoextra)
  data.pca <- prcomp(data, center=TRUE)
  fviz_eig(data.pca)
  summary(data.pca)

  ## Get data
  #TODO get PCA values
  pca.var <- get_pca_var(data.pca)
  summary(pca.var)


#MOdels
library(caret)
require(caTools)
set.seed(101) 
feature_variance <- caret::nearZeroVar(data, saveMetrics = TRUE)

x <- data[, -29]
x <- x[, -21]

sample = sample.split(x$Cover_Type, SplitRatio = .75)
x_train = subset(x, sample == TRUE)
x_val  = subset(x, sample == FALSE)


method = 'pcar'
method = c('naive_bayes','kknn', 'C50' )
levels(x_train[,-55])
levels(x_train$Cover_Type)
train.control <- trainControl(method="cv", number=10, classProbs= TRUE, summaryFunction = multiClassSummary)
metric <- "logLoss"

y_train <-  make.names(as.character(x_train$Cover_Type))
m_kknn <- train(x_train[,-55], y_train, method="kknn", metric=metric, preProcess = "pca", 
                trControl=train.control )


print(m_kknn)
