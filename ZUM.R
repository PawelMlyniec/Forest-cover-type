# Load data
data <- read.csv(paste("data\\train.csv",sep=''), sep=",")

#MOdels
library(caret)
require(caTools)
set.seed(101) 
#feature_variance <- caret::nearZeroVar(data, saveMetrics = TRUE)


# Data pre-processing
X <- data[, -29]
x <- X[, -21]

sample = sample.split(X$Cover_Type, SplitRatio = .75)
X_train = subset(x, sample == TRUE)
X_val  = subset(x, sample == FALSE)

y_train = make.names(as.character(X_train$Cover_Type))
X_train = X_train[, -55]

y_val = make.names(as.character(X_val$Cover_Type))
X_val = X_val[, -55]

train_control <- trainControl(method="cv", number=10, classProbs= TRUE, summaryFunction = multiClassSummary)
metric <- "logLoss"

# Naive Bayes
m_nb <- train(X_train, y_train, method="naive_bayes", metric=metric, preProcess = "pca", 
                trControl=train_control )
print(m_nb)

# K Nearest Neighbours
m_kknn <- train(X_train, y_train, method="kknn", metric=metric, preProcess = "pca", 
              trControl=train_control )
print(m_kknn)

# C5.0 Tree
m_c50 <- train(X_train, y_train, method="c5.0", metric=metric, preProcess = "pca", 
              trControl=train_control )
print(m_c50)
