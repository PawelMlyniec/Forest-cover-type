# Load data
data <- read.csv(paste("data\\train.csv",sep=''), sep=",")

#summary
summary(data)

#delete ID column
data <- data[,2:56]

#set Cover_type as categorical
data$Cover_Type <- as.factor(data$Cover_Type)

#class balance
table(data$Cover_Type)

# calculate skewness
#install.packages("moments")
library(moments)
skewness(data) #problem with soil_type 8, 25 or 7 and 15

#corellation between arguments
library(reshape2)
library(ggplot2)

cormat <- round(cor(data[,1:10]),2)

## Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri  <- get_upper_tri(cormat)

melted_cormat <- melt(upper_tri , na.rm = TRUE)

ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

#Box plot
for (column in 1:54){
  boxplot(data[,column]~Cover_Type,
          data=data,
          main=paste("Boxplot for", names(data)[column],  "against Cover Type"),
          xlab="Cover Type",
          ylab=names(data)[column],
          col="orange",
          border="brown"
  )
}

#Lasso
library(dplyr)
library(broom)

data_matrix <- as.matrix(data[,1:54])
data_lasso <- glmnet(
  x = data_matrix,
  y=data[,55],
  alpha=1
  )
plot(data_lasso, xvar="lambda", label=TRUE)

data_cv_lasso <- cv.glmnet(x = data_matrix, y=data[,55], alpha=1 )
plot(data_cv_lasso)

min(data_cv_lasso$cvm)
data_cv_lasso$lambda.min
data_cv_lasso$cvm[data_cv_lasso$lambda == data_cv_lasso$lambda.1se]
data_cv_lasso$lambda.1se

data_lasso_min <- glmnet(
  x = data_matrix,
  y=data[,55],
  alpha = 1
)

plot(data_lasso_min, xvar = "lambda")
abline(v = log(data_cv_lasso$lambda.min), col = "red", lty = "dashed")
abline(v = log(data_cv_lasso$lambda.1se), col = "red", lty = "dashed")

coef(data_cv_lasso, s = "lambda.1se") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)


#MOdels
library(caret)
require(caTools)
library(mlbench)
set.seed(101) 
#feature_variance <- caret::nearZeroVar(data, saveMetrics = TRUE)

#Data pre-porcessing V2
X <- data[, -29]
X <- X[, -21]

sample = sample.split(X$Cover_Type, SplitRatio = .75)
X_train = subset(X, sample == TRUE)
X_val  = subset(X, sample == FALSE)

table(X_train$Cover_Type)
table(X_val$Cover_Type)

y_train = make.names(as.character(X_train$Cover_Type))
X_train = X_train[, -53]

y_val = make.names(as.character(X_val$Cover_Type))
X_val = X_val[, -53]
y_val = as.factor(y_val)

# Enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

# Setup train control
train_control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs= TRUE, summaryFunction = multiClassSummary)
metric <- "logLoss"

# Setup text file logging
my_log = file("train_log_n10_r3_center_scale_v1.txt")
sink(my_log, append=TRUE, type="output")
sink(my_log, append=TRUE, type="message")

# Naive Bayes
start_time = Sys.time()
m_nb <- train(X_train, y_train, method="naive_bayes", metric=metric, preProcess=c("center", "scale"), 
                trControl=train_control )
end_time = Sys.time()
print("Elapsed training time:")
print(end_time - start_time)
print(m_nb)
pred_nb <- predict(m_nb, newdata=X_val)
end_time = Sys.time()
print("Elapsed predict time:")
print(end_time - start_time)
confusionMatrix(data=pred_nb, reference=y_val)

# K Nearest Neighbours
start_time = Sys.time()
m_kknn <- train(X_train, y_train, method="kknn", metric=metric, preProcess=c("center", "scale"), 
              trControl=train_control )
end_time = Sys.time()
print("Elapsed training time:")
print(end_time - start_time)
print(m_kknn)
start_time = Sys.time()
pred_kknn = predict(m_kknn, newdata=X_val)
end_time = Sys.time()
print("Elapsed predict time:")
print(end_time - start_time)
confusionMatrix(data=pred_kknn, reference=y_val)

# C5.0 Tree
start_time = Sys.time()
m_c50 <- train(X_train, y_train, method="C5.0Tree", metric=metric, preProcess=c("center", "scale"), 
              trControl=train_control )
end_time = Sys.time()
print("Elapsed training time:")
print(end_time - start_time)
print(m_c50)
start_time = Sys.time()
pred_c50 = predict(m_c50, newdata=X_val)
end_time = Sys.time()
print("Elapsed predict time:")
print(end_time - start_time)
confusionMatrix(data=pred_c50, reference=y_val)

# eXtreme Gradient Boosting
start_time = Sys.time()
m_xgbtree <- train(X_train, y_train, method="xgbTree", metric=metric, preProcess=c("center", "scale"), 
               trControl=train_control)
end_time = Sys.time()
print("Elapsed training time:")
print(end_time - start_time)
print(m_xgbtree)
start_time = Sys.time()
pred_xgbtree = predict(m_xgbtree, newdata=X_val)
end_time = Sys.time()
print("Elapsed predict time:")
print(end_time - start_time)
confusionMatrix(data=pred_xgbtree, reference=y_val)

# close file logger
closeAllConnections()
# stop parallel processing cluster
stopCluster(cl)

