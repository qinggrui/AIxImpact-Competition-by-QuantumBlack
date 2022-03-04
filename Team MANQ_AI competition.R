library(data.table)
library(ggplot2)
library(nnet)
library(caTools)
library(class)
library(dplyr)
library(rpart)
library(rpart.plot)
library(pdp)
library(vip)
library(reshape)
library(tidyquant)
library(zoo)
library(Metrics)
library(e1071)
#string and text mining packages
library(stringr)
library(quanteda)
library(MLmetrics)
library(DBI)
library(arules)

DF =  fread("C:/Users/Qing Rui/Desktop/AI Impact competition/US_E-commerce_records_2020.csv", stringsAsFactors = F)
summary(DF)
View(DF)
DF$Margin = DF$Sales-DF$Profit
View(DF)
DF$Region=factor(DF$Region)
DF$Category=factor(DF$Category)
summary(DF)

#create column for profit margin and analyse profit, discount, qty, sales and margin using svm
#justify svm data for best fit

set.seed(2022)
train <- sample.split(Y = DF$Sales, SplitRatio = 0.7)
trainset <- subset(DF, train == T)
trainset
testset <- subset(DF, train == F)
testset

library(caret)

str(train)
set.seed(2022)
intrain <- createDataPartition(y = DF$Sales, p= 0.7, list = FALSE)
training <- DF[intrain,]
testing <- DF[-intrain,]

summary(testing)
dim(training) #check dimensions of training dataset
dim(testing) #check dimensions of testing dataset

anyNA(training) #check for NA values

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3) #re-sampling and cross-validation of training dataset

#SVM for East Region
set.seed(2022)
svm_LinearE <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Region=="East",],
                    method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
print(svm_LinearE)
varImp(svm_LinearE, scale = FALSE) #Shows variable importance 

summary(test_predE)
test_predE <- predict(svm_LinearE, newdata = testing[Region=="East",])
class(test_predE)
test1 = cbind(test_predC, test_predE, test_predS, test_predW)
test2 = t(test1)
colnames(test2)[0] = c('Region')
library(EnvStats)
plot(test2, lwd = 2, col = "red",
     main = "Multiple curves", xlab = "")

epdfPlot(test_predE, epdf.col = "red", main = "Optimal Sales target for East Region",
         xlab = 'East Region Predicted Sales', ylab='Optimal Sales Freq Value')

#SVM for West Region
set.seed(2022)
svm_LinearW <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Region=="West",],
                    method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
print(svm_LinearW)
varImp(svm_LinearW, scale = FALSE) #Shows variable importance 


test_predW <- predict(svm_LinearW, newdata = testing[Region=="West",])
class(test_predW)
summary(test_predW)
epdfPlot(test_predW, epdf.col = "blue", main = "Optimal Sales target for West Region",
         xlab = 'West Region Predicted Sales', ylab='Optimal Sales Freq Value') 

#SVM for Central Region
set.seed(2022)
svm_LinearC <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Region=="Central",],
                    method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
print(svm_LinearC)
varImp(svm_LinearC, scale = FALSE) #Shows variable importance 


test_predC <- predict(svm_LinearC, newdata = testing[Region=="Central",])
class(test_predC)
summary(test_predC)
epdfPlot(test_predC, epdf.col = "orange",main = "Optimal Sales target for Central Region",
         xlab = 'Central Region Predicted Sales', ylab='Optimal Sales Freq Value') 

#SVM for South Region
set.seed(2022)
svm_LinearS <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Region=="South",],
                     method = "svmLinear",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
print(svm_LinearS)
varImp(svm_LinearS, scale = FALSE) #Shows variable importance 


test_predS <- predict(svm_LinearS, newdata = testing[Region=="South",])
class(test_predS)
summary(test_predS)
epdfPlot(test_predS, epdf.col = "green", main = "Optimal Sales target for South Region",
         xlab = 'South Region Predicted Sales', ylab='Optimal Sales Freq Value') 





#SVM for Technology Category
set.seed(2022)
svm_LinearT <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Category=="Technology",],
                     method = "svmLinear",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
print(svm_LinearT)
varImp(svm_LinearT, scale = FALSE) #Shows variable importance 


test_predT <- predict(svm_LinearT, newdata = testing[Category=="Technology",])
class(test_predT)
summary(test_predT)
epdfPlot(test_predT, epdf.col = "purple", main = "Optimal Sales target for Technology Category",
         xlab = 'Technology Category Predicted Sales (USD$)', ylab='Optimal Sales Freq Value') 



#SVM for Furniture Category
set.seed(2022)
svm_LinearF <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Category=="Furniture",],
                     method = "svmLinear",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
print(svm_LinearF)
varImp(svm_LinearF, scale = FALSE) #Shows variable importance 


test_predF <- predict(svm_LinearF, newdata = testing[Category=="Furniture",])
class(test_predF)
summary(test_predF)
epdfPlot(test_predF, epdf.col = "green", main = "Optimal Sales target for Furniture Category",
         xlab = 'Furniture Category Predicted Sales (USD$)', ylab='Optimal Sales Freq Value')


#SVM for Office supplies Category
set.seed(2022)
svm_LinearOS <- train(Sales ~ Quantity+ Profit+Margin+ Discount, data = training[Category=="Office Supplies",],
                     method = "svmLinear",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
print(svm_LinearOS)
varImp(svm_LinearOS, scale = FALSE) #Shows variable importance 


test_predOS <- predict(svm_LinearOS, newdata = testing[Category=="Office Supplies",])
class(test_predOS)
summary(test_predOS)
epdfPlot(test_predOS, epdf.col = "pink", main = "Optimal Sales target for Office Supplies Category",
         xlab = 'Office Supplies Category Predicted Sales (USD$)', ylab='Optimal Sales Freq Value')

##Visualisation for data comparison
summary(DF[Category=='Office Supplies'])
summary(DF[Category=='Technology'])
summary(DF[Category=='Furniture'])
Office_Supplies= c(122.926, 3.834, 0.16,19.849, 103.077)
Variables= c('Sales', 'Quantity', 'Discount', 'Profit', 'Margin')
Technology = c(453.47, 3.787, 0.1303, 81.225, 354.241)
Furniture = c(313.976, 3.552, 0.1699, 4.4, 309.576)
ab3= data.frame(Variables, Office_Supplies, Furniture, Technology)

Category = c('Office Supplies', 'Furniture', 'Technology')
Sales=c(122.926, 313.976, 453.47)
Quantity = c(3.834, 3.552, 3.787)
Discount = c(0.16, 0.1699, 0.1303)
Profit = c(19.849, 4.4, 81.225)
Margin = c(103.077, 309.576, 354.241)
ab3 = data.frame(Category, Sales, Quantity, Discount, Profit, Margin)

Variables= c('Sales', 'Quantity', 'Discount', 'Profit', 'Margin','Sales', 'Quantity', 'Discount', 'Profit', 'Margin'
             ,'Sales', 'Quantity', 'Discount', 'Profit', 'Margin')
Category = c('Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 
             'Furniture','Furniture','Furniture','Furniture','Furniture',
             'Technology','Technology','Technology','Technology','Technology')
Values = c(122.926, 3.834, 0.16,19.849, 103.077, 313.976, 3.552, 0.1699, 4.4, 309.576, 453.47, 3.787, 0.1303, 81.225, 354.241)
ab3= data.frame(Variables, Category, Values)
ggplot(ab3, aes(clarity, fill = cut)) + 
  geom_bar(position = 'identity', alpha = 0.3)

ggplot(ab3, aes(x = Category, y = Values, fill = Variables)) +
  geom_col() +
  guides(fill = guide_legend(reverse = TRUE))
ggplot(data = ab3, aes(x = Category, y = Values, fill = Variables)) + 
  geom_col(position = 'stack', identity = 0.3)


summary(DF[Region=='Central'])
summary(DF[Region=='East'])
summary(DF[Region=='South'])
summary(DF[Region=='West'])
Variables= c('Sales', 'Quantity', 'Discount', 'Profit', 'Margin','Sales', 'Quantity', 'Discount', 'Profit', 'Margin'
             ,'Sales', 'Quantity', 'Discount', 'Profit', 'Margin','Sales', 'Quantity', 'Discount', 'Profit', 'Margin')
Regions = c('Central','Central','Central','Central','Central',
             'East','East','East','East','East',
             'South','South','South','South','South',
             'West','West','West','West','West')
Values = c(189.072, 3.702, 0.2396, 9.706, 179.367, 231.360, 3.704, 0.1478, 36.081, 195.279,
           237.270, 3.697, 0.1554, 17.083, 220.187, 228.43, 3.9, 0.1052, 40.008, 188.42)
ab4= data.frame(Variables, Regions, Values)
ggplot(data = ab4, aes(x = Regions, y = Values, fill = Variables)) + 
  geom_col(position = 'stack')


###EXTRA STUFF###
test_pred=matrix(test_pred)
sm.density(test_pred)
test_pred


summary(test_pred)
plot(test_pred, main='SVM', ylab='Predicted Sales', xlab = 'Data index')
class(test_pred)
library(sm)
sm.density(test_pred)


test_pred=data.frame(test_pred)
ggplot(test_pred, aes(x=test_pred)) + 
  geom_density()
cm = table(test_pred[, 3], test_pred)
cm

testset3.error <- test_pred - testing$ESGScore
testset3.error

############################################# Market Basket Analysis ###########################################################
install.packages("RJDBC")
install.packages("RODBC")
install.packages("arules")


library(data.table)
library(DBI)
library(dplyr)
library(dbplyr)
library(odbc)
require(RJDBC)
require(RODBC)
library(arules)

df1=fread("C:/Users/Qing Rui/Desktop/AI Impact competition/US_E-commerce_records_2020.csv", stringsAsFactors = F)
df1$`Product ID` <- factor(df1$`Product ID`)
df1$`Order ID` <- factor(df1$`Order ID`)
class(df1$`Product ID`)
class(df1$`Order ID`)

AggPosData <- split(df1$`Product ID`,df1$`Order ID`)
View(AggPosData)

Txns <- as(AggPosData,'transactions')
Txns

## Support: How frequently an itemset occurs in the transaction as a percentage of all transactions
## Confidence: Ratio of the number of transactions that include items A and B to the number of transactions that include items in A
## how often items in B appear in transactions that contain A only. It is a conditional probability.
## Lift: ratio of confidence to expected confidence | rate of confidence that B will be purchased given that A was purchased


Rules <- apriori(Txns,parameter = list(supp=0, conf=0.5,minlen=2))

AprioriFile <- DATAFRAME((Rules))
print(AprioriFile)


