library(caret)

loan_train<-read.csv("train_u6lujuX_CVtuZ9i.csv",stringsAsFactors = T)
str(loan_train)
sum(is.na(loan_train))
preprovalues <- preProcess(loan_train,
                           method=c("knnImpute",
                                    "center",
                                    "scale"))

library('RANN')
train_processed <- predict(preprovalues,
                           loan_train)


sum(is.na(train_processed))

train_processed$Loan_Status <- ifelse(loan_train$Loan_Status=='N',0,1)

train_processed$Loan_ID <- NULL

dmy <- dummyVars("~.",train_processed,fullRank = TRUE)

sapply(loan_train, class)

trained_transformed <- data.frame(predict(dmy,train_processed))
trained_transformed$Loan_Status <- as.factor(trained_transformed$Loan_Status)

intrain <- createDataPartition(trained_transformed$Loan_Status,
                               p=0.70,
                               list = FALSE)
trainData <- trained_transformed[intrain,]
testData <- trained_transformed[-intrain,]


control <- rfeControl(functions=rfFuncs,method="repeatedcv",repeats = 3,verbose=FALSE)

loan_pred_profile3 <- rfe(Loan_Status ~.,data=trainData,rfeControl=control)


trainData<- trainData[c("Credit_History","LoanAmount","ApplicantIncome","Loan_Amount_Term","CoapplicantIncome","Loan_Status")]

fitcontrol <- trainControl(method = "repeatedcv",number = 5,repeats = 3,savePredictions = 'final',verboseIter = FALSE)
model_rf <- train(Loan_Status~.,data=trainData,
                  method='rf',trControl=fitcontrol,
                  tuneLength=10)




testData$pred<-predict(model_rf,testData)

confusionMatrix(testData$Loan_Status,testData$pred)


model_knn <- train(Loan_Status~., data=trainData,method="knn",trControl=fitcontrol,tuneLength=10)
testData$pred_knn<- predict(model_knn,testData)
confusionMatrix(testData$Loan_Status,testData$pred_knn)


model_glm <- train(Loan_Status~., data=trainData,method="glm",trControl=fitcontrol,tuneLength=10)
testData$pred_glm<-predict(model_glm,testData)
confusionMatrix(testData$Loan_Status,testData$pred_glm)

testData$pred_rf.prob <-predict(model_rf,testData,type = "prob")
testData$pred_knn.prob<- predict(model_knn,testData,type="prob")
testData$pred_glm.prob <- predict(model_glm,testData,type="prob")



testData$pred.prob.all <- (testData$pred_rf.prob$`1`+testData$pred_knn.prob$`1`+testData$pred_glm.prob$`1`)/3

testData$pred.allavg.Loan.status <-as.factor(ifelse(testData$pred.prob.all > 0.5,'Y','N'))

testData$pred <- as.factor(ifelse(testData$pred==1,"Y","N"))
testData$pred_knn <- as.factor(ifelse(testData$pred_knn==1,"Y","N"))
testData$pred_glm <- as.factor(ifelse(testData$pred_glm==1,"Y","N"))



testData$pred_majority <- as.factor(ifelse(testData$pred=='Y' & testData$pred_knn=='Y','Y',
                                          ifelse(testData$pred=='Y' & testData$pred_glm=='Y','Y',
                                                 ifelse(testData$pred_knn=='Y' & testData$pred_glm=='Y','Y','N'))))



testData$pred_rf.prob <-predict(model_rf,testData,type = "prob")
testData$pred_knn.prob<- predict(model_knn,testData,type="prob")
testData$pred_glm.prob <- predict(model_glm,testData,type="prob")


testData$pred.weighted.avg <- (testData$pred_rf.prob$`1`*0.25)+(testData$pred_knn.prob$`1`*0.25)+(testData$pred_glm.prob$`1`*0.25)
testData$pred.weighted.avg.final <- as.factor(ifelse(testData$pred.weighted.avg > 0.5,'Y','N'))

fitControl <- trainControl(method="cv",number = 10,savePredictions = 'final',classProbs = TRUE)


model_knn <- train(Loan_Status~., data=trainData,method="knn",trControl=fitcontrol,tuneLength=10)
model_glm <- train(Loan_Status~., data=trainData,method="glm",trControl=fitcontrol,tuneLength=10)
model_rf <- train(Loan_Status~.,data=trainData,
                  method='rf',trControl=fitcontrol,
                  tuneLength=10)

testData$OOF_pred_rf <- predict(model_rf,testData,type="prob")  
testData$OOF_pred_glm <- predict(model_glm,testData,type="prob")
testData$OOF_pred_knn <- predict(model_knn,testData,type="prob")

#Predictors for top layer models 
predictors_top<-c('OOF_pred_rf','OOF_pred_knn','OOF_pred_glm') 

#GBM as top layer model 
model_gbm<- 
  train(trainData,trainData$Loan_Status,method='gbm',trControl=fitcontrol,tuneLength=3)


model_glm<-
  train(trainData,trainData$Loan_Status,method='glm',trControl=fitcontrol,tuneLength=3)


testDatam<- testData[c("Credit_History","LoanAmount","ApplicantIncome","Loan_Amount_Term","CoapplicantIncome","Loan_Status")]

#predict using GBM top layer model
testData$gbm_stacked<-predict(model_gbm,testDatam)

#predict using logictic regression top layer model
testDatam$glm_stacked<-predict(model_glm,testDatam)

