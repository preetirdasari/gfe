### Final Code ### 

### Grammatical Facial Expressions - Dasari, Preeti ###

## Required Packages ##

require(car)
require(caTools)
require(randomForest)
require(glmnet)
require(caret)
require(tidyverse)
require(dplyr)
require(class)
require(ggplot2)
library(e1071)
library(kernlab)
library(gridExtra)

## Importing Data ##

setwd("/Users/PreetiRDasari/Desktop/Baruch/Fall 2019/STA 9891 - ML/grammatical_facial_expression")

gfe <- read.csv("GFE_Data.csv")
gfe <- gfe[,3:303]

table(gfe$GFE) ## Imbalanced Data 


# Setting up the data

X             =    model.matrix(GFE~., gfe)[, -301]
y             =    factor(gfe$GFE)
n             =    dim(X)[1] 
p             =    dim(X)[2] 


# Standardizing

X <- scale(X)
gfe2 <- data.frame(X, as.factor(gfe$GFE))
gfe2 <- gfe2[,-1]
names(gfe2)[names(gfe2) == "as.factor.gfe.GFE."] <- "GFE"


### Part I: BALANCED, 0.5N ###

## Error Arrays

S = 100

Err.rf.1        =    matrix(0, nrow = S, ncol = 6) 
Err.svm.1       =    matrix(0, nrow = S, ncol = 7) 
Err.logistic.1  =    matrix(0, nrow = S, ncol = 6) 
Err.lasso.1     =    matrix(0, nrow = S, ncol = 7) 
Err.ridge.1     =    matrix(0, nrow = S, ncol = 7) 


for (s in 1:S) {
  
  # Splitting
  
  sample1 = sample.split(gfe2, SplitRatio = 1/2)
  
  train1 = subset(gfe2, sample1==TRUE)
  test1 = subset(gfe2, sample1==FALSE)
  
  
  train1$GFE <- as.factor(train1$GFE)
  test1$GFE <- as.factor(test1$GFE)
  
  x.train1 = model.matrix(GFE~., train1)[,-301]
  x.test1 = model.matrix(GFE~., test1)[,-301]
  
  y.train1 = train1 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test1 = test1 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Balancing Data 
  
  bal.train1 <- upSample(x.train1, y.train1, list=FALSE, yname = "GFE")
  bal.test1 <- upSample(x.test1, y.test1, list=FALSE, yname = "GFE")
  bal.train1 <- bal.train1[,-1]
  bal.test1 <- bal.test1[,-1]
  
  
  x.train1 <- model.matrix(GFE~., bal.train1)[,-300]
  x.test1 <-  model.matrix(GFE~., bal.test1)[,-300]
  
  y.train1 = bal.train1 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test1 = bal.test1 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Fitting Random Forest
  
  rf1 <- randomForest(GFE ~., data = bal.train1, ntrees = 300, mtry = sqrt(p))
  
  rf.train.hat1       =     predict(rf1, newdata = bal.train1, type = "class")
  rf.test.hat1        =     predict(rf1, newdata = bal.test1, type = "class")
  
  ## Random Forest train and test error rates 
  Err.rf.1[s,1]       =     mean(y.train1 != rf.train.hat1)
  Err.rf.1[s,2]       =     mean(y.test1 != rf.test.hat1)
  
  ## Random Forest false positive and negative train error
  Err.rf.1[s,3]  =  mean(1 == rf.train.hat1[y.train1==0]) # false positive
  Err.rf.1[s,4]  =  mean(0 == rf.train.hat1[y.train1==1]) # false negative
  
  ## Random Forest false positive and negative test error
  Err.rf.1[s,5]  =  mean(1 == rf.test.hat1[y.test1==0]) # false positive
  Err.rf.1[s,6]  =  mean(0 == rf.test.hat1[y.test1==1]) # false negative
  
  
  # Fitting Radial SVM
  
    trctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
    
    svm_cv1 <- train(GFE ~., data = bal.train1, method = "svmRadial",
                    trControl=trctrl1,
                    tuneLength = 10,
                    cost = 10^seq(-2,2,length.out = 5),
                    sigma = c(0.1, 0.5, 1, 2, 5, 10))
  
                    
  svm.train.hat1 <- predict(svm_cv1, newdata = bal.train1)
  svm.test.hat1 <- predict(svm_cv1, newdata = bal.test1)
  
  ## SVM train, test and CV error rates 
  Err.svm.1[s,1]       =     mean(y.train1 != svm.train.hat1)
  Err.svm.1[s,2]       =     mean(y.test1 != svm.test.hat1)
  svm_results1          =     data.frame(svm_cv1$results)
  Err.svm.1[s,3]       =    1 - max(svm_results1$Accuracy)
  
  
  ## SVM false positive and negative train error
  Err.svm.1[s,4]  =  mean(1 == svm.train.hat1[y.train1==0]) # false positive
  Err.svm.1[s,5]  =  mean(0 == svm.train.hat1[y.train1==1]) # false negative
  
  ## SVM false positive and negative test error
  Err.svm.1[s,6]  =  mean(1 == svm.test.hat1[y.test1==0]) # false positive
  Err.svm.1[s,7]  =  mean(0 == svm.test.hat1[y.test1==1]) # false negative
  
  # Fitting Logistic
  
  logistic_model1 <- glm(GFE ~., data = bal.train1, family = binomial(link = "logit"))
    
  log.train.prob1 <- predict(logistic_model1, newx = x.train1, type = "response")
  log.test.prob1 <- predict(logistic_model1, data = x.test1, type = "response")
  log.train.hat1 <- ifelse(log.train.prob1 > 0.5, 1, 0)
  log.test.hat1 <- ifelse(log.test.prob1 > 0.5, 1, 0)
  
  ## Logistic train and test error rates 
  Err.logistic.1[s,1] =     mean(y.train1 != log.train.hat1)
  Err.logistic.1[s,2] =     mean(y.test1 != log.test.hat1)
  
  
  ## Logistic false positive and negative train error
  Err.logistic.1[s,3]  =  mean(1 == log.train.hat1[y.train1==0]) # false positive
  Err.logistic.1[s,4]  =  mean(0 == log.train.hat1[y.train1==1]) # false negative
  
  ## Logistic false positive and negative test error
  Err.logistic.1[s,5]  =  mean(1 == log.test.hat1[y.test1==0]) # false positive
  Err.logistic.1[s,6]  =  mean(0 == log.test.hat1[y.test1==1]) # false negative
  
  # Fitting Logistic Lasso
  
  lasso1 <- cv.glmnet(x.train1, y.train1, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
  lasso_lamda1 = lasso1$lambda.min #optimal lambda for lasso
  lasso_model1 <- glmnet(x.train1, y.train1, lambda = lasso_lamda1, family="binomial")
  
  lasso.train.hat1 <- predict(lasso_model1, newx = x.train1, type = "class")
  lasso.test.hat1 <- predict(lasso_model1, newx = x.test1, type = "class")
  
  Err.lasso.1[s,1]    =     mean(y.train1 != lasso.train.hat1)
  Err.lasso.1[s,2]    =     mean(y.test1 != lasso.test.hat1)
  Err.lasso.1[s,3]    =     min(lasso1$cvm)
  
  ## Logistic false positive and negative train error
  Err.lasso.1[s,4]  =  mean(1 == lasso.train.hat1[y.train1==0]) # false positive
  Err.lasso.1[s,5]  =  mean(0 == lasso.train.hat1[y.train1==1]) # false negative
  
  # column 5 of Err = false positive test error 
  # column 6 of Err = false negative test error
  Err.lasso.1[s,6]  =  mean(1 == lasso.test.hat1[y.test1==0]) # false positive
  Err.lasso.1[s,7]  =  mean(0 == lasso.test.hat1[y.test1==1]) # false negative
  
  
  
  # Fitting Logistic Ridge 
  
  ridge1 <-  cv.glmnet(x.train1, y.train1, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
  ridge_lambda1 = ridge1$lambda.min
  ridge_model1 <-  glmnet(x.train1, y.train1, lambda = ridge_lambda1, family = "binomial")
  
  ridge.train.hat1 <-   predict(ridge_model1, newx = x.train1, type = "class")
  ridge.test.hat1  <-   predict(ridge_model1, newx = x.test1, type = "class")
  
  Err.ridge.1[s,1]    =     mean(y.train1 != ridge.train.hat1)
  Err.ridge.1[s,2]    =     mean(y.test1 != ridge.test.hat1)
  Err.ridge.1[s,3]    =     min(ridge1$cvm)
  
  
  ## Ridge false positive and negative train error
  Err.ridge.1[s,4]  =  mean(1 == ridge.train.hat1[y.train1==0]) # false positive
  Err.ridge.1[s,5]  =  mean(0 == ridge.train.hat1[y.train1==1]) # false negative
  
  # column 5 of Err = false positive test error 
  # column 6 of Err = false negative test error
  Err.ridge.1[s,6]  =  mean(1 == ridge.test.hat1[y.test1==0]) # false positive
  Err.ridge.1[s,7]  =  mean(0 == ridge.test.hat1[y.test1==1]) # false negative
  
}


## Error Box Plots 

err.train1     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                 rep("Ridge", S), rep("SVM", S)) , 
                               c(Err.lasso.1[,1], Err.logistic.1[,1], Err.rf.1[,1], Err.ridge.1[,1], Err.svm.1[,1]))
names(err.train1) <- c("Method", "Train.Err")   

err.test1     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                rep("Ridge", S), rep("SVM", S)) , 
                              c(Err.lasso.1[,2], Err.logistic.1[,2], Err.rf.1[,2], Err.ridge.1[,2], Err.svm.1[,2]))

names(err.test1) <- c("Method", "Test")   

err.cv1     =     data.frame(c(rep("Lasso", S), rep("Ridge", S), rep("SVM", S)) , 
                            c(Err.lasso1[,3], Err.ridge.1[,3], Err.svm.1[,3]))

names(err.cv1) <- c("Method", "CV.Err")   

# train false positive (fp)
err.train.fp1        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                       rep("Ridge", S), rep("SVM", S)) , 
                                     c(Err.lasso.1[,4], Err.logistic.1[,3], Err.rf.1[,3], Err.ridge.1[,4], Err.svm.1[,4]))

names(err.train.fp1) <- c("Method", "Train.fp")   


# train false negative (fn)
err.train.fn1        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                       rep("Ridge", S), rep("SVM", S)) , 
                                     c(Err.lasso.1[,5], Err.logistic.1[,4], Err.rf.1[,4], Err.ridge.1[,5], Err.svm.1[,5]))

names(err.train.fn1) <- c("Method", "Train.fn")   

# test false positive (fp)
err.test.fp1        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                      rep("Ridge", S), rep("SVM", S)) , 
                                    c(Err.lasso.1[,6], Err.logistic.1[,5], Err.rf.1[,5], Err.ridge.1[,6], Err.svm.1[,6]))

names(err.test.fp1) <- c("Method", "Test.fp")   

# train false negative (fn)
err.test.fn1        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                      rep("Ridge", S), rep("SVM", S)) , 
                                    c(Err.lasso.1[,7], Err.logistic.1[,6], Err.rf.1[,6], Err.ridge.1[,7], Err.svm.1[,7]))

names(err.test.fn1) <- c("Method", "Test.fn")   


p1 = ggplot(err.train1)   +     aes(x=Method, y = Train.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("Training Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))


p2 = ggplot(err.test1)   +     aes(x=Method, y = Test, fill=Method) +   geom_boxplot()  +
  ggtitle("Testing Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))


p3 = ggplot(err.cv1)   +     aes(x=Method, y = CV.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("CV Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))


p4 = ggplot(err.train.fp1)   +     aes(x=Method, y = Train.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("Train fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.4)  


p5 = ggplot(err.train.fn1)   +     aes(x=Method, y = Train.fn, fill=Method) +   geom_boxplot()  +  
  ggtitle("Train fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.4)  

p6 = ggplot(err.test.fp1)   +     aes(x=Method, y = Test.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("Test fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.4)  

p7 = ggplot(err.test.fn1)   +     aes(x=Method, y = Test.fn, fill=Method) +   geom_boxplot()  + 
  ggtitle("Test fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.4)  


grid.arrange(p1, p4, p5, p2, p6, p7, ncol=2)

p3

### Fitting Single Models - 0.5N ###


# Random Forest 


rf_time1 <- system.time({ 
rf_s1 <- randomForest(GFE ~., data = bal.train1, ntrees = 300, mtry = sqrt(p))

})

rf_time1 ["elapsed"]

## SVM CV

svm_time1 <- system.time({ 
  tune.svm1  =   tune(svm, GFE~., data=bal.train1, kernel = "radial",
                     ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                   gamma = c(0.1, 0.5, 1, 2, 5, 10)))
  svm.fit1 = tune.svm1$best.model
})

svm_time1["elapsed"] 

## SVM fit

svm_time2 <- system.time({ 
svm.fit2 = svm(x.train1, y.train1, kernel = "radial")
})

svm_time2["elapsed"] 

## Lasso CV

lasso_time1 <- system.time({ 
  m = 25
  lasso_s1 <- cv.glmnet(x.train1, y.train1, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
  lasso_lamda.1 = lasso_s1$lambda.min #optimal lambda for lasso
  lasso_model.1 <- glmnet(x.train1, y.train1, lambda = lasso_s1$lambda, alpha = 1, family="binomial")
})

lasso_time1["elapsed"]

## Lasso Single

lasso_time2 <- system.time({ 
  lasso_s2 <- glmnet(x.train1, y.train1, alpha = 1, family="binomial")
})

lasso_time2["elapsed"]

## Ridge CV

ridge_time1 <- system.time({
  m = 25
  ridge_s1 <-  cv.glmnet(x.train1, y.train1, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
  r_lambda.1 = ridge_s1$lambda.min
  ridge_model.1 <-  glmnet(x.train1, y.train1, lambda = ridge_s1$lambda, alpha = 0, family = "binomial")
})
ridge_time1["elapsed"]

## Ridge Single

ridge_time2 <- system.time({
  ridge_s2 <-  glmnet(x.train1, y.train1, alpha = 0, family = "binomial")
})
ridge_time2 ["elapsed"]


## 10 fold CV Curves


plot(tune.svm1)

n.lambdas     =    dim(lasso_model.1$beta)[2]

lasso.ratio1    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  lasso.ratio1[i] <- sum(abs(lasso_model.1$beta[,i]))/sum(abs(lasso_s2$beta))
}


n.lambdas1   =    dim(ridge_model.1$beta)[2]

ridge.ratio1    =    rep(0, n.lambdas1)
for (i in 1:n.lambdas1) {
  ridge.ratio1[i] <- sum(abs(ridge_model.1$beta[,i]))/sum(abs(ridge_s2$beta))
}

lasso.cvm <- lasso_s1$cvm
lasso.sd <- lasso_s1$cvsd
ridge.cvm <- ridge_s1$cvm
ridge.sd <- ridge_s1$cvsd


cv.curves.1 <- data.frame(c(rep("lasso", length(lasso.ratio1)),  rep("ridge", length(ridge.ratio1))), 
                          c(lasso.ratio1, ridge.ratio1),
                          c(lasso.cvm, ridge.cvm),
                          c(lasso.sd, ridge.sd))


colnames(cv.curves.1) =     c("method", "ratio", "cv", "sd")


cv.plot1      =     ggplot(cv.curves.1, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
cv.plot1      =     cv.plot1  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
cv.plot1      =     cv.plot1  + theme(legend.text = element_text(colour="black", size=14, face="bold", family = "Courier")) 
cv.plot1      =     cv.plot1  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
cv.plot1      =     cv.plot1  + theme(legend.title=element_blank()) 
cv.plot1      =     cv.plot1  + scale_color_discrete(breaks=c("lasso", "ridge"))
cv.plot1      =     cv.plot1  + theme(axis.title.x = element_text(size=24),
                                        axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                        axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
cv.plot1      =     cv.plot1  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
cv.plot1      =     cv.plot1  + ggtitle("10-fold Curves for 0.5n")
cv.plot1



## Part II: BALANCED, 0.9N

Err.rf.2        =    matrix(0, nrow = S, ncol = 6) 
Err.svm.2       =    matrix(0, nrow = S, ncol = 7) 
Err.logistic.2  =    matrix(0, nrow = S, ncol = 6) 
Err.lasso.2     =    matrix(0, nrow = S, ncol = 7) 
Err.ridge.2     =    matrix(0, nrow = S, ncol = 7) 

for (s in 1:S) {
  
  # Splitting
  
  sample2 = sample.split(gfe2, SplitRatio = 9/10)
  
  train.2 = subset(gfe2, sample2==TRUE)
  test.2 = subset(gfe2, sample2==FALSE)
  
  
  train.2$GFE <- as.factor(train.2$GFE)
  test.2$GFE <- as.factor(test.2$GFE)
  
  x.train.2 = model.matrix(GFE~., train.2)[,-301]
  x.test.2 = model.matrix(GFE~., test.2)[,-301]
  
  y.train.2 = train.2 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test.2 = test.2 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Dealing with Imbalanced data
  
  bal.train.2 <- upSample(x.train.2, y.train.2, list=FALSE, yname = "GFE")
  
  bal.test.2 <- upSample(x.test.2, y.test.2, list=FALSE, yname = "GFE")
  bal.train.2 <- bal.train.2[,-1]
  bal.test.2 <- bal.test.2[,-1]
  
  
  x.train.2 <- model.matrix(GFE~., bal.train.2)[,-300]
  x.test.2 <-  model.matrix(GFE~., bal.test.2)[,-300]
  
  y.train.2 = bal.train.2 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test.2 = bal.test.2 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Fitting Random Forest
  
  rf2 <- randomForest(GFE ~., data = bal.train.2, ntrees = 300, mtry = sqrt(p))
  
  
  rf.train.2.hat       =     predict(rf2, newdata = bal.train.2, type = "class")
  rf.test.2.hat        =     predict(rf2, newdata = bal.test.2, type = "class")
  
  Err.rf.2[s,1]       =     mean(y.train.2 != rf.train.2.hat)
  Err.rf.2[s,2]       =     mean(y.test.2 != rf.test.2.hat)
  
  ## Random Forest false positive and negative train.2 error
  Err.rf.2[s,3]  =  mean(1 == rf.train.2.hat[y.train.2==0]) # false positive
  Err.rf.2[s,4]  =  mean(0 == rf.train.2.hat[y.train.2==1]) # false negative
  
  # column 5 of Err = false positive test.2 error 
  # column 6 of Err = false negative test.2 error
  Err.rf.2[s,5]  =  mean(1 == rf.test.2.hat[y.test.2==0]) # false positive
  Err.rf.2[s,6]  =  mean(0 == rf.test.2.hat[y.test.2==1]) # false negative
  
  
  
  # Fitting Radial SVM
  
  
  trctrl2 <- train.2Control(method = "repeatedcv", number = 10, repeats = 3)
  
  svm_cv2 <- train.2(GFE ~., data = bal.train.2, method = "svmRadial",
                     trControl=trctrl,
                     tuneLength = 10,
                     cost = 10^seq(-2,2,length.out = 5),
                     sigma = c(0.1, 0.5, 1, 2, 5, 10)
  )
  
  
  svm.train.2.hat <- predict(svm_cv2, newdata = bal.train.2)
  svm.test.2.hat <- predict(svm_cv2, newdata = bal.test.2)
  
  Err.svm.2[s,1]       =     mean(y.train.2 != svm.train.2.hat)
  Err.svm.2[s,2]       =     mean(y.test.2 != svm.test.2.hat)
  svm_results2        =     data.frame(svm_cv$results)
  Err.svm.2[s,3]       =    1 - max(svm_results2$Accuracy)
  
  ## SVM false positive and negative train.2 error
  Err.svm.2[s,4]  =  mean(1 == svm.train.2.hat[y.train.2==0]) # false positive
  Err.svm.2[s,5]  =  mean(0 == svm.train.2.hat[y.train.2==1]) # false negative
  
  # column 5 of Err = false positive test.2 error 
  # column 6 of Err = false negative test.2 error
  Err.svm.2[s,6]  =  mean(1 == svm.test.2.hat[y.test.2==0]) # false positive
  Err.svm.2[s,7]  =  mean(0 == svm.test.2.hat[y.test.2==1]) # false negative
  
  
  
  # Fitting Logistic
  
  logistic_model2 <- glm(GFE ~., data = bal.train.2, family = binomial(link = "logit"))
  
  log.train.2.prob <- predict(logistic_model2, newx = x.train.2, type = "response")
  log.test.2.prob <- predict(logistic_model2, data = x.test.2, type = "response")
  log.train.2.hat <- ifelse(log.train.2.prob > 0.5, 1, 0)
  log.test.2.hat <- ifelse(log.test.2.prob > 0.5, 1, 0)
  
  Err.logistic.2[s,1] =     mean(y.train.2 != log.train.2.hat)
  Err.logistic.2[s,2] =     mean(y.test.2 != log.test.2.hat)
  
  ## Logistic false positive and negative train.2 error
  Err.logistic.2[s,3]  =  mean(1 == log.train.2.hat[y.train.2==0]) # false positive
  Err.logistic.2[s,4]  =  mean(0 == log.train.2.hat[y.train.2==1]) # false negative
  
  # column 5 of Err = false positive test.2 error 
  # column 6 of Err = false negative test.2 error
  Err.logistic.2[s,5]  =  mean(1 == log.test.2.hat[y.test.2==0]) # false positive
  Err.logistic.2[s,6]  =  mean(0 == log.test.2.hat[y.test.2==1]) # false negative
  
  
  # Fitting Logistic Lasso
  
  m = 25
  lasso2 <- cv.glmnet(x.train.2, y.train.2, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
  lasso_lamda2 = lasso2$lambda.min #optimal lambda for lasso
  lasso_model2 <- glmnet(x.train.2, y.train.2, lambda = lasso_lamda2, family="binomial")
  lasso.train.2.hat <- predict(lasso_model2, newx = x.train.2, type = "class")
  lasso.test.2.hat <- predict(lasso_model2, newx = x.test.2, type = "class")
  
  Err.lasso.2[s,1]    =     mean(y.train.2 != lasso.train.2.hat)
  Err.lasso.2[s,2]    =     mean(y.test.2 != lasso.test.2.hat)
  Err.lasso.2[s,3]    =     min(lasso2$cvm)
  
  
  ## Logistic false positive and negative train.2 error
  Err.lasso.2[s,4]  =  mean(1 == lasso.train.2.hat[y.train.2==0]) # false positive
  Err.lasso.2[s,5]  =  mean(0 == lasso.train.2.hat[y.train.2==1]) # false negative
  
  # column 5 of Err = false positive test.2 error 
  # column 6 of Err = false negative test.2 error
  Err.lasso.2[s,6]  =  mean(1 == lasso.test.2.hat[y.test.2==0]) # false positive
  Err.lasso.2[s,7]  =  mean(0 == lasso.test.2.hat[y.test.2==1]) # false negative
  
  
  
  # Fitting Logistic Ridge 
  
  m = 25
  
  ridge2 <-  cv.glmnet(x.train.2, y.train.2, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
  ridge_lambda2 = ridge2$lambda.min
  ridge_model2 <-  glmnet(x.train.2, y.train.2, lambda = ridge_lambda2, family = "binomial")
  ridge.train.2.hat <-   predict(ridge_model2, newx = x.train.2, type = "class")
  ridge.test.2.hat  <-   predict(ridge_model2, newx = x.test.2, type = "class")
  
  Err.ridge.2[s,1]    =     mean(y.train.2 != ridge.train.2.hat)
  Err.ridge.2[s,2]    =     mean(y.test.2 != ridge.test.2.hat)
  Err.ridge.2[s,3]    =     min(ridge2$cvm)
  
  
  ## Ridge false positive and negative train.2 error
  Err.ridge.2[s,4]  =  mean(1 == ridge.train.2.hat[y.train.2==0]) # false positive
  Err.ridge.2[s,5]  =  mean(0 == ridge.train.2.hat[y.train.2==1]) # false negative
  
  ## Ridge false positive and negative test.2 error
  Err.ridge.2[s,6]  =  mean(1 == ridge.test.2.hat[y.test.2==0]) # false positive
  Err.ridge.2[s,7]  =  mean(0 == ridge.test.2.hat[y.test.2==1]) # false negative
  
}

## Error Box Plots


err.train.2     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                   rep("Ridge", S), rep("SVM", S)) , 
                                 c(Err.lasso.2[,1], Err.logistic.2[,1], Err.rf.2[,1], Err.ridge.2[,1], Err.svm.2[,1]))
names(err.train.2) <- c("Method", "train.2.Err")   

err.test.2     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                  rep("Ridge", S), rep("SVM", S)) , 
                                c(Err.lasso.2[,2], Err.logistic.2[,2], Err.rf.2[,2], Err.ridge.2[,2], Err.svm.2[,2]))

names(err.test.2) <- c("Method", "test.2.Err")   

err.cv     =     data.frame(c(rep("Lasso", S), rep("Ridge", S), rep("SVM", S)) , 
                            c(Err.lasso.2[,3], Err.rf.2[,3], Err.svm.2[,3]))

names(err.cv) <- c("Method", "CV.Err")   

# train.2 false positive (fp)
err.train.2.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.2[,4], Err.logistic.2[,3], Err.rf.2[,3], Err.ridge.2[,4], Err.svm.2[,4]))

names(err.train.2.fp) <- c("Method", "train.2.fp")   


# train.2 false negative (fn)
err.train.2.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.2[,5], Err.logistic.2[,4], Err.rf.2[,4], Err.ridge.2[,5], Err.svm.2[,5]))

names(err.train.2.fn) <- c("Method", "train.2.fn")   

# test.2 false positive (fp)
err.test.2.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.2[,6], Err.logistic.2[,5], Err.rf.2[,5], Err.ridge.2[,6], Err.svm.2[,6]))

names(err.test.2.fp) <- c("Method", "test.2.fp")   

# train.2 false negative (fn)
err.test.2.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.2[,7], Err.logistic.2[,6], Err.rf.2[,6], Err.ridge.2[,7], Err.svm.2[,7]))
names(err.test.2.fn) <- c("Method", "test.2.fn") 

p1_0.9n = ggplot(err.train.2)   +     aes(x=Method, y = train.2.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("train.2ing Error Rates for GFE (0.9n)") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))


p2_0.9n = ggplot(err.test.2)   +     aes(x=Method, y = test.2.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("test.2ing Error Rates for GFE (0.9n)") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))


p3_0.9n = ggplot(err.cv)   +     aes(x=Method, y = CV.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("CV Error Rates for GFE (0.9n)") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))

p4_0.9n = ggplot(err.train.2.fp)   +     aes(x=Method, y = train.2.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.2 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.6)  

p5_0.9n = ggplot(err.train.2.fn)   +     aes(x=Method, y = train.2.fn, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.2 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

p6_0.9n = ggplot(err.test.2.fp)   +     aes(x=Method, y = test.2.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("test.2 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

p7_0.9n = ggplot(err.test.2.fn)   +     aes(x=Method, y = test.2.fn, fill=Method) +   geom_boxplot()  + 
  ggtitle("test.2 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

grid.arrange(p1_0.9n, p4_0.9n, p5_0.9n, p2_0.9n, p6_0.9n, p7_0.9n, ncol=2)

p3_0.9n

### Fitting Single Models - 0.9N ###


# Random Forest 


rf_time2 <- system.time({ 
  rf_s2 <- randomForest(GFE ~., data = bal.train.2, ntrees = 300, mtry = sqrt(p))
  
})

rf_time2 ["elapsed"]

## SVM CV

svm_time3 <- system.time({ 
  tune.svm2  =   tune(svm, GFE~., data=bal.train.2, kernel = "radial",
                      ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                    gamma = c(0.1, 0.5, 1, 2, 5, 10)))
  svm.fit3 = tune.svm2$best.model
})

svm_time3["elapsed"] 

## SVM fit

svm_time4 <- system.time({ 
  svm.fit4 = svm(x.train.2, y.train.2, kernel = "radial")
})

svm_time4["elapsed"] 

## Lasso CV

lasso_time3 <- system.time({ 
  m = 25
  lasso_s3 <- cv.glmnet(x.train.2, y.train.2, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
  lasso_lamda.2 = lasso_s3$lambda.min #optimal lambda for lasso
  lasso_model.2 <- glmnet(x.train.2, y.train.2, lambda = lasso_s3$lambda, alpha = 1, family="binomial")
})

lasso_time3["elapsed"]

## Lasso Single

lasso_time4 <- system.time({ 
  lasso_s4 <- glmnet(x.train.2, y.train.2, alpha = 1, family="binomial")
})

lasso_time4["elapsed"]

## Ridge CV

ridge_time3 <- system.time({
  m = 25
  ridge_s3 <-  cv.glmnet(x.train.2, y.train.2, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
  r_lambda.2 = ridge_s3$lambda.min
  ridge_model.2 <-  glmnet(x.train.2, y.train.2, lambda = ridge_s3$lambda, alpha = 0, family = "binomial")
  })
ridge_time3["elapsed"]

## Ridge Single

ridge_time4 <- system.time({
  ridge_s4 <-  glmnet(x.train.2, y.train.2, alpha = 0, family = "binomial")
})
ridge_time4 ["elapsed"]


## 10 fold CV Curves


plot(tune.svm2)


n.lambdas     =    dim(lasso_model.2$beta)[2]

lasso.ratio2    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  lasso.ratio2[i] <- sum(abs(lasso_model.2$beta[,i]))/sum(abs(lasso_s4$beta))
}


n.lambdas2   =    dim(ridge_model.2$beta)[2]
ridge.ratio2    =    rep(0, n.lambdas2)
for (i in 1:n.lambdas2) {
  ridge.ratio2[i] <- sum(abs(ridge_model.2$beta[,i]))/sum(abs(ridge_s4$beta))
}

lasso.cvm1 <- lasso_s3$cvm

lasso.sd1 <- lasso_s3$cvsd

ridge.cvm1 <- ridge_s3$cvm

ridge.sd1 <- ridge_s3$cvsd


cv.curves.2 <- data.frame(c(rep("lasso", length(lasso.ratio2)),  rep("ridge", length(ridge.ratio2))), 
                          c(lasso.ratio2, ridge.ratio2),
                          c(lasso.cvm1, ridge.cvm1),
                          c(lasso.sd1, ridge.sd1))


colnames(cv.curves.2) =     c("method", "ratio", "cv", "sd")


cv.plot2      =     ggplot(cv.curves.2, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
cv.plot2      =     cv.plot2  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
cv.plot2      =     cv.plot2  + theme(legend.text = element_text(colour="black", size=14, face="bold", family = "Courier")) 
cv.plot2      =     cv.plot2  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
cv.plot2      =     cv.plot2  + theme(legend.title=element_blank()) 
cv.plot2      =     cv.plot2  + scale_color_discrete(breaks=c("lasso", "ridge"))
cv.plot2      =     cv.plot2  + theme(axis.title.x = element_text(size=24),
                                      axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                      axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
cv.plot2      =     cv.plot2  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
cv.plot2      =     cv.plot2  + ggtitle("10-fold Curves for 0.9n")
cv.plot2


## PART III: Imbalanced, 0.5N

Err.rf.3        =    matrix(0, nrow = S, ncol = 6) 
Err.svm.3       =    matrix(0, nrow = S, ncol = 7) 
Err.logistic.3  =    matrix(0, nrow = S, ncol = 6) 
Err.lasso.3     =    matrix(0, nrow = S, ncol = 7) 
Err.ridge.3     =    matrix(0, nrow = S, ncol = 7) 


for (s in 1:S) {
  
  # Splitting
  
  sample3 = sample.split(gfe2, SplitRatio = 1/2)
  
  train.3 = subset(gfe2, sample3==TRUE)
  test.3 = subset(gfe2, sample3==FALSE)
  
  
  train.3$GFE <- as.factor(train.3$GFE)
  test.3$GFE <- as.factor(test.3$GFE)
  
  x.train.3 = model.matrix(GFE~., train.3)[,-301]
  x.test.3 = model.matrix(GFE~., test.3)[,-301]
  
  y.train.3 = train.3 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test.3 = test.3 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Fitting Random Forest
  
  rf3 <- randomForest(GFE ~., data = train.3, ntrees = 300, mtry = sqrt(p))
  
  rf.train.3.hat       =     predict(rf3, newdata = train.3, type = "class")
  rf.test.3.hat        =     predict(rf3, newdata = test.3, type = "class")
  
  ## Random Forest train.3 and test.3 error rates 
  Err.rf.3[s,1]       =     mean(y.train.3 != rf.train.3.hat)
  Err.rf.3[s,2]       =     mean(y.test.3 != rf.test.3.hat)
  
  ## Random Forest false positive and negative train.3 error
  Err.rf.3[s,3]  =  mean(1 == rf.train.3.hat[y.train.3==0]) # false positive
  Err.rf.3[s,4]  =  mean(0 == rf.train.3.hat[y.train.3==1]) # false negative
  
  ## Random Forest false positive and negative test.3 error
  Err.rf.3[s,5]  =  mean(1 == rf.test.3.hat[y.test.3==0]) # false positive
  Err.rf.3[s,6]  =  mean(0 == rf.test.3.hat[y.test.3==1]) # false negative
  
  # Fitting Radial SVM
  
  trctrl <- train.3Control(method = "repeatedcv", number = 10, repeats = 3)
  
  svm_cv <- train.3(GFE ~., data = train.3, method = "svmRadial",
                    trControl=trctrl,
                    tuneLength = 10,
                    cost = 10^seq(-2,2,length.out = 5),
                    sigma = c(0.1, 0.5, 1, 2, 5, 10)
  )
  
  svm.train.3.hat <- predict(svm_cv3, newdata = train.3)
  svm.test.3.hat <- predict(svm_cv3, newdata = test.3)
  
  Err.svm.3[s,1]       =     mean(y.train.3 != svm.train.3.hat)
  Err.svm.3[s,2]       =     mean(y.test.3 != svm.test.3.hat)
  svm_results3        =     data.frame(svm_cv3$results)
  Err.svm.3[s,3]       =    1 - max(svm_results3$Accuracy)
  
  
  ## SVM false positive and negative train.3 error
  Err.svm.3[s,4]  =  mean(1 == svm.train.3.hat[y.train.3==0]) # false positive
  Err.svm.3[s,5]  =  mean(0 == svm.train.3.hat[y.train.3==1]) # false negative
  
  # column 5 of Err = false positive test.3 error 
  # column 6 of Err = false negative test.3 error
  Err.svm.3[s,6]  =  mean(1 == svm.test.3.hat[y.test.3==0]) # false positive
  Err.svm.3[s,7]  =  mean(0 == svm.test.3.hat[y.test.3==1]) # false negative
  
  # Fitting Logistic
  
  logistic_model3 <- glm(GFE ~., data = train.3, family = binomial(link = "logit"))
  
  log.train.3.prob <- predict(logistic_model3, newx = x.train.3, type = "response")
  log.test.3.prob <- predict(logistic_model3, data = x.test.3, type = "response")
  log.train.3.hat <- ifelse(log.train.3.prob > 0.5, 1, 0)
  log.test.3.hat <- ifelse(log.test.3.prob > 0.5, 1, 0)
  
  Err.logistic.3[s,1] =     mean(y.train.3 != log.train.3.hat)
  Err.logistic.3[s,2] =     mean(y.test.3 != log.test.3.hat)
  
  
  ## Logistic false positive and negative train.3 error
  Err.logistic.3[s,3]  =  mean(1 == log.train.3.hat[y.train.3==0]) # false positive
  Err.logistic.3[s,4]  =  mean(0 == log.train.3.hat[y.train.3==1]) # false negative
  
  ## Logistic false positive and negative train.3 error
  Err.logistic.3[s,5]  =  mean(1 == log.test.3.hat[y.test.3==0]) # false positive
  Err.logistic.3[s,6]  =  mean(0 == log.test.3.hat[y.test.3==1]) # false negative
  
  # Fitting Logistic Lasso
  
  m = 25
  lasso3 <- cv.glmnet(x.train.3, y.train.3, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
  lasso_lamda3 = lasso3$lambda.min #optimal lambda for lasso
  lasso_model3 <- glmnet(x.train.3, y.train.3, lambda = lasso_lamda3, family="binomial")
  lasso.train.3.hat <- predict(lasso_model3, newx = x.train.3, type = "class")
  lasso.test.3.hat <- predict(lasso_model3, newx = x.test.3, type = "class")
  
  Err.lasso.3[s,1]    =     mean(y.train.3 != lasso.train.3.hat)
  Err.lasso.3[s,2]    =     mean(y.test.3 != lasso.test.3.hat)
  Err.lasso.3[s,3]    =     min(lasso3$cvm)
  
  ## Logistic false positive and negative train.3 error
  Err.lasso.3[s,4]  =  mean(1 == lasso.train.3.hat[y.train.3==0]) # false positive
  Err.lasso.3[s,5]  =  mean(0 == lasso.train.3.hat[y.train.3==1]) # false negative
  
  ## Logistic false positive and negative test.3 error
  Err.lasso.3[s,6]  =  mean(1 == lasso.test.3.hat[y.test.3==0]) # false positive
  Err.lasso.3[s,7]  =  mean(0 == lasso.test.3.hat[y.test.3==1]) # false negative
  
  
  
  # Fitting Logistic Ridge 
  
  m = 25
  ridge3 <-  cv.glmnet(x.train.3, y.train.3, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
  ridge_lambda3 = ridge3$lambda.min
  ridge_model3 <-  glmnet(x.train.3, y.train.3, lambda = ridge_lambda3, family = "binomial")
  
  ridge.train.3.hat <-   predict(ridge_model3, newx = x.train.3, type = "class")
  ridge.test.3.hat  <-   predict(ridge_model3, newx = x.test.3, type = "class")
  
  Err.ridge.3[s,1]    =     mean(y.train.3 != ridge.train.3.hat)
  Err.ridge.3[s,2]    =     mean(y.test.3 != ridge.test.3.hat)
  Err.ridge.3[s,3]    =     min(ridge3$cvm)
  
  
  ## Ridge false positive and negative train.3 error
  Err.ridge.3[s,4]  =  mean(1 == ridge.train.3.hat[y.train.3==0]) # false positive
  Err.ridge.3[s,5]  =  mean(0 == ridge.train.3.hat[y.train.3==1]) # false negative
  
  # column 5 of Err = false positive test.3 error 
  # column 6 of Err = false negative test.3 error
  Err.ridge.3[s,6]  =  mean(1 == ridge.test.3.hat[y.test.3==0]) # false positive
  Err.ridge.3[s,7]  =  mean(0 == ridge.test.3.hat[y.test.3==1]) # false negative
  
}


## Box Plots

err.train.3     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                   rep("Ridge", S), rep("SVM", S)) , 
                                 c(Err.lasso.3[,1], Err.logistic.3[,1], Err.rf.3[,1], Err.ridge.3[,1], Err.svm.3[,1]))
names(err.train.3) <- c("Method", "train.3.Err")   

err.test.3     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                  rep("Ridge", S), rep("SVM", S)) , 
                                c(Err.lasso.3[,2], Err.logistic.3[,2], Err.rf.3[,2], Err.ridge.3[,2], Err.svm.3[,2]))

names(err.test.3) <- c("Method", "test.3")   

err.cv3     =     data.frame(c(rep("Lasso", S), rep("Ridge", S), rep("SVM", S)) , 
                             c(Err.lasso.3[,3], Err.rf.3[,3], Err.svm.3[,3]))

names(err.cv3) <- c("Method", "CV.Err")   

# train.3 false positive (fp)
err.train.3.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.3[,4], Err.logistic.3[,3], Err.rf.3[,3], Err.ridge.3[,4], Err.svm.3[,4]))

names(err.train.3.fp) <- c("Method", "train.3.fp")   


# train.3 false negative (fn)
err.train.3.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.3[,5], Err.logistic.3[,4], Err.rf.3[,4], Err.ridge.3[,5], Err.svm.3[,5]))

names(err.train.3.fn) <- c("Method", "train.3.fn")   

# test.3 false positive (fp)
err.test.3.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.3[,6], Err.logistic.3[,5], Err.rf.3[,5], Err.ridge.3[,6], Err.svm.3[,6]))

names(err.test.3.fp) <- c("Method", "test.3.fp")   

# train.3 false negative (fn)
err.test.3.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.3[,7], Err.logistic.3[,6], Err.rf.3[,6], Err.ridge.3[,7], Err.svm.3[,7]))

names(err.test.3.fn) <- c("Method", "test.3.fn")   


p1_imbal = ggplot(err.train.3)   +     aes(x=Method, y = train.3.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("train.3ing Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))

p2_imbal = ggplot(err.test.3)   +     aes(x=Method, y = test.3, fill=Method) +   geom_boxplot()  +
  ggtitle("test.3ing Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))

p3_imbal = ggplot(err.cv3)   +     aes(x=Method, y = CV.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("CV Error Rates for GFE 0.5n (Imbalanced)") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.125)

p4_imbal = ggplot(err.train.3.fp)   +     aes(x=Method, y = train.3.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.3 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.4)  

p5_imbal = ggplot(err.train.3.fn)   +     aes(x=Method, y = train.3.fn, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.3 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.4)  

p6_imbal = ggplot(err.test.3.fp)   +     aes(x=Method, y = test.3.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("test.3 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

p7_imbal = ggplot(err.test.3.fn)   +     aes(x=Method, y = test.3.fn, fill=Method) +   geom_boxplot()  + 
  ggtitle("test.3 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

grid.arrange(p1_imbal, p4_imbal, p5_imbal, p2_imbal, p6_imbal, p7_imbal, ncol=3)

p3_imbal

## PART IV: Imbalanced, 0.9N ###

Err.rf.4        =    matrix(0, nrow = S, ncol = 6) 
Err.svm.4       =    matrix(0, nrow = S, ncol = 7) 
Err.logistic.4  =    matrix(0, nrow = S, ncol = 6) 
Err.lasso.4     =    matrix(0, nrow = S, ncol = 7) 
Err.ridge.4     =    matrix(0, nrow = S, ncol = 7) 


for (s in 1:S) {
  
  # Splitting
  
  sample = sample.split(gfe2, SplitRatio = 9/10)
  
  train.4 = subset(gfe2, sample==TRUE)
  test.4 = subset(gfe2, sample==FALSE)
  
  
  train.4$GFE <- as.factor(train.4$GFE)
  test.4$GFE <- as.factor(test.4$GFE)
  
  x.train.4 = model.matrix(GFE~., train.4)[,-301]
  x.test.4 = model.matrix(GFE~., test.4)[,-301]
  
  y.train.4 = train.4 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  y.test.4 = test.4 %>%
    select(GFE) %>%
    unlist() %>%
    as.factor()
  
  
  # Fitting Random Forest
  
    rf4 <- randomForest(GFE ~., data = train.4, ntrees = 300, mtry = sqrt(p))
 
  rf.train.4.hat       =     predict(rf4, newdata = train.4, type = "class")
  rf.test.4.hat        =     predict(rf4, newdata = test.4, type = "class")
  
  ## Random Forest train.4 and test.4 error rates 
  Err.rf.4[s,1]       =     mean(y.train.4 != rf.train.4.hat)
  Err.rf.4[s,2]       =     mean(y.test.4 != rf.test.4.hat)
  
  ## Random Forest false positive and negative train.4 error
  Err.rf.4[s,3]  =  mean(1 == rf.train.4.hat[y.train.4==0]) # false positive
  Err.rf.4[s,4]  =  mean(0 == rf.train.4.hat[y.train.4==1]) # false negative
  
  # column 5 of Err = false positive test.4 error 
  # column 6 of Err = false negative test.4 error
  Err.rf.4[s,5]  =  mean(1 == rf.test.4.hat[y.test.4==0]) # false positive
  Err.rf.4[s,6]  =  mean(0 == rf.test.4.hat[y.test.4==1]) # false negative
  
  # Fitting Radial SVM
  
    trctrl4 <- train.4Control(method = "repeatedcv", number = 10, repeats = 3)
    
    svm_cv4 <- train.4(GFE ~., data = train.4, method = "svmRadial",
                      trControl=trctrl,
                      tuneLength = 10,
                      cost = 10^seq(-2,2,length.out = 5),
                      sigma = c(0.1, 0.5, 1, 2, 5, 10)
    )
  
  svm.train.4.hat <- predict(svm_cv4, newdata = train.4)
  svm.test.4.hat <- predict(svm_cv4, newdata = test.4)
  
  Err.svm.4[s,1]       =     mean(y.train.4 != svm.train.4.hat)
  Err.svm.4[s,2]       =     mean(y.test.4 != svm.test.4.hat)
  svm_results4        =     data.frame(svm_cv4$results)
  Err.svm.4[s,3]       =    1 - max(svm_results4$Accuracy)
  
  
  ## SVM false positive and negative train.4 error
  Err.svm.4[s,4]  =  mean(1 == svm.train.4.hat[y.train.4==0]) # false positive
  Err.svm.4[s,5]  =  mean(0 == svm.train.4.hat[y.train.4==1]) # false negative
  
  # column 5 of Err = false positive test.4 error 
  # column 6 of Err = false negative test.4 error
  Err.svm.4[s,6]  =  mean(1 == svm.test.4.hat[y.test.4==0]) # false positive
  Err.svm.4[s,7]  =  mean(0 == svm.test.4.hat[y.test.4==1]) # false negative
  
  # Fitting Logistic
  
    logistic_model4 <- glm(GFE ~., data = train.4, family = binomial(link = "logit"))
  
  log.train.4.prob <- predict(logistic_model4, newx = x.train.4, type = "response")
  log.test.4.prob <- predict(logistic_model4, data = x.test.4, type = "response")
  log.train.4.hat <- ifelse(log.train.4.prob > 0.5, 1, 0)
  log.test.4.hat <- ifelse(log.test.4.prob > 0.5, 1, 0)
  
  Err.logistic.4[s,1] =     mean(y.train.4 != log.train.4.hat)
  Err.logistic.4[s,2] =     mean(y.test.4 != log.test.4.hat)
  
  
  ## Logistic false positive and negative train.4 error
  Err.logistic.4[s,3]  =  mean(1 == log.train.4.hat[y.train.4==0]) # false positive
  Err.logistic.4[s,4]  =  mean(0 == log.train.4.hat[y.train.4==1]) # false negative
  
  # column 5 of Err = false positive test.4 error 
  # column 6 of Err = false negative test.4 error
  Err.logistic.4[s,5]  =  mean(1 == log.test.4.hat[y.test.4==0]) # false positive
  Err.logistic.4[s,6]  =  mean(0 == log.test.4.hat[y.test.4==1]) # false negative
  
  # Fitting Logistic Lasso
  
  m = 25
    lasso4 <- cv.glmnet(x.train.4, y.train.4, family="binomial", alpha = 1, nfolds = 10, type.measure="class")
    lasso_lamda4 = lasso4$lambda.min #optimal lambda for lasso
    lasso_model4 <- glmnet(x.train.4, y.train.4, lambda = lasso_lamda4, family="binomial")
  
  lasso.train.4.hat <- predict(lasso_model4, newx = x.train.4, type = "class")
  lasso.test.4.hat <- predict(lasso_model4, newx = x.test.4, type = "class")
  
  Err.lasso.4[s,1]    =     mean(y.train.4 != lasso.train.4.hat)
  Err.lasso.4[s,2]    =     mean(y.test.4 != lasso.test.4.hat)
  Err.lasso.4[s,3]    =     min(lasso4$cvm)
  
  ## Logistic false positive and negative train.4 error
  Err.lasso.4[s,4]  =  mean(1 == lasso.train.4.hat[y.train.4==0]) # false positive
  Err.lasso.4[s,5]  =  mean(0 == lasso.train.4.hat[y.train.4==1]) # false negative
  
  # column 5 of Err = false positive test.4 error 
  # column 6 of Err = false negative test.4 error
  Err.lasso.4[s,6]  =  mean(1 == lasso.test.4.hat[y.test.4==0]) # false positive
  Err.lasso.4[s,7]  =  mean(0 == lasso.test.4.hat[y.test.4==1]) # false negative
  
  
  
  # Fitting Logistic Ridge 
  
  m = 25

    ridge4 <-  cv.glmnet(x.train.4, y.train.4, family = "binomial", alpha = 0,  nfolds = 10, type.measure="class")
    ridge_lambda4 = ridge4$lambda.min
    ridge_model4 <-  glmnet(x.train.4, y.train.4, lambda = ridge_lambda4, family = "binomial")
 
  ridge.train.4.hat <-   predict4(ridge_model4, newx = x.train.4, type = "class")
  ridge.test.4.hat  <-   predict4(ridge_model4, newx = x.test.4, type = "class")
  
  Err.ridge.4[s,1]    =     mean(y.train.4 != ridge.train.4.hat)
  Err.ridge.4[s,2]    =     mean(y.test.4 != ridge.test.4.hat)
  Err.ridge.4[s,3]    =     min(ridge4$cvm)
  
  
  ## Ridge false positive and negative train.4 error
  Err.ridge.4[s,4]  =  mean(1 == ridge.train.4.hat[y.train.4==0]) # false positive
  Err.ridge.4[s,5]  =  mean(0 == ridge.train.4.hat[y.train.4==1]) # false negative
  
  # column 5 of Err = false positive test.4 error 
  # column 6 of Err = false negative test.4 error
  Err.ridge.4[s,6]  =  mean(1 == ridge.test.4.hat[y.test.4==0]) # false positive
  Err.ridge.4[s,7]  =  mean(0 == ridge.test.4.hat[y.test.4==1]) # false negative
  
}


## Box Plots

err.train.4     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                   rep("Ridge", S), rep("SVM", S)) , 
                                 c(Err.lasso.4[,1], Err.logistic.4[,1], Err.rf.4[,1], Err.ridge.4[,1], Err.svm.4[,1]))
names(err.train.4) <- c("Method", "train.4.Err")   

err.test.4     =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                  rep("Ridge", S), rep("SVM", S)) , 
                                c(Err.lasso.4[,2], Err.logistic.4[,2], Err.rf.4[,2], Err.ridge.4[,2], Err.svm.4[,2]))

names(err.test.4) <- c("Method", "test.4")   

err.cv4     =     data.frame(c(rep("Lasso", S), rep("Ridge", S), rep("SVM", S)) , 
                            c(Err.lasso.4[,3], Err.rf.4[,3], Err.svm.4[,3]))

names(err.cv4) <- c("Method", "CV.Err")   

# train.4 false positive (fp)
err.train.4.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.4[,4], Err.logistic.4[,3], Err.rf.4[,3], Err.ridge.4[,4], Err.svm.4[,4]))

names(err.train.4.fp) <- c("Method", "train.4.fp")   


# train.4 false negative (fn)
err.train.4.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                         rep("Ridge", S), rep("SVM", S)) , 
                                       c(Err.lasso.4[,5], Err.logistic.4[,4], Err.rf.4[,4], Err.ridge.4[,5], Err.svm.4[,5]))

names(err.train.4.fn) <- c("Method", "train.4.fn")   

# test.4 false positive (fp)
err.test.4.fp        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.4[,6], Err.logistic.4[,5], Err.rf.4[,5], Err.ridge.4[,6], Err.svm.4[,6]))

names(err.test.4.fp) <- c("Method", "test.4.fp")   

# train.4 false negative (fn)
err.test.4.fn        =     data.frame(c(rep("Lasso", S), rep("Logistic", S), rep("RF", S),
                                        rep("Ridge", S), rep("SVM", S)) , 
                                      c(Err.lasso.4[,7], Err.logistic.4[,6], Err.rf.4[,6], Err.ridge.4[,7], Err.svm.4[,7]))

names(err.test.4.fn) <- c("Method", "test.4.fn")   


p1_imbal2 = ggplot(err.train.4)   +     aes(x=Method, y = train.4.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("train.4ing Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))

p2_imbal2 = ggplot(err.test.4)   +     aes(x=Method, y = test.4, fill=Method) +   geom_boxplot()  +
  ggtitle("test.4ing Error Rates for GFE") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))

p3_imbal2 = ggplot(err.cv4)   +     aes(x=Method, y = CV.Err, fill=Method) +   geom_boxplot()  +
  ggtitle("CV Error Rates for GFE 0.9n (Imbalanced)") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.125)

p4_imbal2 = ggplot(err.train.4.fp)   +     aes(x=Method, y = train.4.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.4 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ylim(0, 0.4)  

p5_imbal2 = ggplot(err.train.4.fn)   +     aes(x=Method, y = train.4.fn, fill=Method) +   geom_boxplot()  +  
  ggtitle("train.4 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.4)  

p6_imbal2 = ggplot(err.test.4.fp)   +     aes(x=Method, y = test.4.fp, fill=Method) +   geom_boxplot()  +  
  ggtitle("test.4 fp errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

p7_imbal2 = ggplot(err.test.4.fn)   +     aes(x=Method, y = test.4.fn, fill=Method) +   geom_boxplot()  + 
  ggtitle("test.4 fn errors") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+  ylim(0, 0.6)  

grid.arrange(p1_imbal2, p4_imbal2, p5_imbal2, p2_imbal2, p6_imbal2, p7_imbal2, ncol=3)

p3_imbal2

