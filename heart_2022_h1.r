library(caret)
library(glmnet)
library(dplyr)

data<-read.csv("C:\\Users\\Yanzheng\\OneDrive - University of Toledo\\Desktop\\DS5220\\project\\heart_2022_no_nans.csv")
dataclear <- data %>%
  mutate(
    HadHeartAttack = factor(HadHeartAttack, levels = c("No", "Yes")),
    HighRisk = ifelse(HighRiskLastYear == "Yes", 1, 0),          
    HighBMI = ifelse(BMI > mean(BMI, na.rm = TRUE), 1, 0),         
    Diabetes = ifelse(HadDiabetes == "Yes", 1, 0)
  )

# Filter only relevant columns
model_data <- dataclear %>%
  select(HadHeartAttack, HighRisk, HighBMI, Diabetes) %>%
  na.omit()


train_control <- trainControl(method = "cv", number = 10)
#cross validation
# Train logistic regression model
set.seed(123)
model <- train(
  HadHeartAttack ~ HighRisk + HighBMI + Diabetes,
  data = model_data,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

print(model)
summary(model$finalModel)

#Logistic Regression
set.seed(123)
trainIndex <- createDataPartition(model_data$HadHeartAttack, p = 0.8, list = FALSE)
train <- model_data[trainIndex, ]
test <- model_data[-trainIndex, ]


model <- glm(HadHeartAttack ~ HighRisk + HighBMI + Diabetes, data = train, family = binomial)

prob <- predict(model, newdata = test, type = "response")
pred_class <- ifelse(prob > 0.5, "Yes", "No")
pred_class <- factor(pred_class, levels = c("No", "Yes"))

# Confusion matrix and performance metrics
confusion <- confusionMatrix(pred_class, test$HadHeartAttack, positive = "Yes")
print(confusion)

#Logistic Regression with Linear Model Selection
x <- model.matrix(HadHeartAttack ~ ., model_data)[, -1]  # remove intercept
y <- model_data$HadHeartAttack

# Encode response as 0/1 for glmnet
y_numeric <- ifelse(y == "Yes", 1, 0)

# Apply cross-validated logistic regression with Lasso
set.seed(123)
cv_model <- cv.glmnet(x, y_numeric, family = "binomial", alpha = 1)  # alpha = 1 for Lasso

# Best lambda value
best_lambda <- cv_model$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Final model with best lambda
lasso_model <- glmnet(x, y_numeric, family = "binomial", alpha = 1, lambda = best_lambda)

# Coefficients selected
coef(lasso_model)
