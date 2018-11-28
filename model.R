library(keras)
library(tensorflow)
library(tidyverse)
library(reticulate)

# use_python("/usr/bin/python")
# py_config()

tf$Session()
demand_wide <- readRDS("demand_wide.RDS")

windowSize <- 52
testSize <- 12
batchSize <- 128


input_demand <- layer_input(shape = c(windowSize, 1L), name = "demand")
demand <- input_demand %>%
  layer_lstm(units = 128, activation = "tanh")
  

input_store <- layer_input(shape = 1L, name = "store") 
store <- input_store %>%
  layer_embedding(input_dim = max(demand_wide$store_number + 1), output_dim = 10) %>%
  layer_flatten()

input_biz_cd <- layer_input(shape = 1L, name = "biz_cd") 
biz_cd <- input_biz_cd %>%
  layer_embedding(input_dim = max(demand_wide$biz_cd_int + 1), output_dim = 25) %>%
  layer_flatten()

input_scale_factor <- layer_input(shape = 1L, name = "scale_factor")
scale_factor <- input_scale_factor %>%
  layer_repeat_vector(testSize)

predictions <- layer_concatenate(list(demand,
                               store, 
                               biz_cd)) %>%
  layer_dense(256, activation = "relu") %>%
  layer_repeat_vector(n = testSize) %>%
  layer_lstm(units = 256, return_sequences = TRUE, activation = "tanh") %>%
  time_distributed(layer_dense(units = 256, activation = "relu")) %>% 
  time_distributed(layer_dense(unit = 2, activation = "softplus"))

output <- list(predictions, scale_factor) %>%
  layer_concatenate()

model <- keras_model(list(input_demand, input_biz_cd, input_scale_factor, input_store), output)

model %>% compile(optimizer = "adam", 
                  loss = py$loss)
summary(model)


model %>%
  fit_generator(generator = custom_data_generator(data = demand_wide, 
                                                  batchSize = batchSize, 
                                                  probs = demand_wide$sampling_probability, 
                                                  windowSize = windowSize, 
                                                  testSize = testSize), 
                steps_per_epoch = nrow(demand_wide) / batchSize, 
                epochs = 25)

 check <- custom_data_generator(data = demand_wide, 
                               batchSize = 1, 
                               probs = demand_wide$sampling_probability, 
                               windowSize = 52, 
                               testSize = 12)
chck <- check()

test <-predict(model, chck[[1]])
mu <- test[,,1] * test[,,3]
alpha <- 1 / (test[,,2] / sqrt(test[,,3]))

chck
rnegbin(12, mu, alpha)

