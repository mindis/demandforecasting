library(keras)
library(tensorflow)
library(tidyverse)
library(reticulate)
use_python("/usr/bin/python3")

# scaled_demand_wide <- readRDS("scaled_demand_wide.RDS")
py_config()
windowSize <- 52
testSize <- 8
batchSize <- 32


input_demand <- layer_input(shape = c(windowSize, 1L), name = "demand")
demand <- input_demand %>%
  layer_lstm(units = 128)
  

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
  layer_repeat_vector(n = testSize) %>%
  layer_lstm(units = 256, return_sequences = TRUE) %>%
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
                                                  batchSize = 64, 
                                                  probs = demand_wide$sampling_probability, 
                                                  windowSize = 52, 
                                                  testSize = 8), 
                steps_per_epoch = nrow(demand_wide) / batchSize, 
                epochs = 1)

check <- custom_data_generator(data = demand_wide, 
                               batchSize = 1, 
                               probs = demand_wide$sampling_probability, 
                               windowSize = 52, 
                               testSize = 8)
chck <- check()

test <-predict(model, chck[[1]])
mu <- test[,,1] * test[,,3]
alpha <- 1 / (test[,,2] / sqrt(test[,,3]))
rnegbin(8, mu, alpha)

