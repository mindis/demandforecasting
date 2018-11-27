

# This data generator will generate data indefinitely by randomly sampling rows according to their sampling probability
# For informaiton on how and why this is being used / calculated see the Amazon DeepAR paper which is referenced 
# in the bitbucket for this project.
custom_data_generator <- function(data, batchSize, probs, windowSize, testSize){
  
  function(){
    
    # Select which rows are being used according to input probs
    rows <- sample(1:nrow(data), batchSize, replace = TRUE, prob = probs)
    
    # Create col indicies for input to x and y Data
    total_time_steps <- length(colnames(data)[grepl("^2", colnames(data))])
    first <- which(colnames(data) %in% colnames(data)[grepl("^2", colnames(data))][1])
    start <- sample(1:(total_time_steps - testSize - windowSize), 1, replace = TRUE) + first
    train_cols <- start:(start + windowSize - 1)
    test_cols <- (start + windowSize):(start + windowSize + testSize - 1)
    
    
    
    # Next create data of covariates. Hardcoded for now
    # TODO generalize covariate selection
    
    # Training covariates (all will be embedded for simplicity)
    stores_train <- matrix(data[rows, ]$store_number)
    biz_cd_train <- matrix(data[rows, ]$biz_cd_int)
    fyWeek_train <- matrix(as.integer(str_sub(colnames(data)[grepl("^2", colnames(data))][train_cols], 5, 7)),
                           ncol = windowSize, nrow = batchSize, byrow = TRUE)
    
    # Test data covariates
    stores_test <- matrix(as.integer(data[rows, ]$store_number))
    biz_cd_test <- matrix(as.integer(data[rows, ]$biz_cd_int))
    fyWeek_test <- matrix(as.integer(str_sub(colnames(data)[grepl("^2", colnames(data))][test_cols], 5, 7)),
                          ncol = testSize, nrow = batchSize, byrow = TRUE)
    
    scale_factor <- matrix(data[rows, ]$scale_factor)
    
    
    x_data <- as.matrix(data[rows, train_cols]) / data$scale_factor[rows]
      #data[, colnames(data)[grepl("^2", colnames(data))]] %>%

    #x_data <- abind::abind(x_data, fyWeek_train, along = 3)
    dim(x_data) <- c(dim(x_data)[1], dim(x_data)[2], 1)
    
    
    y_data <- as.matrix(data[rows, test_cols])
    #y_data <- abind::abind(y_data, fyWeek_test, along = 3)
    dim(y_data) <- c(dim(y_data)[1], dim(y_data)[2], 1)
    
    
    list(list(x_data, biz_cd_train, scale_factor, stores_train), 
         list(y_data))
  }
}


# Here is a little sample to see how the output data looks.
# call data() after assignment to generate sample data

data <- custom_data_generator(demand_wide,
                              batchSize = 2,
                              probs = demand_wide$sampling_probability,
                              windowSize = 5,
                              testSize = 3)


py_run_string("import tensorflow as tf")
py_run_string("import tensorflow_probability as tfp")
py_run_string(
  "def loss(y_true, y_pred):
    y_pred = y_pred + 1e-7
    y_true = y_true + 1e-7

    # Need to extract input vectors and set the shapes of tensors
    mu = tf.reshape(y_pred[:,:,0], [-1])
    alpha = tf.reshape(y_pred[:,:,1], [-1])
    scale = tf.reshape(y_pred[:,:,2], [-1])
    y_true = tf.reshape(y_true[:,:,0], [-1])

    # need to rescale mu and alpha
    mu = mu * scale
    alpha = alpha / tf.sqrt(scale)
    
    # Using tf probability to calculate log loss
    loss = tfp.distributions.NegativeBinomial(mu, alpha).log_prob(y_true)

    return -tf.reduce_sum(loss, axis=-1)"
)



