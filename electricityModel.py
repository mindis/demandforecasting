import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import time
import os

class Inputs(tf.keras.layers.Layer):
    def __init__(self, emb_inputDim, emb_outputDim, windowSize, predictSize):
        super(Inputs, self).__init__()
        
        self.embed = tf.keras.layers.Embedding(emb_inputDim, emb_outputDim)
        self.flatten = tf.keras.layers.Flatten()
        self.repeat = tf.keras.layers.RepeatVector(windowSize)
    
    def call(self, ts, ID, yvals):
        ts = tf.expand_dims(ts, 2)
        ts = tf.cast(ts, tf.float32)

        ID = self.embed(ID)
        ID = self.flatten(ID)
        ID = self.repeat(ID)
        ID = tf.cast(ID, tf.float32)
                
        merged = tf.concat([ts, ID], axis=2)
        return merged, yvals

class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.gru1 = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    def call(self, x):
        x, _ = self.gru(x)
        x, _ = self.gru1(x)
        x, states = self.gru2(x)
        return states

class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True)
        
        self.dense1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation="softplus")
                )
        self.dense2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation="softplus")
                )
                
    def call(self, hidden_state, y, training=True):
        if training:
            x, _ = self.gru(y, initial_state=hidden_state)
            mu = self.dense1(x)
            alpha = self.dense2(x)
            out = tf.concat([mu, alpha], 2)
            return out
        else:
            out = np.zeros(y.shape)
            
            # Feed only starting values for predictions
            out[:,0,:] = y[:,0,:] 
            
            for k in range(out.shape[1]):
                x, _ = self.gru(y, initial_state=hidden_state)
                mu = self.dense1(x)
                alpha = self.dense2(x)
                mod = tfp.distributions.NegativeBinomial(mu, alpha)
                out[:,k+1,:] = mod.sample()
            
            return out
                
                
                
            

class deepAR(tf.keras.Model):
    def __init__(self, emb_inputDim, emb_outputDim, windowSize, predictSize, 
                 enc_units, dec_units, training=True):
        super(deepAR, self).__init__()
        self.training = training
        self.inputs = Inputs(emb_inputDim, emb_outputDim, 
                             windowSize, predictSize)
        self.encoder = Encoder(enc_units)
        self.decoder = Decoder(dec_units)
    
    def call(self, ts, ID, yvals):
        merged, yvals = self.inputs(ts, ID, yvals)
        states = self.encoder(merged)
        yvals = tf.cast(yvals, tf.float32)
        pred = self.decoder(states, yvals, training = self.training)
        return pred

def neg_bin_loss(y_true, y_pred):
    y_pred = y_pred + 1e-7
    y_true = y_true + 1e-7
    mu = tf.reshape(y_pred[:,:,0], [-1])
    alpha = tf.reshape(y_pred[:,:,1], [-1])
    y_true = tf.reshape(y_true, [-1])
    loss = tfp.distributions.NegativeBinomial(mu, alpha).log_prob(y_true)
    return -tf.reduce_sum(loss, axis=-1)

def normal_loss(y_true, y_pred):
    y_pred = y_pred + 1e-7
    y_true = y_true + 1e-7
    mu = tf.reshape(y_pred[:,:,0], [-1])
    sigma = tf.reshape(y_pred[:,:,1], [-1])
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    
    loss = tfp.distributions.Normal(loc = mu, scale = sigma).log_prob(y_true)
    return -tf.reduce_sum(loss, axis=-1)

def series_to_supervised(data, n_in=168, n_out=24, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def data_prep(data):
    for i in range(data.shape[1]):
        if i == 0:
            d1 = np.array(data)[:,i][:,np.newaxis]
            output = [np.array(series_to_supervised(d1))]
        else:
            d1 = np.array(data)[:,i][:,np.newaxis]
            out = np.array(series_to_supervised(d1))
            output.append(out)
    return np.array(output)

def ND(z, zhat):
    sum(abs(z - zhat)) / sum(z)
    

data = np.load("electricity.npy")

windowSize = 168
predictSize = 24
batchSize = 50
learningRate = 1e-3
epochs = 1000
emb_inputDim = 370
emb_outputDim = 20
enc_units = 40
dec_units = 40


data_test = data[data.shape[0] - (windowSize+predictSize):, :]
input_data = data_prep(data[:data.shape[0] - (predictSize+windowSize), :])
data_train = input_data.reshape(input_data.shape[0]*input_data.shape[1], 
                                predictSize+windowSize)

x_train = data_train[:, :windowSize]
y_train_in = data_train[:, (windowSize-1):-1, np.newaxis]
y_train_out = data_train[:, windowSize:]
id_train = np.repeat(range(input_data.shape[0]), input_data.shape[1])


model = deepAR(emb_inputDim, emb_outputDim, windowSize, predictSize, 
                 enc_units, dec_units, training=True)

steps = x_train.shape[0] // batchSize
optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
train_rmse = tf.keras.metrics.RootMeanSquaredError(name = "train_accuracy")


@tf.function
def train_step(ts, ID, yvals_in, yvals_out):
    loss = 0
    with tf.GradientTape() as tape:
        params = model(ts, ID, yvals_in)
        loss = normal_loss(yvals_out, params)
        
        predictions = tfp.distributions.Normal(params[:,:,0], params[:,:,1]).sample()
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_rmse(yvals_out, predictions)
    return loss

tf_train = tf.data.Dataset.from_tensor_slices((x_train, id_train, y_train_in))
tf_labels = tf.data.Dataset.from_tensor_slices(y_train_out)

tf_dataset = tf.data.Dataset.zip((tf_train, tf_labels)).batch(batchSize)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

for epoch in range(epochs):
    start = time.time()
    total_loss = 0
    
    for (batch, ((ts, id_val, y_in), y_out)) in enumerate(tf_dataset.take(steps)):
        batch_loss = train_step(ts, id_val, y_in, y_out)
        total_loss += batch_loss
        
        if batch % 1000 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                  batch, batch_loss.numpy()))
        
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f} RMSE {:.4f}'.format(epoch + 1,
              total_loss / steps, train_rmse.result()))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        