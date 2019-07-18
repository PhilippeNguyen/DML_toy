
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
import argparse
from distutils.util import strtobool

def get_loss(dist,squeeze=True):
    def loss (y_true,y_pred):
        if squeeze:
            y_true = tf.squeeze(y_true,axis=1) #needed for keras/tfp bug fix?
        neg_log_likelihood = -tf.reduce_mean(dist.log_prob(y_true))
        return neg_log_likelihood
    return loss

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--mode", action="store", dest="mode",
    default='dml',
    help=("What kind of model to use, either 'regression'(for linear regression), "
    "'softmax'(for softmax regression), "
    "or 'dml' (for discretize mixture of logistics")
)
parser.add_argument(
    "--num_mixtures", action="store", dest="num_mixtures",
    default=6,type=int,
    help=("Int, Number of mixtures if using dml")
)
parser.add_argument(
    "--batch_size", action="store", dest="batch_size",
    default=512,type=int,
    help=("Int, batch_size for training.")
)
parser.add_argument(
    "--plot", action="store", dest="plot",
    default=True,type=strtobool,
    help=("bool, if true, plot results of the model")
)
args = parser.parse_args()
sess= K.get_session()
mode = args.mode
assert mode in ('regression','softmax','dml')

#generate data
((img_train,class_train),(img_test,class_test)) = load_data()
y_train = img_train[:,14,14]
y_test = img_test[:,14,14]
x_train = K.one_hot(class_train,10).eval(session=sess)
x_test = K.one_hot(class_test,10).eval(session=sess)

data_shape = x_train.shape[1:]
if mode == 'regression':
    out_size = 1
elif mode == 'softmax':
    out_size = 256
elif mode == 'dml':
    out_size = 3* args.num_mixtures
    
#Build model
input_layer = keras.layers.Input(shape=data_shape)
x_out = keras.layers.Dense(out_size,activation='linear')(input_layer)
model = keras.models.Model([input_layer],[x_out])
model_output = model.output

if mode == 'regression':
    dist = tfd.Normal(loc=model_output,scale=0.1) #don't really care about scale (should be 0)
    loss = get_loss(dist,squeeze=False)
elif mode == 'softmax':
    dist = tfd.Categorical(logits=model_output)
    loss = get_loss(dist,squeeze=True)
elif mode == 'dml':
    loc, unconstrained_scale, logits = tf.split(model_output,
                                            num_or_size_splits=3,
                                            axis=-1)

    scale = tf.nn.softplus(unconstrained_scale)
    
    discretized_logistic_dist = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(
            distribution=tfd.Logistic(loc=loc, scale=scale),
            bijector=tfb.AffineScalar(shift=-0.5)),
        low=0.,
        high=255.)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=discretized_logistic_dist)
    loss = get_loss(dist,squeeze=True)

model.compile(optimizer=keras.optimizers.Adam(lr=0.01),loss=loss)   

#train model 
early_stop = keras.callbacks.EarlyStopping(patience=5)
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),
          epochs=500,batch_size=args.batch_size,callbacks=[early_stop])

#plotting
if args.plot:
    target_val = tf.placeholder(tf.float32)
    log_prob_fn = dist.log_prob(target_val)
    prob_indices = np.arange(0,256).astype(np.float32)
    for val in range(10):
        if mode == 'regression':
            class_input = np.zeros((1,10))
        else:
            class_input = np.zeros((256,10))
        class_input[:,val] = 1
        log_probs = sess.run(log_prob_fn,feed_dict={model.input:class_input,
                                                    target_val:prob_indices})
        probs = np.exp(np.squeeze(log_probs))
        plt.bar(prob_indices,probs,width=5)
        plt.title('pdf of class: ' +str(val))
        plt.xlabel('pixel value')
        plt.show()