import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle
import time

# TODO: Load traffic signs data.

with open('./train.p', mode='rb') as f:
    data = pickle.load(f)

X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)


nb_classes = 43
mean_0 = np.mean(X_train[:,:,0])
mean_1 = np.mean(X_train[:,:,1])
mean_2 = np.mean(X_train[:,:,2])

X_train = (X_train - np.array([mean_0,mean_1,mean_2]) ) / 128
X_valid = (X_valid - np.array([mean_0,mean_1,mean_2])  ) / 128

# TODO: Split data into training and validation sets.

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, nb_classes)
resized = tf.image.resize_images(x, (227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
W9 = tf.Variable(tf.truncated_normal(shape, mean=0, stddev = 0.1))
b9 = tf.Variable(tf.zeros(nb_classes))
logits = tf.add(tf.matmul(fc7,W9),b9)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
epochs = 10
batch_size = 128

#logits, reg = MyNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, var_list=[W9, b9])

saver = tf.train.Saver()

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

import time
start = time.time()

best_acc = 0 # best net so far

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train,y_train)
        for offset in range(0,num_examples, batch_size):
            end = offset+batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        train_accuracy = evaluate(X_train, y_train)
        
        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            saver.save(sess, './mynet')
            print("Model saved.")
        
        print("Epoch {}".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
        
end = time.time()
print()
print("Running time was {} seconds".format(end-start))
print("The best model has a validation accuracy of {}.".format(best_acc))