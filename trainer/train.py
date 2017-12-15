import json
import tensorflow as tf
import datetime
import time
import datasetutils
from trainutils import train
from mynet import create_mynet
from alexnet import create_alexnet
import sys

#batch_size = 16
#learning_rate = 1e-4

learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])

net_name = "alexnet"

train_path = './training_data'

# Prepare input data
classes = datasetutils.read_classes(train_path)
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2

# image settings
img_size = 224
num_channels = 3

with open('../meta/meta.json', 'w') as f:
    json.dump({'classes': classes, 'img_size': img_size, 'num_channels': num_channels}, f)

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = datasetutils.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


report_file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d_%H+%M+%S')

with open('./train_info/{0}.json'.format(report_file_name), 'w') as f:
    json.dump({'batch_size': batch_size,
               'learning_rate': learning_rate,
               'img_size': img_size,
               'num_classes': num_classes,
               'net_name': net_name,
               'validation_size': validation_size
               }, f)


mynet = create_alexnet(x, num_classes)

y_pred = tf.nn.softmax(mynet, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=mynet, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

train(epoch_amount=50,
      data=data,
      session=session,
      x=x,
      y_true=y_true,
      batch_size=batch_size,
      accuracy=accuracy,
      cost=cost,
      optimizer=optimizer,
      report_file_name=report_file_name)
