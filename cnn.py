import numpy as np
import tensorflow as tf
from load_data import load_data
import cv2

def build_cnn(features, labels, mode):
	input_layer = tf.reshape(features['x'], [-1, 48, 100, 3])

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu)

	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
	pool2_flat = tf.reshape(pool2, [-1, 12 * 25 * 64])

	dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)

	dropout = tf.layers.dropout(inputs=dense2, rate=.3, training=mode == tf.estimator.ModeKeys.TRAIN)

	logits = tf.layers.dense(inputs=dropout, units=2)

	predictions = {
		'classes': tf.argmax(input=logits, axis=1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
      	loss=loss,
		global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.INFO)
data, aug_data, labels = load_data()
ind = int(len(data) / 10) * 8
train_data, train_labels = aug_data[:ind], labels[:ind]
test_data, test_labels = aug_data[ind:], labels[ind:]
model = tf.estimator.Estimator(model_fn=build_cnn, model_dir='./model')
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'x': test_data},
	y=test_labels,
	num_epochs=1,
	shuffle=False)


total_steps = 0
model_list = []

max_steps = 2500
step_size = 50

eval_results = model.evaluate(input_fn=eval_input_fn)
model_list.append(eval_results)
while total_steps < max_steps:
    total_steps += step_size
    model.train(
        input_fn=train_input_fn,
        steps=step_size,
        hooks=[logging_hook])
    eval_results = model.evaluate(input_fn=eval_input_fn)
    model_list.append(eval_results)
    print("After {} steps: ".format(total_steps))
    print(eval_results)

stepsRank = sorted(model_list, key=lambda a: -a['accuracy'])

bestModel = stepsRank[0]['global_step']
print("The highest accuracy is: {}. The best model is at step {}"\
        .format(stepsRank[0]['accuracy'], bestModel))

with open('./model/checkpoint') as checkpointFile:
    lines = checkpointFile.readlines()
write = open('./model/checkpoint', 'wb')
write.write('model_checkpoint_path: "model.ckpt-{}"\n'.format(bestModel))
for i in range(len(lines)-1):
    write.write(lines[1+i])
write.close()
print("Model {} is written into ./model/checkpoint metadata.".format(bestModel))



"""
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'x': test_data},
	num_epochs=1,
	shuffle=False)

predictions = model.predict(input_fn=predict_input_fn)
predictions = [pred['classes'] for pred in predictions]
print('Test Samples: ' + str(len(test_data)))
print('Correct Predictions: ' + str(np.sum(predictions == test_labels)))
for ind, pred in enumerate(predictions):
	print('Prediction: ' + str(pred) + '\tActual: ' + str(test_labels[ind]))
	cv2.imshow('d', test_data[ind])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
"""
