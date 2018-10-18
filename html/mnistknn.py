import numpy as np
import tensorflow as tf
#steps followed
#1.download the mnist dataset(test and train) into tensorflow libraries also setting it into batches
#2.calculate the L1 distance
#3.Make predictions and measure the accuracy

#step 1
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_data/",one_hot=True)

training_digits,training_labels=mnist.train.next_batch(5000)
test_digits,test_labels=mnist.test.next_batch(500)

training_digits_placeholder=tf.placeholder("float",[None,784])
test_digits_placeholder=tf.placeholder("float",[784])

#step2:calculating the distance between the test data and all traing sets.

l1_distance=tf.abs(tf.add(training_digits_placeholder,tf.negative(test_digits_placeholder)))
#nearest neighbor calculation using l1 distance as shown above
distance=tf.reduce_sum(l1_distance,axis=1)

#prediction:get the index of the datapoint with the shortest distance
pred=tf.arg_min(distance,0)

accuracy=0.
correctly_predicted=0
init=tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)


	for i in range(len(test_digits)):
		nn_index=sess.run(pred,feed_dict={training_digits_placeholder:training_digits,test_digits_placeholder:test_digits[i,:]})

		#get the nearest class label and compare it with its true label
		print("Test",i,"prediction",np.argmax(training_labels[nn_index]),"true label",np.argmax(test_labels[i]))
		if np.argmax(training_labels[nn_index]== np.argmax(test_labels[i])):
			#accuracy=accuracy+1.0/len(test_digits)
			correctly_predicted+=1
			accuracy=correctly_predicted/len(test_digits)*100
		

	print("training done!")
	print("Accuracy is:",accuracy)



