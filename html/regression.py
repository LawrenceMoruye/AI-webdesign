import tensorflow as tf

w=tf.Variable([.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)

x=tf.placeholder(tf.float32)
linear_model=w*x+b
y=tf.placeholder(tf.float32)
#calculating the loss
loss=tf.reduce_sum(tf.square(linear_model-y))
#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

#training data
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]


init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#passing the data 1000 times
	for i in range(1000):
		sess.run(train,{x:x_train,y:y_train})#kiva loans,kiva

	#evaluating the training accuracy
	curl_w,cur_b,curl_loss=sess.run([w,b,loss],{x:x_train,y:y_train})
	print("w: %s b: %s loss: %s"%(curl_w,cur_b,curl_loss))#printing out the values that minimizes the loss function
