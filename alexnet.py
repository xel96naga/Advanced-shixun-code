from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
mnist = input_data.read_data_sets(".\MNIST_data", one_hot=True)


import tensorflow.compat.v1 as tf #使用TensorFlow 1版本比较好
tf.disable_v2_behavior()

#初始化
learning_rate = 0.01 #学习率
batch_size = 100 #每轮学习多少次
epochs = 10 #进行的学习周期数
dropout = 0.8 #防止过拟合问题

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#生成卷积
def Convolution(_name, _input, _w, _b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input, _w, strides=[1,1,1,1], padding="SAME"), _b), name=_name)

#池化操作
def Pool(_name, _input, k):
    return tf.nn.max_pool(_input, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME", name=_name)

#归一化
def Normalize(_name, _input, lsize=4):
    return tf.nn.lrn(_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=_name)

#构建AlexNet模型
def alexnet(_x, _weights, _bias, _dropout):
    #第一层卷积
    _x = tf.reshape(_x, [-1,28,28,1])
    conv1 = Convolution("conv1", _x, _weights['wc1'], _bias['bc1'])
    pool1 = Pool("pool1", conv1, k=2)
    Normalize1 = Normalize("Normalize1", pool1, lsize=4)
    Normalize1 = tf.nn.dropout(Normalize1, _dropout)

    #第二层卷积
    conv2 = Convolution("conv2", Normalize1, _weights['wc2'], _bias['bc2'])
    pool2 = Pool("pool2", conv2, k=2)
    Normalize2 = Normalize("Normalize2", pool2, lsize=4)
    Normalize2 = tf.nn.dropout(Normalize2, _dropout)

    #第三层卷积
    conv3 = Convolution("conv3", Normalize2, _weights['wc3'], _bias['bc3'])
    pool3 = Pool("pool3", conv3, k=2)
    Normalize3 = Normalize("Normalize3", pool3, lsize = 4)
    Normalize3 = tf.nn.dropout(Normalize3, _dropout)

    #两层全连接
    dense1 = tf.reshape(Normalize3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + bias['bd1'], name="fc1")

    desen2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + bias['bd2'], name="fc2")

    #输出
    out = tf.nn.bias_add(tf.matmul(desen2, _weights['out']), bias['out'], name="out")
    return out

weights = {
    'wc1' : tf.Variable(tf.random_normal([3,3,1,64])),
    'wc2' : tf.Variable(tf.random_normal([3,3,64,128])),
    'wc3' : tf.Variable(tf.random_normal([3,3,128,256])),
    'wd1' : tf.Variable(tf.random_normal([4*4*256,1024])),
    'wd2' : tf.Variable(tf.random_normal([1024,1024])),
    'out' : tf.Variable(tf.random_normal([1024,10]))
}

bias = {
    'bc1' : tf.Variable(tf.random_normal([64])),
    'bc2' : tf.Variable(tf.random_normal([128])),
    'bc3' : tf.Variable(tf.random_normal([256])),
    'bd1' : tf.Variable(tf.random_normal([1024])),
    'bd2' : tf.Variable(tf.random_normal([1024])),
    'out' : tf.Variable(tf.random_normal([10]))
}


#构建模型，并锁定输出
pred = alexnet(x, weights, bias, keep_prob)

#交叉熵损失，数值越小，说明变化越小，模型参数趋于稳定
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

#准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    for each in range(epochs):
        total_cost = 0
        for i in range(total_batch):
            train_x, train_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:train_x, y:train_y, keep_prob:dropout})
            ave_cost = sess.run(cost, feed_dict={x: train_x, y: train_y, keep_prob: 1.})
            total_cost += ave_cost/total_batch
        print("轮数:%d 熵损失:%f" %(each, total_cost))
    print("准确度：", sess.run(accuracy, feed_dict={x:mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob: 1.}))