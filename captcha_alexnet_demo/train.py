import os
import tensorflow as tf 
from nets import nets_factory
import numpy as np
import image_reader as ir

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord_dir', '/ ', 'This is the tfrecord directory of the picture')
tf.app.flags.DEFINE_string('model_dir', '/ ', 'This is the saved model directory of the net')
tf.app.flags.DEFINE_integer('batch_size', 10, 'This is the batch size')
tf.app.flags.DEFINE_integer('train_num', 1000, 'This is the number of train')
tf.app.flags.DEFINE_integer('print_loss_accuracy_interval', 10, 'This is the interval of printing')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'This is the rate of learning')

# 不同字符数量
CHAR_SET_LEN = 10

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  
y0 = tf.placeholder(tf.float32, [None]) 
y1 = tf.placeholder(tf.float32, [None]) 
y2 = tf.placeholder(tf.float32, [None]) 
y3 = tf.placeholder(tf.float32, [None])

image, label0, label1, label2, label3 = ir.read_and_decode(FLAGS.tfrecord_dir)

#使用shuffle_batch可以随机打乱
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size =FLAGS.batch_size,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)

#定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=True)

# inputs: a tensor of size [batch_size, height, width, channels]
X = tf.reshape(x, [FLAGS.batch_size, 224, 224, 1])
# 数据输入网络得到输出值
logits0,logits1,logits2,logits3,end_points = train_network_fn(X)
    
# 把标签转成one_hot的形式
one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    
  
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0,labels=one_hot_labels0)) 
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1,labels=one_hot_labels1)) 
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2,labels=one_hot_labels2)) 
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3,labels=one_hot_labels3)) 
  
total_loss = (loss0+loss1+loss2+loss3)/4.0
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(total_loss) 
    
# 计算准确率
correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
    
correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
    
correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
    
correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32)) 
    
# 用于保存模型
saver = tf.train.Saver()

def main(unuse_args):
    
    with tf.Session() as sess:
     
        # 初始化
        sess.run(tf.global_variables_initializer())
        
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(FLAGS.train_num):
            b_image, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
            sess.run(optimizer, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3})  

            if i % FLAGS.print_loss_accuracy_interval == 0:
                
                acc0,acc1,acc2,acc3,TotalLoss = sess.run([accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
                                                                                                                    y0: b_label0,
                                                                                                                    y1: b_label1,
                                                                                                                    y2: b_label2,
                                                                                                                    y3: b_label3})      
                print ("times:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f" % (i,TotalLoss,acc0,acc1,acc2,acc3))
                 
                
        saver.save(sess, FLAGS.model_dir)                      
        # 通知其他线程关闭
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()





