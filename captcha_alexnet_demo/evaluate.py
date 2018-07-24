import tensorflow as tf 
import image_reader as ir
from nets import nets_factory



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord_dir', '/ ', 'This is the tfrecord directory of the picture')
tf.app.flags.DEFINE_integer('test_size', 10, 'This is the batch size')



# 不同字符数量
CHAR_SET_LEN = 10

BATCH_SIZE=1

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  
y0 = tf.placeholder(tf.float32, [None]) 
y1 = tf.placeholder(tf.float32, [None]) 
y2 = tf.placeholder(tf.float32, [None]) 
y3 = tf.placeholder(tf.float32, [None])

image, label0, label1, label2, label3 = ir.read_and_decode(FLAGS.tfrecord_dir)

#使用shuffle_batch可以随机打乱
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size =BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)



#定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)

# inputs: a tensor of size [batch_size, height, width, channels]
X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
# 数据输入网络得到输出值
logits0,logits1,logits2,logits3,end_points = train_network_fn(X)

# 预测值
predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])  
predict0 = tf.argmax(predict0, 1)  

predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])  
predict1 = tf.argmax(predict1, 1)  

predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])  
predict2 = tf.argmax(predict2, 1)  

predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])  
predict3 = tf.argmax(predict3, 1) 
    
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
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(total_loss) 
    
# 计算准确率
correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
    
correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
    
correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
    
correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32)) 
    



def main(unuse_args):
    
    with tf.Session() as sess:
         
        sess.run(tf.global_variables_initializer())
        
        #载入模型
        my_saver = tf.train.import_meta_graph('C:/Users/asus-/Desktop/captcha_demo/model/Alexnet.meta')
        my_saver.restore(sess,tf.train.latest_checkpoint('C:/Users/asus-/Desktop/captcha_demo/model/'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(FLAGS.test_size):
            b_image, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
            print('%d times label:%d,%d,%d,%d' % ((i+1) ,b_label0, b_label1 ,b_label2 ,b_label3))
            sess.run(train_step, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3}) 
            label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x: b_image})
            print('predict:',label0,label1,label2,label3)
            
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    tf.app.run()
