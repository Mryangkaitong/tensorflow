import tensorflow as tf

with tf.Session() as sess:
    
    my_saver = tf.train.import_meta_graph('C:/Users/asus-/Desktop/captcha_demo/model/Alexnet.meta')
    my_saver.restore(sess,tf.train.latest_checkpoint('C:/Users/asus-/Desktop/captcha_demo/model/'))
    graph = tf.get_default_graph()
    writer_test=tf.summary.FileWriter('C:/Users/asus-/Desktop/logs/',sess.graph)
         
