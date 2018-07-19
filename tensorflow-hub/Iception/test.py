import tensorflow as tf
import os
import numpy as np


image_path='C:/Users/asus-/Desktop/tensorflow-hub/test/'

lines = tf.gfile.GFile('C:/Users/asus-/Desktop/tensorflow-hub/output/output_labels.txt').readlines()
uid_to_human = {}
#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]

input_height = 299
input_width = 299
input_mean = 0
input_std = 255


def read_tensor_from_image_file(file_name,input_height=299,input_width=299,input_mean=0,input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


with tf.Session() as sess:
    
    with tf.gfile.FastGFile('C:/Users/asus-/Desktop/tensorflow-hub/output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    writer=tf.summary.FileWriter('C:/Users/asus-/Desktop/logs',sess.graph)
   

 
with tf.Session() as sess:
    #图片输入
    image_input=sess.graph.get_tensor_by_name('Placeholder:0')
    #预测输出
    softmax=sess.graph.get_tensor_by_name('final_result:0')
 
    for rootdir,childdir,files in os.walk(image_path):
        for file in files: 
            t = read_tensor_from_image_file(os.path.join(rootdir,file),input_height=input_height,input_width=input_width,input_mean=input_mean,input_std=input_std)

            predctions=sess.run(softmax,feed_dict={image_input:t})
            predctions=np.squeeze(predctions)
            #[::-1]是将结果逆序
            top_n=predctions.argsort()[::-1]

            print('%s：：：：',os.path.join(rootdir,file))
            for id in top_n:
                print('%s of score is %f'% (id_to_string(id),predctions[id]))
 
            print('--------------------------------------')
            
           
