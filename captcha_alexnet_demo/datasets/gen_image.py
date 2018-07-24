import tensorflow as tf
from captcha.image import ImageCaptcha  
import random
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', '/ ', 'This is the saved directory of the picture')
tf.app.flags.DEFINE_integer('Captcha_size', 3, 'This is the number of characters of captcha')
tf.app.flags.DEFINE_integer('image_num', 1000, 'This is the number of pictures generated ,but less than  image_num')


#验证码内容
Captcha_content = ['0','1','2','3','4','5','6','7','8','9']

# 生成字符
def random_captcha_text():
    captcha_text = []
    for i in range(FLAGS.Captcha_size):
        ch = random.choice(Captcha_content)
        captcha_text.append(ch)
    return captcha_text
 
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    image.write(captcha_text, FLAGS.output_dir + captcha_text + '.jpg')  


def main(unuse_args):
    for i in range(FLAGS.image_num ):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, FLAGS.image_num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print("Finish!!!!!!!!!!!")
                        
if __name__ == '__main__':
    tf.app.run()



   
