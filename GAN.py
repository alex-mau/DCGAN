#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:15:08 2018

@author: Alex Mau
"""
import tensorflow as tf
import numpy as np
import PIL
loc=1
n=0
def load_img(x):
    # x is address of an image
    img=PIL.Image.open(x)
    img=img.resize((image_size,image_size))
    y=np.array(img)
    return y
def save_img(x):
    global n
    #x is an array of an image
    img=tf.keras.preprocessing.image.array_to_img(x)
    img.save('./generated/'+str(n)+'.jpg','JPEG')
    n=n+1
def get_batch(x):
    global loc
    x_train=np.zeros([x,image_size,image_size,n_channels],'float32')
    for i in range(x):
        if loc<=n_samples:
            name=str(loc)
            name=name.zfill(6)
            name='./img_align_celeba/'+name+'.jpg'
        else:
            loc=1
            name=str(loc)
            name=name.zfill(6)
            name='./img_align_celeba/'+name+'.jpg'
        sample=load_img(name)
        x_train[i]=sample
        loc=loc+1
    x_train=x_train/127.5-1
    noise=np.random.uniform(-1,1,[x,n_noise]).astype('float32')
    
    return noise,x_train
# http://stackoverflow.com/a/34634291/2267819
def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
	with tf.variable_scope(scope):
		#beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
		#gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=decay)
 
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
         
		mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
	return normed
def generator(x):
    g_out0=tf.reshape(tf.matmul(x,g_W0),[batch_size,4,4,1024])
    g_out0=batch_norm(g_out0,g_beta0,g_gamma0,z)
    g_out0=tf.nn.relu(g_out0)

    g_out1=tf.nn.conv2d_transpose(g_out0,g_W1,output_shape=[batch_size,8,8,512],strides=[1,2,2,1],padding='SAME')
    g_out1=batch_norm(g_out1,g_beta1,g_gamma1,z)
    g_out1=tf.nn.relu(g_out1)

    g_out2=tf.nn.conv2d_transpose(g_out1,g_W2,output_shape=[batch_size,16,16,256],strides=[1,2,2,1],padding='SAME')
    g_out2=batch_norm(g_out2,g_beta2,g_gamma2,z)
    g_out2=tf.nn.relu(g_out2)

    g_out3=tf.nn.conv2d_transpose(g_out2,g_W3,output_shape=[batch_size,32,32,128],strides=[1,2,2,1],padding='SAME')
    g_out3=batch_norm(g_out3,g_beta3,g_gamma3,z)
    g_out3=tf.nn.relu(g_out3)

    g_out4=tf.nn.conv2d_transpose(g_out3,g_W4,output_shape=[batch_size,64,64,3],strides=[1,2,2,1],padding='SAME')
    g_out4=tf.nn.tanh(g_out4)
    return g_out4
def discriminator(x):
    d_out0=tf.nn.conv2d(x,d_W0,strides=[1,2,2,1],padding='SAME')
    d_out0=batch_norm(d_out0,d_beta0,d_gamma0,z)
    d_out0=tf.nn.leaky_relu(d_out0)
    
    d_out1=tf.nn.conv2d(d_out0,d_W1,strides=[1,2,2,1],padding='SAME')
    d_out1=batch_norm(d_out1,d_beta1,d_gamma1,z)
    d_out1=tf.nn.leaky_relu(d_out1)
    
    d_out2=tf.nn.conv2d(d_out1,d_W2,strides=[1,2,2,1],padding='SAME')
    d_out2=batch_norm(d_out2,d_beta2,d_gamma2,z)
    d_out2=tf.nn.leaky_relu(d_out2)
    
    d_out3=tf.nn.conv2d(d_out2,d_W3,strides=[1,2,2,1],padding='SAME')
    d_out3=batch_norm(d_out3,d_beta3,d_gamma3,z)
    d_out3=tf.nn.leaky_relu(d_out3)
    
    d_out4=tf.reshape(d_out3,[batch_size,4*4*1024])
    d_out4=tf.matmul(d_out4,d_W4)
    return d_out4
image_size=64
n_channels=3
n_noise=100
n_samples=202599
is_training=False

if is_training==True:
    batch_size=16
    tf.reset_default_graph()
    with tf.name_scope('is_training'):
        z=tf.placeholder(tf.bool)
    with tf.name_scope('Noise'):
        x=tf.placeholder(tf.float32,[batch_size,n_noise])
    with tf.name_scope('Generator'):
        g_W0=tf.Variable(tf.random_uniform([n_noise,4*4*1024]),name='g_W0')
        g_beta0=tf.Variable(tf.constant(0.0,shape=[1024]),name='g_beta0')
        g_gamma0=tf.Variable(tf.random_normal([1024],stddev=0.02),name='g_gamma0')
        g_W1=tf.Variable(tf.truncated_normal([5,5,512,1024],stddev=0.05),name='g_W1')
        g_beta1=tf.Variable(tf.constant(0.0,shape=[512]),name='g_beta1')
        g_gamma1=tf.Variable(tf.random_normal([512],stddev=0.02),name='g_gamma1')
        g_W2=tf.Variable(tf.truncated_normal([5,5,256,512],stddev=0.05),name='g_W2')
        g_beta2=tf.Variable(tf.constant(0.0,shape=[256]),name='g_beta2')
        g_gamma2=tf.Variable(tf.random_normal([256],stddev=0.02),name='g_gamma2')
        g_W3=tf.Variable(tf.truncated_normal([5,5,128,256],stddev=0.05),name='g_W3')
        g_beta3=tf.Variable(tf.constant(0.0,shape=[128]),name='g_beta3')
        g_gamma3=tf.Variable(tf.random_normal([128],stddev=0.02),name='g_gamma3')
        g_W4=tf.Variable(tf.truncated_normal([5,5,3,128],stddev=0.05),name='g_W4')
    
        g_var=[g_W0,g_beta0,g_gamma0,
               g_W1,g_beta1,g_gamma1,
               g_W2,g_beta2,g_gamma2,
               g_W3,g_beta3,g_gamma3,
               g_W4]
        
        g_pred=generator(x)
    with tf.name_scope('Images'):
        y=tf.placeholder(tf.float32,[batch_size,image_size,image_size,n_channels])
    with tf.name_scope('Discriminator'):
        d_W0=tf.Variable(tf.truncated_normal([5,5,3,128],stddev=0.005),name='d_W0')
        d_beta0=tf.Variable(tf.constant(0.0,shape=[128]),name='d_beta0')
        d_gamma0=tf.Variable(tf.random_normal([128],stddev=0.02),name='d_gamma0')
        d_W1=tf.Variable(tf.truncated_normal([5,5,128,256],stddev=0.005),name='d_W1')
        d_beta1=tf.Variable(tf.constant(0.0,shape=[256]),name='d_beta1')
        d_gamma1=tf.Variable(tf.random_normal([256],stddev=0.02),name='d_gamma1')
        d_W2=tf.Variable(tf.truncated_normal([5,5,256,512],stddev=0.005),name='d_W2')
        d_beta2=tf.Variable(tf.constant(0.0,shape=[512]),name='d_beta2')
        d_gamma2=tf.Variable(tf.random_normal([512],stddev=0.02),name='d_gamma3')
        d_W3=tf.Variable(tf.truncated_normal([5,5,512,1024],stddev=0.005),name='d_W3')
        d_beta3=tf.Variable(tf.constant(0.0,shape=[1024]),name='d_beta3')
        d_gamma3=tf.Variable(tf.random_normal([1024],stddev=0.02),name='d_gamma3')
        d_W4=tf.Variable(tf.random_uniform([4*4*1024,1]),name='d_W4')
        
        d_var=[d_W0,d_beta0,d_gamma0,
               d_W1,d_beta1,d_gamma1,
               d_W2,d_beta2,d_gamma2,
               d_W3,d_beta3,d_gamma3,
               d_W4]
        
        d_fake_pred=discriminator(g_pred)
        d_real_pred=discriminator(y)
    with tf.name_scope('Loss'):    
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits =d_real_pred, labels = tf.ones_like(d_real_pred)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_pred, labels = tf.zeros_like(d_fake_pred)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_pred, labels = tf.ones_like(d_fake_pred)))
        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss',d_loss)
        tf.summary.scalar('g_loss',g_loss)
    with tf.name_scope('G_optimizer'):
        g_optimizer=tf.train.AdamOptimizer(0.0002,0.5).minimize(loss=g_loss,var_list=g_var)
    with tf.name_scope('D_optimizer'):
        d_optimizer=tf.train.AdamOptimizer(0.0002,0.5).minimize(loss=d_loss,var_list=d_var)
    
    init=tf.global_variables_initializer()  
    merge=tf.summary.merge_all()
    sess=tf.Session()
    sess.run(init)
    writer=tf.summary.FileWriter('./logs',sess.graph)
    saver=tf.train.Saver()
    
    ckpt=tf.train.get_checkpoint_state('./DCGAN_model')
    if ckpt!=None:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Model restored')
    else:
        print('Created a new model')
    for i in range(30000):
        xx,yy=get_batch(batch_size)
        sess.run(d_optimizer,{x:xx,y:yy,z:is_training})
        sess.run(g_optimizer,{x:xx,z:is_training})
        loss1=sess.run(d_loss,{x:xx,y:yy,z:is_training})
        loss2=sess.run(g_loss,{x:xx,z:is_training})
        print('d_loss=',loss1,'g_loss=',loss2)
        if i%100==0:
            merged=sess.run(merge,{x:xx,y:yy,z:is_training})
            writer.add_summary(merged,i)
            tmp=sess.run(g_pred,{x:xx,z:is_training})
            for j in tmp:
                save_img(j)
    
    saver.save(sess,'./DCGAN_model/model')
else:
    tf.reset_default_graph()
    batch_size=1
    with tf.name_scope('is_training'):
        z=tf.placeholder(tf.bool)
    with tf.name_scope('Noise'):
        x=tf.placeholder(tf.float32,[1,n_noise])
    with tf.name_scope('Generator'):
        g_W0=tf.Variable(tf.random_uniform([n_noise,4*4*1024]),name='g_W0')
        g_beta0=tf.Variable(tf.constant(0.0,shape=[1024]),name='g_beta0')
        g_gamma0=tf.Variable(tf.random_normal([1024],stddev=0.02),name='g_gamma0')
        g_W1=tf.Variable(tf.truncated_normal([5,5,512,1024],stddev=0.05),name='g_W1')
        g_beta1=tf.Variable(tf.constant(0.0,shape=[512]),name='g_beta1')
        g_gamma1=tf.Variable(tf.random_normal([512],stddev=0.02),name='g_gamma1')
        g_W2=tf.Variable(tf.truncated_normal([5,5,256,512],stddev=0.05),name='g_W2')
        g_beta2=tf.Variable(tf.constant(0.0,shape=[256]),name='g_beta2')
        g_gamma2=tf.Variable(tf.random_normal([256],stddev=0.02),name='g_gamma2')
        g_W3=tf.Variable(tf.truncated_normal([5,5,128,256],stddev=0.05),name='g_W3')
        g_beta3=tf.Variable(tf.constant(0.0,shape=[128]),name='g_beta3')
        g_gamma3=tf.Variable(tf.random_normal([128],stddev=0.02),name='g_gamma3')
        g_W4=tf.Variable(tf.truncated_normal([5,5,3,128],stddev=0.05),name='g_W4')
    
        g_pred=generator(x)
    with tf.name_scope('Discriminator'):
        d_W0=tf.Variable(tf.truncated_normal([5,5,3,128],stddev=0.005),name='d_W0')
        d_beta0=tf.Variable(tf.constant(0.0,shape=[128]),name='d_beta0')
        d_gamma0=tf.Variable(tf.random_normal([128],stddev=0.02),name='d_gamma0')
        d_W1=tf.Variable(tf.truncated_normal([5,5,128,256],stddev=0.005),name='d_W1')
        d_beta1=tf.Variable(tf.constant(0.0,shape=[256]),name='d_beta1')
        d_gamma1=tf.Variable(tf.random_normal([256],stddev=0.02),name='d_gamma1')
        d_W2=tf.Variable(tf.truncated_normal([5,5,256,512],stddev=0.005),name='d_W2')
        d_beta2=tf.Variable(tf.constant(0.0,shape=[512]),name='d_beta2')
        d_gamma2=tf.Variable(tf.random_normal([512],stddev=0.02),name='d_gamma3')
        d_W3=tf.Variable(tf.truncated_normal([5,5,512,1024],stddev=0.005),name='d_W3')
        d_beta3=tf.Variable(tf.constant(0.0,shape=[1024]),name='d_beta3')
        d_gamma3=tf.Variable(tf.random_normal([1024],stddev=0.02),name='d_gamma3')
        d_W4=tf.Variable(tf.random_uniform([4*4*1024,1]),name='d_W4')
        
        d_pred=discriminator(g_pred)
        
        sess=tf.Session()
        saver=tf.train.Saver()
        ckpt=tf.train.get_checkpoint_state('./DCGAN_model')
        saver.restore(sess,ckpt.model_checkpoint_path)
    
    for i in range(1000):
        noise=np.random.uniform(-1,1,[1,n_noise]).astype('float32')
        dis=sess.run(d_pred,{x:noise,z:is_training})
        sigmoid=1.0/(1.0+np.exp(-dis))
        print('similarity=',sigmoid)
        if sigmoid>0.001:
            tmp=sess.run(g_pred,{x:noise,z:is_training})
            tmp=np.reshape(tmp,[64,64,3])
            save_img(tmp)
        
    
    
    
    
    



























