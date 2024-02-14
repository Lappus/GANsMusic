import tensorflow as tf

from Generator import *
from Discriminator import *

class GAN(tf.keras.Model):

    def __init__(self):
        super(GAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def WGAN_GP_train_d_step(self, real_image, batch_size, step):
        LAMBDA = 10
        noise = tf.random.normal([batch_size, self.generator.noise_dim])
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        ###################################
        # Train D
        ###################################
        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = self.generator(noise, training=True)
                fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
                fake_mixed_pred = self.discriminator(fake_image_mixed, training=True)
                
            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            
            fake_pred = self.discriminator(fake_image, training=True)
            real_pred = self.discriminator(real_image, training=True)
            
            D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
        # Calculate the gradients for discriminator
        D_gradients = d_tape.gradient(D_loss, self.discriminator.trainable_variables)
        # Apply the gradients to the optimizer
        self.discriminator.optimizer.apply_gradients(zip(D_gradients, self.discriminator.trainable_variables))

        # Write loss values to tensorboard
        #if step % 10 == 0:
        #    with train_summary_writer.as_default():
        #        tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)

    @tf.function
    def WGAN_GP_train_g_step(self, real_image, batch_size, step):
        LAMBDA = 10
        noise = tf.random.normal([batch_size, self.generator.noise_dim])
        ###################################
        # Train G
        ###################################
        with tf.GradientTape() as g_tape:
            fake_image = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_image, training=True)
            G_loss = -tf.reduce_mean(fake_pred)
        # Calculate the gradients for generator
        G_gradients = g_tape.gradient(G_loss, self.generator.trainable_variables)
        # Apply the gradients to the optimizer
        self.generator.optimizer.apply_gradients(zip(G_gradients,self.generator.trainable_variables))
        # Write loss values to tensorboard
        #if step % 10 == 0:
        #    with train_summary_writer.as_default():
        #        tf.summary.scalar('G_loss', G_loss, step=step)
            
