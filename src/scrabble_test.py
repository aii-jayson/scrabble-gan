# import sys
# sys.path.extend(['/home/ubuntu/workspace/scrabble-gan'])

import os
import random

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import trange

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_random_word_list
from src.bigacgan.net_architecture import make_generator
from src.bigacgan.net_loss import hinge, not_saturating

gin.external_configurable(hinge)
gin.external_configurable(not_saturating)
gin.external_configurable(spectral_norm)

from src.dinterface.dinterface import init_reading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@gin.configurable
def setup_optimizer(g_lr, d_lr, r_lr, beta_1, beta_2, loss_fn, disc_iters):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    recognizer_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters


@gin.configurable('shared_specs')
def get_shared_specs(epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention):
    return epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention


@gin.configurable('io')
def setup_io(base_path, checkpoint_dir, gen_imgs_dir, gen_data_dir, model_dir, raw_dir, read_dir, input_dim, buf_size, n_classes,
             seq_len, char_vec, bucket_size):
    gen_path = base_path + gen_imgs_dir
    gen_data_pth = base_path + gen_data_dir
    ckpt_path = base_path + checkpoint_dir
    m_path = base_path + model_dir
    raw_dir = base_path + raw_dir
    read_dir = base_path + read_dir
    return input_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path,gen_data_pth, m_path, raw_dir, read_dir, char_vec


def main():
    # init params
    gin.parse_config_file('/root/scrabble-gan/src/scrabble_gan.gin') #'/Users/jsoncunanan/Documents/GitHub/scrabble-gan/src/scrabble_gan.gin'
    epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention = get_shared_specs()
    in_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path, gen_data_path, m_path, raw_dir, read_dir, char_vec = setup_io()

    # convert IAM Handwriting dataset (words) to GAN format
    if not os.path.exists(read_dir):
        print('converting iamDB-Dataset to GAN format...')
        init_reading(raw_dir, read_dir, in_dim, bucket_size)


    # load random words into memory (used for word generation by G)
    random_words = load_random_word_list(read_dir, bucket_size, char_vec)

    rnd_wrd = [char_vec.index(char) for char in 'withaMac'] #
    # load and preprocess dataset (python generator)

    # init generator, discriminator and recognizer
    generator = make_generator(latent_dim, in_dim, embed_y, gen_path, kernel_reg, g_bw_attention, n_classes)
    generator.load_weights(m_path + "generator_15.tf")

    seed = tf.random.normal([num_gen, latent_dim], seed=711)


    random_bucket_idx = random.randint(4, bucket_size - 1)
    #single preds
    #particular styles
    # style_1 = [seed[i, :] for i in [7,9,10,20,21, 25, 34, 41]]#[0,2,6,7,9,12,16,19]
    # style_1 = tf.convert_to_tensor(style_1)
    # labels = np.array([rnd_wrd for _ in range(1)], np.int32)
    # style_1 = [seed[9, :] for _ in range(1)]    # fixed style
    # style_1 = tf.convert_to_tensor(style_1)
    # test_input = [style_1, labels]

    #multiple preds
    labels = np.array([random.choice(random_words[random_bucket_idx]) for _ in range(num_gen)], np.int32)
    test_input = [seed, labels]

    predictions = generator(test_input, training=False)
    predictions = (predictions + 1) / 2.0

    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)

    def make_dataset(num_gen, num_batch=2):
        lines = []
        for _ in trange(num_batch):

            seed = tf.random.normal([num_gen, latent_dim])
            random_bucket_idx = random.randint(4, bucket_size - 1)
            labels = np.array([random.choice(random_words[random_bucket_idx]) for _ in range(num_gen)], np.int32)
            test_input = [seed, labels]

            preds = generator(test_input, training=False)
            preds = (preds + 1) / 2.0

            for i in trange(preds.shape[0]):
                filename = str("".join([char_vec[label] for label in labels[i]]))
                tf.keras.preprocessing.image.save_img(gen_data_path + filename + ".png", preds[i])
                lines.append( 'test/' + filename + '.png\t' + filename + '\n')

        anno = open(gen_data_path+"gt.txt", "w")
        anno.writelines(lines)
        anno.close()

    make_dataset(num_gen)
    # print(predictions.numpy()[0].shape)

if __name__ == "__main__":
    main()

    # plt.imshow(predictions[0, :, :, 0], cmap='gray') lines.append('test/' + filename + '.png\t' + filename + '\n')
    # plt.axis('off')
    #
    # plt.show()

    # x=0
    # y=81
    # for k in range(9):
    #
    #     for i in range(x,y):
    #         plt.subplot(9, 9, i + 1 - x)
    #         plt.imshow(predictions[i, :, :, 0], cmap='gray')
    #         plt.axis('off')
    #
    #     plt.show()
    #     x+=81
    #     y+=81

    # for i in range(predictions.shape[0]):
    #     plt.subplot(predictions.shape[0]//9, 9, i + 1)
    #     plt.imshow(predictions[i, :, :, 0], cmap='gray')
    #     plt.axis('off')
    #
    # plt.show()
