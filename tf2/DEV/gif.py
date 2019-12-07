import os
import imageio as iio

dir_png = 'res/cineca_res/test2_cineca/'

imgs = []

for i in os.listdir(dir_png):
    if i.endswith('.png'):
        img_path = os.path.join(dir_png, i)
        imgs.append(iio.imread(img_path))

iio.mimsave(dir_png+'evolution.gif', imgs, fps=2)