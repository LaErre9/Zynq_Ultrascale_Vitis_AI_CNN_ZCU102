import torch
import torchvision

import argparse
import os
import shutil
import sys
import cv2
import random, PIL, numpy
import numpy as np
from tqdm import tqdm

from common import gen_transform


DIVIDER = '-----------------------------------------'


def generate_images(dset_dir, num_images, dest_dir):

  classes = ['0','1','2','3','4','5','6','7','8','9',
             '10','11','12','13','14','15','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29',
             '30','31','32','33','34','35','36','37','38','39',
             '40','41','42']

  # Ottenere la lista di tutte le immagini nella cartella
  all_images = []
  for filename in os.listdir(dset_dir):
      dir_path = os.path.join(dset_dir, filename)
      for image in os.listdir(dir_path):
          image_path = os.path.join(dir_path, image)
          all_images.append((image_path, int(filename)))  # Salva il percorso dell'immagine e l'etichetta associata

  # Mescola le immagini in modo casuale
  random.shuffle(all_images)

  # Selezionare casualmente un campione di immagini
  random_images = random.sample(all_images, num_images)

  i = 0
  for image_path,true_label in random_images:
    img = numpy.array(PIL.Image.open(image_path))
    img_file=os.path.join(dest_dir, classes[true_label]+'_'+str(i)+'.jpg')
    cv2.imwrite(img_file, img)
    i += 1

  return



def make_target(build_dir,target,num_images,app_dir):

    dset_dir = build_dir + '/dataset/images'
    comp_dir = build_dir + '/compiled_model'
    target_dir = build_dir + '/target_' + target

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_traffic_sign.py'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/CNN_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)

    # create images
    dest_dir = target_dir + '/images'
    shutil.rmtree(dest_dir, ignore_errors=True)  
    os.makedirs(dest_dir)
    generate_images(dset_dir, num_images, dest_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',  type=str,  default='build', help='Path to build folder. Default is build')
    ap.add_argument('-t', '--target',     type=str,  default='zcu102', choices=['zcu102','zcu104','u50','vck190'], help='Target board type (zcu102,zcu104,u50,vck190). Default is zcu102')
    ap.add_argument('-n', '--num_images', type=int,  default=10000, help='Number of test images. Default is 10000')
    ap.add_argument('-a', '--app_dir',    type=str,  default='application', help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --target       : ', args.target)
    print (' --num_images   : ', args.num_images)
    print (' --app_dir      : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir, args.target, args.num_images, args.app_dir)


if __name__ ==  "__main__":
    main()
