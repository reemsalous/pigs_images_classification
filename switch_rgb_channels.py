import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob

def swap_rgb_channels(filename):  
  filepath = filename
  orig_files = [file for file in glob.glob(filepath+"/*.png")]
  new_files = [os.path.join(filepath, os.path.basename(f)) for f in orig_files]

  for orig_f,new_f in zip(orig_files,new_files):
    img = cv2.imread(orig_f)
    print (img)
    #print('Shape:', img.shape)
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
  #cv2.imshow('blue_and_red_swapped', img)
  #cv2.waitKey()
  #cv2.destroyAllWindows()
    cv2.imwrite(new_f, img)


def main():
  for i in range(4):
    filename_train = "fli_train/" + str(i)
    filename_valid = "fli_val/" + str(i)
    swap_rgb_channels(filename_train)
    swap_rgb_channels(filename_valid)

if __name__ == '__main__':
  main()