import cv2
import imageio
import os
import argparse
from glob import glob


def main(args): 
    
    images_path = [image for image in glob(os.path.join(args.images_path,'*')) if image.endswith('.png') and 'loss' not in image]
    images_path = sorted(images_path, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    output_file = os.path.join(args.images_path, 'epoch_video.gif')
    images = list()
    for image_path in images_path: 
        image = cv2.imread(image_path)[45:440, 130:525, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    imageio.mimsave(output_file, images, duration=args.duration)

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='path of the image for making interpolation video')
    parser.add_argument('--duration', type=int, default=1, help='frames per second for the generated interpolation video')
    parser.add_argument('--save_path', type=str, default='.', help='path to save the generated interpolation video')
    args = parser.parse_args()
    
    main(args)

