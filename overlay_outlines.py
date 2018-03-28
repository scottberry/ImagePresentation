import argparse
import imageio
import mahotas as mh
import numpy as np
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='overlay_outlines',
        description=('Takes a channel image and a label image'
                     ' and overlays the segmentation as outlines'
                     )
    )
    parser.add_argument('channel_image', help='path to channel image')
    parser.add_argument('segmentation', help='path to label image')

    return(parser.parse_args())


def create_overlay_image_plot(
        channel_image, label_image,
        color='rgb(255, 191, 0)', thickness=3):

    # threshold label_image
    label_image = np.array(label_image)
    thresh_image = label_image > 0
    outlines = mh.morph.dilate(mh.labeled.bwperim(thresh_image))

    channel_image = channel_image.convert("RGBA")
    overlay = Image.fromarray(np.uint8(outlines))
    overlay = overlay.convert("RGBA")

    channel_image.paste(overlay)

    return channel_image

def main(args):

    channel_image = Image.open(args.channel_image)
    segmentation = Image.open(args.segmentation)

    overlay_image = create_overlay_image_plot(channel_image,segmentation)
    overlay_image.save('tmp.png')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
