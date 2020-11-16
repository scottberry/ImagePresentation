import os
import argparse
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
    parser.add_argument(
        '-o','--output_dir',
        default=os.getcwd(),
        help='where to save the composite image'
    )
    parser.add_argument(
        '-t','--thin',
        default=False,
        action='store_true',
        help='make the outlines thinner'
    )

    return(parser.parse_args())


def create_overlay_image_plot(
        channel_image, label_image,
        color='rgb(255, 191, 0)', thin=False):

    # threshold label_image and generate outlines
    label_image = np.array(label_image)
    if (thin):
        outlines = mh.labeled.bwperim(label_image > 0) * 255
    else:
        outlines = mh.morph.dilate(mh.labeled.bwperim(label_image > 0)) * 255
    overlay = Image.fromarray(np.uint8(outlines))
    outlines_transparent = Image.new(
        mode='RGBA', size=outlines.shape[::-1], color=(0, 0, 0, 0)
    )

    channel_image = channel_image.convert("RGBA")
    outlines_transparent.paste(channel_image, (0,0))
    outlines_transparent.paste(overlay, (0,0), mask=overlay)

    return outlines_transparent


def main(args):

    channel_image = Image.open(args.channel_image)
    segmentation = Image.open(args.segmentation)

    composite_filename = (
        os.path.splitext(
            os.path.basename(args.channel_image)
        )[0] + '_Overlay.png'
    )

    overlay_image = create_overlay_image_plot(channel_image,segmentation,thin=args.thin)
    overlay_image.save(os.path.join(args.output_dir,composite_filename))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
