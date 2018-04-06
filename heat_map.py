import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
from tmclient import TmClient
from operator import itemgetter
from PIL import Image

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s | %(levelname)s'
           ' | %(module)s %(funcName)s | %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='heat_map',
        description=('Generates a heatmap for a sinmgle site'
                     ' from a processed dataset on a TissueMAPS'
                     ' instance.')
    )
    parser.add_argument(
        '-H', '--host', required=True,
        help='name of TissueMAPS server host'
    )
    parser.add_argument(
        '-P', '--port', type=int, default=80,
        help='number of the port to which the server listens (default: 80)'
    )
    parser.add_argument(
        '-u', '--user', dest='username', required=True,
        help='name of TissueMAPS user'
    )
    parser.add_argument(
        '--password',
        help='password of TissueMAPS user'
    )
    parser.add_argument(
        '-v', '--verbosity', action='count', default=0,
        help='increase logging verbosity'
    )
    parser.add_argument(
        '-x', '--well-pos-x', metavar='X', dest='well_pos_x',
        type=int, required=True,
        help='zero-based x cooridinate of acquisition site within the well'
    )
    parser.add_argument(
        '-y', '--well-pos-y', metavar='Y', dest='well_pos_y',
        type=int, required=True,
        help='zero-based y cooridinate of acquisition site within the well'
    )
    parser.add_argument(
        '-w', '--well', metavar='WELL', dest='well_name', required=True,
        help='name of the well'
    )
    parser.add_argument(
        '-p', '--plate', metavar='PLATE', dest='plate_name', required=True,
        help='name of the plate'
    )
    parser.add_argument(
        '-o', '--object-type', metavar='OBJECT-TYPE',
        dest='mapobject_type_name', required=True,
        help='name of the object type'
    )
    parser.add_argument(
        '-f', '--feature', metavar='FEATURE',
        dest='feature_name', required=True,
        help='name of the feature'
    )
    parser.add_argument(
        '--output_dir', metavar='PATH',
        dest='output_dir', required=False, default=os.getcwd(),
        help='destination directory to save the heat map'
    )
    parser.add_argument(
        '-e', '--experiment', metavar='EXPERIMENT',
        dest='experiment_name',
        required=True, help='name of the experiment'
    )
    parser.add_argument(
        '--scale_max', metavar='SCALE_MAX',
        dest='scale_max', default=None,
        required=False, help='maximum value on heatmap'
    )
    parser.add_argument(
        '--scale_min', metavar='SCALE_MIN',
        dest='scale_min', default=None,
        required=False, help='minimum value on heatmap'
    )

    return(parser.parse_args())


def main(args):

    logger.debug('Connecting to tissuemaps host')
    tm = TmClient(
        host=args.host,
        port=args.port,
        experiment_name=args.experiment_name,
        username=args.username,
        password=args.password
    )
    logger.debug('Downloading feature values')
    feature_values = tm.download_feature_values(
        mapobject_type_name=args.mapobject_type_name,
        plate_name=args.plate_name,
        well_name=args.well_name,
        well_pos_y=args.well_pos_y,
        well_pos_x=args.well_pos_x
    )
    logger.debug('Downloading metadata')
    object_metadata = tm.download_object_metadata(
        mapobject_type_name=args.mapobject_type_name,
        plate_name=args.plate_name,
        well_name=args.well_name,
        well_pos_y=args.well_pos_y,
        well_pos_x=args.well_pos_x
    )
    logger.debug('Downloading segmentation')
    label_image = tm.download_segmentation_image(
        mapobject_type_name=args.mapobject_type_name,
        plate_name=args.plate_name,
        well_name=args.well_name,
        well_pos_y=args.well_pos_y,
        well_pos_x=args.well_pos_x
    )

    logger.debug('Checking that feature exists')
    if args.feature_name not in set(feature_values.columns.get_values()):
        logger.error('%s not found in list of feature names for this experiment', args.feature_name)
        logger.error('\n'.join(list(feature_values.columns.get_values())))

    logger.debug('Combining feature values and metadata')
    values = feature_values[args.feature_name]
    metadata = object_metadata[['label','is_border']]
    combined = pd.concat([values.reset_index(drop=True), metadata], axis=1)

    logger.debug('Extracting border and non-border cells')
    border_cells = combined[combined['is_border'] == 1][['label',args.feature_name]].set_index('label').T.to_dict('list')
    non_border_cells = combined[combined['is_border'] == 0][['label',args.feature_name]].set_index('label').T.to_dict('list')

    logger.debug('Generating heatmap')
    heat_map = np.copy(label_image)
    for key, value in border_cells.iteritems():
        heat_map[label_image == key] = 0
    for key, value in non_border_cells.iteritems():
        heat_map[label_image == key] = value

    min_value = min(non_border_cells.iteritems(), key=itemgetter(1))[1][0]
    max_value = max(non_border_cells.iteritems(), key=itemgetter(1))[1][0]
    logger.info('Minimum value of %s is %s', args.feature_name, min_value)
    logger.info('Maximum value of %s is %s', args.feature_name, max_value)

    min_rescale_value = min_value if args.scale_min is None else float(args.scale_min)
    max_rescale_value = max_value if args.scale_max is None else float(args.scale_max)

    logger.debug('Re-scaling and converting heatmap to 8-bit tiff')
    heat_map = heat_map.astype('float64')
    heat_map -= min_rescale_value
    heat_map /= max_rescale_value
    heat_map = mpl.cm.viridis(heat_map)  # RGBA

    logger.debug('Colouring border cells')
    for channel in range(0,3):
        for key, value in border_cells.iteritems():
            heat_map[:,:,channel][label_image == key] = 0.2
    logger.debug('Colouring background')
    for channel in range(0,3):
        heat_map[:,:,channel][label_image == 0] = 0.0
    heat_map = np.uint8(heat_map * 255.0)
    heat_map_image = Image.fromarray(heat_map)

    heat_map_basename = (
        args.experiment_name +
        '_' + args.plate_name +
        '_' + args.well_name +
        '_y' + str(args.well_pos_y) +
        '_x' + str(args.well_pos_x) +
        '_z000' +
        '_t000' + '_HeatMap_' +
        args.mapobject_type_name +
        '_' + args.feature_name
    )
    heat_map_path = os.path.join(
        args.output_dir,
        heat_map_basename + '.tif'
    )
    logger.debug('Saving image to %s',heat_map_path)
    heat_map_image.save(heat_map_path)

    logger.debug('Generating heat map legend')
    fig = plt.figure(figsize=(1, 4))
    ax1 = fig.add_axes([0.05, 0.80, 0.4, 0.9])
    norm = mpl.colors.Normalize(vmin=min_rescale_value,
                                vmax=max_rescale_value)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=mpl.cm.viridis,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label(args.feature_name)

    heat_map_legend_path = os.path.join(
        args.output_dir,
        heat_map_basename + '_ColorScale.tif'
    )
    fig.savefig(heat_map_legend_path, bbox_inches='tight')

    return


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
