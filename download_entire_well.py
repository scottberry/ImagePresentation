from tmclient import TmClient

dir_path = '/home/ubuntu/pelkmans-sc-storage/20170608-Kim2-EU-PCNA-Beads/TIFF_C07'
pword = xxx
uname = xxx
experiment = '20170608-Kim2-EU-PCNA-Beads'
well_name = 'C07'
plate_name = 'plate01'
channel_name = 'SE'


rest_api = TmClient(
    host='app.tissuemaps.org',
    port=80,
    experiment_name=experiment,
    username=uname,
    password=pword
)

sites = rest_api.get_sites(plate_name='plate01', well_name='C07')

for i in range(len(sites)):
    print sites[i]
    SE = rest_api.download_channel_image_file(
        channel_name=channel_name,
        plate_name=plate_name,
        well_name=well_name,
        well_pos_y=sites[i][u'y'],
        well_pos_x=sites[i][u'x'],
        correct=True,
        cycle_index=0,
        tpoint=0,
        zplane=0,
        directory=dir_path
    )
