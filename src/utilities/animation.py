import argparse
from glob import glob
from PIL import Image
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    image_filenames = glob(f'{args.path}/*.png')

    df = pd.DataFrame(columns=['image_filename'])
    df['image_filename'] = image_filenames
    df['id'] = df['image_filename'].apply(lambda x: str(x).split('/')[-1].split('_')[0])
    df['organ'] = df['image_filename'].apply(lambda x: str(x).split('/')[-1].split('_')[1])
    df['fold'] = df['image_filename'].apply(lambda x: int(str(x).split('/')[-1].split('_')[2].replace('fold', '')))
    df['epoch'] = df['image_filename'].apply(lambda x: int(str(x).split('/')[-1].split('_')[3].replace('epoch', '')))

    for image_id, df_image in df.groupby('id'):
        df_image = df_image.sort_values(by='epoch', ascending=True).reset_index(drop=True)
        frames = (Image.open(f) for f in df_image['image_filename'])
        frame = next(frames)
        frame.save(
            fp=f'/home/gunes/Desktop/{image_id}.gif',
            format='GIF',
            append_images=frames,
            save_all=True,
            duration=30,
            loop=0
        )
