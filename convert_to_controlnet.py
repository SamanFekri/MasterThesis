import pandas as pd
import os
import time
from tqdm import tqdm

input_dir = '.'
output_dir = {
    'dir': 'output',
    'source': 'source',
    'target': 'target',
    'prompt': 'prompt.json',
}
extension = '.parquet'
# read file in the directory filter it only file with extension
files = [file for file in os.listdir(input_dir) if file.endswith(extension)]

# make the output directory if it doesn't exist
if not os.path.exists(output_dir['dir']):
    os.mkdir(output_dir['dir'])
# make the output/input directory if it doesn't exist
if not os.path.exists(os.path.join(output_dir['dir'], output_dir['source'])):
    os.mkdir(os.path.join(output_dir['dir'], output_dir['source']))
# make the output/target directory if it doesn't exist
if not os.path.exists(os.path.join(output_dir['dir'], output_dir['target'])):
    os.mkdir(os.path.join(output_dir['dir'], output_dir['target']))


# show a progress bar and iterate over the files and update the progress bar with file name
files_pbar = tqdm(files, desc='Processing Files')
result = []
for file in files_pbar:
    files_pbar.set_postfix(file=file)
    # read the parquet file
    df = pd.read_parquet(file)
    # make a progress bar for the rows
    rows_pbar = tqdm(list(df.iterrows()), desc='Processing Rows', leave=False)
    for index, row in rows_pbar:
        # save the bytes and use the path for filename input
        output_path = os.path.join(output_dir['dir'], output_dir['source'], row['input_image']['path'])
        with open(output_path, 'wb') as f:
            f.write(row['input_image']['bytes'])
        # save the bytes and use the path for filename target
        output_path = os.path.join(output_dir['dir'], output_dir['target'], row['edited_image']['path'])
        
        with open(output_path, 'wb') as f:
            f.write(row['edited_image']['bytes'])
        result.append({
            'source': f"{output_dir['source']}/{row['input_image']['path']}",
            'target': f"{output_dir['target']}/{row['edited_image']['path']}",
            'prompt': row['edit_prompt'],
        })

# save result as jsonnl in output_dir promot
with open(os.path.join(output_dir['dir'], output_dir['prompt']), 'w') as f:
    for line in result:
        f.write(f"{line}\n")



# df = pd.read_parquet('t1.parquet')
# print(df.columns)

# print(df['input_image'][0]['bytes'])
# save the bytes and use the path for filename
# with open(df['edited_image'][0]['path'], 'wb') as f:
    # f.write(df['edited_image'][0]['bytes'])
# read the bytes of the image and convert them to jpg bytes
# img = Image.frombytes('RGB', df['input_image'][0]['bytes'], 'raw')
# img.show()

# img.show()