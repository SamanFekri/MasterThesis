import pandas as pd
import os
import time
from tqdm import tqdm
import json

input_dir = '../pix2pix_lite/downlaods'
output_dir = {
    'dir': 'output',
    'source': 'source',
    'target': 'target',
    'prompt': 'prompt.json',
}
extension = '.parequet'
# read file in the directory filter it only file with extension
print(os.listdir(input_dir))
files = [file for file in os.listdir(input_dir) if not file.endswith('.json', '.lock')]

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
z = 0
result = []
for file in files_pbar:
    print(file)
    if z == 60:
        break
    z += 1
    files_pbar.set_postfix(file=file)
    # read the parquet file
    df = pd.read_parquet(f'{input_dir}/{file}')
    # make a progress bar for the rows
    rows_pbar = tqdm(list(df.iterrows()), desc='Processing Rows', leave=False)
    for index, row in rows_pbar:
        # save the bytes and use the path for filename input
        output_path = os.path.join(output_dir['dir'], output_dir['source'], row['original_image']['path'])

        with open(output_path, 'wb') as f:
            f.write(row['original_image']['bytes'])
        # save the bytes and use the path for filename target
        output_path = os.path.join(output_dir['dir'], output_dir['target'], row['edited_image']['path'])
        
        with open(output_path, 'wb') as f:
            f.write(row['edited_image']['bytes'])
        result.append({
            'source': f"{output_dir['source']}/{row['original_image']['path']}",
            'target': f"{output_dir['target']}/{row['edited_image']['path']}",
            'prompt': row['edit_prompt'],
            'original_prompt': row['original_prompt'],
            'edited_prompt': row['edited_prompt'],
        })
    # remove the file after processing
    os.remove(f'{input_dir}/{file}')
    
    # save result as jsonnl in output_dir promot
    with open(os.path.join(output_dir['dir'], output_dir['prompt']), 'w') as f:
        for line in result:
            # make sure line when converts to json use " for the key and value not '
            line = json.dumps(line)
            f.write(f"{line}\n")
