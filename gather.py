import yaml
import json
import os

# read yaml file
with open('splitter.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

# read the validation file
with open(config['splitter']['validation_path'], 'r') as file:
    # this is josn new line file
    validation = file.readlines()
    print(len(validation))
    # convert each line to json
    validation = [json.loads(line) for line in validation]

    # if original directory does not exist create it
    if not os.path.exists(config['gather']['original']):
        os.makedirs(config['gather']['original'])
    # if edited directory does not exist create it
    if not os.path.exists(config['gather']['edited']):
        os.makedirs(config['gather']['edited'])
    
    # for each line in validation copy the image that is in the dataset_path + source to the original directory
    for index, line in enumerate(validation):
        # get the source path
        source_path = os.path.join(config['gather']['dataset_path'], line['source'])
        # get the destination path
        destination_path = os.path.join(config['gather']['original'], f'{index + 1:03d}.png')
        # copy the image from source to destination
        os.system(f'cp {source_path} {destination_path}')


        # do the same for the edited image
        source_path = os.path.join(config['gather']['dataset_path'], line['target'])
        destination_path = os.path.join(config['gather']['edited'], f'{index + 1:03d}.png')
        os.system(f'cp {source_path} {destination_path}')

        print(f'Copied index {index}')
    print('Done')