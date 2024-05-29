import json
import yaml

with open('splitter.yaml', 'r') as file:
    config = yaml.safe_load(file)
    print(config)


def split_json_file(input_file, train_file, validation_file, num_validation_items):
    """
    Splits a JSON file with newline-separated entries into train and validation files.

    Args:
        input_file (str): Path to the input JSON file.
        train_file (str): Path to the output train file.
        validation_file (str): Path to the output validation file.
        num_validation_items (int): Number of items desired in the validation set.
    """

    with open(input_file, 'r') as f_in:
        json_lines = f_in.readlines()

    num_lines = len(json_lines)
    if num_validation_items > num_lines:
        raise ValueError("Number of validation items cannot exceed total number of lines")

    # Randomly shuffle the lines to ensure unbiased selection
    import random
    random.shuffle(json_lines)

    with open(train_file, 'w') as f_train, open(validation_file, 'w') as f_val:
        f_train.writelines(json_lines[:-num_validation_items])
        f_val.writelines(json_lines[-num_validation_items:])


# Example usage
input_file = config['splitter']['data_path']
train_file = config['splitter']['train_path']
validation_file = config['splitter']['validation_path']
num_validation_items = config['splitter']['validation_size']

split_json_file(input_file, train_file, validation_file, num_validation_items)
print(f"Split complete! Train data in '{train_file}', validation data in '{validation_file}'.")

# read the validation file and tell me number of lines
with open(validation_file, 'r') as f:
    json_lines = f.readlines()
    print(f"Number of lines in validation file: {len(json_lines)}")

# read the train file and tell me number of lines
with open(train_file, 'r') as f:
    json_lines = f.readlines()
    print(f"Number of lines in train file: {len(json_lines)}")
