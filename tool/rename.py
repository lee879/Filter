import os

def batch_rename(folder_path, new_prefix):
    i = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            i += 1
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_prefix + str(i) + '.jpg'))

folder_path = '/path/to/folder'
new_prefix = 'new_image_'
batch_rename(folder_path, new_prefix)
