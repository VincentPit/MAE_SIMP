import requests
import zipfile
import os

url = 'http://images.cocodataset.org/zips/unlabeled2017.zip'

zip_file_path = 'unlabeled2017.zip'
extract_dir = 'unlabeled2017' 


print('Starting download...')
response = requests.get(url, stream=True)
with open(zip_file_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)
print('Download complete.')

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

print('Extracting files...')
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print('Extraction complete.')

os.remove(zip_file_path)
print('ZIP file removed.')

print(f'All files extracted to {extract_dir}')
