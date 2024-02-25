import shutil, os
from fastapi import UploadFile


def save_file(file: UploadFile):
    file_location = f'files/{file.filename}'
    with open(file_location, 'wb+') as file_object:
        shutil.copyfileobj(file.file, file_object)
    return file_location


def delete_file(file_path):
    os.remove(file_path)