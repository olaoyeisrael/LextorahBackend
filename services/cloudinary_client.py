import cloudinary
import cloudinary.uploader
import cloudinary.api

from dotenv import load_dotenv

import os
load_dotenv()


cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
api_key = os.getenv('CLOUDINARY_API_KEY')
api_secret = os.getenv('CLOUDINARY_API_SECRET')


cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)
def upload_image(file_path, public_id=None, folder=None):
    options = {}
    if public_id:
        options['public_id'] = public_id
    if folder:
        options['folder'] = folder

    response = cloudinary.uploader.upload(file_path, **options)
    return response

def upload_video(filename: str):
    pass

def delete_image(public_id, options=None):
    response = cloudinary.uploader.destroy(public_id, **(options or {}))
    return response
def get_image_url(public_id, options=None):
    url = cloudinary.CloudinaryImage(public_id).build_url(**(options or {}))
    return url
def list_images(options=None):
    response = cloudinary.api.resources(**(options or {}))
    return response
def uploadMaterialToCloudinary(temp_file):
    result = cloudinary.uploader.upload(temp_file, folder="materials", resource_type="auto")
    print(result)
    return result['secure_url']

 