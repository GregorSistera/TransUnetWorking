import os
from google.cloud import storage

def download_single_file(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

if __name__ == "__main__":
    bucket_name = "vit_models"
    source_blob_name = "imagenet21k/R50+ViT-B_16.npz"
    destination_file_name = "./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"

    download_single_file(bucket_name, source_blob_name, destination_file_name)
