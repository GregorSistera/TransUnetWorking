import os
from google.cloud import storage


def download_blob(bucket, blob, destination_folder):
    # Create local directories if needed
    local_path = os.path.join(destination_folder, blob.name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob.name} to {local_path}")


def download_bucket(bucket_name, destination_folder):
    # Create anonymous client for public bucket access
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs()  # lists all files

    for blob in blobs:
        # Skip "folder placeholders" (zero size)
        if not blob.name.endswith('/'):
            download_blob(bucket, blob, destination_folder)


if __name__ == "__main__":
    bucket_name = "vit_models"
    destination_folder = "./model"
    download_bucket(bucket_name, destination_folder)
