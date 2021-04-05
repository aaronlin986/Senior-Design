from google.cloud import storage
import os
import subprocess


def upload_blob(input_filename,source_file_path):
    """Uploads a file to the bucket."""
    bucket_name = "ese440_test_images"
    # source_file_name = "local/path/to/file"
    destination_blob_name = input_filename # blob = file in google cloud storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)  # name the new file in google cloud

    blob.upload_from_filename(source_file_path)

    print(
        "File {} uploaded to {}.".format(
            input_filename, destination_blob_name
        )
    )


def parallel_upload_google_cloud(folder_path, bucket_address):
    # command: gsutil -m cp -r dir bucket-address
    # upload_command = 'gsutil -m cp -r {0} gs://{1}'.format(folder_path, bucket_address)
    bucket_address = 'gs://'+bucket_address
    subprocess.run(['gsutil', '-m', 'cp', '-r', folder_path, bucket_address])


# return all the file name in the storage bucket
def list_blobs(bucket_name):
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)


# loads the google cloud credentials
def configure():
    import os
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"


configure()  # configure is required for all google cloud related code
gs_bucket_name = "ese440_test_images"
folder_name = "/home/ese440/PycharmProjects/ESE440/image_samples/"

# parallel_upload_google_cloud(folder_name,bucket_name)
parallel_upload_google_cloud(folder_name,gs_bucket_name)

# read_images(folder_name)
