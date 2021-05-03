from google.cloud import storage
import os
import subprocess


# this file contains functions that handle download and upload on google cloud storage

def upload_blob(input_filename, source_file_path):
    # blob = file in google cloud storage
    """Uploads a file to the bucket."""
    bucket_name = "ese440_audios"
    destination_blob_name = input_filename

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)  # name the new file in google cloud

    blob.upload_from_filename(source_file_path)

    print(
        "File {} uploaded to {}.".format(
            input_filename, destination_blob_name
        )
    )

    return destination_blob_name


def parallel_upload_google_cloud(folder_path, bucket_address):
    # command: gsutil -m cp -r dir bucket-address
    bucket_address = 'gs://'+bucket_address
    subprocess.run(['gsutil', '-m', 'cp', '-r', folder_path, bucket_address])


def json_file_download(folder_path, bucket_address):
    bucket_address = 'gs://' + bucket_address
    subprocess.run(['gsutil', '-m', 'cp', '-r', bucket_address, folder_path])


# return all the file name in the storage bucket
def list_blobs(bucket_name):
    storage_client = storage.Client()
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

json_src_folder = "ese440_batch_output"
json_des_name = "/home/ese440/PycharmProjects/ESE440/annotated_json/"



