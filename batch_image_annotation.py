from google.cloud import vision,storage

from file_uploader import list_blobs
# loads the google cloud credentials


def configure():
    import os
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ese440/PycharmProjects/ESE440/resources/ese440-ee15808ee7b1.json"


def sample_async_batch_annotate_images(
    output_uri="gs://ese440_batch_output/",
):
    """Perform async batch image annotation."""
    image_folder_path="gs://ese440_test_images/"  # path to the image folder
    bucket_name = "ese440_test_images"
    client = vision.ImageAnnotatorClient()

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    requests = []
    features = [
        {"type_": vision.Feature.Type.FACE_DETECTION}, # detect only face
    ]
    for blob in blobs:
        image = {"source": {"image_uri": image_folder_path+blob.name}}
        requests.append({'image': image, "features": features})

    gcs_destination = {"uri": output_uri}
    # The max number of responses to output in each JSON file
    batch_size = 100
    output_config = {"gcs_destination": gcs_destination,
                     "batch_size": batch_size}
    print('copy maybe?')
    operation = client.async_batch_annotate_images(requests=requests, output_config=output_config)

    print("Waiting for operation to complete...")
    response = operation.result(90)
    # The output is written to GCS with the provided output_uri as prefix
    gcs_output_uri = response.output_config.gcs_destination.uri
    print("Output written to GCS with prefix: {}".format(gcs_output_uri))


configure()
sample_async_batch_annotate_images()
