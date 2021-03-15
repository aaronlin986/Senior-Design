# Reference
# https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries?authuser=2#client-libraries-install-python
from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket_name = "your-bucket-name"
    source_file_name = "local/path/to/file"
    destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech_v1p1beta1 as speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)

    # if it is a MP3 file, the encoding and the sample_rate have to be specified
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    result = operation.result(timeout=90)

    for result in result.results:
        alternative = result.alternatives[0]
        print("Transcript: {}".format(alternative.transcript))
        print("Confidence: {}".format(alternative.confidence))

        # https://cloud.google.com/speech-to-text/docs/basics#time-offsets
        # Map the time and word to the graph time axis
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time

            print(
                f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}"
            )

gci_uri = "gs://ese440_audios/my_result.mp3"
transcribe_gcs(gci_uri)

def run:


if __name__ == "__main__":
    run()