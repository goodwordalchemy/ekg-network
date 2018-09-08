from google.cloud import storage

BUCKET_NAME = 'ekg-network'

def get_bucket():
    client = storage.Client()

    bucket = client.get_bucket(BUCKET_NAME)

    return bucket


def list_blob_names():
    bucket = get_bucket()

    blobs = bucket.list_blobs()

    names = [b.name for b in blobs]

    return names




# # Then do other things...
# blob = bucket.get_blob('remote/path/to/file.txt')

# with open('tmp', 'wb') as f:
#     blob.download_to_file(f)



if __name__ == '__main__':
    print(list_blob_names())
