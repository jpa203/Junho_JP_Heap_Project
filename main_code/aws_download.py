import boto3
import os

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='',  # e.g., 'us-east-1'
)

s3 = session.resource('s3')

bucket_name = ''

# Local directory to download the directory
local_directory = ''

# Download only pdf file from a specific directory from the s3 bucket
def download_dir(bucket_name, local_dir, prefix=""):
    bucket = s3.Bucket(bucket_name)
    
    for obj in bucket.objects.filter(Prefix=prefix):
        # Only process .pdf files
        if obj.key.endswith('.pdf'):
            local_file = os.path.join(local_dir, obj.key)
            
            if not os.path.exists(os.path.dirname(local_file)):
                os.makedirs(os.path.dirname(local_file))
            
            bucket.download_file(obj.key, local_file)

            
download_dir(bucket_name, local_directory, prefix="Deep_Learning/")

