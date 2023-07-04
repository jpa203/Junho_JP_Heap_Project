import boto3
import os

session = boto3.Session(
    aws_access_key_id='AKIASPKJ6RJC6Q3CVKMU',
    aws_secret_access_key='9KcpuJcqjU+p40YbJ0qz/64Si6T52X4rp8OL/Qn0',
    region_name='us-east-1',  # e.g., 'us-east-1'
)

s3 = session.resource('s3')

bucket_name = 'junho-heap-project'
local_directory = '/Users/junhoeum/Desktop/experimentation'

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

