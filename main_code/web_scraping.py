import urllib.request as libreq
import feedparser
import requests
import io
import configparser
import boto3
import tempfile


parser = configparser.ConfigParser()
parser.read('pipeline.conf')
access_key = parser.get("aws_boto_credentials","access_key")
secret_key = parser.get("aws_boto_credentials", "secret_key")
bucket_name = parser.get("aws_boto_credentials", "bucket_name")

s3 = boto3.client('s3', aws_access_key_id = access_key, 
                  aws_secret_access_key = secret_key)



# def pdf_etl():

# Base api query url
base_url = 'http://export.arxiv.org/api/query?'

# Search parameters
search_query = 'all:deeplearning' # search for electron in all fields
start = 0                    # retreive the first 5 results
max_results = 100

query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                    start,
                                                    max_results)

# perform a GET request using the base_url and query

with libreq.urlopen(base_url+query) as url:
    response = url.read()

# parse the response using feedparser
feed = feedparser.parse(response)

# print out feed information
print ('Feed title: %s' % feed.feed.title)
print ('Feed last updated: %s' % feed.feed.updated)

# print opensearch metadata
print ('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
print ('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
print ('startIndex for this query: %s'   % feed.feed.opensearch_startindex)
print()

pdf_lst = []

# Run through each entry, and print out information
for entry in feed.entries:
    print ('e-print metadata')
    print ('arxiv-id: %s' % entry.id.split('/abs/')[-1])
    print ('Published: %s' % entry.published)
    print ('Title:  %s' % entry.title)
    print ('Primary Category: %s' % entry.tags[0]['term'])
    print()


    # get the links to the abs page and pdf for this e-print

    for link in entry.links:
        if link.rel == 'alternate':
            print ('abs page link: %s' % link.href)
        elif link.title == 'pdf':

            pdf = link.href

            pdf_lst.append(pdf)
    
            print ('pdf link: %s' % link.href)



i = 1

for doc in pdf_lst:
    response = requests.get(doc)
    pdf_content = response.content
    pdf_file = io.BytesIO(pdf_content)

    with open(f"/Users/jazzopardi/Desktop/Juno Project/venvName/pdf_data/{search_query}_file_{i}.pdf", "wb") as f:
        f.write(pdf_file.read())



    i += 1




# for doc in pdf_lst:
#     response = requests.get(doc)
#     pdf_content = response.content
#     pdf_file = io.BytesIO(pdf_content)
    
#     with tempfile.TemporaryFile() as temp_file:
#         try: 
#             bucket_name = 'junho-heap-project'
#             destination_folder = 'Deep_Learning/'
#             temp_file.write(pdf_content)
#             temp_file.seek(0)
#             s3.upload_fileobj(temp_file, bucket_name, f'{destination_folder}deep_learning{i}.pdf')
#             print(f'deep_learning{i} - success')
#             i += 1 
#         except Exception as e:
#             print(e)




