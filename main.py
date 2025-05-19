import os
from webdav3.client import Client

# خواندن متغیرهای محیطی
options = {
    'webdav_hostname': os.getenv("WEBDAV_URL"),
    'webdav_login':    os.getenv("WEBDAV_USER"),
    'webdav_password': os.getenv("WEBDAV_PASSWORD")
}

client = Client(options)

# مثال: فهرست محتویات ریشه
print("Contents of root:")
print(client.list('/'))

# مثال: آپلود یک فایل محلی به WebDAV
local_path  = 'localfile.txt'
remote_path = '/remote/localfile.txt'
client.upload_sync(remote_path=remote_path, local_path=local_path)
print(f"Uploaded {local_path} → {remote_path}")

# مثال: دانلود یک فایل از WebDAV
client.download_sync(remote_path=remote_path, local_path='downloaded.txt')
print(f"Downloaded {remote_path} → downloaded.txt")
