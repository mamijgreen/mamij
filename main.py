from webdav3.client import Client

# ——— پیکربندی اتصال ———
options = {
    # آدرس پایه WebDAV (بدون مسیر ریشه)
    'webdav_hostname': 'https://gift.nodisk.ir',
    # مسیر ریشه باکت توی WebDAV
    'webdav_root':    '/remote.php/dav/files/09962165421',
    # نام کاربری نودیسک (معمولاً شماره موبایل یا ایمیل)
    'webdav_login':   '09962165421',
    # پسورد اکانت نودیسک
    'webdav_password':'156907'
}

client = Client(options)

# ——— لیست محتویات ریشه ———
print("Contents of root:")
for item in client.list('/'):
    print(" -", item)

# ——— آپلود یک فایل ———
local_path  = 'localfile.txt'           # فایل محلی که می‌خوای آپلود کنی
remote_path = '/uploads/localfile.txt'  # مسیری که توی WebDAV می‌خوای ذخیره بشه
client.upload_sync(remote_path=remote_path, local_path=local_path)
print(f"Uploaded {local_path} → {remote_path}")

# ——— دانلود یک فایل ———
download_dest = 'downloaded.txt'
client.download_sync(remote_path=remote_path, local_path=download_dest)
print(f"Downloaded {remote_path} → {download_dest}")
