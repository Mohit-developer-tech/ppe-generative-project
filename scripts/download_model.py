import os
import urllib.request
import time

dest_dir = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--distilbert-base-uncased",
    "snapshots",
    "12040accade4e8a0f71eabdb258fecc2e7e948be"
)
os.makedirs(dest_dir, exist_ok=True)
dest_file = os.path.join(dest_dir, "model.safetensors")

if os.path.exists(dest_file) and os.path.getsize(dest_file) > 250000000:
    print(f"File already downloaded: {dest_file} ({os.path.getsize(dest_file)} bytes)")
    exit(0)

url = "https://huggingface.co/distilbert-base-uncased/resolve/main/model.safetensors"
print(f"Downloading from {url} to {dest_file}...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

t0 = time.time()
with urllib.request.urlopen(req, timeout=60) as response, open(dest_file + ".tmp", "wb") as out_file:
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB
    last_print = time.time()
    
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        out_file.write(chunk)
        downloaded += len(chunk)
        if time.time() - last_print > 3:
            pct = (downloaded / total_size) * 100 if total_size else 0
            speed = (downloaded / (time.time() - t0)) / (1024 * 1024)
            print(f"Downloaded: {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%) - Speed: {speed:.2f} MB/s", flush=True)
            last_print = time.time()

if os.path.exists(dest_file):
    os.remove(dest_file)
os.rename(dest_file + ".tmp", dest_file)
print(f"Download complete in {time.time()-t0:.1f}s! File size: {os.path.getsize(dest_file)} bytes", flush=True)

