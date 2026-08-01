import os
import sys
import csv
import zipfile
import urllib.request

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://docs.google.com/spreadsheets/'
}

def download_file(url: str, dest_path: str):
    """Downloads a file with proper HTTP headers and progress reporting."""
    print(f"\nDownloading {os.path.basename(dest_path)} from {url}...")
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.getheader('Content-Length', 0))
            bytes_downloaded = 0
            block_size = 8192 * 4

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                bytes_downloaded += len(buffer)
                percent = int(bytes_downloaded * 100 / total_size) if total_size > 0 else 0
                mb_downloaded = bytes_downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r Progress: {percent}% [{mb_downloaded:.2f} MB / {mb_total:.2f} MB]")
                sys.stdout.flush()

        print("\n[+] Download complete!")
        return True
    except Exception as e:
        print(f"\n[-] Failed to download: {e}")
        return False

def unzip_file(zip_path: str, extract_to: str):
    """Extracts a zip file to the specified directory."""
    print(f"Extracting {os.path.basename(zip_path)} -> {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[+] Extraction complete!")
        return True
    except Exception as e:
        print(f"[-] Extraction failed: {e}")
        return False

def main():
    csv_file = "spreadsheet_data.csv"
    if not os.path.exists(csv_file):
        print("Spreadsheet CSV file not found!")
        return

    data_dir = os.path.abspath("data")
    downloads_dir = os.path.join(data_dir, "zips")
    os.makedirs(downloads_dir, exist_ok=True)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    # Essential files to download first
    essential_files = [
        "clip-features-32-aic25-b1.zip",
        "map-keyframes-aic25-b1.zip",
        "media-info-aic25-b1.zip",
        "objects-aic25-b1.zip"
    ]

    print("=" * 60)
    print("  AIC 2026 DATASET AUTOMATED DOWNLOADER")
    print("=" * 60)

    for row in rows:
        if len(row) >= 3:
            filename = row[1]
            url = row[2]
            if filename in essential_files:
                dest_zip = os.path.join(downloads_dir, filename)
                if not os.path.exists(dest_zip):
                    success = download_file(url, dest_zip)
                else:
                    print(f"\nFile {filename} already downloaded. Skipping download.")
                    success = True

                if success:
                    unzip_file(dest_zip, data_dir)

    print("\n" + "=" * 60)
    print("  ESSENTIAL DATASETS DOWNLOADED & EXTRACTED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
