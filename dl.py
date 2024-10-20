import os
import requests
from bs4 import BeautifulSoup
import time

def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")

def main():
    base_url = "https://ftp.tugraz.at/outgoing/ITSG/GRACE/ITSG-Grace2018/monthly/monthly_n96/"

    # Get the directory listing
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all .gfc files in the directory
    gfc_files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href.endswith('.gfc'):
            full_path = os.path.join(base_url, href)
            gfc_files.append((full_path, href))

    # Download each file
    for url, filename in gfc_files:
        download_file(url, filename)

if __name__ == "__main__":
    main()
