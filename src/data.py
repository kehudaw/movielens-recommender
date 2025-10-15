import os, zipfile, requests

DATA_DIR = "data"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-100k")

def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXTRACT_DIR):
        print("Already present:", EXTRACT_DIR); return
    print("Downloading MovieLens 100K ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f: f.write(r.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf: zf.extractall(DATA_DIR)
    print("Extracted to", EXTRACT_DIR)

def main():
    try:
        download_movielens()
    except Exception as e:
        print("[WARN] Could not auto-download:", e)
        print("Manual: download ml-100k.zip from", URL, "and extract into data/")

if __name__ == "__main__":
    main()
