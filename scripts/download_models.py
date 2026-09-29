"""Download pinned HF snapshots with streaming GET (works with HEAD-limited mirrors)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def ranged_download(url, path, size):
    chunk = 8 * 1024 * 1024
    def fetch(start):
        end = min(size - 1, start + chunk - 1)
        for attempt in range(3):
            try:
                r = requests.get(url, headers={"Range": "bytes=%d-%d" % (start, end)}, timeout=(20, 60))
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get("Content-Range", "").split("/")[0] != "bytes %d-%d" % (start, end) or len(r.content) != end - start + 1:
                    raise ValueError("Invalid range response")
                return start, r.content
            except requests.RequestException:
                if attempt == 2:
                    raise
        raise RuntimeError("Download failed")
    with path.open("wb") as f, ThreadPoolExecutor(max_workers=4) as pool:
        for i, (start, content) in enumerate(pool.map(fetch, range(0, size, chunk))):
            f.seek(start)
            f.write(content)
            if i % 16 == 0:
                print("Progress: %.1f%%" % (100 * (start + len(content)) / size), flush=True)


def download(repo):
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    response = requests.get(endpoint + "/api/models/" + repo, timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    response = requests.get(endpoint + "/api/models/" + repo + "/tree/" + revision, params={"recursive": "true"}, timeout=60)
    response.raise_for_status()
    out = ROOT / "data/models" / repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for item in sorted(response.json(), key=lambda x: (x["path"].endswith((".safetensors", ".bin")), x["path"])):
        name = item["path"]
        if not name.endswith((".json", ".safetensors", ".txt", ".model", "merges.txt")) and name != "pytorch_model.bin":
            continue
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != item["size"]:
            print("Downloading", repo, name, item["size"], flush=True)
            for attempt in range(3):
                try:
                    url = endpoint + "/" + repo + "/resolve/" + revision + "/" + name
                    if os.environ.get("QWEN_MODELSCOPE") == "1" and repo.startswith("Qwen/") and name.endswith(".safetensors"):
                        # Accept only byte-identical weights against the pinned HF LFS hash below.
                        url = "https://modelscope.cn/models/" + repo + "/resolve/master/" + name
                        ranged_download(url, path.with_suffix(path.suffix + ".part"), item["size"])
                        path.with_suffix(path.suffix + ".part").replace(path)
                        break
                    with requests.get(url,
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        with path.with_suffix(path.suffix + ".part").open("wb") as f:
                            for block in response.iter_content(4 * 1024 * 1024):
                                f.write(block)
                    path.with_suffix(path.suffix + ".part").replace(path)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        if path.stat().st_size != item["size"]:
            raise ValueError("Size mismatch: " + name)
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        expected = item.get("lfs", {}).get("oid")
        if expected and actual != expected:
            raise ValueError("SHA256 mismatch: " + name)
        records.append({"file": name, "sha256": actual})
    (out / "download_provenance.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": records}, indent=2))
    print("Ready:", out, flush=True)


if __name__ == "__main__":
    for repo in sys.argv[1:] or ["openai/clip-vit-base-patch32", "Qwen/Qwen2.5-VL-3B-Instruct"]:
        download(repo)
