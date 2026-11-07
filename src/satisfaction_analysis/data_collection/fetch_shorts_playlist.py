# -*- coding: utf-8 -*-
import subprocess
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = PROJECT_ROOT / "data/raw/chat/shorts_playlist_all.jsonl"
YTDLP_BIN = PROJECT_ROOT / ".venv/bin/yt-dlp"

def main():
    ytdlp = str(YTDLP_BIN) if YTDLP_BIN.exists() else "yt-dlp"
    url = "https://www.youtube.com/@hiroyukike/shorts"
    print(f"Fetching shorts list from {url}...")
    
    cmd = [
        ytdlp,
        "--flat-playlist",
        "-J",
        url
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"Error fetching playlist: {proc.stderr}")
        sys.exit(1)
        
    data = json.loads(proc.stdout)
    entries = data.get("entries", [])
    print(f"Found {len(entries)} entries. Writing to {OUTPUT_FILE}...")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            vid = e.get("id")
            title = e.get("title", "")
            if vid:
                f.write(json.dumps({"id": vid, "title": title}, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"Successfully saved {count} shorts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

