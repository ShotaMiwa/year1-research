# -*- coding: utf-8 -*-
"""
目視検証レポート用の軽量HTTPサーバー。
静的HTMLの配信に加えて、判定結果（decisions）をローカルファイル（data/processed/verified_alignments.json）に
リアルタイムで自動同期・保存・読み込みするAPIを提供する。
"""

import http.server
import socketserver
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_PAPER = PROJECT_ROOT / "docs/paper"
DECISIONS_FILE = PROJECT_ROOT / "data/processed/verified_alignments.json"

PORT = 8765

class VerificationHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_PAPER), **kwargs)

    def do_GET(self):
        if self.path == '/api/decisions':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if DECISIONS_FILE.exists():
                data = DECISIONS_FILE.read_text(encoding='utf-8')
            else:
                data = "{}"
            self.wfile.write(data.encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/decisions':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                decisions_data = json.loads(body)
                DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(DECISIONS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(decisions_data, f, ensure_ascii=False, indent=2)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), VerificationHandler) as httpd:
        print(f"🚀 目視検証同期サーバー起動: http://localhost:{PORT}/verification_report.html")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()

