import socket
import uvicorn
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from dashboard.app import app

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 8000))
    sock.listen(128)
    print("[*] Dashboard server bound to 0.0.0.0:8000 via SO_REUSEADDR socket", flush=True)

    config = uvicorn.Config(app, log_level="info", access_log=True)
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
