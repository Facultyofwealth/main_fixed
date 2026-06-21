import multiprocessing
import socket
import threading
import time
import webbrowser

import uvicorn
from main_fixed import app


HOST = "0.0.0.0"
PORT = 8000
REMOTE_PORT = 8001
DESKTOP_URL = f"http://127.0.0.1:{PORT}/"


def get_lan_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def wait_for_server_then_open():
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                webbrowser.open(DESKTOP_URL)
                return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    lan_ip = get_lan_ip()
    remote_url = f"http://{lan_ip}:{REMOTE_PORT}/remote"

    print("")
    print("In the Beginning is starting...")
    print(f"Desktop transcription: {DESKTOP_URL}")
    print(f"Phone remote:          {remote_url}")
    print("")
    print("Keep this window open while using the app.")
    print("Use the localhost desktop URL for transcription, and the phone URL/QR for remote.")
    print("")

    multiprocessing.freeze_support()
    threading.Thread(target=wait_for_server_then_open, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
