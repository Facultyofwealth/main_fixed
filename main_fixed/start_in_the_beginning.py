import multiprocessing
import socket
import threading
import time

import uvicorn
from main_fixed import app

# pywebview is only available on desktop (local .exe / dev machine).
# On Railway (headless, no display) this import will fail — that's expected
# and handled gracefully: the app falls back to plain Uvicorn.
try:
    import webview
    PYWEBVIEW_AVAILABLE = True
except Exception:
    PYWEBVIEW_AVAILABLE = False


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


def run_uvicorn():
    """Run Uvicorn in a background thread (used when pywebview takes the main thread)."""
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


def wait_for_server(timeout=90):
    """Block until the server is accepting connections, or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


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

    if PYWEBVIEW_AVAILABLE:
        # pywebview MUST run on the main thread — move Uvicorn to background.
        server_thread = threading.Thread(target=run_uvicorn, daemon=True)
        server_thread.start()

        # Wait for Uvicorn to be ready before opening the window.
        if wait_for_server():
            window = webview.create_window(
                "In The Beginning",
                DESKTOP_URL,
                width=1280,
                height=800,
                resizable=True,
                fullscreen=False,
            )
            webview.start()
        else:
            print("ERROR: Server did not start in time.")
    else:
        # Headless / Railway — just run Uvicorn normally on the main thread.
        uvicorn.run(app, host=HOST, port=PORT, reload=False)