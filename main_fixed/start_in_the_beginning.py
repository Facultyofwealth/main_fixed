import multiprocessing
import socket
import threading
import time

import uvicorn
from main_fixed import app, set_webview_runtime

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
    """
    Return the best LAN IP for the phone remote URL.

    Priority (highest → lowest):
      3 — 192.168.x.x  (standard home / office WiFi router)
      3 — 172.20.x.x   (iPhone personal hotspot) — same priority as above
      2 — 172.16-31.x.x other private ranges (excludes Docker 172.17/18)
      1 — 10.x.x.x     (VPN / corporate / virtual adapters — last resort)

    Falls back to 127.0.0.1 only if nothing else is found.
    """
    import ipaddress

    def score(ip_str):
        try:
            ip = ipaddress.IPv4Address(ip_str)
        except ValueError:
            return -1
        if ip.is_loopback:
            return -1
        # High priority: real WiFi (192.168.x.x) or iPhone hotspot (172.20.x.x)
        if ip_str.startswith("192.168."):
            return 3
        if ip_str.startswith("172.20."):
            return 3
        # Medium: other private 172.16-31 ranges (skip Docker 172.17/172.18/172.19)
        if ip_str.startswith("172."):
            second_octet = int(ip_str.split(".")[1])
            if 16 <= second_octet <= 31 and second_octet not in (17, 18, 19):
                return 2
        # Low: 10.x.x.x — VPN / virtual adapters
        if ip_str.startswith("10."):
            return 1
        return 0

    candidates = []

    # Method 1: collect all IPs via hostname resolution
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            s = score(ip)
            if s >= 0:
                candidates.append((s, ip))
    except Exception:
        pass

    # Method 2: UDP trick to find the default-route IP
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        s = score(ip)
        if s >= 0:
            candidates.append((s, ip))
    except Exception:
        pass

    if candidates:
        # Return the IP with the highest score (stable sort keeps first found on tie)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

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
            # The control UI always opens on the PRIMARY monitor. The projector
            # output is a separate frameless fullscreen window that the backend
            # opens on the operator's saved target screen (POST /output/on).
            primary_screen = None
            try:
                primary_screen = next(
                    (s for s in webview.screens if int(s.x) == 0 and int(s.y) == 0), None
                )
            except Exception:
                primary_screen = None

            window_kwargs = dict(
                width=1280,
                height=800,
                resizable=True,
                fullscreen=False,
            )
            if primary_screen is not None:
                window_kwargs["screen"] = primary_screen

            window = webview.create_window("In The Beginning", DESKTOP_URL, **window_kwargs)

            # Hand pywebview to the backend so it can create/destroy the output
            # window from the /output/on and /output/off endpoints.
            set_webview_runtime(webview, window)
            webview.start()
        else:
            print("ERROR: Server did not start in time.")
    else:
        # Headless / Railway — just run Uvicorn normally on the main thread.
        uvicorn.run(app, host=HOST, port=PORT, reload=False)