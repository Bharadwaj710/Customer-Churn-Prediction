"""
tests/capture_dashboard_screenshots.py
--------------------------------------
Automated headless screenshot generator for the Customer Churn Intelligence Dashboard.
Launches local server, renders each tab in headless Edge/Chrome, and saves high-resolution
PNG screenshots in outputs/figures/dashboard/.
"""

import os
import sys
import time
import subprocess
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "figures", "dashboard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Locate Edge or Chrome executable
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_PATH):
    EDGE_PATH = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

BROWSER_PATH = EDGE_PATH if os.path.exists(EDGE_PATH) else CHROME_PATH
print(f"Using browser executable: {BROWSER_PATH}")

# Start Dash server in subprocess
print("Starting Dash server on port 8050...")
server_process = subprocess.Popen(
    [sys.executable, os.path.join(BASE_DIR, "app.py")],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for server to become responsive
server_ready = False
for attempt in range(15):
    time.sleep(1)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8050/", timeout=2) as response:
            if response.status == 200:
                server_ready = True
                print("Dash server is ready and responding at http://127.0.0.1:8050/")
                break
    except Exception:
        pass

if not server_ready:
    server_process.kill()
    raise RuntimeError("Server failed to start within timeout.")

# Capture Screenshots for all views
tabs_to_capture = [
    ("module_a_executive_overview.png", "http://127.0.0.1:8050/"),
    ("module_b_customer_analytics.png", "http://127.0.0.1:8050/analytics"),
    ("module_c_model_evaluation.png", "http://127.0.0.1:8050/models"),
    ("module_d_e_churn_simulator.png", "http://127.0.0.1:8050/simulator"),
]

try:
    for filename, url in tabs_to_capture:
        dest_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Capturing screenshot: {filename} from {url}...")
        
        cmd = [
            BROWSER_PATH,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1920,1200",
            "--virtual-time-budget=5000",
            f"--screenshot={dest_path}",
            url,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        
        if os.path.exists(dest_path):
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"  [SUCCESS] Saved {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  [WARNING] Failed to capture {filename}: {res.stderr.decode('utf-8', errors='ignore')}")

finally:
    print("Shutting down Dash server subprocess...")
    server_process.kill()
    server_process.wait()
    print("Dashboard capture completed.")

