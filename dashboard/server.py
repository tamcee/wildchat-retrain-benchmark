"""
Lightweight high-performance HTTP server for the retrain benchmark web dashboard.
Serves static assets from dashboard/web/ on port 8501.
"""
import os
import sys
import argparse
from http.server import SimpleHTTPRequestHandler, HTTPServer


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        # If an old Streamlit tab is polling _stcore, redirect to root
        if self.path.startswith("/_stcore"):
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dashboard development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run(port=8501, host="127.0.0.1"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"\n========================================================")
    print(f"✨ Retraining Benchmark Dashboard is Live!")
    print(f"👉 Local URL: http://{host}:{port}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the modern retrain benchmark dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()
    run(port=args.port, host=args.host)
