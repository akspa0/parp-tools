#!/usr/bin/env python3
"""
Simple HTTP server for WoWRollback viewer
Serves the latest comparison viewer on http://localhost:8080
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

PORT = 8080

def find_latest_viewer():
    """Find the latest comparison viewer directory"""
    comparisons_dir = Path("rollback_outputs/comparisons")
    
    if not comparisons_dir.exists():
        print("❌ No comparisons found. Run regeneration first.")
        sys.exit(1)
    
    # Get the first (and likely only) comparison directory
    comparison_dirs = list(comparisons_dir.iterdir())
    
    if not comparison_dirs:
        print("❌ No comparison directories found.")
        sys.exit(1)
    
    viewer_dir = comparison_dirs[0] / "viewer"
    
    if not viewer_dir.exists():
        print(f"❌ Viewer directory not found: {viewer_dir}")
        sys.exit(1)
    
    return viewer_dir

def main():
    viewer_dir = find_latest_viewer()
    
    # Change to viewer directory
    os.chdir(viewer_dir)
    
    print(f"🌐 Starting web server...")
    print(f"📁 Serving: {viewer_dir.absolute()}")
    print(f"🔗 URL: http://localhost:{PORT}")
    print(f"\n✅ Server running. Press Ctrl+C to stop.\n")
    
    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            # Open browser
            webbrowser.open(f"http://localhost:{PORT}/index.html")
            
            # Serve forever
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped.")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"\n❌ Port {PORT} is already in use. Stop the other server first.")
            sys.exit(1)
        raise

if __name__ == "__main__":
    main()
