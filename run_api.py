"""Start the NIDS API server.

Usage (from the project root):
    python run_api.py

Optional environment overrides: NIDS_HOST (default 127.0.0.1),
NIDS_PORT (default 8000).
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.nids.app:app",
        host=os.getenv("NIDS_HOST", "127.0.0.1"),
        port=int(os.getenv("NIDS_PORT", "8000")),
        log_level="info",
    )
