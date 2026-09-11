"""Collect reproducible server or robot deployment information."""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent


def command_output(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_info() -> dict[str, str | None]:
    return {
        "branch": command_output("git", "branch", "--show-current"),
        "commit": command_output("git", "rev-parse", "HEAD"),
    }


def local_ips() -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0].split("%", 1)[0]
            if address not in {"127.0.0.1", "::1"}:
                addresses.add(address)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            if address != "127.0.0.1":
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def software_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "opencv": None,
        "onnxruntime": None,
        "available_execution_providers": [],
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["available_execution_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return info


def health_url(server_url: str) -> str:
    normalized = server_url.rstrip("/")
    if normalized.endswith("/infer"):
        return normalized[: -len("/infer")] + "/health"
    return normalized + "/health"


def probe_health(server_url: str) -> dict[str, Any]:
    url = health_url(server_url)
    try:
        with urlopen(url, timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"url": url, "ok": response.status == 200, "status": response.status, "body": body}
    except HTTPError as exc:
            return {"url": url, "ok": False, "status": exc.code, "error": "HTTPError"}
    except (URLError, OSError) as exc:
        return {"url": url, "ok": False, "status": None, "error": type(exc).__name__}


def camera_info(index: int, width: int, height: int) -> dict[str, Any]:
    devices = sorted(str(path) for path in Path("/dev").glob("video*"))
    result: dict[str, Any] = {
        "requested_index": index,
        "detected_devices": devices,
        "opened": False,
        "requested_resolution": [width, height],
        "actual_resolution": None,
    }
    try:
        import cv2
    except ImportError:
        result["error"] = "opencv is not installed"
        return result

    capture = cv2.VideoCapture(index)
    try:
        result["opened"] = bool(capture.isOpened())
        if result["opened"]:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            result["actual_resolution"] = [
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ]
    finally:
        capture.release()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect deployment handoff information")
    parser.add_argument("--role", required=True, choices=("server", "robot"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "role": args.role,
        "git": git_info(),
        "system": {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or None,
            "local_ipv4_addresses": local_ips(),
        },
        "software": software_info(),
        "server": {
            "inference_url": args.server_url,
            "health": probe_health(args.server_url),
        },
    }
    if args.role == "server":
        providers = report["software"]["available_execution_providers"]
        report["server"]["configured_execution_provider"] = (
            "CPUExecutionProvider" if "CPUExecutionProvider" in providers else None
        )
    else:
        report["robot"] = {
            "camera": camera_info(args.camera, args.width, args.height),
            "frame_skip": args.frame_skip,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved report: {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
