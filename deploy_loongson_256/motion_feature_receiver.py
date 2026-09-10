"""Reference receiver for the independent motion-control process."""
from __future__ import annotations

import argparse
import json
import socket
import time


def safety_stop_reason(
    message: dict, now: float, stale_after: float
) -> str | None:
    """Return the reason motion must stop, or ``None`` for a safe target."""
    try:
        timestamp = float(message["timestamp"])
    except (KeyError, TypeError, ValueError):
        return "invalid timestamp"
    if now - timestamp > stale_after:
        return "target missing or stale"
    if not bool(message.get("target_valid")):
        return "target missing or stale"
    if bool(message.get("fall")):
        return "fall detected"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--stale-after", type=float, default=0.5)
    args = parser.parse_args()
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind((args.host, args.port))
        receiver.settimeout(args.stale_after)
        print(f"[MOTION] listening on {args.host}:{args.port}")
        print("[MOTION] STOP (awaiting valid tracking data)")
        last_received_at = None
        last_stop_reason = "awaiting valid tracking data"
        while True:
            try:
                data, address = receiver.recvfrom(8192)
            except socket.timeout:
                now = time.monotonic()
                if (
                    last_received_at is not None
                    and now - last_received_at > args.stale_after
                    and last_stop_reason != "target missing or stale"
                ):
                    print("[MOTION] STOP (target missing or stale)")
                    last_stop_reason = "target missing or stale"
                continue
            try:
                message = json.loads(data.decode("utf-8"))
                now = time.monotonic()
                last_received_at = now
                stop_reason = safety_stop_reason(message, now, args.stale_after)
                if stop_reason is not None:
                    print(f"[MOTION] STOP ({stop_reason})")
                    last_stop_reason = stop_reason
                else:
                    print(
                        f"[MOTION] target={message['horizontal_error']:.3f} "
                        f"distance={message['distance_ratio']:.3f} "
                        f"fall={message['fall']} from={address}"
                    )
                    last_stop_reason = None
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"[MOTION][ERROR] invalid message: {exc}")


if __name__ == "__main__":
    main()
