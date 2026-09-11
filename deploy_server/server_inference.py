"""FastAPI single-image inference service for the server-side pose model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
LOONGSON_INFERENCE_DIR = PROJECT_ROOT / "deploy_loongson_256"
DEFAULT_MODEL = PROJECT_ROOT / "yolo11n-pose-256.onnx"
INPUT_SIZE = 256
DEFAULT_PROVIDER = "CPUExecutionProvider"


def load_reused_pipeline() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load the existing 256 deployment preprocessing/postprocessing helpers."""
    path = str(LOONGSON_INFERENCE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    from onnx_inference import postprocess, preprocess

    return preprocess, postprocess


def resolve_model(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    return model_path


def to_jsonable(array: np.ndarray) -> list[Any]:
    return np.asarray(array).tolist()


class InferenceService:
    """Decode an image, run CPU ONNX Runtime, and return JSON-compatible data."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        session: Any | None = None,
        preprocess_fn: Callable[..., Any] | None = None,
        postprocess_fn: Callable[..., Any] | None = None,
    ) -> None:
        if provider != DEFAULT_PROVIDER:
            raise ValueError(
                "Only CPUExecutionProvider is supported by this service; "
                "GPU providers are not selected automatically."
            )

        self.model_path = resolve_model(model_path)
        self.provider = provider
        self.preprocess = preprocess_fn
        self.postprocess = postprocess_fn

        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required to start the service") from exc
            if provider not in ort.get_available_providers():
                raise RuntimeError(f"{provider} is unavailable")
            session = ort.InferenceSession(
                str(self.model_path), providers=[DEFAULT_PROVIDER]
            )
        self.session = session
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        session_providers = self.session.get_providers()
        if not session_providers or session_providers[0] != DEFAULT_PROVIDER:
            raise RuntimeError(
                f"CPUExecutionProvider was not selected first: {session_providers}"
            )
        if self.preprocess is None or self.postprocess is None:
            self.preprocess, self.postprocess = load_reused_pipeline()

    def infer_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError("image body is empty")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("request body is not a valid JPEG or PNG image")

        blob, ratio, padding = self.preprocess(frame, INPUT_SIZE)
        started = time.perf_counter()
        output = self.session.run(self.output_names, {self.input_name: blob})[0]
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        boxes, keypoints, scores = self.postprocess(
            output, frame.shape[:2], ratio, padding, INPUT_SIZE
        )
        return {
            "boxes": to_jsonable(boxes),
            "keypoints": to_jsonable(keypoints),
            "scores": to_jsonable(scores),
            "inference_time_ms": inference_time_ms,
        }


def create_app(service: InferenceService | None = None):
    """Create the FastAPI app without loading the model at module import time."""
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required to create the HTTP service; install requirements.txt"
        ) from exc

    app = FastAPI(title="YOLO11n-Pose Server Inference")
    app.state.inference_service = service

    @app.get("/health")
    async def health() -> dict[str, str]:
        if app.state.inference_service is None:
            return {"status": "not_ready"}
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)) -> dict[str, Any]:
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=415, detail="only JPEG and PNG are supported")
        current_service = app.state.inference_service
        if current_service is None:
            raise HTTPException(status_code=503, detail="inference service is not ready")
        try:
            return current_service.infer_bytes(await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--provider", choices=[DEFAULT_PROVIDER], default=DEFAULT_PROVIDER
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn

    service = InferenceService(args.model, args.provider)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
