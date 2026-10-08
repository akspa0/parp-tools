"""ComfyUI Orchestration Layer for SAM 3.1 and Mothership VLM Inference.

Provides a robust client for interacting with a local ComfyUI instance (default
http://127.0.0.1:8199) hosting SAM 3.1 segmentation and Mothership VLM custom nodes.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI reports an execution error or failure."""
    pass


class ComfyUIConnectionError(ComfyUIError):
    """Raised when unable to communicate with the ComfyUI endpoint."""
    pass


class ComfyUIOrchestrator:
    """Client for dispatching SAM 3.1 and Mothership VLM workflows to ComfyUI."""

    def __init__(self, base_url: str = "http://127.0.0.1:8199", client_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())

    # -------------------------------------------------------------------------
    # Core HTTP API
    # -------------------------------------------------------------------------

    def _request(
        self,
        endpoint: str,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        method: Optional[str] = None,
    ) -> bytes:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_headers = headers or {}
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return resp.read()
        except urllib.error.HTTPError as ex:
            err_body = ""
            try:
                err_body = ex.read().decode("utf-8")
            except Exception:
                pass
            raise ComfyUIConnectionError(f"HTTP {ex.code} error from {url}: {err_body or ex.reason}") from ex
        except urllib.error.URLError as ex:
            raise ComfyUIConnectionError(f"Failed to connect to ComfyUI at {url}: {ex}") from ex

    def check_health(self) -> Dict[str, Any]:
        """Check ComfyUI server health and GPU availability."""
        raw = self._request("system_stats")
        return json.loads(raw.decode("utf-8"))

    def get_object_info(self, node_class: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve schema information for available node classes."""
        endpoint = f"object_info/{node_class}" if node_class else "object_info"
        raw = self._request(endpoint)
        return json.loads(raw.decode("utf-8"))

    def upload_image(
        self,
        image_data: Union[bytes, np.ndarray, Image.Image, Path, str],
        filename: Optional[str] = None,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> str:
        """Upload an image to ComfyUI's input directory. Returns the uploaded filename."""
        if isinstance(image_data, (str, Path)):
            path = Path(image_data)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img_bytes = path.read_bytes()
            if not filename:
                filename = path.name
        elif isinstance(image_data, np.ndarray):
            if image_data.dtype != np.uint8:
                image_data = np.clip(image_data * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(image_data)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            if not filename:
                filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        elif isinstance(image_data, Image.Image):
            buf = io.BytesIO()
            image_data.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            if not filename:
                filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        elif isinstance(image_data, bytes):
            img_bytes = image_data
            if not filename:
                filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        else:
            raise TypeError(f"Unsupported image_data type: {type(image_data)}")

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body_parts: List[bytes] = []

        # Subfolder part
        if subfolder:
            body_parts.extend([
                f"--{boundary}\r\n".encode("utf-8"),
                b'Content-Disposition: form-data; name="subfolder"\r\n\r\n',
                subfolder.encode("utf-8"),
                b"\r\n",
            ])

        # Overwrite part
        body_parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\n',
            (b"true" if overwrite else b"false"),
            b"\r\n",
        ])

        # File part
        body_parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"),
            b"Content-Type: image/png\r\n\r\n",
            img_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ])

        full_body = b"".join(body_parts)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

        resp_raw = self._request("upload/image", data=full_body, headers=headers)
        result = json.loads(resp_raw.decode("utf-8"))
        return result.get("name", filename)

    def queue_prompt(self, workflow_prompt: Dict[str, Any]) -> str:
        """Submit a prompt graph workflow for execution. Returns prompt_id."""
        payload = {
            "prompt": workflow_prompt,
            "client_id": self.client_id,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        resp_raw = self._request("prompt", data=body, headers=headers)
        result = json.loads(resp_raw.decode("utf-8"))
        if "prompt_id" not in result:
            raise ComfyUIError(f"ComfyUI prompt submission failed: {result}")
        return result["prompt_id"]

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Fetch history entry for a given prompt_id."""
        try:
            raw = self._request(f"history/{prompt_id}")
            history = json.loads(raw.decode("utf-8"))
            return history.get(prompt_id)
        except ComfyUIConnectionError:
            return None

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll ComfyUI until prompt execution finishes or fails."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            history_entry = self.get_history(prompt_id)
            if history_entry is not None:
                status = history_entry.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise ComfyUIError(f"ComfyUI execution error for prompt {prompt_id}: {messages}")
                return history_entry
            time.sleep(poll_interval)
        raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout_seconds:.1f}s")

    def get_output_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image artifact from ComfyUI /view."""
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        return self._request(f"view?{query}")

    # -------------------------------------------------------------------------
    # Workflow Generators
    # -------------------------------------------------------------------------

    @staticmethod
    def build_sam_segmentation_workflow(
        image_name: str,
        threshold: float = 0.35,
        refine_iterations: int = 2,
        prompt_text: str = "building, house, roof, tree, structure, road, doodad",
        prefix: str = "sam_mask",
    ) -> Dict[str, Any]:
        """Build workflow graph for SAM 3.1 segmentation with optional text conditioning.

        Nodes:
          1: LoadImage(image=image_name)
          2: CRTAutoDLSAM3BodyCheckpoint() -> outputs [MODEL, CLIP, VAE]
          3: SAM3_Detect(model=[2, 0], image=[1, 0], conditioning=[6, 0], ...) -> outputs [MASK, BOUNDING_BOX]
          4: MaskToImage(mask=[3, 0]) -> outputs IMAGE
          5: SaveImage(images=[4, 0], filename_prefix=prefix)
          6: CLIPTextEncode(clip=[2, 1], text=prompt_text)
        """
        workflow: Dict[str, Any] = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "CRTAutoDLSAM3BodyCheckpoint",
                "inputs": {},
            },
            "3": {
                "class_type": "SAM3_Detect",
                "inputs": {
                    "model": ["2", 0],
                    "image": ["1", 0],
                    "threshold": float(threshold),
                    "refine_iterations": int(refine_iterations),
                    "individual_masks": False,
                },
            },
            "4": {
                "class_type": "MaskToImage",
                "inputs": {"mask": ["3", 0]},
            },
            "5": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["4", 0],
                    "filename_prefix": prefix,
                },
            },
        }

        if prompt_text:
            workflow["6"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["2", 1],
                    "text": prompt_text,
                },
            }
            workflow["3"]["inputs"]["conditioning"] = ["6", 0]

        return workflow

    @staticmethod
    def build_mothership_vlm_workflow(
        image_name: str,
        prompt: str,
        system_prompt: str = "",
        slot: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        seed: int = -1,
        max_dim: int = 1024,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Build workflow graph for Mothership VLM (Gemma / PaliGemma) inference.

        Nodes:
          1: LoadImage(image=image_name)
          2: MothershipGemmaVLM(image=[1, 0], prompt=prompt, ...)
          3: MothershipStringSink(text=[2, 0])
        """
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "MothershipGemmaVLM",
                "inputs": {
                    "image": ["1", 0],
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "slot": int(slot),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "seed": int(seed),
                    "max_dim": int(max_dim),
                    "timeout": float(timeout),
                },
            },
            "3": {
                "class_type": "MothershipStringSink",
                "inputs": {
                    "text": ["2", 0],
                },
            },
        }

    # -------------------------------------------------------------------------
    # High-Level Orchestrated Tasks
    # -------------------------------------------------------------------------

    def segment_objects(
        self,
        image_data: Union[bytes, np.ndarray, Image.Image, Path, str],
        threshold: float = 0.35,
        refine_iterations: int = 2,
        prompt_text: str = "building, house, roof, tree, structure, road, doodad",
        timeout_seconds: float = 60.0,
    ) -> np.ndarray:
        """Run SAM 3.1 object segmentation on an image. Returns binary mask (H, W) uint8 (0 or 255)."""
        uploaded_name = self.upload_image(image_data)
        prefix = f"sam_{uuid.uuid4().hex[:6]}"
        workflow = self.build_sam_segmentation_workflow(
            image_name=uploaded_name,
            threshold=threshold,
            refine_iterations=refine_iterations,
            prompt_text=prompt_text,
            prefix=prefix,
        )
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_completion(prompt_id, timeout_seconds=timeout_seconds)

        outputs = history.get("outputs", {})
        save_node_out = outputs.get("5", {})
        images = save_node_out.get("images", [])
        if not images:
            raise ComfyUIError(f"No output image produced for SAM segmentation: {outputs}")

        img_meta = images[0]
        raw_mask_bytes = self.get_output_image(
            filename=img_meta["filename"],
            subfolder=img_meta.get("subfolder", ""),
            folder_type=img_meta.get("type", "output"),
        )
        mask_pil = Image.open(io.BytesIO(raw_mask_bytes)).convert("L")
        mask_arr = np.array(mask_pil, dtype=np.uint8)
        # Binarize mask
        return np.where(mask_arr > 127, 255, 0).astype(np.uint8)

    def analyze_vlm(
        self,
        image_data: Union[bytes, np.ndarray, Image.Image, Path, str],
        prompt: str,
        system_prompt: str = "",
        slot: int = 1,
        temperature: float = 0.3,
        timeout_seconds: float = 60.0,
    ) -> str:
        """Run Mothership VLM inference on an image crop. Returns generated text response."""
        uploaded_name = self.upload_image(image_data)
        workflow = self.build_mothership_vlm_workflow(
            image_name=uploaded_name,
            prompt=prompt,
            system_prompt=system_prompt,
            slot=slot,
            temperature=temperature,
        )
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_completion(prompt_id, timeout_seconds=timeout_seconds)

        outputs = history.get("outputs", {})
        sink_out = outputs.get("3", {})
        # Check text or sink return values
        text_res = sink_out.get("text", "")
        if isinstance(text_res, list) and len(text_res) > 0:
            return str(text_res[0])
        elif isinstance(text_res, str):
            return text_res
        return str(sink_out)
