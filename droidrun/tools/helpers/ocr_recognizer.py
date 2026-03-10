"""OCR Recognizer - Use EasyOCR to recognize text elements from screenshots.

This module provides OCR-based UI tree recognition functionality, extracting
text elements from screenshots and building a hierarchical UI tree structure.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import easyocr
import numpy as np

try:
    import os
    import torch

    if torch.backends.mps.is_available():
        # Enable MPS fallback for operations not yet supported on MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        _GPU_AVAILABLE = True
        _GPU_BACKEND = "mps"
    elif torch.cuda.is_available():
        _GPU_AVAILABLE = True
        _GPU_BACKEND = "cuda"
    else:
        _GPU_AVAILABLE = False
        _GPU_BACKEND = "cpu"
except ImportError:
    _GPU_AVAILABLE = False
    _GPU_BACKEND = "cpu"

logger = logging.getLogger("droidrun")


# ══════════════════════════════════════════════════════════════
# 1. Image Preprocessing
# ══════════════════════════════════════════════════════════════
class ImagePreprocessor:
    """Image preprocessor class to enhance OCR recognition rate"""

    SCALE_FACTOR = 1.5
    REGION_SIZE = 512

    def enhance_for_ocr(self, image: np.ndarray) -> List[tuple]:
        """Generate multiple image variants to improve recognition rate"""
        variants = []

        # Original image
        variants.append(("original", image))

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variants.append(("gray", gray))

        # Enhanced contrast
        enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
        variants.append(("enhanced", enhanced))

        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("binary", binary))

        # Gaussian blur denoising
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        variants.append(("blurred", blurred))

        return variants

    def split_regions(self, image: np.ndarray) -> List[tuple]:
        """Split image into multiple regions to capture small text"""
        h, w = image.shape[:2]
        regions = []

        for y in range(0, h, self.REGION_SIZE):
            for x in range(0, w, self.REGION_SIZE):
                region = image[y : y + self.REGION_SIZE, x : x + self.REGION_SIZE]
                if region.size > 0:
                    regions.append((region, x, y))

        return regions


# ══════════════════════════════════════════════════════════════
# 2. OCR Text Detection
# ══════════════════════════════════════════════════════════════
class OCRDetector:
    """OCR text detector class using EasyOCR to recognize text elements"""

    SCALE_FACTOR = 1.5

    def __init__(self):
        logger.info(
            "Initializing EasyOCR engine (GPU=%s, backend=%s)...",
            _GPU_AVAILABLE,
            _GPU_BACKEND,
        )
        self.engine = easyocr.Reader(["en", "ch_sim"], gpu=_GPU_AVAILABLE)
        logger.info("EasyOCR engine initialized")

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect text elements from screenshot"""
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        img_h, img_w = original_img.shape[:2]
        logger.info(f"Image dimensions: {img_w} x {img_h}")

        preprocessor = ImagePreprocessor()
        all_elements = []

        # Phase 1: Multi-variant OCR (capture main text in full image)
        # Images are NOT scaled here, so coordinates are already in original space (scale=1.0)
        logger.info("Phase 1: Multi-variant OCR detection")
        variants = preprocessor.enhance_for_ocr(original_img)
        for variant_name, variant_img in variants:
            logger.debug(
                f"OCR variant: {variant_name} ({variant_img.shape[1]}x{variant_img.shape[0]})"
            )
            all_elements += self._collect_from_ocr(variant_img, 0, 0, scale=1.0)

        # Phase 2: Regional OCR (capture small text missed in full image)
        # Images are scaled up by SCALE_FACTOR, so coordinates must be divided back down
        logger.info("Phase 2: Regional OCR detection")
        scaled_img = cv2.resize(
            original_img,
            None,
            fx=self.SCALE_FACTOR,
            fy=self.SCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )
        for region_img, offset_x, offset_y in preprocessor.split_regions(scaled_img):
            all_elements += self._collect_from_ocr(
                region_img, offset_x, offset_y, scale=self.SCALE_FACTOR
            )

        # Deduplicate and merge
        all_elements = self._deduplicate(all_elements, img_w, img_h)
        logger.info(
            f"OCR detected {len(all_elements)} text elements (after deduplication)"
        )
        return all_elements

    def _collect_from_ocr(
        self, image: np.ndarray, offset_x: int, offset_y: int, scale: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Execute OCR on image, map coordinates back to original size and return element list.

        Args:
            image: Image to run OCR on.
            offset_x: X offset of this image region within the scaled image.
            offset_y: Y offset of this image region within the scaled image.
            scale: The scale factor applied to the original image before OCR.
                   Use 1.0 when the image is the original (unscaled) size,
                   use SCALE_FACTOR when the image was upscaled before OCR.
        """
        elements = []
        for text, x1, y1, x2, y2, conf in self._ocr_image(image):
            # Map from scaled/regional coordinates back to original image coordinates
            ox1 = int((x1 + offset_x) / scale)
            oy1 = int((y1 + offset_y) / scale)
            ox2 = int((x2 + offset_x) / scale)
            oy2 = int((y2 + offset_y) / scale)
            elements.append(self._make_element(text, ox1, oy1, ox2, oy2, conf))
        return elements

    def _ocr_image(self, image: np.ndarray) -> List[tuple]:
        """Execute OCR on single image, return (text, x1, y1, x2, y2, confidence) list"""
        return self._ocr_easyocr(image)

    def _ocr_easyocr(self, image: np.ndarray) -> List[tuple]:
        results = self.engine.readtext(
            image,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            width_ths=0.7,
        )
        parsed = []
        for bbox, text, conf in results:
            conf_f = float(conf)
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))
            if conf_f < 0.05 or not text.strip():
                logger.debug(
                    "OCR raw (FILTERED): text=%r conf=%.3f bbox=[%d,%d,%d,%d]",
                    text.strip(), conf_f, x1, y1, x2, y2,
                )
                continue
            logger.debug(
                "OCR raw (KEPT):     text=%r conf=%.3f bbox=[%d,%d,%d,%d]",
                text.strip(), conf_f, x1, y1, x2, y2,
            )
            parsed.append((text.strip(), x1, y1, x2, y2, conf_f))
        return parsed

    @staticmethod
    def _deduplicate(
        elements: List[Dict[str, Any]], img_w: int, img_h: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate by confidence descending order, IOU or containment relationship"""
        if not elements:
            return []

        elements = sorted(elements, key=lambda e: e.get("confidence", 0), reverse=True)
        kept = []

        for candidate in elements:
            cb = candidate["boundsInScreen"]
            is_duplicate = False
            for existing in kept:
                eb = existing["boundsInScreen"]
                ix1 = max(cb["left"], eb["left"])
                iy1 = max(cb["top"], eb["top"])
                ix2 = min(cb["right"], eb["right"])
                iy2 = min(cb["bottom"], eb["bottom"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_c = (cb["right"] - cb["left"]) * (cb["bottom"] - cb["top"])
                area_e = (eb["right"] - eb["left"]) * (eb["bottom"] - eb["top"])
                union = area_c + area_e - inter

                iou = inter / union if union > 0 else 0
                if iou > 0.3:
                    is_duplicate = True
                    break

                min_area = min(area_c, area_e)
                if min_area > 0 and inter / min_area > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                candidate["boundsInScreen"] = {
                    "left": max(0, cb["left"]),
                    "top": max(0, cb["top"]),
                    "right": min(img_w, cb["right"]),
                    "bottom": min(img_h, cb["bottom"]),
                }
                kept.append(candidate)

        return kept

    @staticmethod
    def _make_element(text, x1, y1, x2, y2, conf) -> Dict[str, Any]:
        pad = 4
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = x2 + pad, y2 + pad
        return {
            "id": f"ocr_{x1}_{y1}",
            "type": "ocr_text",
            "text": text,
            "contentDescription": text,
            "confidence": round(float(conf), 3),
            "boundsInScreen": {"left": x1, "top": y1, "right": x2, "bottom": y2},
            "center": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
            "isClickable": True,
            "isVisibleToUser": True,
            "children": [],
        }


# ══════════════════════════════════════════════════════════════
# 3. Build Hierarchical Tree
# ══════════════════════════════════════════════════════════════
class UITreeBuilder:
    """UI tree builder class to build hierarchical UI tree in Portal format"""

    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h

    def build_tree(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build tree based on containment: large boxes contain small boxes → parent-child relationship"""
        sorted_elems = sorted(
            elements,
            key=lambda e: (
                (e["boundsInScreen"]["right"] - e["boundsInScreen"]["left"])
                * (e["boundsInScreen"]["bottom"] - e["boundsInScreen"]["top"])
            ),
        )

        nodes = [dict(e) for e in sorted_elems]
        for node in nodes:
            node.setdefault("children", [])
            node["_assigned"] = False

        for i, child in enumerate(nodes):
            cb = child["boundsInScreen"]
            best_parent = None
            best_area = float("inf")
            for j, parent in enumerate(nodes):
                if i == j:
                    continue
                pb = parent["boundsInScreen"]
                if (
                    pb["left"] <= cb["left"]
                    and pb["top"] <= cb["top"]
                    and pb["right"] >= cb["right"]
                    and pb["bottom"] >= cb["bottom"]
                ):
                    parent_area = (pb["right"] - pb["left"]) * (
                        pb["bottom"] - pb["top"]
                    )
                    if parent_area < best_area:
                        best_area = parent_area
                        best_parent = j
            if best_parent is not None:
                nodes[best_parent]["children"].append(child["id"])
                nodes[i]["_assigned"] = True

        root_children = [n for n in nodes if not n["_assigned"]]
        for node in nodes:
            node.pop("_assigned", None)

        id_map = {n["id"]: n for n in nodes}

        def resolve(node):
            node["children"] = [
                resolve(id_map[cid])
                for cid in node.get("children", [])
                if cid in id_map
            ]
            return node

        root_children = [resolve(n) for n in root_children]

        return {
            "id": "root",
            "type": "root",
            "className": "ScreenRoot",
            "text": "",
            "contentDescription": "Screen root node",
            "boundsInScreen": {
                "left": 0,
                "top": 0,
                "right": self.img_w,
                "bottom": self.img_h,
            },
            "center": {"x": self.img_w // 2, "y": self.img_h // 2},
            "isClickable": False,
            "isVisibleToUser": True,
            "children": root_children,
        }


# ══════════════════════════════════════════════════════════════
# 4. Main Coordinator Class
# ══════════════════════════════════════════════════════════════
class OCRRecognizer:
    """OCR recognizer main coordinator class that encapsulates the entire OCR workflow"""

    def __init__(self):
        self._ocr_detector: Optional[OCRDetector] = None

    def recognize_ui_tree(
        self,
        screenshot_bytes: bytes,
        screen_width: int,
        screen_height: int,
    ) -> Dict[str, Any]:
        """Recognize UI tree from screenshot bytes

        Args:
            screenshot_bytes: Screenshot byte stream
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            UI tree dictionary (a11y_tree)

        Raises:
            RuntimeError: If OCR recognition fails
        """
        try:
            # Initialize OCR detector
            if self._ocr_detector is None:
                self._ocr_detector = OCRDetector()

            # Save byte stream to temporary image file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(screenshot_bytes)

            try:
                # Recognize text elements via OCR
                ocr_elements = self._ocr_detector.detect(tmp_path)

                # Build UI tree
                ui_tree = UITreeBuilder(screen_width, screen_height).build_tree(
                    ocr_elements
                )

                logger.debug(f"OCR UI tree recognized: {len(ocr_elements)} elements")

                # Return only UI tree
                return ui_tree

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"OCR-based UI tree recognition failed: {e}")
            raise RuntimeError(f"Failed to recognize UI tree via OCR: {e}") from e
