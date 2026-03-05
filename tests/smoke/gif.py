"""GIF generation from screenshot bytes."""

import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger("smoke")


def create_gif(
    screenshots: list[bytes], output_path: Path, duration: int = 1000
) -> Path | None:
    """Create an animated GIF from a list of PNG screenshot bytes.

    Returns the output path on success, None if no screenshots.
    """
    if not screenshots:
        logger.warning("No screenshots to create GIF")
        return None

    images = []
    for raw in screenshots:
        try:
            images.append(Image.open(io.BytesIO(raw)))
        except Exception:
            continue

    if not images:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass

    logger.info(f"GIF saved: {output_path} ({len(images)} frames)")
    return output_path
