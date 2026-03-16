from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapidraw_basic_color.engine import BasicColorRenderer
from rapidraw_basic_color.io import load_image
from rapidraw_basic_color.params import BasicColorParams



def run_test() -> None:
    input_path = ROOT / "examples" / "demo_input.png"
    params_path = ROOT / "examples" / "params.basic_color.json"
    assert input_path.exists(), f"missing demo input: {input_path}"
    assert params_path.exists(), f"missing params: {params_path}"

    renderer = BasicColorRenderer()
    params = BasicColorParams.from_json_file(params_path)
    image = load_image(input_path)
    result = renderer.render_array(image, params, debug=True)

    assert result.image_srgb.shape == image.shape
    assert np.isfinite(result.image_srgb).all()
    assert np.min(result.image_srgb) >= 0.0
    assert np.max(result.image_srgb) <= 1.0
    assert result.recorder is not None and len(result.recorder.stages) == 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "out.jpg"
        dbg_dir = Path(tmp_dir) / "dbg"
        renderer.render_file(input_path, params, out_path, debug_dir=dbg_dir)
        assert out_path.exists()
        assert len(list(dbg_dir.glob("*.png"))) == 4

    print("Basic+Color only smoke test passed.")


if __name__ == "__main__":
    run_test()
