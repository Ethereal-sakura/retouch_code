from __future__ import annotations

import argparse

from .engine import BasicColorRenderer
from .params import BasicColorParams


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。
    
    必选参数：
      --input   输入图像路径（支持 TIF/PNG/JPG）
      --params  JSON 参数文件路径（格式见 examples/params.basic_color.json）
      --output  输出图像路径
    
    可选参数：
      --quality              JPEG 输出质量，默认 95
      --save-intermediates-dir  保存 4 个中间阶段 PNG 的目录
    """
    parser = argparse.ArgumentParser(description="RapidRAW Basic + Color only renderer (non-RAW)")
    parser.add_argument("--input", required=True, help="Input image path: tif/png/jpg")
    parser.add_argument("--params", required=True, help="JSON params file for Basic + Color only")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for JPG output")
    parser.add_argument("--save-intermediates-dir", default=None, help="Optional directory for 4 intermediate PNGs")
    return parser


def main() -> int:
    """CLI 入口：解析参数 → 加载 JSON → 执行渲染 → 保存输出。
    
    返回值为进程退出码（0 表示成功），由 main_basic_color.py 传给 sys.exit。
    """
    parser = build_parser()
    args = parser.parse_args()
    params = BasicColorParams.from_json_file(args.params)
    renderer = BasicColorRenderer()
    renderer.render_file(
        input_path=args.input,
        params=params,
        output_path=args.output,
        debug_dir=args.save_intermediates_dir,
        quality=int(args.quality),
    )
    print(f"Basic+Color rendered: {args.input} -> {args.output}")
    return 0
