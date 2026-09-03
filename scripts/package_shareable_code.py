#!/usr/bin/env python3
"""Build a shareable source snapshot without data, weights, outputs, or secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_FILES = (
    ".gitignore",
    "README.md",
    "environment.yml",
    "requirements.txt",
    "requirements-train.txt",
    "constraints-sam3.txt",
    "pyproject.toml",
    "zd.txt",
)

SAM3_ROOT_FILES = (
    ".gitignore",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "README_TRAIN.md",
    "RELEASE_SAM3p1.md",
    "pyproject.toml",
)

CALIB_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)

SENSITIVE_NAMES = {
    ".camera_password",
    ".env",
    "credentials.json",
    "secrets.json",
}

SKIP_DIRECTORY_NAMES = {
    ".agents",
    ".codex",
    ".git",
    ".github",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "logs",
    "outputs",
}

SKIP_SUFFIXES = {
    ".avi",
    ".bmp",
    ".docx",
    ".engine",
    ".gif",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".tflite",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sensitive(path: Path) -> bool:
    """Reject common credential filenames and private-key suffixes."""
    name = path.name.lower()
    return (
        name in SENSITIVE_NAMES
        or name.startswith(".env.")
        or path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
    )


def common_allowed(relative: Path) -> bool:
    """Apply global cache, binary-data, model, and credential exclusions."""
    if any(part in SKIP_DIRECTORY_NAMES or part.endswith(".egg-info") for part in relative.parts):
        return False
    if is_sensitive(relative):
        return False
    return relative.suffix.lower() not in SKIP_SUFFIXES


def yoloe_allowed(relative: Path) -> bool:
    """Keep YOLOE runtime source while dropping upstream demos and evaluation data."""
    if not common_allowed(relative):
        return False
    parts = relative.parts
    if len(parts) > 1 and parts[0] == "yoloe" and parts[1] in {
        "docs",
        "examples",
        "figures",
        "tests",
    }:
        return False
    excluded_nested = {"data", "images", "ios_app", "notebooks", "results", "sav_dataset"}
    if parts and parts[0] == "yoloe" and "third_party" in parts:
        if any(part in excluded_nested for part in parts):
            return False
    return True


def copy_path(
    source: Path,
    destination: Path,
    source_root: Path,
    out_dir: Path,
    records: list[dict[str, Any]],
    predicate=common_allowed,
) -> None:
    """Recursively copy an allowed file or directory and record every file."""
    if source.is_dir():
        for child in sorted(source.rglob("*")):
            if not child.is_file():
                continue
            relative_to_source = child.relative_to(source_root)
            if not predicate(relative_to_source):
                continue
            target = destination / child.relative_to(source)
            copy_path(child, target, source_root, out_dir, records, predicate)
        return
    relative_to_source = source.relative_to(source_root)
    if not predicate(relative_to_source):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "source_path": str(relative_to_source),
            "packaged_path": str(destination.relative_to(out_dir)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def write_share_readme(out_dir: Path) -> Path:
    """Create a concise handoff guide for recipients."""
    path = out_dir / "共享版使用说明.md"
    path.write_text(
        """# 水下海缆检测代码共享版

本包是2026-09-03的当前代码快照，用于共享、复核和在其他设备上部署。包内不含
原始数据、模型权重、运行结果、Git历史或任何本机密码。

## 包含内容

- `scripts/`：抽帧、预处理、SAM3/YOLOE识别、分类、评估、tracking与可视化脚本；
- `configs/`：分类、运行、闭环及提示词配置；
- `tests/`：项目测试；
- `docs/`：算法与运行方案说明；
- `sam3/`：SAM3运行所需上游源码及许可证；
- `yoloe/`：YOLOE运行所需上游源码、依赖源码及许可证；
- `tools/calib/`：相机采集和标定工具源码；
- `manifest.json`：共享包逐文件SHA-256清单。

## 未包含内容

以下内容体积较大或包含本机信息，需要接收方自行准备：

- `data/`、`outputs/`、`deliverables/`；
- `ckpts/` 下的SAM3和YOLOE模型；
- `mobileclip_blt.ts`；
- `.camera_password`、环境变量、Git历史和缓存；
- 中期报告Word文件、结果图片和视频。

## 环境安装

通用环境：

```bash
conda env create -f environment.yml
conda activate yoloe
```

SAM3建议单独建立环境。DGX Spark/GB10可按照根目录 `README.md` 使用CUDA 13
对应的PyTorch；其他GPU应安装与其CUDA版本匹配的PyTorch。随后安装SAM3源码：

```bash
pip install -e sam3
```

## 模型放置

```text
ckpts/sam3/sam3.pt
ckpts/yoloe/yoloe-v8l-seg.pt
mobileclip_blt.ts
```

模型权重不属于代码包，需根据各模型许可由接收方自行下载。SAM3-only方案不需要
YOLOE权重；YOLOE文本提示方案需要YOLOE权重和 `mobileclip_blt.ts`。

## 快速检查

```bash
python -m py_compile scripts/*.py
python -m unittest discover -s tests
```

主要运行命令和参数见根目录 `README.md`。其中 `/home/nvidia/DATA/UW/...` 是原机器
示例路径，在其他机器上应替换为实际数据路径。需要Codex分类的路径使用Codex Pro
登录态；`classify_samples_gpt.py` 才需要设置 `OPENAI_API_KEY`。

## 第三方许可

SAM3与YOLOE为随包提供的第三方源码快照，各自的许可文件位于 `sam3/LICENSE` 和
`yoloe/LICENSE`；其 `third_party/` 目录内的组件保留各自许可证。
""",
        encoding="utf-8",
    )
    return path


def write_checkpoint_readme(out_dir: Path) -> Path:
    """Create empty checkpoint directories without bundling weights."""
    path = out_dir / "ckpts" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "sam3").mkdir(parents=True, exist_ok=True)
    (path.parent / "yoloe").mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# 模型权重目录

模型权重未包含在共享代码包中。请自行放置：

```text
ckpts/sam3/sam3.pt
ckpts/yoloe/yoloe-v8l-seg.pt
```

YOLOE文本提示所需的 `mobileclip_blt.ts` 放在项目根目录。
""",
        encoding="utf-8",
    )
    return path


def add_generated_record(path: Path, out_dir: Path, records: list[dict[str, Any]]) -> None:
    """Add a generated documentation file to the package manifest."""
    records.append(
        {
            "source_path": None,
            "packaged_path": str(path.relative_to(out_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for name in ROOT_FILES:
        copy_path(source_root / name, out_dir / name, source_root, out_dir, records)
    for name in ("scripts", "configs", "tests"):
        copy_path(source_root / name, out_dir / name, source_root, out_dir, records)

    docs_source = source_root / "docs"
    for source in sorted(docs_source.glob("*.md")):
        copy_path(source, out_dir / "docs" / source.name, source_root, out_dir, records)

    for name in SAM3_ROOT_FILES:
        copy_path(
            source_root / "sam3" / name,
            out_dir / "sam3" / name,
            source_root,
            out_dir,
            records,
        )
    copy_path(
        source_root / "sam3" / "sam3",
        out_dir / "sam3" / "sam3",
        source_root,
        out_dir,
        records,
    )

    copy_path(
        source_root / "yoloe",
        out_dir / "yoloe",
        source_root,
        out_dir,
        records,
        yoloe_allowed,
    )

    for name in CALIB_ROOT_FILES:
        copy_path(
            source_root / "tools" / "calib" / name,
            out_dir / "tools" / "calib" / name,
            source_root,
            out_dir,
            records,
        )
    for name in ("camera_calibration", "config", "scripts", "tests"):
        copy_path(
            source_root / "tools" / "calib" / name,
            out_dir / "tools" / "calib" / name,
            source_root,
            out_dir,
            records,
        )

    share_readme = write_share_readme(out_dir)
    checkpoint_readme = write_checkpoint_readme(out_dir)
    add_generated_record(share_readme, out_dir, records)
    add_generated_record(checkpoint_readme, out_dir, records)

    records.sort(key=lambda row: row["packaged_path"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": "UW Detection shareable source snapshot",
        "source_root": str(source_root),
        "summary": {
            "file_count_excluding_manifest": len(records),
            "total_size_bytes_excluding_manifest": sum(
                int(row["size_bytes"]) for row in records
            ),
        },
        "included": [
            "project scripts, configs, tests, and Markdown documentation",
            "SAM3 runtime source and license",
            "YOLOE runtime source, selected dependencies, and licenses",
            "camera calibration source, configs, and tests",
        ],
        "excluded": [
            "data, outputs, deliverables, checkpoints, and mobileclip_blt.ts",
            "Git metadata, caches, generated media, and report binaries",
            "credential files including tools/calib/.camera_password",
        ],
        "files": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total_files = len(records) + 1
    print(
        f"Shareable code packaged: {total_files} files, "
        f"{sum(int(row['size_bytes']) for row in records) / 1024 ** 2:.2f} MiB "
        f"plus manifest -> {out_dir}"
    )


if __name__ == "__main__":
    main()
