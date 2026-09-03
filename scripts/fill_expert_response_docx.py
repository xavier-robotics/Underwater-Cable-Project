#!/usr/bin/env python3
"""Fill responses 2 and 4 in the expert-response DOCX without extra packages."""

from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "docs/20260813 课题二专家意见回复-待补充.docx"
DEFAULT_OUTPUT = REPO_ROOT / "docs/20260813 课题二专家意见回复-问题2和4已补充.docx"

RESPONSES = {
    "2": [
        (
            "针对专家意见，已对识别算法的自主化边界、数据集规模、实际场景差异和艇载部署状态进行补充。"
            "当前采用“本地开源基础模型+项目自主工程算法”的技术路线：SAM3使用本地开源预训练权重，"
            "不调用云端接口；数据整理与增强、海缆和损伤提示设计、三分类决策、悬空支撑几何判定、"
            "10 Hz视频跟踪、结果评测及运行脚本均由承研方完成，可在本地计算平台独立运行。"
            "因此，现阶段实现的是工程算法链路自主可控，不将开源底座表述为完全自主训练模型。"
        ),
        (
            "当前中期数据集按原视频可解码图像帧统计，共14785帧、585.19秒。"
            "按“损伤优先”的三分类口径，外部损伤11254帧、裸露且无外部损伤3101帧、"
            "悬空且无外部损伤430帧。“无故障海缆”不是额外第四类，按无外部损伤统计为3531帧，"
            "即裸露3101帧与悬空430帧之和；若将“既不裸露、不悬空、也无损伤”定义为独立正常类，"
            "则当前数据为0帧，已列为后续补充对象。上述图像帧来源于30个独立视频样本，"
            "每段视频对应一张样本级真值图；连续帧存在时序相关性，因此帧数用于说明数据规模，"
            "不将14785帧等同于14785个相互独立样本。"
            "现有30张图三分类正确27张，准确率90.00%；30段视频10 Hz抽帧5862帧，海缆有效跟踪5704帧，"
            "覆盖率97.30%。上述结果属于同一水池数据上的中期工程验证，不作为真实海域泛化精度。"
        ),
        (
            "水池数据与实际海域的主要差异包括：水质和光照较可控、背景以池砖为主、样本材质和外观较单一，"
            "缺少泥沙埋覆、生物附着、海流扰动、高浑浊、复杂海床及多型号海缆。"
            "现阶段已形成离线可执行程序和艇载部署方案，但尚未完成实艇端实时联调；10 Hz为输出采样频率，"
            "不等同于当前端到端实时推理速度。下一阶段将使用项目标注数据训练自主参数的轻量检测/状态模型，"
            "把SAM3转为离线标注与基线工具，并开展量化加速、实时视频流接入和艇载算力平台联调，完成实艇部署验证。"
        ),
    ],
    "4": [
        (
            "当前模拟样本采用深色长圆柱海缆样件在水池中进行水下成像，覆盖贴底裸露、支撑悬空和局部外部损伤三种状态。"
            "采集过程中包含水下绿偏、光照和视角变化、反光、池底纹理及局部遮挡等视觉干扰。"
            "样本在目标细长形态、暗色护套外观、贴底/跨距姿态、局部表面异常以及连续视频视角变化等方面，"
            "覆盖了真实海缆巡检的主要视觉识别要素，可用于检验“发现海缆—判断状态—连续跟踪”的完整算法链路。"
        ),
        (
            "模拟样本与真实样本仍存在明确差异：当前部分外部损伤以金属贴片等可控标志模拟，悬空状态利用白色支撑杆构造，"
            "池底为规则瓷砖，水质和照明相对稳定；尚未完整复现真实海缆的多层护套、铠装或导体外露、裂纹、磨损和腐蚀，"
            "也未覆盖泥沙埋覆、生物附着、海流扰动、高浑浊和复杂海床。现有记录没有形成模拟件与实缆在直径、材料、"
            "表面反射特性等方面的系统等效性检测，因此不能将模拟样本表述为与真实海缆物理结构完全等同。"
        ),
        (
            "综上，当前模拟样本对受控水池条件下的水下视觉特征和巡检作业流程具有工程真实性，适用于中期功能验证、"
            "算法对比和系统联调；其真实性边界是“场景与视觉任务等效”，不是“故障材料与真实海域完全等效”。"
            "下一阶段将补充退役或实物海缆的裂口、磨损、铠装外露等真实缺陷样本及近海/海域视频，按真实尺寸和材料制作"
            "更多缺陷样件，增加浑浊、泥沙、生物附着和多海床背景，并使用与当前水池数据隔离的盲测集开展对比验证。"
        ),
    ],
}

ROW_PATTERN = re.compile(r"<w:tr(?:\s[^>]*)?>.*?</w:tr>", re.DOTALL)
CELL_PATTERN = re.compile(r"<w:tc(?:\s[^>]*)?>.*?</w:tc>", re.DOTALL)
TEXT_PATTERN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
TC_PROPERTIES_PATTERN = re.compile(r"<w:tcPr>.*?</w:tcPr>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    """Parse input and output DOCX paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def visible_text(fragment: str) -> str:
    """Return concatenated Word text nodes from an XML fragment."""
    return "".join(html.unescape(text) for text in TEXT_PATTERN.findall(fragment))


def response_paragraph(text: str) -> str:
    """Build one black Song-style response paragraph matching the source table."""
    escaped = html.escape(text, quote=False)
    return (
        '<w:p><w:pPr><w:spacing w:line="300" w:lineRule="auto"/>'
        '<w:rPr><w:rFonts w:ascii="宋体" w:eastAsia="宋体" w:hAnsi="宋体" '
        'w:hint="eastAsia"/><w:sz w:val="24"/><w:szCs w:val="28"/>'
        '</w:rPr></w:pPr><w:r><w:rPr><w:rFonts w:ascii="宋体" '
        'w:eastAsia="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
        '<w:sz w:val="24"/><w:szCs w:val="28"/></w:rPr>'
        f'<w:t>{escaped}</w:t></w:r></w:p>'
    )


def replace_response_cell(row_xml: str, sequence: str, paragraphs: list[str]) -> str:
    """Replace the third cell in a numbered table row."""
    cells = list(CELL_PATTERN.finditer(row_xml))
    if len(cells) < 3 or visible_text(cells[0].group(0)).strip() != sequence:
        return row_xml
    response_cell = cells[2].group(0)
    if visible_text(response_cell).strip() != "XXX":
        raise ValueError(
            f"Response {sequence} is not an untouched XXX placeholder: "
            f"{visible_text(response_cell)!r}"
        )
    opening_end = response_cell.find(">") + 1
    opening = response_cell[:opening_end]
    properties_match = TC_PROPERTIES_PATTERN.search(response_cell)
    properties = properties_match.group(0) if properties_match else ""
    replacement = opening + properties
    replacement += "".join(response_paragraph(text) for text in paragraphs)
    replacement += "</w:tc>"
    start, end = cells[2].span()
    return row_xml[:start] + replacement + row_xml[end:]


def update_document_xml(document_xml: bytes) -> bytes:
    """Replace exactly responses 2 and 4 and validate the resulting XML."""
    source = document_xml.decode("utf-8")
    replacements = {sequence: 0 for sequence in RESPONSES}

    def update_row(match: re.Match[str]) -> str:
        row_xml = match.group(0)
        cells = list(CELL_PATTERN.finditer(row_xml))
        if not cells:
            return row_xml
        sequence = visible_text(cells[0].group(0)).strip()
        if sequence not in RESPONSES:
            return row_xml
        replacements[sequence] += 1
        return replace_response_cell(row_xml, sequence, RESPONSES[sequence])

    updated = ROW_PATTERN.sub(update_row, source)
    if replacements != {"2": 1, "4": 1}:
        raise ValueError(f"Unexpected replacement counts: {replacements}")
    ElementTree.fromstring(updated.encode("utf-8"))
    if "XXX" in visible_text(updated):
        raise ValueError("An XXX placeholder remains after replacement")
    return updated.encode("utf-8")


def write_updated_docx(input_path: Path, output_path: Path, force: bool) -> None:
    """Copy the DOCX package and replace only word/document.xml."""
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.docx")
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")
    with zipfile.ZipFile(input_path, "r") as source, zipfile.ZipFile(
        temporary, "w"
    ) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "word/document.xml":
                payload = update_document_xml(payload)
            target.writestr(info, payload)
    with zipfile.ZipFile(temporary, "r") as verification:
        broken = verification.testzip()
        if broken is not None:
            raise zipfile.BadZipFile(f"CRC check failed for {broken}")
        ElementTree.fromstring(verification.read("word/document.xml"))
    temporary.replace(output_path)


def main() -> None:
    """Create the completed copy and print the inserted response text."""
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().absolute()
    if not input_path.is_file():
        raise SystemExit(f"Input DOCX does not exist: {input_path}")
    write_updated_docx(input_path, output_path, args.force)
    print(f"Created: {output_path}")
    for sequence, paragraphs in RESPONSES.items():
        print(f"\n问题 {sequence} 回复：")
        for paragraph in paragraphs:
            print(paragraph)


if __name__ == "__main__":
    main()
