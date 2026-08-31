"""Build a sanitized, hash-bound provenance register for the primary Ge/Si TCAD data.

The source tree contains commercial TCAD project files and logs with local
machine/licensing details.  This script inventories them without copying those
files into the publication artifact.  Only the frozen parameter tables are
copied; all published paths are logical, source-root-relative locators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "results" / "tcad_provenance"
CAMPAIGN_RELATIVE = Path("Data") / "Data_20260617_0243_181408"
CAMPAIGN_DATE = "20260617"
SOURCE_SNAPSHOTS = (
    "DataGenerator.py",
    "DeviceGePhotodetector.py",
    "ParallelSimulator.py",
    "SupplementChargeAcData.py",
    "TerminalChargeDataGenerator.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_locator(path: Path, source_root: Path) -> str:
    return (Path("<FROZEN_TCAD_ROOT>") / path.relative_to(source_root)).as_posix()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def read_log_summary(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    version = lines[0].strip() if lines else "not recovered"
    grid_shapes = sorted(
        {
            "x".join(match)
            for match in re.findall(
                r"Simulation size in gridpoints:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)",
                text,
            )
        }
    )
    vertices = sorted({int(value) for value in re.findall(r"\+ vertices:\s*(\d+)", text)})
    elements = sorted({int(value) for value in re.findall(r"\+ elements:\s*(\d+)", text)})
    if "FDTD Solver" in version:
        solver = "FDTD"
    elif "Charge Transport Solver" in version:
        solver = "TCAD"
    else:
        solver = "unknown"
    return {
        "solver": solver,
        "software_version": version,
        "gridpoint_shapes": ";".join(grid_shapes),
        "vertices": ";".join(str(value) for value in vertices),
        "elements": ";".join(str(value) for value in elements),
    }


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(source_root: Path, output: Path) -> None:
    source_root = source_root.resolve()
    campaign = source_root / CAMPAIGN_RELATIVE
    raw_dir = campaign / "data"
    native_dir = source_root / "ParaSimulationFile"
    required = (campaign, raw_dir, native_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source path(s): " + ", ".join(missing))

    output.mkdir(parents=True, exist_ok=True)
    para_list = next(campaign.glob("para_list_*.csv"))
    para_info = next(campaign.glob("para_info_*.csv"))
    packaged_para_list = output / "primary_condition_table.csv"
    packaged_para_info = output / "primary_sampling_parameters.csv"
    shutil.copy2(para_list, packaged_para_list)
    shutil.copy2(para_info, packaged_para_info)

    raw_rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.csv"), key=lambda item: item.name.lower()):
        raw_rows.append(
            {
                "logical_path": relative_locator(path, source_root),
                "bytes": path.stat().st_size,
                "rows": csv_rows(path),
                "sha256": sha256(path),
                "status": "recovered_local_hash_registered_not_packaged",
            }
        )
    write_csv(
        output / "primary_raw_output_inventory.csv",
        ("logical_path", "bytes", "rows", "sha256", "status"),
        raw_rows,
    )

    native_rows: list[dict[str, object]] = []
    for path in sorted(native_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".fsp", ".ldev", ".log"}:
            continue
        date_match = re.search(r"_(\d{8})_", path.name)
        run_date = date_match.group(1) if date_match else "unknown"
        relation = "primary_campaign_date_match" if run_date == CAMPAIGN_DATE else "earlier_development_context"
        summary = read_log_summary(path) if path.suffix.lower() == ".log" else {
            "solver": "FDTD" if path.suffix.lower() == ".fsp" else "TCAD",
            "software_version": "embedded native project; inspect with licensed TCAD software",
            "gridpoint_shapes": "",
            "vertices": "",
            "elements": "",
        }
        native_rows.append(
            {
                "logical_path": relative_locator(path, source_root),
                "file_type": path.suffix.lower().lstrip("."),
                "run_date": run_date,
                "campaign_relation": relation,
                "bytes": path.stat().st_size,
                "modified_utc": iso_mtime(path),
                "sha256": sha256(path),
                "solver": summary["solver"],
                "software_version": summary["software_version"],
                "gridpoint_shapes": summary["gridpoint_shapes"],
                "vertices": summary["vertices"],
                "elements": summary["elements"],
                "publication_status": "hash_registered_not_packaged_commercial_binary_or_sensitive_log",
            }
        )
    write_csv(
        output / "native_session_inventory.csv",
        (
            "logical_path",
            "file_type",
            "run_date",
            "campaign_relation",
            "bytes",
            "modified_utc",
            "sha256",
            "solver",
            "software_version",
            "gridpoint_shapes",
            "vertices",
            "elements",
            "publication_status",
        ),
        native_rows,
    )

    snapshot_rows: list[dict[str, object]] = []
    for name in SOURCE_SNAPSHOTS:
        path = source_root / name
        if not path.exists():
            continue
        snapshot_rows.append(
            {
                "logical_path": relative_locator(path, source_root),
                "bytes": path.stat().st_size,
                "modified_utc": iso_mtime(path),
                "sha256": sha256(path),
                "status": "recovered_later_working_snapshot_not_bound_to_primary_campaign",
            }
        )
    write_csv(
        output / "source_snapshot_inventory.csv",
        ("logical_path", "bytes", "modified_utc", "sha256", "status"),
        snapshot_rows,
    )

    native_counts = Counter(row["file_type"] for row in native_rows)
    primary_native_counts = Counter(
        row["file_type"] for row in native_rows if row["campaign_relation"] == "primary_campaign_date_match"
    )
    software_versions = sorted(
        {
            str(row["software_version"])
            for row in native_rows
            if row["file_type"] == "log" and str(row["software_version"]).startswith("commercial TCAD")
        }
    )

    canonical_files = {
        "primary_aggregated_table": ROOT / "artifacts" / "results" / "device_modeling" / "cleaned_data.csv",
        "apparent_capacitance_manifest": ROOT / "artifacts" / "results" / "apparent_capacitance_correction" / "manifest.json",
        "apparent_capacitance_table": ROOT / "artifacts" / "results" / "apparent_capacitance_correction" / "primary_ge_si_apparent_capacitance.csv",
        "terminal_charge_provenance": ROOT / "artifacts" / "results" / "terminal_charge_model" / "terminal_charge_model_provenance.json",
    }
    canonical = {}
    for key, path in canonical_files.items():
        canonical[key] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": "tcad_provenance_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_taxonomy": {
            "recovered": "The exact retained file is present and hash registered.",
            "partial": "Only a subset or later unbound snapshot is present.",
            "unavailable": "The item is absent from the recovered snapshot and is not asserted.",
        },
        "primary_campaign": {
            "logical_root": (Path("<FROZEN_TCAD_ROOT>") / CAMPAIGN_RELATIVE).as_posix(),
            "condition_count": csv_rows(para_list),
            "raw_condition_file_count": len(raw_rows),
            "condition_table": {
                "packaged_path": packaged_para_list.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(para_list),
                "packaged_sha256": sha256(packaged_para_list),
            },
            "sampling_parameters": {
                "packaged_path": packaged_para_info.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(para_info),
                "packaged_sha256": sha256(packaged_para_info),
            },
        },
        "native_session_recovery": {
            "all_retained_counts": dict(sorted(native_counts.items())),
            "primary_campaign_date_match_counts": dict(sorted(primary_native_counts.items())),
            "interpretation": (
                "The primary-date subset contains four FDTD projects, four TCAD projects, and eight logs. "
                "It is partial evidence, not a 160-condition native-session archive. Other retained sessions "
                "are earlier development context and are not attributed to the primary campaign."
            ),
            "mesh_boundary": (
                "Mesh state is embedded in retained .fsp/.ldev projects; no separate complete mesh archive "
                "was recovered. Sanitized log-derived grid/element summaries are in native_session_inventory.csv."
            ),
            "software_versions": software_versions,
        },
        "canonical_derivatives": canonical,
        "recoverability": [
            {
                "item": "frozen primary condition table and sampling-parameter table",
                "status": "recovered_and_packaged",
                "evidence": "primary_condition_table.csv; primary_sampling_parameters.csv",
            },
            {
                "item": "160 extracted per-condition primary output tables",
                "status": "recovered_local_hash_registered_not_packaged",
                "evidence": "primary_raw_output_inventory.csv",
            },
            {
                "item": "native FDTD/TCAD projects and logs for primary campaign",
                "status": "partial",
                "evidence": "4 .fsp, 4 .ldev, and 8 logs match the campaign date; see native_session_inventory.csv",
            },
            {
                "item": "mesh state",
                "status": "partial",
                "evidence": "embedded in retained native sessions; sanitized grid/element summaries recovered from logs",
            },
            {
                "item": "current TCAD generation/source scripts",
                "status": "recovered_later_snapshot",
                "evidence": "source_snapshot_inventory.csv",
            },
            {
                "item": "original random-sampling seed",
                "status": "unavailable",
                "evidence": "no seed call or run record was recovered; reproduction starts from the frozen table",
            },
            {
                "item": "exact source-code revision executed for the primary campaign",
                "status": "unavailable",
                "evidence": "no version-control binding was recovered and current snapshots postdate the campaign",
            },
            {
                "item": "native sessions and logs for all 160 primary conditions",
                "status": "unavailable",
                "evidence": "only the primary-date subset listed above was recovered",
            },
        ],
        "publication_boundary": {
            "not_packaged": [
                "commercial TCAD .fsp/.ldev binaries",
                "raw logs containing host, license, and local-path details",
                "large local per-condition raw tables already represented by hash inventory and canonical tables",
            ],
            "claim": (
                "The artifact supports exact replay from frozen tables and canonical derivatives. It does not "
                "claim byte-for-byte regeneration of every original TCAD solve from a sampling seed and complete native deck archive."
            ),
        },
    }
    (output / "tcad_provenance_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    provenance_note = """# TCAD 来源追溯边界

本目录区分已恢复证据、部分恢复证据和当前快照中不可恢复的历史信息。冻结的
160-condition 参数表及其采样参数表按原字节打包；160 个逐条件提取文件、保留的
TCAD native sessions、后期源码快照和 canonical derivative artifacts 均在
CSV/JSON 清单中登记 SHA-256。

原始 sampling seed 和主 campaign 实际执行的精确 source-code revision 未能恢复。
只有 4 个 FDTD 与 4 个 TCAD native projects 匹配主 campaign 日期，因此不能把
native archive 描述为完整。Mesh state 保存在已留存的 native projects 中；清单只
发布由日志提取并脱敏的 grid/vertex/element 摘要。商业二进制 projects 以及包含主机、
许可和本地路径信息的原始日志不纳入匿名 artifact。论文所报机器学习实验从打包的冻结
condition/canonical tables 开始复现，不声称能够重新生成每一次原始 TCAD solve 的逐字节
等价副本。
"""
    (output / "PROVENANCE.md").write_text(provenance_note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source_root, args.output)


if __name__ == "__main__":
    main()
