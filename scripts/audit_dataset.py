"""Create an audit and manifest from the extracted official 72-capture partition."""

import argparse
import json
from pathlib import Path

from src.data.manifest import DatasetAuditError, audit_digiface
from src.utils.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--seed", default=20261111, type=int)
    parser.add_argument("--archives", type=Path, help="Optional JSON mapping part directory to archive and SHA-256")
    args = parser.parse_args()
    archives = json.loads(args.archives.read_text(encoding="utf-8")) if args.archives else None
    try:
        report = audit_digiface(args.data_root, args.manifest, args.seed, archives=archives)
    except DatasetAuditError as exc:
        write_json(args.audit, exc.report)
        parser.exit(2, f"Dataset audit FAIL: {exc}\n")
    report["manifest_sha256"] = sha256_file(args.manifest)
    write_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
