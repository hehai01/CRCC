# -*- coding: utf-8 -*-
"""Convert the original LLCM layout to TokenMatcher/Market1501-style folders.

Original LLCM (from official LLCM dataset):
    LLCM/idx/train_vis.txt       (relative image path, identity label)
    LLCM/idx/train_nir.txt
    LLCM/idx/test_id.txt
    LLCM/vis/, nir/, test_vis/, test_nir/

Generated (inside the same LLCM root):
    rgb_modify/{bounding_box_train,query,bounding_box_test}/<pid>_c<cam>_<unique>.<ext>
    ir_modify/{bounding_box_train,query,bounding_box_test}/<pid>_c<cam>_<unique>.<ext>

Train images come ONLY from idx/train_vis.txt and idx/train_nir.txt.
Test images come from the original test_vis/test_nir folders, filtered by test_id.txt.
The train index labels and test folder IDs are not assumed to use the same
numeric namespace. Their numeric overlap is diagnostic ONLY; identical source
image paths are a hard failure. In case of genuinely shared global identity
labels, use --strict-id-disjoint to enforce ID-level disjointness.
The test copies serve for Dataset statistics/inspection; for publication-quality
LLCM evaluation, use the official 10-trial random-gallery protocol, not a fixed
Market1501 gallery split.

Usage (PowerShell):
    python prepare_llcm.py --data-dir data/LLCM
    python prepare_llcm.py --data-dir data/LLCM --copy-mode hardlink
    python prepare_llcm.py --data-dir data/LLCM --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
CAM_DIR_RE = re.compile(r'^cam(\d+)$', re.IGNORECASE)
CAM_FILE_RE = re.compile(r'(?:^|_)c(\d+)(?:_|\.)', re.IGNORECASE)
PID_FILE_RE = re.compile(r'^(\d+)_c\d+(?:_|\.)', re.IGNORECASE)


def resolve_relative(root: Path, raw: str) -> Path:
    """Resolve LLCM idx paths safely, accepting Windows or POSIX separators."""
    rel = raw.strip().replace('\\', '/')
    if rel.startswith('/') or re.match(r'^[A-Za-z]:', rel):
        raise ValueError('Expected relative path in LLCM idx: ' + raw)
    while rel.startswith('./'):
        rel = rel[2:]
    parts = Path(rel).parts
    if '..' in parts or not parts:
        raise ValueError('Unsafe relative path in LLCM idx: ' + raw)
    path = root.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError('Listed image does not exist: ' + str(path))
    return path


def parse_idx(idx_file: Path, root: Path) -> list[tuple[Path, int]]:
    """Read official 'relative_path pid' lists. Labels are metadata only."""
    if not idx_file.is_file():
        raise FileNotFoundError('Missing official LLCM index: ' + str(idx_file))
    entries = []
    seen = set()
    with idx_file.open('r', encoding='utf-8-sig') as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Last field is the ID; preceding text is a relative image path.
            try:
                rel, pid_txt = line.rsplit(maxsplit=1)
                pid = int(pid_txt)
            except (ValueError, TypeError):
                raise ValueError(f'Bad index line {idx_file}:{no}: {line!r}')
            if pid < 0:  # Consistent with Market1501 junk ID policy
                continue
            img = resolve_relative(root, rel)
            if img.suffix.lower() not in IMAGE_EXTS:
                raise ValueError('Not a supported image type: ' + str(img))
            if img in seen:
                raise ValueError(f'Duplicate image in {idx_file}: {img}')
            seen.add(img)
            entries.append((img, pid))
    if not entries:
        raise RuntimeError('No usable entries in ' + str(idx_file))
    return entries


def camera_id(path: Path) -> int:
    """Obtain camera from a camN path component and/or *_cN_* filename."""
    from_dirs = [int(m.group(1)) for p in path.parts if (m := CAM_DIR_RE.fullmatch(p))]
    from_file = CAM_FILE_RE.search(path.name)
    filename_cam = int(from_file.group(1)) if from_file else None
    dir_cam = from_dirs[-1] if from_dirs else None
    if filename_cam is not None and dir_cam is not None and filename_cam != dir_cam:
        raise ValueError(f'Camera mismatch between directory and filename: {path}')
    cam = dir_cam if dir_cam is not None else filename_cam
    if cam is None or not 1 <= cam <= 9:
        raise ValueError(f'Cannot determine valid LLCM camera (1..9) for: {path}')
    return cam


def parse_test_ids(idx_file: Path) -> set[int] | None:
    if not idx_file.exists():
        return None
    text = idx_file.read_text(encoding='utf-8-sig')
    ids = {int(x) for x in re.findall(r'\d+', text)}
    if not ids:
        raise ValueError(f'Empty LLCM test identity list: {idx_file}')
    return ids


def infer_pid_for_test(path: Path) -> int:
    if path.parent.name.isdigit():
        return int(path.parent.name)
    m = PID_FILE_RE.match(path.name)
    if m:
        return int(m.group(1))
    raise ValueError('Cannot infer test identity from LLCM image path: ' + str(path))


def read_test_entries(root: Path, modality: str, test_ids: set[int] | None):
    """Scan original official test image trees; use indices only as fallback."""
    folder = root / ('test_vis' if modality == 'rgb' else 'test_nir')
    if folder.is_dir():
        entries = []
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                pid = infer_pid_for_test(p)
                if test_ids is None or pid in test_ids:
                    entries.append((p, pid))
        if entries:
            return entries
    # Some distributions provide test_vis.txt and test_nir.txt instead.
    idx_file = root / 'idx' / ('test_vis.txt' if modality == 'rgb' else 'test_nir.txt')
    if idx_file.is_file():
        entries = parse_idx(idx_file, root)
        return [(p, pid) for p, pid in entries if test_ids is None or pid in test_ids]
    raise RuntimeError(f'No test samples for {modality}: expected {folder} or {idx_file}')


def make_name(src: Path, pid: int, cam: int, root: Path) -> str:
    """Market1501 prefix plus stable digest prevents filename collisions."""
    relative = src.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode('utf-8')).hexdigest()[:10]
    stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', src.stem)[:95]
    return f'{pid:04d}_c{cam}_{digest}_{stem}{src.suffix.lower()}'


def export(entries, root: Path, out_dir: Path, mode: str, dry_run: bool) -> dict:
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    destinations = set()
    for src, pid in entries:
        cam = camera_id(src)
        dst = out_dir / make_name(src, pid, cam, root)
        if dst in destinations:
            raise RuntimeError('Output collision: ' + str(dst))
        destinations.add(dst)
        if not dry_run and not dst.exists():
            if mode == 'copy':
                shutil.copy2(src, dst)
            elif mode == 'hardlink':
                os.link(src, dst)  # must be on the same filesystem
            else:
                raise ValueError('Unknown copy mode: ' + mode)
        records.append((pid, cam))
    return {'images': len(records), 'ids': len({p for p, _ in records}),
            'cameras': sorted({c for _, c in records})}


def prepare(root: Path, copy_mode='copy', dry_run=False,
            strict_id_disjoint=False) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError('LLCM root does not exist: ' + str(root))

    rgb_train = parse_idx(root / 'idx' / 'train_vis.txt', root)
    ir_train = parse_idx(root / 'idx' / 'train_nir.txt', root)
    test_ids = parse_test_ids(root / 'idx' / 'test_id.txt')
    rgb_test = read_test_entries(root, 'rgb', test_ids)
    ir_test = read_test_entries(root, 'ir', test_ids)

    train_ids = {pid for _, pid in rgb_train + ir_train}
    actual_test_ids = {pid for _, pid in rgb_test + ir_test}
    id_overlap = train_ids & actual_test_ids

    # LLCM's train-index identity labels and test identity directory names may
    # refer to different label spaces. Equal numbers alone are not proof of
    # identity leakage. Do NOT alter test IDs or use train IDs for supervision.
    print(f'Index ID statistics: train={len(train_ids)}, test={len(actual_test_ids)}')
    if id_overlap:
        print(f'NOTE: {len(id_overlap)} numeric IDs occur in both the training '
              f'index labels and test directory labels (examples: '
              f'{sorted(id_overlap)[:20]}). This does not prove person leakage '
              f'if the label spaces were independently remapped.')
        if strict_id_disjoint:
            raise RuntimeError(
                'Overlapping numeric train/test IDs with --strict-id-disjoint. '
                'Only enable this option if both sets use identical global PID labels.'
            )

    # Canonical image source paths MUST be disjoint even if numeric labels overlap.
    train_paths = {p.resolve() for p, _ in rgb_train + ir_train}
    test_paths = {p.resolve() for p, _ in rgb_test + ir_test}
    duplicated_paths = train_paths & test_paths
    if duplicated_paths:
        examples = list(sorted(map(str, duplicated_paths)))[:5]
        raise RuntimeError(f'Train/test IMAGE path overlap detected: {examples}')
    print('Training and testing image source paths: disjoint (verified).')

    summaries = {}
    for modality, train_entries, test_entries in (
        ('rgb', rgb_train, rgb_test), ('ir', ir_train, ir_test)
    ):
        base = root / ('rgb_modify' if modality == 'rgb' else 'ir_modify')
        summaries[modality] = {
            'train': export(train_entries, root, base / 'bounding_box_train', copy_mode, dry_run),
            'query': export(test_entries, root, base / 'query', copy_mode, dry_run),
            'gallery': export(test_entries, root, base / 'bounding_box_test', copy_mode, dry_run)
        }
    print('LLCM to Market1501-style conversion', '(DRY RUN)' if dry_run else '')
    print('Dataset root:', root)
    for modality, sets in summaries.items():
        print(f'  {modality.upper()}:')
        for split, info in sets.items():
            print(f"    {split:7}  {info['ids']:4} IDs  {info['images']:6} images  "
                  f"cams={info['cameras']}")
    print('Note: query and bounding_box_test contain the full test image pool. '
          'Use the official 10-trial random-gallery evaluation in test_llcm.py.')
    return summaries


def main():
    parser = argparse.ArgumentParser(description='Prepare original LLCM for TokenMatcher')
    parser.add_argument('--data-dir', default='data/LLCM', help='Original LLCM directory')
    parser.add_argument('--copy-mode', choices=('copy', 'hardlink'), default='copy',
                        help='copy is portable; hardlink saves disk space on same filesystem')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and show counts only')
    parser.add_argument('--strict-id-disjoint', action='store_true',
                        help='Also reject overlapping PID numbers ONLY if train/test use one global ID namespace')
    args = parser.parse_args()
    prepare(Path(args.data_dir), copy_mode=args.copy_mode,
            dry_run=args.dry_run, strict_id_disjoint=args.strict_id_disjoint)


if __name__ == '__main__':
    main()
