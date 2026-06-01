import sys
from pathlib import Path
import shutil
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_dataset_scan import find_band_roots, scan_dataset


def qe_text(total, absolute, energy=-1.0):
    return f"""
    Program PWSCF
    !    total energy              =   {energy:.8f} Ry
         total magnetization       =   {total:.2f} Bohr mag/cell
         absolute magnetization    =   {absolute:.2f} Bohr mag/cell
    convergence has been achieved
    JOB DONE.
    """


def write_band(root, mags):
    for i, (total, absolute) in enumerate(mags, start=1):
        image_dir = root / f"image_{i:02d}"
        image_dir.mkdir(parents=True)
        (image_dir / "espresso.pwo").write_text(qe_text(total, absolute, energy=-float(i)), encoding="utf-8")


def test_dataset_scan_finds_and_classifies_band_roots():
    tmp_root = Path(__file__).resolve().parent / f"_tmp_dataset_scan_{uuid.uuid4().hex}"
    try:
        good = tmp_root / "good_band"
        bad = tmp_root / "bad_band"
        write_band(good, [(1.13, 2.0), (1.14, 2.1), (1.13, 2.0)])
        write_band(bad, [(1.13, 2.0), (1.13, 2.7), (1.13, 2.8)])

        roots = find_band_roots(tmp_root)
        rows = scan_dataset(tmp_root)

        assert roots == [bad, good]
        assert {row.band_root: row.verdict for row in rows} == {
            str(bad): "NO-GO_SINGLE_SHEET",
            str(good): "GO",
        }
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
