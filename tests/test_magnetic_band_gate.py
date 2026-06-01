import sys
from pathlib import Path
import shutil
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.magnetic_band_gate import BandImage, analyze_band_images, discover_band_outputs, load_band
from prodromos.magnetic_output_parser import MagneticOutputSummary


def make_image(index, total, absolute, energy=0.0, converged=True):
    return BandImage(
        label=f"image_{index:02d}",
        index=index,
        path=f"image_{index:02d}/espresso.pwo",
        summary=MagneticOutputSummary(
            engine="qe",
            path=f"image_{index:02d}/espresso.pwo",
            scf_converged=converged,
            energy_eV=energy,
            energy_unit="eV",
            total_magnetization_uB=total,
            absolute_magnetization_uB=absolute,
        ),
    )


def test_band_gate_go_for_smooth_single_sheet_band():
    images = [
        make_image(1, 1.13, 2.0, 0.0),
        make_image(2, 1.14, 2.1, 0.1),
        make_image(3, 1.12, 2.2, 0.0),
    ]

    result = analyze_band_images(images)

    assert result.verdict == "GO"
    assert result.sheet_crossing is False
    assert result.endpoint_split is False
    assert result.roles == ["ok", "ok", "ok"]


def test_band_gate_no_go_for_adjacent_abs_jump():
    images = [
        make_image(1, 1.13, 2.0, 0.0),
        make_image(2, 1.13, 2.1, 0.2),
        make_image(3, 1.13, 2.8, 0.1),
    ]

    result = analyze_band_images(images)

    assert result.verdict == "NO-GO_SINGLE_SHEET"
    assert result.sheet_crossing is True
    assert result.crossing_edge == 1
    assert result.delta_abs_adj[1] == pytest.approx(0.7)
    assert result.roles[1] == "spin_split"


def test_band_gate_review_for_missing_magnetization():
    images = [
        make_image(1, 1.13, 2.0),
        make_image(2, None, None),
    ]

    result = analyze_band_images(images)

    assert result.verdict == "REVIEW"
    assert result.sheet_crossing is False
    assert any("missing" in reason for reason in result.reasons)


def test_band_gate_review_for_incomplete_output_even_when_split_visible():
    images = [
        make_image(1, 1.67, 2.4, converged=False),
        make_image(2, 1.13, 2.0),
    ]

    result = analyze_band_images(images)

    assert result.verdict == "REVIEW"
    assert result.endpoint_split is True
    assert any("incomplete" in reason for reason in result.reasons)


def test_discover_band_outputs_sorts_image_directories():
    tmp_root = Path(__file__).resolve().parent / f"_tmp_band_gate_{uuid.uuid4().hex}"
    try:
        for name in ["image_10", "image_02", "image_01"]:
            image_dir = tmp_root / name
            image_dir.mkdir(parents=True)
            (image_dir / "espresso.pwo").write_text("Program PWSCF", encoding="utf-8")

        outputs = discover_band_outputs(tmp_root)

        assert [path.parent.name for path in outputs] == ["image_01", "image_02", "image_10"]
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


@pytest.mark.requires_data
def test_marc_tier1_band_corpus_if_available():
    root = Path(r"D:\home\ignat\project-third-matter\results\dft_datasets\2026-05-28_marc_VFe_tier1_v4c\neb_done")
    if not root.exists():
        pytest.skip("local harvested DFT corpus is not available")

    result = analyze_band_images(load_band(root))

    assert result.verdict == "NO-GO_SINGLE_SHEET"
    assert result.sheet_crossing is True
    assert result.endpoint_split is True
    assert result.crossing_edge == 3
    assert result.delta_abs_adj[3] == pytest.approx(0.55)
