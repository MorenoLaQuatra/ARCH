from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from arch_eval.evaluation.classification.music.fma_small import FMASmall


def write_tracks_csv(dataset_root: Path, rows: List[Tuple[int, str]]) -> None:
    metadata_directory = dataset_root / "fma_metadata"
    metadata_directory.mkdir(parents=True)
    columns = pd.MultiIndex.from_tuples([("track", "genre_top")])
    tracks = pd.DataFrame(
        [[label] for _, label in rows],
        index=[track_id for track_id, _ in rows],
        columns=columns,
    )
    tracks.to_csv(metadata_directory / "tracks.csv")


def create_official_layout_track(dataset_root: Path, track_id: int) -> Path:
    track_filename = f"{track_id:06d}.mp3"
    track_path = (
        dataset_root / "fma_small" / track_filename[:3] / track_filename
    )
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.touch()
    return track_path


@pytest.mark.parametrize(
    ("track_id", "relative_path"),
    [
        (2, "000/000002.mp3"),
        (6407, "006/006407.mp3"),
        (100001, "100/100001.mp3"),
    ],
)
def test_audio_path_matches_official_fma_layout(
    track_id: int,
    relative_path: str,
) -> None:
    assert Path(FMASmall._audio_path("/audio", track_id)) == Path(
        "/audio"
    ) / relative_path


def test_audio_path_accepts_legacy_flat_layout(tmp_path: Path) -> None:
    flat_track = tmp_path / "000002.mp3"
    flat_track.touch()

    assert Path(FMASmall._audio_path(str(tmp_path), 2)) == flat_track


def test_audio_path_prefers_official_layout_over_flat_fallback(
    tmp_path: Path,
) -> None:
    nested_track = tmp_path / "000" / "000002.mp3"
    nested_track.parent.mkdir()
    nested_track.touch()
    flat_track = tmp_path / "000002.mp3"
    flat_track.touch()

    assert Path(FMASmall._audio_path(str(tmp_path), 2)) == nested_track


def test_load_data_accepts_legacy_flat_layout(tmp_path: Path) -> None:
    rows = [
        (track_id, "Rock" if index % 2 else "Jazz")
        for index, track_id in enumerate(range(2, 12))
    ]
    write_tracks_csv(tmp_path, rows)
    audio_directory = tmp_path / "fma_small"
    audio_directory.mkdir()
    for track_id, _ in rows:
        (audio_directory / f"{track_id:06d}.mp3").touch()

    dataset = FMASmall(f"{tmp_path}/")

    assert (
        len(dataset.train_paths)
        + len(dataset.validation_paths)
        + len(dataset.test_paths)
        == 10
    )
    assert dataset.num_classes == 2


def test_load_data_filters_before_encoding_classes(tmp_path: Path) -> None:
    present_rows = [
        (2, "Rock"),
        (5, "Jazz"),
        (10, "Rock"),
        (140, "Jazz"),
        (141, "Rock"),
        (1234, "Jazz"),
        (6407, "Rock"),
        (99999, "Jazz"),
        (100001, "Rock"),
        (155066, "Jazz"),
    ]
    write_tracks_csv(tmp_path, present_rows + [(200001, "Absent class")])
    expected_paths = {
        create_official_layout_track(tmp_path, track_id)
        for track_id, _ in present_rows
    }

    dataset = FMASmall(f"{tmp_path}/")

    loaded_paths = {
        Path(path)
        for path in (
            dataset.train_paths
            + dataset.validation_paths
            + dataset.test_paths
        )
    }
    loaded_labels = set(
        list(dataset.train_labels)
        + list(dataset.validation_labels)
        + list(dataset.test_labels)
    )
    assert loaded_paths == expected_paths
    assert loaded_labels == {0, 1}
    assert dataset.num_classes == 2
    assert len(dataset.train_paths) == 8
    assert len(dataset.validation_paths) == 1
    assert len(dataset.test_paths) == 1


def test_load_data_explains_missing_official_layout(tmp_path: Path) -> None:
    write_tracks_csv(tmp_path, [(2, "Rock")])
    (tmp_path / "fma_small").mkdir()

    with pytest.raises(FileNotFoundError, match="fma_small/000/000002.mp3"):
        FMASmall(f"{tmp_path}/")
