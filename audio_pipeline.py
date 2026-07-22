"""Build one canonical mono 16 kHz WAV from ordered timeline source records."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import math
import os
import subprocess
import wave

import requests


SAMPLE_RATE = 16_000
DOWNLOAD_CONCURRENCY = max(1, min(20, int(os.environ.get("DOWNLOAD_CONCURRENCY", "20"))))
FFMPEG_CONCURRENCY = max(4, min(6, int(os.environ.get("FFMPEG_CONCURRENCY", "5"))))
FFMPEG_TIMEOUT_SECONDS = int(os.environ.get("FFMPEG_TIMEOUT_SECONDS", "900"))
FFMPEG_BINARY = os.environ.get("FFMPEG_BINARY", "ffmpeg")
MAX_SOURCE_RECORDS = int(os.environ.get("MAX_SOURCE_RECORDS", "500"))


def _finite_number(value, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def normalize_source_records(raw_records: list) -> list:
    """Validate records and return them in final timeline order."""
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("source_records must be a non-empty array")
    if len(raw_records) > MAX_SOURCE_RECORDS:
        raise ValueError(f"source_records cannot exceed {MAX_SOURCE_RECORDS} records")

    normalized = []
    for input_index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"source_records[{input_index}] must be an object")

        raw_sources = raw_record.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"source_records[{input_index}].sources must be non-empty")

        sources = []
        for source_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, dict):
                raise ValueError(
                    f"source_records[{input_index}].sources[{source_index}] must be an object"
                )
            url = str(raw_source.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    f"source_records[{input_index}].sources[{source_index}].url must be HTTP(S)"
                )
            start = max(0.0, _finite_number(raw_source.get("start", 0), "source start"))
            end = _finite_number(raw_source.get("end"), "source end")
            if end <= start:
                raise ValueError(
                    f"source_records[{input_index}].sources[{source_index}] has end <= start"
                )
            sources.append({"url": url, "start": start, "end": end})

        clip_duration = sum(source["end"] - source["start"] for source in sources)
        raw_keep_ranges = raw_record.get("keep_ranges") or []
        if not isinstance(raw_keep_ranges, list):
            raise ValueError(f"source_records[{input_index}].keep_ranges must be an array")
        keep_ranges = []
        for range_index, raw_range in enumerate(raw_keep_ranges):
            if not isinstance(raw_range, dict):
                raise ValueError(
                    f"source_records[{input_index}].keep_ranges[{range_index}] must be an object"
                )
            start = max(0.0, _finite_number(raw_range.get("start", 0), "keep range start"))
            end = min(clip_duration, _finite_number(raw_range.get("end"), "keep range end"))
            if end > start:
                keep_ranges.append({"start": start, "end": end})

        if raw_keep_ranges and not keep_ranges:
            raise ValueError(f"source_records[{input_index}] has no valid keep_ranges")

        order = _finite_number(raw_record.get("order", input_index), "record order")
        normalized.append(
            {
                "clip_id": str(raw_record.get("clip_id", input_index)),
                "order": order,
                "input_index": input_index,
                "sources": sources,
                "keep_ranges": keep_ranges,
                "clip_duration": clip_duration,
            }
        )

    normalized.sort(key=lambda record: (record["order"], record["input_index"]))
    return normalized


def _download_destination(url: str, download_dir: Path) -> Path:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}:
        suffix = ".audio"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return download_dir / f"{digest}{suffix}"


def _download_url(url: str, destination: Path) -> Path:
    with requests.get(url, timeout=(20, 180), stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return destination


def download_sources(records: list, work_dir: Path) -> dict:
    """Download each unique source URL with at most 20 concurrent requests."""
    download_dir = work_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    urls = list(
        dict.fromkeys(source["url"] for record in records for source in record["sources"])
    )
    print(f"[CANONICAL] Downloading {len(urls)} unique source(s), concurrency={DOWNLOAD_CONCURRENCY}")

    downloaded = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as executor:
        futures = {
            executor.submit(_download_url, url, _download_destination(url, download_dir)): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                downloaded[url] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to download audio source: {url}") from exc
    return downloaded


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _build_filter_graph(record: dict) -> tuple[str, str]:
    filters = []
    source_labels = []
    for index, source in enumerate(record["sources"]):
        label = f"src{index}"
        filters.append(
            f"[{index}:a]asetpts=PTS-STARTPTS,aresample={SAMPLE_RATE},"
            f"aformat=sample_fmts=s16:channel_layouts=mono[{label}]"
        )
        source_labels.append(f"[{label}]")

    if len(source_labels) == 1:
        base_label = "src0"
    else:
        filters.append(
            f"{''.join(source_labels)}concat=n={len(source_labels)}:v=0:a=1[base]"
        )
        base_label = "base"

    keep_ranges = record["keep_ranges"]
    if not keep_ranges:
        return ";".join(filters), base_label

    if len(keep_ranges) == 1:
        keep_range = keep_ranges[0]
        filters.append(
            f"[{base_label}]atrim=start={_format_seconds(keep_range['start'])}:"
            f"end={_format_seconds(keep_range['end'])},asetpts=PTS-STARTPTS[out]"
        )
        return ";".join(filters), "out"

    split_labels = "".join(f"[split{index}]" for index in range(len(keep_ranges)))
    filters.append(f"[{base_label}]asplit={len(keep_ranges)}{split_labels}")
    kept_labels = []
    for index, keep_range in enumerate(keep_ranges):
        filters.append(
            f"[split{index}]atrim=start={_format_seconds(keep_range['start'])}:"
            f"end={_format_seconds(keep_range['end'])},asetpts=PTS-STARTPTS[keep{index}]"
        )
        kept_labels.append(f"[keep{index}]")
    filters.append(
        f"{''.join(kept_labels)}concat=n={len(kept_labels)}:v=0:a=1[out]"
    )
    return ";".join(filters), "out"


def _process_record(record_index: int, record: dict, downloaded: dict, output_dir: Path) -> Path:
    output_path = output_dir / f"{record_index:06d}.wav"
    filter_graph, output_label = _build_filter_graph(record)
    command = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y"]
    for source in record["sources"]:
        command.extend(
            [
                "-ss",
                _format_seconds(source["start"]),
                "-t",
                _format_seconds(source["end"] - source["start"]),
                "-i",
                str(downloaded[source["url"]]),
            ]
        )
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            f"[{output_label}]",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-threads",
            "1",
            str(output_path),
        ]
    )
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"FFmpeg failed for clip {record['clip_id']}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out for clip {record['clip_id']}") from exc
    return output_path


def process_records(records: list, downloaded: dict, work_dir: Path) -> list:
    """Process 4-6 records concurrently while retaining ordered result slots."""
    output_dir = work_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = [None] * len(records)
    print(
        f"[CANONICAL] Processing {len(records)} record(s), "
        f"ffmpeg_concurrency={FFMPEG_CONCURRENCY}"
    )
    with ThreadPoolExecutor(max_workers=FFMPEG_CONCURRENCY) as executor:
        futures = {
            executor.submit(_process_record, index, record, downloaded, output_dir): index
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            index = futures[future]
            processed[index] = future.result()
    return processed


def write_canonical_wav(processed_paths: list, destination: Path) -> float:
    """Append processed clips with one writer, strictly in final timeline order."""
    total_frames = 0
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        for processed_path in processed_paths:
            with wave.open(str(processed_path), "rb") as reader:
                if (
                    reader.getnchannels() != 1
                    or reader.getsampwidth() != 2
                    or reader.getframerate() != SAMPLE_RATE
                ):
                    raise RuntimeError(f"Non-canonical FFmpeg output: {processed_path}")
                while True:
                    frames = reader.readframes(65_536)
                    if not frames:
                        break
                    writer.writeframesraw(frames)
                total_frames += reader.getnframes()
    return round(total_frames / SAMPLE_RATE, 3)


def build_canonical_wav(raw_records: list, work_dir: Path) -> tuple[Path, float, int]:
    records = normalize_source_records(raw_records)
    downloaded = download_sources(records, work_dir)
    processed = process_records(records, downloaded, work_dir)
    canonical_path = work_dir / "canonical.wav"
    duration = write_canonical_wav(processed, canonical_path)
    print(
        f"[CANONICAL] Ordered WAV complete: records={len(records)}, duration={duration}s"
    )
    return canonical_path, duration, len(records)
