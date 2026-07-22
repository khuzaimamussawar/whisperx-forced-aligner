# whisperx-forced-aligner

RunPod serverless pipeline that assembles one canonical timeline WAV, transcribes it with Whisper large-v3, and runs WhisperX forced alignment once.

## Processing pipeline

1. Receive ordered timeline `source_records` from SceneBuilder.
2. Download unique R2 audio URLs with up to 20 concurrent downloads.
3. Run 4-6 concurrent FFmpeg jobs. Each job slices its source audio, applies `keep_ranges`, and outputs mono 16 kHz PCM.
4. Append every processed record, in final timeline order, through one canonical-WAV writer.
5. Run one Whisper transcription and one WhisperX alignment job on that completed WAV.
6. Return the same flat `{word, start, end}` timestamps consumed by SceneBuilder today.

The service never runs WhisperX once per clip. It always aligns the single completed canonical WAV.

## API

### Input

```json
{
  "input": {
    "source_records": [
      {
        "clip_id": "scene-1",
        "order": 0,
        "sources": [
          {
            "url": "https://r2.example.com/audio-a.wav",
            "start": 12.5,
            "end": 19.75
          }
        ],
        "keep_ranges": [
          { "start": 0.0, "end": 2.4 },
          { "start": 3.1, "end": 7.25 }
        ]
      }
    ],
    "language": "en"
  }
}
```

`keep_ranges` are optional and are relative to the record after its source slices have been joined. A record may contain multiple `sources`; those slices are joined in their listed order before its `keep_ranges` are applied. Records are written to the canonical WAV by `order`, with input position used as the stable tie-breaker.

SceneBuilder always forwards the canonical snake_case contract shown above. For compatibility with older callers, the input boundary also accepts clip_id, id, or clipId and falls back to order when no identifier exists. It also accepts sourceRecords, audioSources, and keepRanges aliases, then normalizes them before processing.

The old `audio_url` input remains available as a single-file compatibility fallback, but SceneBuilder's Whisper X path uses `source_records`.

### Output

```json
{
  "words": [
    { "word": "William", "start": 0.1, "end": 0.42 },
    { "word": "the", "start": 0.42, "end": 0.55 }
  ],
  "duration": 10.24,
  "word_count": 2,
  "source_record_count": 200
}
```

## Concurrency settings

| Variable | Default | Behavior |
|---|---:|---|
| `DOWNLOAD_CONCURRENCY` | `20` | Clamped to a maximum of 20 concurrent unique-source downloads |
| `FFMPEG_CONCURRENCY` | `5` | Clamped to the required 4-6 concurrent processing jobs |
| `FFMPEG_TIMEOUT_SECONDS` | `900` | Timeout for one record's FFmpeg job |
| `WHISPER_BATCH_SIZE` | `8` | Whisper large-v3 transcription batch size |

## Deployment

The Docker image already includes FFmpeg. GitHub Actions builds and publishes the image when `main` changes. Configure these repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Configure SceneBuilder/RunPod with:

- `RUNPOD_WHISPERX_ENDPOINT_ID`
- `RUNPOD_API_KEY`

The image pre-downloads Whisper large-v3, the English wav2vec2 alignment model, and the NLTK tokenizer required by WhisperX.
