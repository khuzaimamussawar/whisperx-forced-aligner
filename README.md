# whisperx-forced-aligner

Whisper X transcription and forced alignment as a RunPod serverless endpoint.

The endpoint accepts an audio URL, transcribes it with Whisper large-v3, then uses WhisperX alignment to return word-level timestamps. It does not use SceneBuilder script text as an alignment source.

## API

### Input

```json
{
  "input": {
    "audio_url": "https://your-cdn.com/audio.mp3",
    "language": "en"
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_url` | string | Yes | - | Public URL to audio (`mp3`, `wav`, `m4a`, `ogg`, or `flac`) |
| `language` | string | No | `en` | Language code for Whisper transcription and WhisperX alignment |

### Output

```json
{
  "words": [
    { "word": "William", "start": 0.1, "end": 0.42 },
    { "word": "the", "start": 0.42, "end": 0.55 },
    { "word": "Conqueror", "start": 0.55, "end": 0.98 }
  ],
  "duration": 10.24,
  "word_count": 3
}
```

Words without a native alignment timestamp are interpolated from neighboring aligned words. If transcription, alignment, or audio loading fails, the endpoint returns an `error` field; SceneBuilder treats that response as a failed sync and does not overwrite existing alignment data.

## Deploy to RunPod

1. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets. The GitHub Actions workflow builds and pushes the image whenever `main` changes.
2. Create a RunPod Serverless endpoint from the pushed image. Use a GPU with at least 16 GB of VRAM.
3. Put the endpoint ID in SceneBuilder's `RUNPOD_WHISPERX_ENDPOINT_ID` secret and its API key in `RUNPOD_API_KEY`.

The service keeps the models loaded while a worker is warm. Set `WHISPER_BATCH_SIZE` in the endpoint environment if needed; it defaults to `8` to fit typical 16 GB serverless GPUs.

## SceneBuilder request shape

```js
await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: { audio_url: audioUrl, language: 'en' }
  })
});
```

## Models baked into the image

| Model | Purpose |
|---|---|
| Whisper large-v3 | Audio transcription |
| wav2vec2 English alignment model | Word-level forced alignment |

The Docker build pre-downloads the English alignment model, Whisper large-v3, and the NLTK tokenizer required by WhisperX alignment.
