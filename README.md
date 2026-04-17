# whisperx-forced-aligner

WhisperX **Stage 2 only** — forced alignment as a RunPod serverless endpoint.

No Whisper transcription. Provide your own script, get precise word-level timestamps back.

---

## What It Does

1. Accepts your script segments + audio URL
2. Pre-normalizes numbers (`"1087"` → `"ten eighty-seven"`) so wav2vec2 can align them
3. Runs **WhisperX Stage 2** (wav2vec2) to force-align your script to the audio
4. Detects speech gaps using **silero-vad** and inserts `[UNMATCHED]` tokens where TTS added extra audio not in the script
5. Returns a flat word list with precise `start`/`end` timestamps

---

## API

### Input

```json
{
  "input": {
    "audio_url": "https://your-cdn.com/audio.mp3",
    "segments": [
      { "text": "William the Conqueror ruled England" },
      { "text": "He crossed the English Channel in 1066" }
    ],
    "language": "en",
    "vad_gap_min": 0.3
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_url` | string | ✅ | — | Public URL to audio (mp3/wav/m4a) |
| `segments` | array | ✅ | — | `[{text}]` — your script split into scenes |
| `language` | string | ❌ | `"en"` | BCP-47 language code |
| `vad_gap_min` | float | ❌ | `0.3` | Minimum gap (seconds) to check for unmatched speech |

### Output

```json
{
  "words": [
    { "word": "William",      "start": 0.10, "end": 0.42 },
    { "word": "the",          "start": 0.42, "end": 0.55 },
    { "word": "Conqueror",    "start": 0.55, "end": 0.98 },
    { "word": "[UNMATCHED]",  "start": 0.98, "end": 3.50 },
    { "word": "He",           "start": 3.50, "end": 3.72 }
  ],
  "duration": 10.24,
  "word_count": 8
}
```

`[UNMATCHED]` tokens mark regions where **VAD detected speech** but no script word was aligned — these are TTS extras or ad-libs that weren't in your script.

---

## Deploy to RunPod

### 1. Build and push Docker image

**Option A — GitHub Actions (recommended)**

1. Create a new public GitHub repo and push this code to `main`
2. Go to **Settings → Secrets** and add:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — a Docker Hub access token (not your password)
3. The workflow at `.github/workflows/docker-publish.yml` will auto-build and push on every commit

**Option B — Build locally**

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/whisperx-forced-aligner:latest .
docker push YOUR_DOCKERHUB_USERNAME/whisperx-forced-aligner:latest
```

---

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Custom Source** → enter your Docker Hub image:
   ```
   YOUR_DOCKERHUB_USERNAME/whisperx-forced-aligner:latest
   ```
4. GPU: **16 GB VRAM** (cheapest tier — wav2vec2 only needs ~2GB but this is the minimum tier)
5. Min workers: `0` (serverless — you only pay when it runs)
6. Max workers: `3` (adjust to your load)
7. Click **Deploy**
8. Copy the **Endpoint ID** — you'll need it in your Cloudflare Worker

---

### 3. Call from Cloudflare Worker

```js
// worker.mjs
const RUNPOD_API_KEY   = env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT  = env.RUNPOD_WHISPERX_ENDPOINT_ID;

const response = await fetch(
  `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/runsync`,
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: {
        audio_url: audioUrl,
        segments: clips.map(c => ({ text: c.scriptSnippet })),
        language: 'en'
      }
    })
  }
);

const data = await response.json();
const words = data.output?.words ?? [];
```

---

## Cost Estimate

| GPU Tier | Cost/sec | 5-min audio job | Per 100 jobs |
|---|---|---|---|
| 16GB VRAM | ~$0.00016/s | ~$0.003 | ~$0.30 |
| A100 80GB | ~$0.00070/s | ~$0.014 | ~$1.40 |

wav2vec2 is very fast — typically **10-25 seconds** processing for a 5-minute audio file.

---

## Models Baked Into Image

| Model | Size | Purpose |
|---|---|---|
| `jonatasgrosman/wav2vec2-large-960h-lv60-self` | ~1.2GB | Forced alignment (English) |
| `silero-vad` | ~3MB | Voice activity detection |

No Whisper model is downloaded. Image is significantly smaller than full WhisperX images.

---

## Notes

- **Numbers**: Automatically converted to spoken form before alignment (`"1066"` → `"one thousand and sixty-six"`). TTS-generated audio should match.
- **Extra TTS audio**: Detected by silero-vad and returned as `[UNMATCHED]` tokens with timestamps.
- **Missing timestamps**: Any word that wav2vec2 can't align gets its timestamp interpolated from surrounding words.
- **Cold start**: First request after idle may take 30-60s to spin up the pod. Subsequent requests on the same pod are fast.
