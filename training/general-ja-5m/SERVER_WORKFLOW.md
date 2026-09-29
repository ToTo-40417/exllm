# Unattended server workflow

The AI server contains one self-contained project root:

```text
~/exllm-general-ja-5m/
├── incoming/       immutable E130 snapshots
├── workspace/      source snapshot and isolated Python environment
├── sources/        downloaded source corpora plus provenance
├── corpus/         normalized JSONL and packed uint16 streams
├── runs/           checkpoints and immutable run manifests
├── jobs/           PID, status and logs for detached jobs
└── artifacts/      evaluated exports ready to return to E130
```

Nothing in this tree depends on LM Studio or another server project. E130 creates
and hashes source snapshots. The server may download allowlisted corpora into
`sources/`, but it may not silently add a source to `sources.json`.

Long operations are started with `tools/server_job.py`. They survive the SSH
session and record their command, PID, timestamps, return code and combined log.

```sh
.venv-server/bin/python tools/server_job.py start \
  --root "$HOME/exllm-general-ja-5m/jobs" \
  --name pack-language -- \
  .venv-server/bin/python tools/pack_text_corpus.py ...

.venv-server/bin/python tools/server_job.py status \
  --root "$HOME/exllm-general-ja-5m/jobs" \
  --name pack-language
```

The final pipeline is allowed to start only when corpus manifests account for
every input and separately report processed, loss-bearing, padding and truncated
tokens. Tokenizer audit fields (byte fallback, tokens per Unicode character and
context-limit incidence) are also required. The measured token budget must be
within the experiment definition.
Intermediate checkpoints are retained until the final EXLLM8 and EXQ12 exports
pass host evaluation and DATAPLUS 6/7 device validation.
