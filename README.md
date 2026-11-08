# Sogni AI — image generation provider

Adds [Sogni AI](https://sogni.ai) as an `image_gen` backend for Hermes, so the
agent can generate and edit images on Sogni's decentralised GPU network.

```
hermes plugins install sogni
```

Once installed and selected as the active `image_gen` provider, the normal tool
surface works unchanged:

```
image_generate(prompt="cinematic moonlit mountain lake")
image_generate(prompt="make it night; add fireflies", image_url="./lake.png")
```

## Why this exists

Hermes ships image backends for FAL, DeepInfra, OpenAI, OpenRouter, xAI and
Krea (Krea.ai). Sogni AI is a **different vendor** — a decentralised GPU network
fronting its own model families plus editing, upscaling, 3D reconstruction and
video — and nothing in Hermes or the plugin catalog could reach it. This plugin
closes that gap.

## Install

1. **The Sogni CLI is a hard requirement.** This plugin drives the vendor's own
   supported client rather than reimplementing Sogni's asynchronous workflow
   protocol:

   ```bash
   npm install -g @sogni-ai/sogni-creative-agent-skill
   sogni-agent-hermes --version
   ```

2. **API key.** Get one from <https://dashboard.sogni.ai> (account menu) and set
   `SOGNI_API_KEY`. Hermes reads it through its secret scope.

3. Enable the plugin and select the provider:

   ```bash
   hermes plugins enable sogni
   hermes config set image_gen.provider sogni
   ```

If the CLI is not on `PATH`, set `SOGNI_AGENT_BIN` to its full path.

## Models

Model ids below were returned by the live API
(`sogni-agent-hermes --search-models`), not copied from documentation. Only
verified ids are advertised — an unverified id would fail at generation time
instead of being honestly absent.

| Provider id | Sogni model id | Notes |
| --- | --- | --- |
| `krea-2-turbo` *(default)* | `krea2_turbo_fp8_scaled` | Fast general purpose |
| `dark-beast-krea2` | `dark_beast_krea2_fp8` | Higher fidelity, more workers |
| `krea-2-identity-edit` | `krea2_identity_edit_v1_2` | Editing; needs a source image |
| `dark-beast-krea2-identity-edit` | `dark_beast_krea2_identity_edit_v1_2` | Editing; needs a source image |

Aspect ratios map to explicit pixels: `landscape` 1216×832, `square` 1024×1024,
`portrait` 832×1216.

## Verified behaviour

- `GET https://api.sogni.ai/v1/models` returns **200** with an honest
  `User-Agent`. The default `Python-urllib` signature is refused with Cloudflare
  **Error 1010** (HTTP 403) — a client-signature block, not an authentication
  failure. **This plugin does not spoof a browser string**; if you see a 403
  from Sogni, that is why a plain client struggles.
- Image model ids and worker counts come from a live `--search-models` call.
- CLI flags confirmed against `--help` on v3.56.1: `-m/--model`, `-w/--width`,
  `-h/--height`, `-n/--count`, `-c/--context`, `--output-format`, `-o/--output`,
  `--json`.

## Capabilities

```yaml
provides_tools:      []
provides_hooks:      []
provides_middleware: []
requires_env:
  - SOGNI_API_KEY
  - SOGNI_AGENT_BIN
```

## Limitations

- **The CLI must be installed.** `is_available()` returns `False` without it, so
  the provider is never selected when it cannot work. This is a Node/npm
  dependency outside Python's reach.
- **No spend cap.** Sogni bills per job in capacity units and the CLI reports
  cost after the fact. There is no honest upfront ceiling to enforce, so an
  unattended run can spend your balance.
- Only the four models listed above are advertised. Sogni fronts more (Z-Image,
  FLUX, Chroma, Qwen-Image, upscalers, 3D, video) — those need their ids
  verified live before being added here.
- The CLI's `--json` schema is the vendor's to change. Result extraction walks
  the common shapes and reports an honest error rather than inventing a success
  if none match.

## Disclosure (catalog rule 13)

What a user should know before installing:

- **Runs a shell command.** Each generation executes the local
  `sogni-agent-hermes` binary as a child process with the prompt and any
  reference image paths as arguments. Nothing else is run, and no shell
  interpreter is involved — arguments are passed as a list, never through a
  shell.
- **Network.** The CLI talks to Sogni's hosted service (`api.sogni.ai`) and the
  GPU network: the prompt, image dimensions, model choice and any reference
  image are sent to Sogni. The result is fetched back from Sogni's storage.
- **Credentials.** `SOGNI_API_KEY` is read through
  `agent.secret_scope.get_secret`, never from a `.env` file, `auth.json` or a
  bare `os.environ` read — a profile with no key of its own will not borrow
  another profile's. The key is passed to the CLI in its environment and is never
  logged or written by this plugin.
- **Reads.** Reference images you pass are read from local paths and handed to
  the CLI for upload.
- **Billing.** Generation is billed to your Sogni balance. Cost is reported, not
  capped.
- **No telemetry, no background processes, no self-update, no stored
  credentials.** Zero Python dependencies — standard library only.

## Files

- `__init__.py` — the provider and `register(ctx)`
- `plugin.yaml` — manifest
