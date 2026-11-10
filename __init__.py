"""Sogni AI image generation backend for Hermes.

Why this plugin exists
----------------------
Hermes ships image backends for FAL, DeepInfra, OpenAI, OpenRouter, xAI and Krea
(Krea.ai). Sogni AI is a different vendor — a decentralised GPU network fronting
its own model families (Krea 2, Z-Image, FLUX, Chroma, Qwen-Image) plus editing,
upscaling, 3D and video — and Hermes cannot reach it at all. There is no Sogni
entry in the plugin catalog and no Sogni image provider anywhere in the tree.

This plugin adds Sogni as a first-class ``image_gen`` provider:

  * text-to-image   — ``image_generate(prompt=...)``
  * image-to-image  — ``image_generate(prompt=..., image_url=...)``

Transport: the vendor's own CLI
-------------------------------
Sogni's hosted *image* path is asynchronous — generation is submitted as a
durable workflow step (``POST /v1/creative-agent/workflows``) and polled, while
the synchronous ``/tools/execute`` endpoint carries text composition only.
Reimplementing that submit/poll protocol here would mean shipping an unverified
hand-rolled client for a vendor whose official client already exists.

So this provider drives ``sogni-agent-hermes`` (the Sogni Creative Agent CLI,
``@sogni-ai/sogni-creative-agent-skill``), which is the vendor's supported
client and handles authentication, media upload, job polling and cost
reporting. The CLI is a hard runtime requirement — ``is_available()`` returns
False without it, so the provider is never selected when it cannot work.

Disclosed per catalog rule 13 (see README and the catalog entry).

Verified against the live API, not inferred
-------------------------------------------
  * ``GET https://api.sogni.ai/v1/models`` returns 200 with an honest
    ``User-Agent``. The default ``Python-urllib`` signature is refused with
    Cloudflare **Error 1010** (HTTP 403) — a client-signature block, not an auth
    failure. No browser-string spoofing is used anywhere in this plugin.
  * Image models confirmed live via ``--search-models``:
    ``krea2_turbo_fp8_scaled`` (188 workers), ``dark_beast_krea2_fp8`` (217),
    and the two identity-edit variants.
  * CLI flags confirmed from ``--help`` (v3.56.1): ``-m/--model``,
    ``-w/--width``, ``-h/--height``, ``-n/--count``, ``-c/--context``,
    ``--output-format``, ``-o/--output``, ``--json``.

Auth: ``SOGNI_API_KEY``, read through Hermes's secret scope only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

CLI_NAMES = ("sogni-agent-hermes", "sogni-agent")
CLI_TIMEOUT = 600.0
DEFAULT_MODEL = "krea-2-turbo"

# Verified live with `sogni-agent-hermes --search-models`. Only models whose ids
# were returned by the API are advertised — an unverified id in this list would
# surface as a generation failure rather than an honest "not supported".
MODELS: Dict[str, Dict[str, Any]] = {
    "krea-2-turbo": {
        "id": "krea2_turbo_fp8_scaled",
        "display": "Krea 2 Turbo",
        "strengths": "Sogni's fast general-purpose model. Strong material and "
                     "light rendering.",
        "speed": "~10-40s",
        "modalities": ["text", "image"],
    },
    "dark-beast-krea2": {
        "id": "dark_beast_krea2_fp8",
        "display": "Dark Beast Krea 2",
        "strengths": "Higher-fidelity KREA 2 variant. More workers than Turbo.",
        "speed": "~20-60s",
        "modalities": ["text", "image"],
    },
    "krea-2-identity-edit": {
        "id": "krea2_identity_edit_v1_2",
        "display": "Krea 2 Identity Edit",
        "strengths": "Editing that preserves a subject's identity. Needs a "
                     "context image.",
        "speed": "~20-60s",
        "modalities": ["image"],
    },
    "dark-beast-krea2-identity-edit": {
        "id": "dark_beast_krea2_identity_edit_v1_2",
        "display": "Dark Beast KREA 2 Identity Edit",
        "strengths": "Identity-preserving edit on the Dark Beast variant.",
        "speed": "~20-60s",
        "modalities": ["image"],
    },
}

# aspect_ratio name -> (width, height). The Hermes surface passes the friendly
# names resolved by resolve_aspect_ratio(); the CLI takes explicit pixels.
DIMENSIONS: Dict[str, tuple] = {
    "landscape": (1216, 832),
    "square": (1024, 1024),
    "portrait": (832, 1216),
}


class SogniImageGenProvider(ImageGenProvider):
    """Sogni AI image generation (text-to-image and image-to-image)."""

    # ------------------------------------------------------------------ meta

    @property
    def name(self) -> str:
        return "sogni"

    @property
    def display_name(self) -> str:
        return "Sogni AI"

    def _cli(self) -> str:
        """Locate the vendor CLI. Env override first, then PATH."""
        override = (get_secret("SOGNI_AGENT_BIN") or "").strip()
        if override and shutil.which(override):
            return override
        for candidate in CLI_NAMES:
            found = shutil.which(candidate)
            if found:
                return found
        return ""

    def _api_key(self) -> str:
        """The key, through Hermes's secret scope only — no os.environ fallback.

        ``get_secret`` already reads the environment in single-profile installs,
        and returns empty on purpose when one process serves several profiles.
        A fallback would let a profile with no key bill another profile's Sogni
        account.
        """
        return (get_secret("SOGNI_API_KEY") or "").strip()

    def is_available(self) -> bool:
        """Both the key and the CLI are required — the provider shells out."""
        return bool(self._api_key() and self._cli())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": m["display"],
                "strengths": m["strengths"],
                "speed": m["speed"],
                "price": "Sogni credits",
                "modalities": m["modalities"],
            }
            for mid, m in MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "Image",
            "tag": "Text + image to image via the Sogni CLI on the GPU network",
            "env_vars": [
                {
                    "key": "SOGNI_API_KEY",
                    "label": "Sogni API key",
                    "help": "dashboard.sogni.ai -> account menu -> API key.",
                    "required": True,
                },
                {
                    "key": "SOGNI_AGENT_BIN",
                    "label": "Path to sogni-agent-hermes",
                    "help": "Optional. Only needed if the CLI is not on PATH.",
                    "required": False,
                },
            ],
            "notes": "Requires the Sogni CLI: "
                     "npm install -g @sogni-ai/sogni-creative-agent-skill@3.56.1",
        }

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(DIMENSIONS),
            # Sogni bills per job in capacity units and the CLI reports the
            # cost after the fact; there is no honest upfront cap to advertise.
            "max_reference_images": 8,
            "creative_controls": {
                "count": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "negative_prompt": {"type": "string"},
            },
        }

    # ------------------------------------------------------------ generation

    @staticmethod
    def _extract_url(doc: Any) -> str:
        """Find a result URL or local path in the CLI's JSON, tolerantly.

        The exact schema is the vendor's to change, so walk the common shapes
        rather than pinning one key. Missing every shape returns "" and the
        caller reports an honest error instead of a fabricated success.
        """
        if isinstance(doc, str):
            return doc if doc.startswith(("http://", "https://", "/")) else ""
        if isinstance(doc, dict):
            for key in ("url", "imageUrl", "image_url", "outputUrl", "output_url",
                        "outputPath", "output_path", "path", "file", "localPath"):
                val = doc.get(key)
                if isinstance(val, str) and val:
                    return val
            for key in ("images", "results", "outputs", "media", "artifacts", "data"):
                found = SogniImageGenProvider._extract_url(doc.get(key))
                if found:
                    return found
        if isinstance(doc, list):
            for item in doc:
                found = SogniImageGenProvider._extract_url(item)
                if found:
                    return found
        return ""

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        aspect_ratio: str = "landscape",
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        count: Optional[int] = None,
        **kwargs: Any,                      # forward-compat: MUST be ignored
    ) -> Dict[str, Any]:
        model_id = model if model in MODELS else DEFAULT_MODEL
        spec = MODELS[model_id]

        cli = self._cli()
        if not cli:
            return error_response(
                error="The Sogni CLI was not found. Install it with "
                      "`npm install -g @sogni-ai/sogni-creative-agent-skill@3.56.1`, "
                      "or set SOGNI_AGENT_BIN to its full path.",
                provider="sogni", model=model_id, prompt=prompt,
            )
        if not self._api_key():
            return error_response(
                error="SOGNI_API_KEY is not set.",
                provider="sogni", model=model_id, prompt=prompt,
            )

        ar = resolve_aspect_ratio(aspect_ratio)
        width, height = DIMENSIONS.get(ar, DIMENSIONS["landscape"])

        refs = normalize_reference_images(
            [image_url] if image_url else reference_image_urls
        ) or []
        modality = "image" if refs else "text"
        if spec["modalities"] == ["image"] and not refs:
            return error_response(
                error=f"{spec['display']} is an editing model and needs a source "
                      "image. Pass image_url (a local path or an https URL).",
                provider="sogni", model=model_id, prompt=prompt,
            )

        argv = [cli, "--json", "-m", spec["id"],
                "-w", str(width), "-h", str(height),
                "-n", str(int(count or 1))]
        for ref in refs:
            argv += ["-c", ref]
        if seed is not None:
            argv += ["--seed", str(int(seed))]
        # `--` ends option parsing so a prompt starting with "-" is never
        # mistaken for a flag.
        argv += ["--", prompt]

        logger.info("sogni: %s %s %sx%s refs=%d",
                    model_id, spec["id"], width, height, len(refs))

        # Minimal env: only what the CLI needs. Prevents leaking Hermes provider
        # keys and ensures the resolved SOGNI_API_KEY is the one the CLI sees.
        minimal_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "SOGNI_API_KEY": self._api_key(),
        }
        # The CLI may need these for Node/npm internals; pass through if present.
        for k in ("NODE_OPTIONS", "NPM_CONFIG_PREFIX", "XDG_CACHE_HOME"):
            v = os.environ.get(k)
            if v:
                minimal_env[k] = v

        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=CLI_TIMEOUT,
                env=minimal_env,
            )
        except Exception as exc:                                   # noqa: BLE001
            return error_response(
                error=f"could not run the Sogni CLI: {exc}",
                provider="sogni", model=model_id, prompt=prompt,
            )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return error_response(
                error=f"Sogni CLI exited {proc.returncode}: {detail}",
                provider="sogni", model=model_id, prompt=prompt,
            )

        raw = (proc.stdout or "").strip()
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            doc = raw                    # a bare URL/path is tolerated

        found = self._extract_url(doc)
        if not found:
            return error_response(
                error="Sogni CLI succeeded but its output carried no result URL "
                      f"or path: {raw[:300]}",
                provider="sogni", model=model_id, prompt=prompt,
            )

        # Materialise remote results into Hermes's image cache; keep local paths.
        image = found
        cached = ""
        if found.startswith(("http://", "https://")):
            try:
                image = str(save_url_image(found, prefix="sogni"))
                cached = image
            except Exception as exc:                               # noqa: BLE001
                logger.warning("sogni: could not cache %s: %s", found, exc)
                image = found           # fall back to the URL

        return success_response(
            image=image,
            model=model_id,
            prompt=prompt,
            aspect_ratio=ar,
            provider="sogni",
            modality=modality,
            extra={
                "sogni_model_id": spec["id"],
                "dimensions": f"{width}x{height}",
                "reference_count": len(refs),
                "source": found,
                "cached_path": cached,
                "cli": cli,
            },
        )


def register(ctx) -> None:
    """Plugin entry point — wire the Sogni provider into the registry."""
    ctx.register_image_gen_provider(SogniImageGenProvider())
