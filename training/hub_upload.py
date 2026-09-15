"""
Uploads checkpoints to a Hugging Face Hub model repo as they are saved
locally (configs/*.yaml's `checkpoint.hf_repo` field, made concrete).

Runs each upload in a background thread so a slow/unreliable network never
blocks the training loop. If a previous upload is still in flight when the
next checkpoint is due, the new one is SKIPPED, not queued -- the
checkpoint after that supersedes it anyway (it's the same latest.pt path
in the repo), and piling up an unbounded queue of stale multi-GB uploads
behind a slow connection would be worse than the Hub occasionally lagging
a few checkpoints behind local disk. Best.pt uploads on improvement follow
the same rule.

Auth: an explicit token if one is passed, otherwise huggingface_hub's own
default resolution (the HF_TOKEN environment variable, or a cached
`huggingface-cli login` / `huggingface_hub.login()` token). This module
never accepts, logs, or embeds a raw token value -- callers should read it
from an environment variable or a secret store (e.g. a Kaggle Secret) and
pass it through, never hardcode it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional


class HubUploader:
    def __init__(self, repo_id: str, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token
        self._lock = threading.Lock()
        self._busy = False
        # Built lazily -- this module has no hard huggingface_hub import at
        # module load time, so it's harmless when hf_repo is unset.
        self._api = None
        self._repo_ready = False

    def _get_api(self):
        from huggingface_hub import HfApi

        if self._api is None:
            self._api = HfApi(token=self.token)
        if not self._repo_ready:
            self._api.create_repo(self.repo_id, repo_type="model", exist_ok=True)
            self._repo_ready = True
        return self._api

    def maybe_upload_async(self, files: dict, commit_message: str) -> bool:
        """`files`: {path_in_repo: local_path}. Returns True if an upload
        was started, False if skipped because a previous one is still
        running."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        thread = threading.Thread(
            target=self._upload, args=(dict(files), commit_message), daemon=True
        )
        thread.start()
        return True

    def _upload(self, files: dict, commit_message: str) -> None:
        try:
            api = self._get_api()
            for path_in_repo, local_path in files.items():
                api.upload_file(
                    path_or_fileobj=str(Path(local_path)),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                )
            print(f"[hub_upload] uploaded {sorted(files)} to {self.repo_id} ({commit_message})", flush=True)
        except Exception as e:
            # A network hiccup or a bad/missing token must never kill the
            # training loop -- the checkpoint is already safe on local disk.
            print(f"[hub_upload] FAILED ({commit_message}): {e}", flush=True)
        finally:
            with self._lock:
                self._busy = False
