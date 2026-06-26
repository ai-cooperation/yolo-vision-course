"""Telegram sink (home profile).

Reads TG_BOT_TOKEN / TG_CHAT_ID from the environment. When unset it runs in
dry-run mode and only logs what it *would* send -- so the pipeline is runnable
on a laptop with no credentials and no external side effects.

Real push (alerts) is sent via the Bot API sendMessage; image/video sending is
left for when the home camera is wired up.
"""
from __future__ import annotations

import os

from core.types import Event


class TelegramSink:
    def __init__(self, kinds=None):
        self.kinds = set(kinds or ["alert"])
        self.token = os.environ.get("TG_BOT_TOKEN")
        self.chat_id = os.environ.get("TG_CHAT_ID")
        self.dry_run = not (self.token and self.chat_id)

    def emit(self, event: Event) -> None:
        if event.kind not in self.kinds:
            return
        text = f"[{event.label}] frame {event.frame_index} {event.data}"
        if self.dry_run:
            print(f"[TG dry-run] would send: {text}")
            return
        import requests

        requests.post(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            json={"chat_id": self.chat_id, "text": text},
            timeout=10,
        )

    def close(self) -> None:
        pass
