# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Streamlit helper functions
"""

from collections.abc import Callable

import streamlit as st

from web.i18n import tr
from pixelle_video.config import config_manager


def safe_rerun():
    """Safe rerun that works with both old and new Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


# Dialog functions are cached by title so Streamlit can keep the same dialog
# open across reruns. The title depends on the active language.
_selfhost_dialogs: dict[str, Callable[[str], None]] = {}


def _render_selfhost_warning(workflow_path: str) -> None:
    """Render the SelfHost checklist inside an already-open dialog."""
    comfyui_config = config_manager.get_comfyui_config()
    comfyui_url = comfyui_config.get("comfyui_url", "http://localhost:8188")
    message = tr(
        "selfhost.warning.message",
        comfyui_url=f"[{comfyui_url}]({comfyui_url})",
        workflow_path=f"workflows/{workflow_path}",
    )
    st.markdown(message.replace("\n", "  \n"))
    st.markdown(tr("selfhost.warning.hint"))
    if st.button(
        tr("selfhost.warning.confirm"),
        key="selfhost_warning_confirm",
        type="primary",
    ):
        st.rerun()


def _selfhost_dialog(title: str) -> Callable[[str], None]:
    dialog = _selfhost_dialogs.get(title)
    if dialog is None:
        @st.dialog(title, width="large")
        def _dialog(workflow_path: str) -> None:
            _render_selfhost_warning(workflow_path)

        dialog = _dialog
        _selfhost_dialogs[title] = dialog
    return dialog


def render_selfhost_workflow_help(workflow_path: str | None, *, button_key: str) -> None:
    """Show a help button that opens the SelfHost checklist.

    The dialog is not opened during page load. It opens only after this button
    is clicked, and only for local ``selfhost/`` workflows.
    """
    if not workflow_path or not workflow_path.startswith("selfhost/"):
        return

    if st.button(
        "?",
        key=button_key,
        help=tr("selfhost.warning.button_help"),
        type="tertiary",
    ):
        _selfhost_dialog(tr("selfhost.warning.title"))(workflow_path)

