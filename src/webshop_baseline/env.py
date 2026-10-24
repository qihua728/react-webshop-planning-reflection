from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ACTION_TO_SUBPAGE = {"Description", "Features", "Reviews", "Attributes"}
ACTION_PATTERN = re.compile(r"^(think|search|click)\[(.*)]$", re.DOTALL)


class WebShopError(RuntimeError):
    """Base error raised by the WebShop adapter."""


class InvalidAction(WebShopError):
    """The action is malformed or invalid for the current page."""


class WebShopRequestError(WebShopError):
    """The WebShop server could not be reached or parsed."""


@dataclass(frozen=True)
class StepResult:
    observation: str
    reward: float
    done: bool


def normalize_action(raw: str) -> str:
    text = raw.strip().strip("`").strip()
    text = re.sub(r"^Action\s*:\s*", "", text, flags=re.IGNORECASE)
    match = ACTION_PATTERN.search(text)
    if not match:
        raise InvalidAction(f"Malformed action: {raw!r}")
    action = f"{match.group(1).lower()}[{match.group(2).strip()}]"
    return action


def _clean_text(value: str) -> str:
    try:
        return value.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value


def _is_visible(element: Any) -> bool:
    ignored = {"style", "script", "head", "title", "meta", "[document]"}
    parent = getattr(element, "parent", None)
    return bool(parent and parent.name not in ignored and not isinstance(element, Comment))


class WebShopEnv:
    """Thin adapter around the HTTP WebShop environment used by ReAct."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 20.0,
        retries: int = 2,
        http_session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.sessions: dict[str, dict[str, Any]] = {}
        self.http = http_session or requests.Session()
        if http_session is None:
            retry = Retry(
                total=retries,
                connect=retries,
                read=retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
            self.http.mount("http://", HTTPAdapter(max_retries=retry))
            self.http.mount("https://", HTTPAdapter(max_retries=retry))

    def reset(self, session_id: str) -> StepResult:
        self.sessions[session_id] = {"session": session_id, "page_type": "init"}
        observation, info = self._render(self.sessions[session_id])
        self.sessions[session_id].update(info)
        return StepResult(observation, float(info.get("reward", 0.0)), False)

    def step(self, session_id: str, raw_action: str) -> StepResult:
        if session_id not in self.sessions:
            raise InvalidAction(f"Unknown session {session_id!r}; call reset first.")
        action = normalize_action(raw_action)
        kind, argument = action.split("[", 1)
        argument = argument[:-1]
        state = self.sessions[session_id]

        if kind == "think":
            return StepResult("OK.", 0.0, False)
        if kind == "search":
            if state["page_type"] != "init" or not argument:
                raise InvalidAction("search[...] is only valid on the initial page.")
            self.sessions[session_id] = {
                "session": session_id,
                "page_type": "search",
                "query_string": argument,
                "page_num": 1,
            }
            state = self.sessions[session_id]
        elif kind == "click":
            done, override = self._click(state, argument)
            observation, info = self._render(state)
            state.update(info)
            return StepResult(
                override or observation,
                float(info.get("reward", 0.0)),
                done,
            )
        else:  # pragma: no cover - guarded by normalize_action
            raise InvalidAction(f"Unsupported action type: {kind}")

        observation, info = self._render(state)
        state.update(info)
        return StepResult(observation, float(info.get("reward", 0.0)), False)

    def _click(self, state: dict[str, Any], button: str) -> tuple[bool, str | None]:
        page = state["page_type"]
        if button == "Buy Now":
            if page != "item":
                raise InvalidAction("Buy Now is only valid on an item page.")
            state["page_type"] = "end"
            return True, None
        if button == "Back to Search":
            if page not in {"search", "item_sub", "item"}:
                raise InvalidAction("Back to Search is invalid on this page.")
            session_id = state["session"]
            state.clear()
            state.update({"session": session_id, "page_type": "init"})
            return False, None
        if button == "Next >":
            raise InvalidAction("The baseline follows the original one-page search limitation.")
        if button == "< Prev":
            if page == "item_sub":
                state["page_type"] = "item"
            elif page == "item":
                state["page_type"] = "search"
                state["options"] = {}
            else:
                raise InvalidAction("< Prev is invalid on this page.")
            return False, None
        if button in ACTION_TO_SUBPAGE:
            if page != "item":
                raise InvalidAction(f"{button} is only valid on an item page.")
            state["page_type"] = "item_sub"
            state["subpage"] = button
            return False, None
        if page == "search":
            if button not in state.get("asins", []):
                raise InvalidAction(f"Unknown product id: {button}")
            state["page_type"] = "item"
            state["asin"] = button
            return False, None
        if page == "item":
            option_types = state.get("option_types", {})
            if button not in option_types:
                raise InvalidAction(f"Unknown item option: {button}")
            state.setdefault("options", {})[option_types[button]] = button
            return False, f"You have clicked {button}."
        raise InvalidAction(f"click[{button}] is invalid on page {page!r}.")

    def _url_for(self, state: dict[str, Any]) -> str:
        page = state["page_type"]
        session_id = state["session"]
        if page == "init":
            return f"{self.base_url}/{session_id}"
        query = state.get("query_string", "")
        page_num = state.get("page_num", 1)
        if page == "search":
            return f"{self.base_url}/search_results/{session_id}/{query}/{page_num}"
        asin = state.get("asin", "")
        options = state.get("options", {})
        if page == "item":
            return f"{self.base_url}/item_page/{session_id}/{asin}/{query}/{page_num}/{options}"
        if page == "item_sub":
            subpage = state.get("subpage", "")
            return f"{self.base_url}/item_sub_page/{session_id}/{asin}/{query}/{page_num}/{subpage}/{options}"
        if page == "end":
            return f"{self.base_url}/done/{session_id}/{asin}/{options}"
        raise WebShopRequestError(f"Unknown WebShop page type: {page}")

    def _render(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        url = self._url_for(state)
        try:
            response = self.http.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebShopRequestError(f"GET {url} failed: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        visible_texts = [text for text in soup.find_all(string=True) if _is_visible(text)]
        if not visible_texts:
            raise WebShopRequestError(f"WebShop returned no visible content for {url}")

        observation_parts: list[str] = []
        option_type = ""
        option_types: dict[str, str] = {}
        asins: list[str] = []
        count = product_count = just_product = 0

        for raw in visible_texts:
            text = str(raw)
            if not text.replace("\n", "").replace("\\n", "").strip():
                continue
            parent = raw.parent
            processed = ""
            if parent.name == "button":
                processed = f"\n[{text}] "
            elif parent.name == "label":
                selected = text in state.get("options", {}).values()
                processed = f"[[{text}]]" if selected else f"[{text}]"
                option_types[text] = option_type
            elif parent.get("class") == ["product-link"]:
                processed = f"\n[{text}] " if product_count < 3 else ""
                product_count += 1
                asins.append(text)
                just_product = 0
            else:
                processed = "\n" + text + " "
                if count < 2 and state["page_type"] != "init":
                    processed = ""
                if just_product <= 2 and product_count >= 4:
                    processed = ""
                option_type = text
                count += 1
            just_product += 1
            observation_parts.append(processed)

        info: dict[str, Any] = {}
        if option_types:
            info["option_types"] = option_types
        if asins:
            info["asins"] = asins

        score_label = "Your score (min 0.0, max 1.0)"
        normalized = [str(item).strip() for item in visible_texts]
        if score_label in normalized:
            index = normalized.index(score_label)
            if index + 1 >= len(normalized):
                raise WebShopRequestError("Score label was present without a score value.")
            try:
                reward = float(normalized[index + 1])
            except ValueError as exc:
                raise WebShopRequestError("WebShop returned an invalid score value.") from exc
            info["reward"] = reward
            return f"Your score (min 0.0, max 1.0): {reward}", info

        return _clean_text("".join(observation_parts)), info

