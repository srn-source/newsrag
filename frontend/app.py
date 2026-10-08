"""Chainlit chat UI for NewsRAG.

A thin client: it only calls the API's POST /ask over HTTP. There is no RAG code
here, so the UI and the API deploy, scale and fail independently.

Run locally, with the API already running on port 8080:
    cd frontend
    uv run chainlit run app.py --port 8000 -w
then open http://localhost:8000

Configuration (environment variables):
    API_URL               where the API is      (default http://localhost:8080)
    API_KEY               sent as X-API-Key     (needed when the API has API_KEY set,
                                                 i.e. on Cloud Run)
    DEMO_PASSWORD         if set, the UI asks for a login: user "demo" + this password.
                          Without it, a public UI would let anyone spend the API's
                          LLM quota through the API key the UI holds.
    CHAINLIT_AUTH_SECRET  required by Chainlit whenever a login is enabled
"""

import os
import secrets
import time

import chainlit as cl
import httpx
from chainlit.input_widget import Select

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")

# Multihop answers take ~20 s. Matches the API's Cloud Run request timeout.
TIMEOUT_SECONDS = 120

EXAMPLES = [
    (
        "Cross-outlet question",
        "Who is the individual associated with the cryptocurrency industry facing a "
        "criminal trial on fraud and conspiracy charges, as reported by both The Verge "
        "and TechCrunch?",
    ),
    (
        "Comparison over time",
        "Between the article published by Sporting News on 2023-10-02 and the one on "
        "2023-11-06, has the explanation of what bettors can place prop bets on "
        "remained consistent or inconsistent?",
    ),
    (
        "Question in Thai",
        "TechCrunch รายงานว่าใครถูกฟ้องคดีฉ้อโกงในวงการคริปโต",
    ),
    (
        "Not in the news (should refuse)",
        "Who won the 2030 FIFA World Cup?",
    ),
]


# --- optional login ------------------------------------------------------------------

if DEMO_PASSWORD:

    @cl.password_auth_callback
    async def login(username: str, password: str) -> cl.User | None:
        valid = username == "demo" and secrets.compare_digest(
            password.encode("utf-8"), DEMO_PASSWORD.encode("utf-8")
        )
        return cl.User(identifier="demo") if valid else None


# --- chat lifecycle ------------------------------------------------------------------


@cl.set_starters
async def starters(current_user: cl.User | None = None, language: str | None = None):
    return [cl.Starter(label=label, message=question) for label, question in EXAMPLES]


@cl.on_chat_start
async def start() -> None:
    settings = await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Retrieval mode",
                values=["multihop", "baseline"],
                initial_value="multihop",
                description="multihop splits the question into one search per outlet or "
                "event (better for comparisons, ~20 s); baseline searches once (~14 s).",
            )
        ]
    ).send()
    cl.user_session.set("mode", settings["mode"])


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("mode", settings["mode"])


@cl.on_message
async def answer(message: cl.Message) -> None:
    mode = cl.user_session.get("mode") or "multihop"

    # A Step shows a "running" indicator in the UI while we wait for the API.
    async with cl.Step(name=f"Searching the news ({mode})", type="tool") as step:
        step.input = message.content
        started = time.perf_counter()
        try:
            data = await ask_api(message.content, mode)
        except ApiError as error:
            step.output = str(error)
            await cl.Message(content=f"**Could not get an answer.** {error}").send()
            return
        elapsed = time.perf_counter() - started
        step.output = f"{' → '.join(data['steps'])}  ·  {elapsed:.1f} s"

    content, elements = render(data)
    await cl.Message(content=content, elements=elements).send()


# --- talking to the API --------------------------------------------------------------


class ApiError(Exception):
    """A failure worth showing to the user, phrased for the user."""


async def ask_api(question: str, mode: str) -> dict:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{API_URL}/ask", json={"question": question, "mode": mode}, headers=headers
            )
    except httpx.TimeoutException as error:
        raise ApiError(f"The API did not answer within {TIMEOUT_SECONDS} s.") from error
    except httpx.HTTPError as error:
        raise ApiError(f"Cannot reach the API at {API_URL}.") from error

    if response.status_code == 401:
        raise ApiError("The API rejected the API key. Check API_KEY.")
    if response.status_code == 422:
        raise ApiError("The question was not accepted (it must be 1-1,000 characters).")
    if response.status_code >= 400:
        detail = response.json().get("detail", "") if response.content else ""
        raise ApiError(f"The API returned {response.status_code}. {detail}")
    return response.json()


def render(data: dict) -> tuple[str, list[cl.Text]]:
    """API response -> chat message: short answer, explanation, numbered sources.

    Each source is also a side panel element named "Source N". Chainlit turns every
    "Source N" in the message into a link that opens that excerpt.
    """
    short_answer, _, explanation = data["answer"].partition("\n")
    lines = [f"**{short_answer.strip()}**"]
    if explanation.strip():
        lines += ["", explanation.strip()]

    if data["status"] == "refused":
        lines += [
            "",
            "_The news corpus (609 articles from late 2023) does not contain an answer, "
            "so none is given rather than a guess._",
        ]
        return "\n".join(lines), []

    elements = []
    if data["citations"]:
        lines += ["", "**Sources**"]
        for citation in data["citations"]:
            name = f"Source {citation['index']}"
            lines.append(
                f"- {name} · {citation['source']} · {citation['published_at']} · "
                f"[{citation['title']}]({citation['url']})"
            )
            elements.append(
                cl.Text(
                    name=name,
                    content=f"**{citation['title']}**\n\n{citation['excerpt']}…",
                    display="side",
                )
            )
    return "\n".join(lines), elements
