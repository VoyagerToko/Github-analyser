import asyncio
import logging
from typing import List, Iterable, Tuple
import re

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
import requests
from serpapi import GoogleSearch

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("agent")

# ---------------------------------
# LLM (Ollama)
# ---------------------------------
llm = ChatOllama(model="mistral")  # e.g., mistral / mistral:latest / your local tag

# ---------------------------------
# CONFIG
# ---------------------------------
SEARCH_MODE = "google"   # "google" | "tavily" | "serpapi"
MAX_LINKS = 3
HEADLESS = False         # set True on servers; False if you want to see the browser
RESULT_CHAR_LIMIT = 7000  # total text fed into LLM after chunking/merge
PER_PAGE_CHAR_LIMIT = 8000  # raw scrape cap before cleaning+chunking
PER_CHUNK_CHAR_LIMIT = 1600 # chunk size for map-reduce summarization
BLOCKED_EXTENSIONS = (".pdf", ".zip", ".rar", ".7z", ".png", ".jpg", ".jpeg", ".gif", ".webp")

# ---------------------------------
# Utilities
# ---------------------------------

def generate_search_query(question: str) -> str:
    """
    Uses the LLM to rephrase the user's natural-language question into an optimized search query.
    """
    prompt = (
        "You are a helpful assistant that creates optimized search queries for search engines.\n"
        "Given a user's natural-language question, create a short, precise search query to find the best results online.\n\n"
        f"User question: {question}\n\n"
        "Search query:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def normalize_urls(search_output, mode: str) -> List[str]:
    """
    Accepts outputs from google/duckduckgo (list[str]), tavily/serpapi (list[tuple]),
    and returns a flat list[str] of URLs, deduped and filtered.
    """
    urls: List[str] = []
    if mode in ("google", "duckduckgo"):
        urls = list(search_output)
    elif mode in ("tavily", "serpapi"):
        # they return [(url, content/snippet), ...]
        urls = [u for (u, _maybe_text) in search_output]
    else:
        urls = list(search_output)

    # basic cleanup & filtering
    cleaned = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        if any(u.lower().endswith(ext) for ext in BLOCKED_EXTENSIONS):
            continue
        if "accounts.google.com" in u or "/preferences" in u:
            continue
        if u in seen:
            continue
        seen.add(u)
        cleaned.append(u)
    return cleaned[:MAX_LINKS]


def clean_html_to_text(html: str) -> str:
    """Remove boilerplate & return readable text."""
    soup = BeautifulSoup(html, "html.parser")

    # remove non-content
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["header", "footer", "nav", "aside"]):
        tag.decompose()

    text_parts: List[str] = []
    # Prefer paragraphs and list items for denser signal
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "article", "section"]):
        t = el.get_text(separator=" ", strip=True)
        if t:
            text_parts.append(t)

    text = "\n".join(text_parts)
    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(s: str, max_chars: int) -> List[str]:
    s = s.strip()
    if len(s) <= max_chars:
        return [s]
    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        # try to break on a sentence boundary
        cut = s.rfind(". ", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.5):
            cut = end
        else:
            cut = cut + 1
        chunks.append(s[start:cut].strip())
        start = cut
    return chunks


def limit_total_text(pages: List[Tuple[str, str]], cap: int) -> List[Tuple[str, str]]:
    """Cap the combined character count across pages to RESULT_CHAR_LIMIT."""
    total = 0
    kept: List[Tuple[str, str]] = []
    for url, text in pages:
        if total >= cap:
            break
        room = cap - total
        kept_text = text[:room]
        kept.append((url, kept_text))
        total += len(kept_text)
    return kept


# ---------------------------------
# Search (Playwright + fallback)
# ---------------------------------
async def google_search(query: str, max_results: int = 5) -> List[str]:
    """Perform Google search with basic CAPTCHA pause; fallback to DuckDuckGo automatically."""
    results: List[str] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=HEADLESS)
            page = await browser.new_page(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            ))

            log.info(f"🌐 Searching Google for: {query}")
            await page.goto("https://www.google.com", timeout=60000)
            await page.wait_for_timeout(1000)

            # quick CAPTCHA detection
            content_l = (await page.content()).lower()
            if "unusual traffic" in content_l or "captcha" in content_l:
                log.warning("⚠️ CAPTCHA detected. Solve in the browser; waiting 30s.")
                await page.wait_for_timeout(30000)
                await page.goto("https://www.google.com", timeout=60000)

            # search box (varies)
            for sel in ("textarea[name='q']", "input[name='q']"):
                try:
                    await page.fill(sel, query)
                    break
                except Exception:
                    continue
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(2500)

            # if blocked again -> fallback
            content_l = (await page.content()).lower()
            if "unusual traffic" in content_l or "captcha" in content_l:
                log.error("🚫 Google blocked search. Falling back to DuckDuckGo.")
                await browser.close()
                return await duckduckgo_search(query, max_results)

            anchors = await page.query_selector_all("a")
            for a in anchors:
                href = await a.get_attribute("href")
                if href and href.startswith("http") and "google.com" not in href:
                    results.append(href)
                if len(results) >= max_results:
                    break

            await browser.close()
            return results

    except Exception as e:
        log.error(f"❌ Google search error: {e}")
        return await duckduckgo_search(query, max_results)


async def duckduckgo_search(query: str, max_results: int = 5) -> List[str]:
    """DuckDuckGo search with stable waits and retry loop."""
    results: List[str] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=HEADLESS)
            page = await browser.new_page(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            ))

            log.info(f"🌐 Searching DuckDuckGo for: {query}")
            await page.goto("https://duckduckgo.com", wait_until="networkidle", timeout=60000)

            # Enter query
            await page.fill("input[name='q']", query)
            await page.keyboard.press("Enter")

            # Wait for results container
            try:
                await page.wait_for_selector("div#links", timeout=10000)
            except Exception:
                log.warning("⚠️ DuckDuckGo results not found immediately, retrying 2s...")
                await page.wait_for_timeout(2000)

            # Grab all result links
            links = await page.query_selector_all("a[data-testid='result-title-a']")
            if not links:
                # Fallback for older layouts
                links = await page.query_selector_all("a.result__a")

            for link in links:
                href = await link.get_attribute("href")
                if href and href.startswith("http"):
                    results.append(href)
                if len(results) >= max_results:
                    break

            await browser.close()
            return results

    except Exception as e:
        log.error(f"❌ DuckDuckGo search failed: {e}")
        return []



# ---------------------------------
# Optional: Tavily / SerpAPI (if you want zero-CAPTCHA)
# ---------------------------------
def tavily_search(query: str, max_links=3):
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": "YOUR_TAVILY_KEY",
        "query": query,
        "num_results": max_links,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    data = resp.json()
    results = []
    for r in data.get("results", []):
        results.append((r["url"], r.get("content", "")))
    return results


def serpapi_search(query: str, max_links=3):
    params = {
        "q": query,
        "hl": "en",
        "num": max_links,
        "api_key": "YOUR_SERPAPI_KEY",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    urls = []
    for r in results.get("organic_results", []):
        urls.append((r["link"], r.get("snippet", "")))
        if len(urls) >= max_links:
            break
    return urls


# ---------------------------------
# Scraping target pages
# ---------------------------------
async def scrape_pages(urls: List[str]) -> List[Tuple[str, str]]:
    """Visit each URL, render, and extract readable text."""
    contents: List[Tuple[str, str]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        ))
        for url in urls:
            try:
                log.info(f"🧭 Visiting {url}")
                await page.goto(url, timeout=45000, wait_until="load")
                # small wait for lazy content
                await page.wait_for_timeout(1200)
                html = await page.content()
                html = html[:PER_PAGE_CHAR_LIMIT]  # safety cap
                text = clean_html_to_text(html)
                if not text:
                    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
                contents.append((url, text))
                log.info(f"✅ Scraped {len(text)} chars from: {url}")
            except Exception as e:
                log.warning(f"⚠️ Failed to scrape {url}: {e}")
        await browser.close()
    return contents


# ---------------------------------
# Summarization (map-reduce over chunks)
# ---------------------------------
def summarize_with_ollama(question: str, pages: List[Tuple[str, str]]) -> str:
    """
    pages: list of (url, text) after scraping.
    We chunk → summarize chunks → combine → final answer with sources.
    """
    # hard-cap total input to keep latency sharp
    capped_pages = limit_total_text(pages, RESULT_CHAR_LIMIT)

    # 1) map: per-chunk summaries
    chunk_summaries: List[Tuple[str, str]] = []  # (url, summary)
    for url, text in capped_pages:
        chunks = chunk_text(text, PER_CHUNK_CHAR_LIMIT)
        for i, ch in enumerate(chunks, 1):
            prompt = (
                "You are a precise analyst. Summarize the key facts relevant to the user's question.\n"
                f"QUESTION:\n{question}\n\n"
                f"PAGE SOURCE: {url}\n"
                "CHUNK:\n"
                f"{ch}\n\n"
                "Return only factual bullets. No fluff."
            )
            resp = llm.invoke(prompt)
            chunk_summaries.append((url, resp.content.strip()))

    # 2) reduce: merge per-source
    merged_by_url: dict[str, List[str]] = {}
    for url, s in chunk_summaries:
        merged_by_url.setdefault(url, []).append(s)

    merged_sources: List[Tuple[str, str]] = []
    for url, parts in merged_by_url.items():
        joined = "\n".join(parts)
        # compress per-source
        prompt = (
            "Compress the following bullets into a concise, non-redundant summary. Keep only facts:\n\n"
            f"{joined}\n\n"
            "Output 5-8 short bullets max."
        )
        resp = llm.invoke(prompt)
        merged_sources.append((url, resp.content.strip()))

    # 3) final: answer + sources
    # build a combined context of per-source summaries
    context = "\n\n".join([f"SOURCE: {u}\n{t}" for (u, t) in merged_sources])
    final_prompt = (
        "You answer using only the provided context. If the context lacks the answer, say so.\n"
        "Be crisp, human-friendly, and avoid speculation. Use plain language.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Now produce:\n"
        "1) A short, direct answer (2–5 sentences).\n"
        "2) If the user asked for comparison or steps, add a compact bullet list.\n"
        "3) Add a 'Sources' section with the URLs you used."
    )
    resp = llm.invoke(final_prompt)
    return resp.content.strip()


# ---------------------------------
# Main runner
# ---------------------------------
async def main():
    user_question = input("Enter your question: ").strip()
    log.info(f"🧠 Original question: {user_question}")

    query = generate_search_query(user_question)
    log.info(f"🔍 Refined search query: {query}")


    # --- Search ---
    if SEARCH_MODE == "google":
        raw = await google_search(query, MAX_LINKS)
        urls = normalize_urls(raw, "google")
    elif SEARCH_MODE == "tavily":
        raw = tavily_search(query, MAX_LINKS)
        urls = normalize_urls(raw, "tavily")
    elif SEARCH_MODE == "serpapi":
        raw = serpapi_search(query, MAX_LINKS)
        urls = normalize_urls(raw, "serpapi")
    else:
        raise ValueError("Invalid SEARCH_MODE. Use 'google', 'tavily', or 'serpapi'.")

    if not urls:
        print("❌ No search results fetched, aborting.")
        return

    log.info(f"🔗 Using {len(urls)} URL(s)")
    # --- Scrape ---
    pages = await scrape_pages(urls)
    if not pages:
        print("❌ Could not scrape any pages.")
        return

    # --- Summarize ---
    print(f"Fetched & scraped {len(pages)} page(s). Summarizing with Mistral…")
    answer = summarize_with_ollama(user_question, pages)


    print("\n---\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
