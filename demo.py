import asyncio
import logging
from typing import List, Iterable, Tuple, Optional
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
import requests
from git import Repo  # GitPython package
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "mistral"
    action: Optional[str] = "direct"

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler('agent.log')  # Also save to file
    ]
)
log = logging.getLogger("agent")

# Enable more detailed logging
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)

# ---------------------------------
# LLMs (Ollama)
# ---------------------------------
general_llm = ChatOllama(model="mistral")   # For general conversation
code_llm = ChatOllama(model="codellama")   # Specialized for code/technical queries
math_llm = ChatOllama(model="llama2")      # Better at mathematical reasoning

# ---------------------------------
# Conversation History
# ---------------------------------
conversation_history = []
MAX_HISTORY_LENGTH = 10  # Maximum number of turns to remember

def determine_query_type(query: str) -> str:
    """
    Determine the type of query to select appropriate LLM.
    Returns: 'code', 'math', or 'general'
    """
    log.info(f"🔍 Analyzing query type for: {query}")
    
    # Code-related indicators
    code_indicators = [
        'code', 'program', 'function', 'algorithm', 'debug', 
        'error', 'compile', 'syntax', 'api', 'framework',
        'git', 'github', 'repository', 'programming'
    ]
    
    # Math/Logic related indicators
    math_indicators = [
        'math', 'calculate', 'solve', 'equation', 'formula',
        'logic', 'proof', 'theorem', 'arithmetic', 'algebra',
        'number', 'computation', 'algorithm complexity'
    ]
    
    query = query.lower()
    
    # Check for code syntax or GitHub URLs
    if any(ind in query for ind in code_indicators) or 'github.com' in query:
        return 'code'
    
    # Check for mathematical or logical queries
    if any(ind in query for ind in math_indicators):
        return 'math'
        
    return 'general'

def get_appropriate_llm(query: str):
    """Get the most appropriate LLM based on query type."""
    query_type = determine_query_type(query)
    selected_llm = {
        'code': code_llm,
        'math': math_llm,
        'general': general_llm
    }.get(query_type, general_llm)
    
    log.info(f"🤖 Selected LLM for query type '{query_type}': {selected_llm.model}")
    return selected_llm

# ---------------------------------
# Conversation Handling
# ---------------------------------
def is_greeting(text: str) -> bool:
    """Check if the input is a greeting."""
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(text.lower().startswith(g) for g in greetings)

def get_conversation_response(text: str, current_llm: ChatOllama, context: str = None) -> str:
    """Generate a conversational response without web search."""
    global conversation_history
    
    base_prompt = (
        "You are a helpful and friendly AI assistant powered by Ollama models (Mistral, CodeLlama, or Llama2). "
        "Respond to the user's message in a natural, conversational way. "
        "Keep the response concise and engaging. When asked about your identity, explain which Ollama model you're using based on the query type. "
        "Use the conversation history to maintain context and provide more relevant responses."
    )
    
    # Build conversation history string
    history_str = ""
    if conversation_history:
        history_str = "Previous conversation:\n"
        for turn in conversation_history:
            history_str += f"User: {turn['user']}\n"
            history_str += f"Assistant: {turn['assistant']}\n"
    
    if context:
        prompt = (
            f"{base_prompt}\n\n"
            f"{history_str}\n"
            f"Context information:\n{context}\n\n"
            f"User: {text}\n"
            "Assistant:"
        )
    else:
        prompt = (
            f"{base_prompt}\n\n"
            f"{history_str}\n"
            f"User: {text}\n"
            "Assistant:"
        )
        
    response = current_llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Update conversation history
    conversation_history.append({
        'user': text,
        'assistant': response_text
    })
    
    # Keep only the last MAX_HISTORY_LENGTH turns
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
    
    return response_text

def is_farewell(text: str) -> bool:
    """Check if the input is a farewell."""
    farewells = ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit']
    return any(text.lower().startswith(f) for f in farewells)

def needs_search(text: str) -> bool:
    """Determine if the query needs web searching."""
    # Questions that typically need search
    search_indicators = [
        'how to', 'what is', 'tell me about', 'explain', 'compare',
        'difference between', 'where can i', 'when did', 'who is',
        'github.com', 'latest', 'recent', 'best', 'top'
    ]
    # Questions that don't need search
    chat_indicators = [
        'how are you', 'what do you think', 'can you help', 
        'i feel', 'thank you', 'thanks', "what's up",
        'nice to meet you', 'pleased to meet you'
    ]
    
    text_lower = text.lower()
    # Don't search if it's clearly conversational
    if any(i in text_lower for i in chat_indicators):
        return False
    # Search if it contains search indicators
    return any(i in text_lower for i in search_indicators)

def get_conversation_response(text: str, llm: ChatOllama, context: str = None) -> str:
    """Generate a conversational response without web search."""
    global conversation_history
    
    base_prompt = (
        "You are a helpful and friendly AI assistant powered by Ollama models (Mistral, CodeLlama, or Llama2). "
        "Respond to the user's message in a natural, conversational way. "
        "Keep the response concise and engaging. When asked about your identity, explain which Ollama model you're using based on the query type. "
        "Use the conversation history to maintain context and provide more relevant responses."
    )
    
    # Build conversation history string
    history_str = ""
    if conversation_history:
        history_str = "Previous conversation:\n"
        for turn in conversation_history:
            history_str += f"User: {turn['user']}\n"
            history_str += f"Assistant: {turn['assistant']}\n"
    
    if context:
        prompt = (
            f"{base_prompt}\n\n"
            f"{history_str}\n"
            f"Context information:\n{context}\n\n"
            f"User: {text}\n"
            "Assistant:"
        )
    else:
        prompt = (
            f"{base_prompt}\n\n"
            f"{history_str}\n"
            f"User: {text}\n"
            "Assistant:"
        )
        
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Update conversation history
    conversation_history.append({
        'user': text,
        'assistant': response_text
    })
    
    # Keep only the last MAX_HISTORY_LENGTH turns
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
    
    return response_text

# ---------------------------------
# CONFIG
# ---------------------------------
MAX_LINKS = 3
HEADLESS = False         # set True on servers; False if you want to see the browser
RESULT_CHAR_LIMIT = 7000  # total text fed into LLM after chunking/merge
PER_PAGE_CHAR_LIMIT = 8000  # raw scrape cap before cleaning+chunking
PER_CHUNK_CHAR_LIMIT = 1600 # chunk size for map-reduce summarization
BLOCKED_EXTENSIONS = (".pdf", ".zip", ".rar", ".7z", ".png", ".jpg", ".jpeg", ".gif", ".webp")

# ---------------------------------
# Utilities
# ---------------------------------

from typing import Optional, Union
import git
import tempfile

def analyze_github_repo(url: str, query: str = "") -> Tuple[str, Optional[str]]:
    """
    Clone and analyze a GitHub repository based on user's query.
    Can analyze specific folders, files, or aspects of the repository.
    """
    try:
        # Use code_llm for all GitHub repo analysis
        code_llm = get_appropriate_llm("github repo analysis")

        # Extract repository information
        temp_dir = tempfile.mkdtemp()
        log.info(f"🔍 Cloning repository: {url}")
        repo = git.Repo.clone_from(url, temp_dir)
        log.info(f"✅ Repository cloned to: {temp_dir}")
        
        # Process the query to understand what user wants to analyze
        analyze_prompt = (
            "You are analyzing a GitHub repository based on the user's query.\n"
            "Extract key information about what they want to analyze.\n\n"
            f"Query: {query}\n\n"
            "Extract:\n"
            "1. Target folders/files to analyze (if any)\n"
            "2. Specific aspects to focus on (e.g., code quality, features)\n"
            "3. Type of analysis needed (e.g., code review, overview)\n"
            "Format: Simple text with key points"
        )
        analysis_request = code_llm.invoke(analyze_prompt).content
        log.info(f"🎯 Analysis focus: {analysis_request}")

        # Initialize repository context
        repo_context = {
            'files': [],
            'readme': None,
            'structure': []
        }
        
        # Get all files and their structure
        all_files = list(Path(temp_dir).rglob("*"))
        code_files = [f for f in all_files if f.suffix in [
            '.py', '.js', '.ts', '.java', '.cpp', '.go', 
            '.rb', '.cs', '.php', '.html', '.css', '.md'
        ]]

        # Extract target folder from query if specified
        target_folder = None
        if "folder:" in query.lower():
            folder_match = re.search(r'folder:\s*([^\s]+)', query.lower())
            if folder_match:
                target_folder = folder_match.group(1)
                code_files = [f for f in code_files if target_folder.lower() in str(f).lower()]
                if not code_files:
                    return "Error", f"No code files found in folder '{target_folder}'"

        # Get README content if it exists
        readme_files = [f for f in all_files if f.name.lower() == 'readme.md']
        if readme_files:
            try:
                with open(readme_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    repo_context['readme'] = f.read()
            except Exception as e:
                log.error(f"Error reading README: {e}")

        # Build repository structure
        base_path = Path(temp_dir)
        for file in code_files:
            try:
                rel_path = str(file.relative_to(base_path))
                if file.is_file():
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:3000]  # First 3000 chars
                        repo_context['files'].append({
                            'path': rel_path,
                            'name': file.name,
                            'content': content
                        })
            except Exception as e:
                log.error(f"Error processing {file}: {e}")
        
        # Generate analysis prompt based on query
        prompt = (
            "You are a software engineering expert analyzing a GitHub repository.\n\n"
            f"User's question: {query}\n\n"
            "Based on this repository's content, provide a detailed answer focusing on what the user asked. "
            "If the user's question is general, provide a comprehensive overview.\n\n"
        )

        # Add repository context to the prompt
        if repo_context['readme']:
            prompt += f"README:\n```markdown\n{repo_context['readme']}\n```\n\n"
        
        prompt += "Repository files:\n"
        for file in repo_context['files']:
            prompt += f"\nFile: {file['path']}\n```\n{file['content']}\n```\n"

        # Add specific analysis instructions
        if "code quality" in query.lower():
            prompt += "\nFocus on code quality aspects such as:\n"
            prompt += "1. Code organization and structure\n"
            prompt += "2. Error handling and edge cases\n"
            prompt += "3. Documentation and comments\n"
            prompt += "4. Best practices compliance\n"
            prompt += "5. Potential improvements\n"
        elif "architecture" in query.lower():
            prompt += "\nFocus on architectural aspects such as:\n"
            prompt += "1. System design and components\n"
            prompt += "2. Dependencies and interfaces\n"
            prompt += "3. Design patterns used\n"
            prompt += "4. Scalability considerations\n"
            prompt += "5. Architecture improvements\n"

        # Always use code_llm for repository analysis
        response = code_llm.invoke(prompt)
        return "Repository Analysis", response.content

    except Exception as e:
        return "Error", f"Failed to analyze repository: {str(e)}"
        
        if not code_files:
            return "Error", f"No code files found in {'the specified folder' if target_folder else 'the repository'}."
        
        # Build folder structure
        structure = []
        for file in base_path.rglob("*"):
            if file.is_file() and not any(p.startswith('.') for p in file.parts):  # Skip hidden files/folders
                rel_path = file.relative_to(base_path)
                structure.append(str(rel_path))
        
        prompt = (
            "You are a software engineering expert analyzing a GitHub repository.\n"
            f"{'Analyzing specific folder: ' + target_folder if target_folder else 'Analyzing entire repository'}\n\n"
            "Folder structure:\n" + "\n".join(f"- {p}" for p in sorted(structure)) + "\n\n"
            "Code files content:\n"
        )
        
        for file in code_files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:2000]  # Read first 2000 chars of each file
                rel_path = file.relative_to(base_path)
                prompt += f"\nFile: {rel_path}\n```\n{content}\n```\n"
            except Exception as e:
                log.error(f"Error reading {file}: {e}")
        
        prompt += "\nProvide:\n"
        if target_folder:
            prompt += (
                f"1. Detailed analysis of the {target_folder} folder contents and purpose\n"
                "2. Key functions and features implemented in this folder\n"
                "3. Dependencies and relationships with other parts of the codebase\n"
                "4. Code quality assessment\n"
                "5. Potential improvements specific to this folder"
            )
        else:
            prompt += (
                "1. Overview of the repository structure and purpose\n"
                "2. Main technologies used\n"
                "3. Key features implemented\n"
                "4. Code quality assessment\n"
                "5. Potential improvements"
            )
        
        # Always use code_llm for repository analysis
        response = code_llm.invoke(prompt)
        return "Repository Analysis", response.content

    except Exception as e:
        return "Error", f"Failed to analyze repository: {str(e)}"

def is_casual_conversation(text: str) -> bool:
    """Determine if this is casual conversation vs. an information-seeking query."""
    casual_patterns = [
        r'^(hi|hey|hello|howdy|what\'s up|good (morning|evening|afternoon)|bye|goodbye)',
        r'^how are you',
        r'^(thanks|thank you)',
        r'^nice to meet you',
        r'what\'s your name',
        r'who are you'
    ]
    text = text.lower().strip()
    return any(re.match(pattern, text) for pattern in casual_patterns)
    return response.content.strip()

def should_search_web(question: str) -> bool:
    """Determine if the question requires web search for accurate answer."""
    log.info("🌐 Evaluating if web search is needed")
    current_llm = get_appropriate_llm(question)
    
    prompt = (
        "You determine if a question needs current web information to be answered accurately.\n"
        "Return only YES or NO.\n\n"
        "Examples that need web search:\n"
        "- Questions about current events, prices, or comparisons\n"
        "- Questions seeking specific facts or statistics\n"
        "- Questions about product reviews or recommendations\n\n"
        "Examples that don't need web search:\n"
        "- General knowledge questions\n"
        "- Casual conversation\n"
        "- Basic how-to questions\n"
        "- Conceptual explanations\n\n"
        f"Question: {question}\n\n"
        "Need web search (YES/NO):"
    )
    response = current_llm.invoke(prompt)
    return response.content.strip().upper() == "YES"

def generate_search_queries(question: str) -> List[str]:
    """
    Uses the LLM to generate search queries or handle GitHub repository analysis.
    Can handle regular web searches or detailed GitHub repository analysis requests.
    """
    log.info("🎯 Generating search queries")
    # First check if this is casual conversation
    if is_casual_conversation(question):
        log.info("💬 Detected casual conversation")
        return ["conversation:" + question]

    # Check if this is a GitHub repository analysis request
    if "github.com" in question.lower() and ("https://" in question.lower() or "http://" in question.lower()):
        url_match = re.search(r'(https?://[^\s<>"]+)', question)
        if url_match:
            url = url_match.group(1)
            query_part = question[question.find(url) + len(url):].strip()
            return [f"analyzing_github_repo:{url}|{query_part}"]

    # Check if we need to search the web
    if not should_search_web(question):
        return ["direct_answer:" + question]

    # Regular web search queries
    # Use appropriate LLM based on question type
    current_llm = get_appropriate_llm(question)
    
    prompt = (
        "You are a helpful assistant that creates optimized search queries for search engines.\n"
        "Given a user's natural-language question, create 2-3 different search queries that focus on different aspects "
        "of the topic to get comprehensive information. Make queries concise and specific.\n"
        "IMPORTANT: Do not number the queries or add quotes. Just write each query on a new line.\n\n"
        f"User question: {question}\n\n"
        "Search queries:"
    )
    response = current_llm.invoke(prompt)
    # Clean up the queries
    queries = []
    for q in response.content.strip().split('\n'):
        q = q.strip()
        if not q:
            continue
        q = re.sub(r'^\s*[\[\(]?\d+[\.\)\]]?\s*', '', q)
        q = q.strip('"\'')
        queries.append(q)
    return queries[:3]  # Limit to max 3 queries


def normalize_urls(search_output, mode: str) -> List[str]:
    """
    Accepts outputs from DuckDuckGo search and returns a flat list[str] of URLs, deduped and filtered.
    """
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
# Search using DuckDuckGo
# ---------------------------------
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
    # Select appropriate LLM based on the question type
    current_llm = get_appropriate_llm(question)
    
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
            resp = current_llm.invoke(prompt)
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
        resp = current_llm.invoke(prompt)
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
    resp = current_llm.invoke(final_prompt)
    return resp.content.strip()


# ---------------------------------
# Main runner
# ---------------------------------
def get_direct_answer(question: str) -> str:
    """Generate a direct answer without web search for general knowledge questions."""
    # Select appropriate LLM based on the question type
    current_llm = get_appropriate_llm(question)
    
    prompt = (
        "You are a knowledgeable AI assistant. Answer this question using your general knowledge.\n"
        "Be helpful, clear, and concise. If you're not completely sure, say so.\n"
        "If the question requires current data or specific facts, recommend doing a web search.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = current_llm.invoke(prompt)
    return response.content.strip()

async def main():
    global conversation_history
    conversation_history = []  # Initialize empty conversation history
    
    print("\n💬 AI Assistant: Hello! I'm here to help. Feel free to ask me anything - from general questions to coding and math problems!\n")
    
    while True:
        user_question = input("You: ").strip()
        if not user_question:
            continue
            
        log.info(f"🧠 Original input: {user_question}")
        
        # Handle farewells
        if is_farewell(user_question):
            # Clear conversation history before exiting
            conversation_history = []
            print("\n💬 AI Assistant: Goodbye! Have a great day!")
            break
            
        # Select appropriate LLM based on query type
        current_llm = get_appropriate_llm(user_question)
        
        # Handle greetings or simple conversation
        if is_greeting(user_question) or not needs_search(user_question):
            response = get_conversation_response(user_question, current_llm)
            print(f"\n💬 AI Assistant: {response}\n")
            continue
            
        # For questions needing search, proceed with query generation
        log.info("🔍 Generating search queries for information retrieval...")
        queries = generate_search_queries(user_question)
        
        if not queries:
            response = get_conversation_response(user_question, current_llm)
            print(f"\n💬 AI Assistant: {response}\n")
            continue

        first_query = queries[0]
        
        try:
            # Handle different types of responses
            if first_query.startswith("conversation:"):
                response = get_conversation_response(user_question, current_llm)
                print(f"\n💬 AI Assistant: {response}\n")
                
            elif first_query.startswith("direct_answer:"):
                question = first_query.split(":", 1)[1]
                response = get_direct_answer(question, current_llm)
                print(f"\n💬 AI Assistant: {response}\n")
                
            elif first_query.startswith("analyzing_github_repo:"):
                repo_info = first_query.split(":", 1)[1]
                if "|" in repo_info:
                    repo_url, query = repo_info.split("|", 1)
                else:
                    repo_url, query = repo_info, user_question
                    
                log.info(f"🔍 Analyzing repository: {repo_url}")
                log.info(f"🔍 Analysis query: {query}")
                
                title, analysis = analyze_github_repo(repo_url, query)
                response = get_conversation_response(user_question, code_llm, context=analysis)
                print(f"\n💬 AI Assistant: {response}\n")
                
            else:
                # Regular web search
                all_urls = []
                for query in queries:
                    log.info(f"🔎 Searching for: {query}")
                    raw = await duckduckgo_search(query, MAX_LINKS)
                    urls = normalize_urls(raw, "duckduckgo")
                    all_urls.extend(urls)
                    
                # Remove duplicates while preserving order
                seen = set()
                urls = [url for url in all_urls if not (url in seen or seen.add(url))]
                urls = urls[:MAX_LINKS]

                if not urls:
                    response = get_conversation_response(user_question, current_llm)
                    print(f"\n💬 AI Assistant: I couldn't find specific information online, but {response}\n")
                    continue

                log.info(f"🔗 Using {len(urls)} URL(s)")
                pages = await scrape_pages(urls)
                
                if not pages:
                    response = get_conversation_response(user_question, current_llm)
                    print(f"\n💬 AI Assistant: While I couldn't access the sources I found, {response}\n")
                    continue

                technical_info = summarize_with_ollama(user_question, pages)
                response = get_conversation_response(user_question, current_llm, context=technical_info)
                print(f"\n💬 AI Assistant: {response}\n")
                
        except Exception as e:
            log.error(f"Error processing request: {str(e)}")
            response = get_conversation_response(
                "I encountered an error processing the previous request. "
                "Can you help provide a friendly response and ask them to try again?",
                general_llm  # Use general LLM for error responses
            )
            print(f"\n💬 AI Assistant: {response}\n")


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # First determine the appropriate model and action based on the message content
        query_type = determine_query_type(request.message)
        model_map = {
            'code': 'codellama',
            'math': 'llama2',
            'general': 'mistral'
        }
        
        # Update the model based on content analysis
        actual_model = model_map[query_type]
        current_llm = {
            "mistral": general_llm,
            "codellama": code_llm,
            "llama2": math_llm
        }[actual_model]

        # Determine if web search is needed
        needs_web_search = should_search_web(request.message)
        is_github = "github.com" in request.message.lower()
        
        if is_github:
            # Handle GitHub repository analysis
            title, analysis = analyze_github_repo(request.message)
            response = get_conversation_response(request.message, code_llm, context=analysis)
            return {
                "response": response,
                "model": "codellama",
                "action": "github"
            }
        elif needs_web_search:
            # Handle web search queries
            queries = generate_search_queries(request.message)
            if not any(q.startswith(("conversation:", "direct_answer:", "analyzing_github_repo:")) for q in queries):
                all_urls = []
                for query in queries[:2]:  # Limit to 2 queries for faster response
                    raw = await duckduckgo_search(query, MAX_LINKS)
                    urls = normalize_urls(raw, "duckduckgo")
                    all_urls.extend(urls)
                
                # Remove duplicates while preserving order
                seen = set()
                urls = [url for url in all_urls if not (url in seen or seen.add(url))]
                urls = urls[:MAX_LINKS]
                
                if urls:
                    pages = await scrape_pages(urls)
                    if pages:
                        technical_info = summarize_with_ollama(request.message, pages)
                        response = get_conversation_response(request.message, current_llm, context=technical_info)
                        return {
                            "response": response,
                            "model": actual_model,
                            "action": "search"
                        }
        
        # Handle direct conversation if no web search needed or if web search failed
        response = get_conversation_response(request.message, current_llm)

        return {
            "response": response,
            "model": actual_model,
            "action": "direct"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
