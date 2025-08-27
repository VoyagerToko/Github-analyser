import asyncio
from playwright.async_api import async_playwright
from langchain_ollama import ChatOllama
import json

# -------------------------------
# LLM INIT
# -------------------------------
llm = ChatOllama(model="mistral")

# -------------------------------
# BROWSER TOOLS
# -------------------------------
class BrowserTools:
    def __init__(self, page):
        self.page = page

    async def goto(self, url):
        try:
            await self.page.goto(url, wait_until="networkidle")
            await asyncio.sleep(1)  # buffer for page load
            return f"🌐 Navigated to {url}"
        except Exception as e:
            return f"⚠️ Could not navigate to {url}: {e}"

    async def click(self, selector, retries=3):
        for attempt in range(retries):
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.click(selector)
                await self.page.wait_for_load_state("networkidle")
                await asyncio.sleep(1)
                return f"✅ Clicked {selector}"
            except Exception:
                await asyncio.sleep(0.5)
        return f"⚠️ Could not click {selector} after {retries} attempts"

    async def type_query_generic(self, query, retries=3):
        # List of common input selectors to try
        input_selectors = [
            "input[name='q']",          # Google
            "input[type='text']",       # generic
            "input[title='Search']"     # generic
        ]
        for attempt in range(retries):
            for sel in input_selectors:
                try:
                    inp = await self.page.query_selector(sel)
                    if inp and await inp.is_visible() and await inp.is_enabled():
                        await inp.scroll_into_view_if_needed()
                        await inp.click(force=True)
                        await inp.fill("")           # clear prefilled text
                        await inp.type(query, delay=50)  # human-like typing
                        await inp.press("Enter")
                        await self.page.wait_for_load_state("networkidle")
                        await asyncio.sleep(1)
                        return f"✅ Typed query '{query}' in '{sel}'"
                except Exception:
                    continue
            await asyncio.sleep(0.5)
        return f"⚠️ Failed to type query '{query}' after {retries} attempts"

    async def read_texts(self, selector=None, retries=3):
        for attempt in range(retries):
            try:
                if selector:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    content = await self.page.text_content(selector)
                    return [content.strip()] if content else ["⚠️ Empty"]
                else:
                    elements = await self.page.locator("body *").all_text_contents()
                    return [t.strip() for t in elements if t.strip()]
            except Exception:
                await asyncio.sleep(0.5)
        return [f"⚠️ Failed to read texts after {retries} attempts"]

# -------------------------------
# RECURSIVE AGENT
# -------------------------------
async def agent_recursive(query, tools: BrowserTools, depth=0, max_depth=2):
    if depth > max_depth:
        return []

    # Step 1: read full page contents
    full_page_texts = await tools.read_texts()
    full_page_content = "\n".join(full_page_texts)

    # Step 2: ask LLM for next steps
    prompt = f"""
You are a web automation agent.
User query: "{query}"
Current FULL page content: "{full_page_content}"

Instructions:
- Respond ONLY in JSON list of steps
- Each step is an object: {{"action": "goto" | "search" | "click" | "read", "url": "...", "selector": "...", "text": "..."}}
"""
    plan = llm.invoke(prompt)
    steps_json = plan.content if hasattr(plan, "content") else str(plan)
    print(f"\n🔹 PLAN JSON (depth {depth}):\n", steps_json)

    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError:
        steps = []

    results = []
    for step in steps:
        action = step.get("action")
        if action == "goto":
            results.append(await tools.goto(step.get("url", "")))
        elif action == "search":
            results.append(await tools.type_query_generic(step.get("text", "")))
        elif action == "click":
            results.append(await tools.click(step.get("selector", "")))
        elif action == "read":
            results.extend(await tools.read_texts(step.get("selector")))
        else:
            results.extend(await tools.read_texts())  # fallback

    # Step 3: recurse if allowed
    if results and depth < max_depth:
        results.extend(await agent_recursive(query, tools, depth=depth+1, max_depth=max_depth))

    return results

# -------------------------------
# MAIN LOOP
# -------------------------------
async def main():
    query = "Latest AI news headlines"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()
        tools = BrowserTools(page)

        results = await agent_recursive(query, tools)
        print("\n✅ Final Results:", results[:50])  # first 50 lines

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
