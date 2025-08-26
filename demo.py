import asyncio
import logging
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Ollama phi model
llm = ChatOllama(model="phi")

async def google_search(query, max_links=3):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                 "Chrome/115.0.0.0 Safari/537.36",
                                     locale="en-US")

        await page.goto("https://www.google.com")

        # Accept cookie popups
        for btn_text in ["I agree", "Accept all"]:
            try:
                await page.click(f'button:has-text("{btn_text}")', timeout=3000)
                logging.info(f"Accepted popup with button: {btn_text}")
                break
            except Exception:
                logging.info(f"No popup with button: {btn_text} found")

        await page.wait_for_timeout(2000)
        await page.screenshot(path="google_page_before_wait.png")
        logging.info("Saved screenshot to google_page_before_wait.png")

        selectors = [
            'input[name="q"]',
            'input[aria-label="Search"]',
            'textarea[name="q"]',  # Added for new Google UI
        ]

        for selector in selectors:
            try:
                await page.wait_for_selector(selector, timeout=7000)
                await page.fill(selector, query)
                logging.info(f"Filled search input using selector: {selector}")
                break
            except Exception as e:
                logging.warning(f"Selector {selector} not found or fill failed: {e}")
        else:
            # Advanced fallback: query all input/textarea elements via Playwright
            logging.warning("Trying advanced fallback with all inputs...")

            all_inputs = await page.query_selector_all("input, textarea")
            logging.info(f"Found {len(all_inputs)} input/textarea elements")

            for i, element in enumerate(all_inputs):
                try:
                    attr_type = await element.get_attribute("type") or ""
                    attr_name = await element.get_attribute("name") or ""
                    aria_label = await element.get_attribute("aria-label") or ""

                    logging.info(f"[{i}] name={attr_name}, type={attr_type}, aria-label={aria_label}")

                    if "search" in aria_label.lower() or attr_type == "text":
                        await element.fill(query)
                        logging.info(f"Successfully filled fallback input: name={attr_name}")
                        break
                except Exception as e:
                    logging.warning(f"Failed to fill input [{i}]: {e}")
            else:
                # If all fails
                html = await page.content()
                with open("google_fallback.html", "w", encoding="utf-8") as f:
                    f.write(html)
                raise Exception("No valid search input found even after advanced fallback.")

        await page.keyboard.press('Enter')
        await page.wait_for_selector('div.g', timeout=5000)

        results = await page.query_selector_all('div.g a')
        urls = []
        for r in results:
            href = await r.get_attribute('href')
            if href and href.startswith("http"):
                urls.append(href)
            if len(urls) >= max_links:
                break

        contents = []
        for url in urls:
            try:
                await page.goto(url)
                await page.wait_for_load_state('load')

                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all('p')
                text = "\n".join(p.get_text() for p in paragraphs)
                contents.append((url, text[:5000]))  # limit size
            except Exception as e:
                logging.warning(f"Failed to fetch or parse {url}: {e}")

        await browser.close()
        return contents



def answer_query_with_ollama(query, contents):
    combined_text = "\n\n".join([f"URL: {url}\nContent:\n{text}" for url, text in contents])
    prompt = f"""
You are a helpful assistant using external web information.

User question: {query}

Based on the following extracted web page contents, provide a concise and accurate answer.

{combined_text}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content


async def main():
    query = input("Enter your question: ")
    logging.info(f"User input query: {query}")
    print(f"Searching Google for: {query}")

    pages = await google_search(query)
    logging.info(f"Fetched {len(pages)} pages")

    if not pages:
        logging.warning("No pages fetched, aborting.")
        print("No pages fetched, aborting.")
        return

    print(f"Fetched {len(pages)} pages. Asking Ollama...")

    answer = answer_query_with_ollama(query, pages)
    print("\n---\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
