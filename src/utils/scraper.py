from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from pydantic import BaseModel
from typing import Optional
import time
import os
import argparse

parser = argparse.ArgumentParser(description="Arguments for scraping")

parser.add_argument("--max_docs", type=int, default=None,
                    help="Number of maximum documents for scraping. Set it None for manual stopping.")

parser.add_argument("--start_id", type=int, default=int,
                    help="Starting point of the scraping.")

parser.add_argument("--base_url", type=str, default="https://e-qanun.az/framework/",
                    help="Base url for scraping.")

parser.add_argument("--save_dir", type=str, default="../../data/scraped_docs/", 
                    help="The path for saving scraped documents.")

args = parser.parse_args()

class ScrapingResult(BaseModel):
    success: bool = False
    page_content  : Optional[str] = None
    text_content  : Optional[str] = None
    error_message : Optional[str] = None

def scrape(
        url : str, 
        wait_time : float = 10000.
        ) -> ScrapingResult :
    """
    Scrapes websites. Designed for working on E-Qanun.

    Args:
        url: str
            The link to the website.
    wait_time : float (optional, default is 10000.)
        Time to wait for a webpage to load, in milliseconds
    
    Returns:
        ResultType: An object that contains the output information.
        This class has these attributes:
            success: bool
                whether the scraping has successfully completed or not
            page_content  : Optional[str]
                The content of the page in raw HTML format
            text_content  : Optional[str]
                The relevant content of the page without HTML formatting.
                Strikethrough portions are directly removed.
            error_message : Optional[str]
                The content of the error if there occurs any.
    """
    result = ScrapingResult()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded")

            page.evaluate("""
                // Trigger common events that might load content
                window.dispatchEvent(new Event('load'));
                window.dispatchEvent(new Event('DOMContentLoaded'));

                // Scroll to trigger lazy loading
                window.scrollTo(0, document.body.scrollHeight);

                // Try to trigger any onload handlers
                if (window.onload) window.onload();
            """)

            page.wait_for_timeout(wait_time)

            # Remove all <s> and <del> elements before extracting the text
            cleaned_text = page.evaluate("""() => {
                const elementsToRemove = document.querySelectorAll('s, del');
                elementsToRemove.forEach(el => el.remove());
                return document.body.innerText;
            }""")

            if cleaned_text:
                result.success = True
            
            # You can still get the original page content if needed
            result.page_content = page.content()
            # removing the ambiguous NBSP character (no-break space) 
            # and the ambiguous dash icon which resulted in
            #  issues in the chunking process.
            result.text_content = cleaned_text  \
                .replace(" ", " ")              \
                .replace("–", "-")

        except Exception as e:
            result.error_message = str(e)
        finally:
            browser.close()

    return result

async def async_scrape(
    url: str,
    wait_time: float = 10000.
    ) -> ScrapingResult:
    """
    Asynchronously scrapes websites. Designed for working on E-Qanun.

    Args:
        url: str
            The link to the website.
        wait_time: float (optional, default is 10000.)
            Time to wait for a webpage to load, in milliseconds.
    
    Returns:
        ScrapingResult: An object that contains the output information.
        This class has these attributes:
            success: bool
                Whether the scraping has successfully completed or not.
            page_content: Optional[str]
                The content of the page in raw HTML format.
            text_content: Optional[str]
                The relevant content of the page without HTML formatting.
                Strikethrough portions are directly removed.
            error_message: Optional[str]
                The content of the error if there occurs any.
    """
    result = ScrapingResult()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded")

            await page.evaluate("""
                // Trigger common events that might load content
                window.dispatchEvent(new Event('load'));
                window.dispatchEvent(new Event('DOMContentLoaded'));

                // Scroll to trigger lazy loading
                window.scrollTo(0, document.body.scrollHeight);

                // Try to trigger any onload handlers
                if (window.onload) window.onload();
            """)

            await page.wait_for_timeout(wait_time)

            # Remove all <s> and <del> elements before extracting the text
            cleaned_text = await page.evaluate("""() => {
                const elementsToRemove = document.querySelectorAll('s, del');
                elementsToRemove.forEach(el => el.remove());
                return document.body.innerText;
            }""")

            if cleaned_text:
                result.success = True
            
            # You can still get the original page content if needed
            result.page_content = await page.content()
            # removing the ambiguous NBSP character (no-break space) 
            # and the ambiguous dash icon which resulted in
            # issues in the chunking process.
            result.text_content = cleaned_text \
                .replace(" ", " ") \
                .replace("–", "-")

        except Exception as e:
            result.error_message = str(e)
        finally:
            await browser.close()

    return result

def is_valid(result):
    return result.success and result.text_content and len(result.text_content.strip()) > 100

def main(max_docs: int, start_id: int,
         base_url: str, save_dir: str,
         wait_seconds: int=5):
    """
    Scraping E-Qanun.az automatically. Scraping is done until the specified number of documents or KeyboardInterrupt.

    Args: 
        max_docs: maximum number of documents for scraping. Set None for manual stopping.
        base_url: Base scraping url
        start_id: Documents are stored in the combination of this value with base url.
        save_dir: The path for saving scraped documents.
        wait_seconds: waiting time for requesting to E-qanun.az
    """
    print("Starting scraper... Press Ctrl+C to stop.")
    count = 0
    current_id = start_id

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        while True:
            url = f"{base_url}{current_id}"
            print(f"[{count}] Scraping: {url}")

            result = scrape(url)

            if is_valid(result):
                filename = os.path.join(save_dir, f"doc_{current_id}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.text_content)
                count += 1
                print(f"Saved → {filename}")
            else:
                print("Skipped (invalid or not found).")

            current_id += 1

            if max_docs and count >= max_docs:
                print(f"Reached max_docs={max_docs}")
                break

            time.sleep(wait_seconds)  # be polite

    except KeyboardInterrupt:
        print("Scraping interrupted manually. Exiting...")

if __name__ == "__main__":
    main(max_docs=args.max_docs, start_id=args.start_id,
         base_url=args.base_url, save_dir=args.save_dir)
