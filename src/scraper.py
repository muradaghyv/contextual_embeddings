from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from pydantic import BaseModel
from typing import Optional

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

if __name__ == "__main__":
    result = scrape("https://e-qanun.az/framework/46944")
    

    with open("scraped_sample_46944.txt", "w") as f:
        f.write(result.text_content)
