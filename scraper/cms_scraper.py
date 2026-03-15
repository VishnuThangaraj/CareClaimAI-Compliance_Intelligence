from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup, Tag
from fake_useragent import UserAgent
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# CMS Target Catalogue

CMS_TARGETS: List[Dict[str, str]] = [
    {
        "name": "CMS LCD Search",
        "url": "https://www.cms.gov/medicare-coverage-database/search/search-criteria.aspx",
        "type": "LCD",
        "parser": "lcd",
    },
    {
        "name": "CMS NCD Index",
        "url": "https://www.cms.gov/medicare-coverage-database/indexes/ncd-alphabetical-index.aspx",
        "type": "NCD",
        "parser": "ncd",
    },
    {
        "name": "CMS Physician Fee Schedule",
        "url": "https://www.cms.gov/medicare/payment/fee-schedules/physician",
        "type": "FEE_SCHEDULE",
        "parser": "fee_schedule",
    },
    {
        "name": "CMS OPPS Addenda",
        "url": "https://www.cms.gov/medicare/payment/prospective-payment-systems/hospital-outpatient/addendum",
        "type": "OPPS",
        "parser": "generic",
    },
    {
        "name": "MLN Matters Articles",
        "url": "https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnmattersarticles",
        "type": "MLN",
        "parser": "mln",
    },
    {
        "name": "CMS ICD-10 Updates",
        "url": "https://www.cms.gov/medicare/coding-billing/icd-10-codes",
        "type": "ICD10",
        "parser": "generic",
    },
]


# Data Models

@dataclass
class ScrapedPolicy:
    policy_id: str
    source_url: str
    source_type: str            
    title: str
    raw_text: str
    html_content: str
    billing_codes: List[str]   
    effective_date: Optional[datetime]
    revision_date: Optional[datetime]
    summary: str
    content_hash: str           
    metadata: Dict[str, Any] = field(default_factory=dict)
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Code Extraction Patterns

CPT_PATTERN     = re.compile(r'\b(\d{5}[A-Z]?)\b')
HCPCS_PATTERN   = re.compile(r'\b([A-V][0-9]{4})\b')
ICD10_PATTERN   = re.compile(r'\b([A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?)\b')
DATE_PATTERNS   = [
    re.compile(r'(\d{1,2}/\d{1,2}/\d{4})'),
    re.compile(r'(\w+ \d{1,2},\s*\d{4})'),
    re.compile(r'(\d{4}-\d{2}-\d{2})'),
]


def extract_billing_codes(text: str) -> List[str]:
    codes: set[str] = set()
    codes.update(CPT_PATTERN.findall(text))
    codes.update(HCPCS_PATTERN.findall(text))
    # ICD-10 — filter out common false positives
    for m in ICD10_PATTERN.findall(text):
        if len(m) >= 3:
            codes.add(m)
    return sorted(codes)


def extract_date(text: str) -> Optional[datetime]:
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if m:
            date_str = m.group(1)
            for fmt in ("%m/%d/%Y", "%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt).replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue
    return None


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# HTTP Session Factory

_ua = UserAgent()

def _make_headers() -> Dict[str, str]:
    return {
        "User-Agent": _ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


# Core Scraper

class CMSScraper:

    def __init__(self, concurrency: int = 5, timeout: int = 30):
        self.concurrency = concurrency
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._seen_hashes: set[str] = set()          # in-memory dedup cache
        self._semaphore = asyncio.Semaphore(concurrency)

    # Public API 

    async def scrape_all(self) -> List[ScrapedPolicy]:
        """Scrape all configured CMS targets concurrently."""
        async with aiohttp.ClientSession(
            headers=_make_headers(),
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(ssl=False, limit=self.concurrency),
        ) as session:
            tasks = [
                self._scrape_target(session, target)
                for target in CMS_TARGETS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        policies: List[ScrapedPolicy] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Scrape task failed: {r}")
            elif r:
                policies.extend(r if isinstance(r, list) else [r])

        logger.info(f"Scraped {len(policies)} new/changed policies")
        return policies

    async def scrape_url(self, url: str, source_type: str = "MANUAL") -> Optional[ScrapedPolicy]:
        """Scrape a single arbitrary CMS URL."""
        async with aiohttp.ClientSession(
            headers=_make_headers(),
            timeout=self.timeout,
        ) as session:
            html = await self._fetch(session, url)
            if not html:
                return None
            return self._parse_generic(html, url, source_type)

    # Internal

    async def _scrape_target(
        self, session: aiohttp.ClientSession, target: Dict[str, str]
    ) -> List[ScrapedPolicy]:
        async with self._semaphore:
            logger.info(f"Scraping: {target['name']}")
            html = await self._fetch(session, target["url"])
            if not html:
                return []

            parser_fn = getattr(self, f"_parse_{target['parser']}", self._parse_generic)
            try:
                result = parser_fn(html, target["url"], target["type"])
            except Exception as e:
                logger.error(f"Parse error [{target['name']}]: {e}")
                return []

            if isinstance(result, list):
                return [p for p in result if self._is_new(p) and self._is_quality(p)]
            elif result and self._is_new(result) and self._is_quality(result):
                return [result]
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status == 200:
                    return await resp.text(errors="replace")
                elif resp.status == 403:
                    # Try Playwright for JS-gated pages
                    return await self._fetch_playwright(url)
                else:
                    logger.warning(f"HTTP {resp.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Fetch error [{url}]: {e}")
            raise

    async def _fetch_playwright(self, url: str) -> Optional[str]:
        """Fallback: headless browser for JavaScript-heavy CMS portals."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30_000)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            logger.error(f"Playwright fallback failed [{url}]: {e}")
            return None

    def _is_new(self, policy: ScrapedPolicy) -> bool:
        if policy.content_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(policy.content_hash)
        return True

    @staticmethod
    def _is_quality(policy: ScrapedPolicy) -> bool:
        """Reject CMS nav/index pages that have no real policy content."""
        junk_titles = {
            "submit feedback", "ask a question", "alphabetical index",
            "ncd alphabetical", "search criteria", "search results",
            "medicare coverage database", "addendum index", "national coverage",
        }
        junk_urls = [
            "alphabetical-index", "indexes/", "search-criteria",
            "search/search", "#ncd", "feedback", "ask-a-question",
        ]
        title = policy.title.strip().lower()
        url   = policy.source_url.lower()
        if len(policy.raw_text.strip()) < 200:
            return False
        if any(j in title for j in junk_titles):
            return False
        if any(p in url for p in junk_urls):
            return False
        if len(policy.raw_text.split()) < 40:
            return False
        return True

    # Parsers─

    def _parse_generic(self, html: str, url: str, source_type: str) -> Optional[ScrapedPolicy]:
        soup = BeautifulSoup(html, "lxml")
        self._clean_soup(soup)

        title = self._extract_title(soup)
        main_content = self._extract_main_content(soup)
        raw_text = main_content.get_text(separator="\n", strip=True)

        if len(raw_text) < 100:
            return None

        policy_id = self._url_to_id(url)
        return ScrapedPolicy(
            policy_id=policy_id,
            source_url=url,
            source_type=source_type,
            title=title,
            raw_text=raw_text,
            html_content=str(main_content),
            billing_codes=extract_billing_codes(raw_text),
            effective_date=extract_date(raw_text),
            revision_date=None,
            summary=raw_text[:500],
            content_hash=compute_hash(raw_text),
        )

    def _parse_lcd(self, html: str, url: str, source_type: str) -> Optional[ScrapedPolicy]:
        soup = BeautifulSoup(html, "lxml")
        self._clean_soup(soup)

        # LCD-specific: extract LCD number, contractor, effective dates
        lcd_number = ""
        lcd_tag = soup.find(string=re.compile(r'L\d{5}'))
        if lcd_tag:
            m = re.search(r'(L\d{5})', str(lcd_tag))
            if m:
                lcd_number = m.group(1)

        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else "LCD Policy"

        # Extract contractor info
        contractor = ""
        for tag in soup.find_all(string=re.compile(r'Contractor', re.I)):
            parent = tag.parent
            if parent:
                contractor = parent.get_text(strip=True)[:100]
                break

        main_div = (
            soup.find("div", {"id": re.compile(r'content|main|policy', re.I)})
            or soup.find("main")
            or soup.body
            or soup
        )
        raw_text = main_div.get_text(separator="\n", strip=True) if main_div else ""

        return ScrapedPolicy(
            policy_id=lcd_number or self._url_to_id(url),
            source_url=url,
            source_type="LCD",
            title=title,
            raw_text=raw_text,
            html_content=str(main_div),
            billing_codes=extract_billing_codes(raw_text),
            effective_date=extract_date(raw_text),
            revision_date=None,
            summary=raw_text[:500],
            content_hash=compute_hash(raw_text),
            metadata={"lcd_number": lcd_number, "contractor": contractor},
        )

    def _parse_ncd(self, html: str, url: str, source_type: str) -> List[ScrapedPolicy]:
        """Parse NCD index page → list of individual NCD links → scrape each."""
        soup = BeautifulSoup(html, "lxml")
        policies = []

        # Extract links to individual NCD pages
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "ncd" in href.lower() and "manual" not in href.lower():
                full_url = urljoin(url, href)
                links.append(full_url)

        # Return lightweight stubs (full scrape happens in batch)
        for link in links[:20]:   # cap for rate-limiting
            policy_id = self._url_to_id(link)
            text = a.get_text(strip=True)
            if text:
                stub = ScrapedPolicy(
                    policy_id=policy_id,
                    source_url=link,
                    source_type="NCD",
                    title=text[:200],
                    raw_text=text,
                    html_content="",
                    billing_codes=extract_billing_codes(text),
                    effective_date=None,
                    revision_date=None,
                    summary=text[:300],
                    content_hash=compute_hash(link),   # URL hash as stub
                )
                policies.append(stub)

        return policies

    def _parse_fee_schedule(self, html: str, url: str, source_type: str) -> Optional[ScrapedPolicy]:
        soup = BeautifulSoup(html, "lxml")
        self._clean_soup(soup)

        # Fee schedule: capture tables with CPT codes and RVU values
        tables = soup.find_all("table")
        table_text = ""
        for tbl in tables[:5]:
            table_text += tbl.get_text(separator="\t", strip=True) + "\n"

        raw_text = soup.get_text(separator="\n", strip=True)
        combined = raw_text + "\n" + table_text

        return ScrapedPolicy(
            policy_id=f"fee_schedule_{datetime.now().strftime('%Y%m')}",
            source_url=url,
            source_type="FEE_SCHEDULE",
            title="CMS Physician Fee Schedule",
            raw_text=combined,
            html_content=html[:50_000],  # cap size
            billing_codes=extract_billing_codes(combined),
            effective_date=extract_date(combined),
            revision_date=None,
            summary=raw_text[:500],
            content_hash=compute_hash(combined),
        )

    def _parse_mln(self, html: str, url: str, source_type: str) -> List[ScrapedPolicy]:
        """Parse MLN Matters article listings."""
        soup = BeautifulSoup(html, "lxml")
        policies = []

        articles = soup.find_all(
            "article",
        ) or soup.find_all("div", class_=re.compile(r"article|item|entry", re.I))

        for art in articles[:10]:
            link_tag = art.find("a", href=True)
            if not link_tag:
                continue
            link_url = urljoin(url, link_tag["href"])
            title = link_tag.get_text(strip=True)
            text = art.get_text(separator="\n", strip=True)

            policies.append(
                ScrapedPolicy(
                    policy_id=self._url_to_id(link_url),
                    source_url=link_url,
                    source_type="MLN",
                    title=title[:200],
                    raw_text=text,
                    html_content=str(art),
                    billing_codes=extract_billing_codes(text),
                    effective_date=extract_date(text),
                    revision_date=None,
                    summary=text[:400],
                    content_hash=compute_hash(text),
                )
            )
        return policies

    # Utilities ─

    @staticmethod
    def _clean_soup(soup: BeautifulSoup) -> None:
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            tag.decompose()

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        for selector in ["h1", "h2", "title"]:
            el = soup.find(selector)
            if el:
                return el.get_text(strip=True)[:300]
        return "Untitled Policy"

    @staticmethod
    def _extract_main_content(soup: BeautifulSoup) -> Tag:
        for selector in [
            {"id": re.compile(r"main|content|body-content|article", re.I)},
            {"role": "main"},
            {"class": re.compile(r"main|content|article|page-body", re.I)},
        ]:
            el = soup.find(attrs=selector)
            if el:
                return el
        return soup.find("main") or soup.find("body") or soup

    @staticmethod
    def _url_to_id(url: str) -> str:
        """Derive a stable, deterministic policy_id from URL."""
        parsed = urlparse(url)
        path_part = parsed.path.strip("/").replace("/", "_").replace(".", "_")
        return f"cms_{path_part}"[:100] or f"cms_{compute_hash(url)[:16]}"
