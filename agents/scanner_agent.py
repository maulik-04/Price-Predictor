import logging
import os
import json
import re
from groq import Groq
from dataclasses import dataclass
from typing import List
import feedparser
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

@dataclass
class Deal:
    title:       str
    description: str
    price:       float
    url:         str

    def describe(self) -> str:
        return f"Title: {self.title}\nDescription: {self.description}\nURL: {self.url}"


RSS_FEEDS = [
    "https://slickdeals.net/newsearch.php?mode=frontpage&searchin=first&RSS=1",
    "https://www.dealnews.com/c142/Electronics/?rss=1",
]

SYSTEM_PROMPT = """You identify the 5 best deals from a list.
Select deals with clear prices and detailed descriptions.
Respond ONLY in this JSON format, no other text:
{
  "deals": [
    {
      "title": "product title",
      "description": "clean product description without deal terms",
      "price": 49.99,
      "url": "https://..."
    }
  ]
}
Rules:
- Price must be the actual product price not discount amount
- Ignore deals where price is unclear
- Description should describe the product not the deal terms
- Return exactly 5 deals"""


class ScannerAgent:
    """
    Scans RSS deal feeds and uses Groq (free)
    to filter and extract the 5 best deals.
    """

    name = "Scanner Agent"

    def __init__(self):
        logger.info(f"Initializing {self.name}")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        logger.info(f"✅ {self.name} ready")

    def fetch_deals(self) -> List[str]:
        """Fetch raw deals from RSS feeds"""
        raw_deals = []

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    raw_deals.append(
                        f"Title: {entry.get('title', '')}\n"
                        f"Description: {entry.get('summary', '')[:300]}\n"
                        f"URL: {entry.get('link', '')}"
                    )
                logger.info(f"Fetched {len(feed.entries)} from {feed_url}")
            except Exception as e:
                logger.warning(f"Failed to fetch {feed_url}: {e}")

        logger.info(f"Total raw deals: {len(raw_deals)}")
        return raw_deals

    def filter_deals(self, raw_deals: List[str]) -> List[Deal]:
        """Use Groq LLM to filter and extract best deals"""
        deals_text = "\n\n---\n\n".join(raw_deals[:30])

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Here are the deals:\n\n{deals_text}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0
            )

            content = response.choices[0].message.content
            data    = json.loads(content)
            deals   = [Deal(**d) for d in data["deals"]]
            logger.info(f"Filtered to {len(deals)} quality deals")
            return deals

        except Exception as e:
            logger.error(f"LLM filtering failed: {e}")
            return []

    def scan(self) -> List[Deal]:
        """Fetch and filter deals"""
        logger.info(f"{self.name} scanning for deals...")
        raw_deals = self.fetch_deals()

        if not raw_deals:
            logger.warning("No deals fetched")
            return []

        deals = self.filter_deals(raw_deals)
        logger.info(f"Found {len(deals)} quality deals")
        return deals
