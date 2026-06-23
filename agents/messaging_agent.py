import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

class MessagingAgent:
    """
    Sends push notifications via Pushover
    when good deals are found.
    """

    name = "Messaging Agent"

    def __init__(self):
        self.user  = os.getenv("PUSHOVER_USER")
        self.token = os.getenv("PUSHOVER_TOKEN")

        if self.user and self.token:
            logger.info(f"✅ {self.name} ready (Pushover configured)")
        else:
            logger.warning(f"⚠️ {self.name} — Pushover not configured")

    def push(self, message: str):
        """Send raw push notification"""
        print(f"📱 Push: {message}")

        if not self.user or not self.token:
            logger.warning("Pushover not configured — skipping push")
            return

        try:
            payload = {
                "user":    self.user,
                "token":   self.token,
                "message": message,
            }
            response = requests.post(PUSHOVER_URL, data=payload)
            if response.status_code == 200:
                logger.info("✅ Push notification sent")
            else:
                logger.error(f"Push failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Push error: {e}")

    def notify(self, description: str, price: float, estimate: float, url: str):
        """Send formatted deal notification"""
        discount = ((estimate - price) / estimate) * 100

        message = (
            f"🎉 DEAL FOUND!\n"
            f"Product: {description[:80]}\n"
            f"Deal Price: ${price:.2f}\n"
            f"Estimated Value: ${estimate:.2f}\n"
            f"Discount: {discount:.0f}% off!\n"
            f"Link: {url}"
        )

        self.push(message)
