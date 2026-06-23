import logging
import modal
import re

logger = logging.getLogger(__name__)

class SpecialistAgent:
    """
    Uses our fine-tuned Llama 3.2-3B model deployed on Modal
    to predict product prices from descriptions.
    """

    name = "Specialist Agent"

    def __init__(self):
        logger.info(f"Initializing {self.name}")
        # Connect to deployed Modal service
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        logger.info(f"{self.name} connected to Modal pricer service")

    def price(self, description: str) -> float:
        """Predict price using fine-tuned Llama model"""
        logger.info(f"{self.name} pricing: {description[:50]}...")
        
        try:
            result = self.pricer.price.remote(description)
            price  = float(result)
            logger.info(f"{self.name} prediction: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return 0.0
