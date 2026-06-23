import logging
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent

logger = logging.getLogger(__name__)

class EnsembleAgent:
    """
    Combines SpecialistAgent and FrontierAgent
    using weighted averaging for best accuracy.
    
    Weights:
    - FrontierAgent (Groq Llama 70B): 70%
    - SpecialistAgent (Fine-tuned Llama 3.2-3B): 30%
    """

    name = "Ensemble Agent"

    def __init__(self, collection=None):
        logger.info(f"Initializing {self.name}")
        self.specialist = SpecialistAgent()
        self.frontier   = FrontierAgent(collection)
        logger.info(f"{self.name} ready")

    def price(self, description: str) -> float:
        """Predict price using weighted ensemble"""
        logger.info(f"{self.name} pricing: {description[:50]}...")

        specialist_price = self.specialist.price(description)
        frontier_price   = self.frontier.price(description)

        # Weighted average
        ensemble_price = (frontier_price * 0.7) + (specialist_price * 0.3)

        logger.info(f"{self.name} results:")
        logger.info(f"  Specialist: ${specialist_price:.2f}")
        logger.info(f"  Frontier:   ${frontier_price:.2f}")
        logger.info(f"  Ensemble:   ${ensemble_price:.2f}")

        return ensemble_price
