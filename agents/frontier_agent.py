import logging
import os
import re
from litellm import completion

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a product pricing expert. 
Estimate the price of the product described. 
Respond with ONLY a number. No $ sign, no explanation, just the number.
Example: 49.99"""

class FrontierAgent:
    """
    Uses Groq's Llama 3.3 70B with RAG context
    to predict product prices.
    """

    name = "Frontier Agent"

    # Free Groq model
    MODEL = "groq/llama-3.3-70b-versatile"

    def __init__(self, collection=None):
        logger.info(f"Initializing {self.name}")
        self.collection = collection
        logger.info(f"✅ {self.name} ready (RAG: {'enabled' if collection else 'disabled'})")

    def similar_items_context(self, description: str) -> str:
        """Get similar items from ChromaDB for RAG context"""
        if not self.collection:
            return ""
        
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            vector  = encoder.encode(description).astype(float).tolist()
            
            results = self.collection.query(
                query_embeddings=vector,
                n_results=5
            )
            
            context = "\n\nFor reference, here are similar products:\n"
            for doc, metadata in zip(
                results['documents'][0],
                results['metadatas'][0]
            ):
                context += f"- {doc[:100]}... Price: ${metadata['price']:.2f}\n"
            
            return context
        except Exception as e:
            logger.warning(f"RAG failed: {e}")
            return ""

    def messages_for(self, description: str) -> list:
        context = self.similar_items_context(description)
        user_message = f"Estimate the price of this product:\n\n{description}{context}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]

    def price(self, description: str) -> float:
        """Predict price using Groq LLM with optional RAG"""
        logger.info(f"{self.name} pricing: {description[:50]}...")

        try:
            response = completion(
                model=self.MODEL,
                messages=self.messages_for(description),
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            result = result.replace("$","").replace(",","").strip()
            match  = re.search(r"[\d.]+", result)
            price  = float(match.group()) if match else 0.0
            logger.info(f"{self.name} prediction: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return 0.0
