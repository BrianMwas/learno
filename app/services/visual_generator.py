"""
Visual generation service for converting descriptions to diagrams.
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from app.core.config import get_settings
import logging
import re

logger = logging.getLogger(__name__)
settings = get_settings()


class VisualGenerator:
    """Generates visual diagrams from text descriptions."""
    
    # Pre-made SVG assets for common biology concepts
    PREMADE_ASSETS = {
        "cell": "cell-structure",
        "mitochondria": "mitochondria",
        "nucleus": "nucleus",
        "ribosome": "ribosome",
        "dna": "dna-helix",
        "chromosome": "chromosome",
        "membrane": "cell-membrane",
        "protein": "protein-structure",
        "enzyme": "enzyme-substrate",
        "photosynthesis": "photosynthesis-cycle",
        "respiration": "cellular-respiration",
        "mitosis": "mitosis-stages",
        "meiosis": "meiosis-stages",
    }
    
    def __init__(self):
        """Initialize the visual generator with LLM."""
        try:
            self.model = init_chat_model(
                f"openai:{settings.OPENAI_MODEL}",
                temperature=0.3,  # Lower temp for more consistent diagrams
                api_key=settings.OPENAI_API_KEY
            )
            logger.info("VisualGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VisualGenerator: {str(e)}")
            raise
    
    def generate_visual(self, description: str, topic: str, content: str) -> dict:
        """
        Generate visual representation from description.
        
        Args:
            description: Visual description from LLM
            topic: Current topic being taught
            content: Full teaching content for context
            
        Returns:
            dict with keys:
                - type: "mermaid" | "svg" | "premade" | "none"
                - data: diagram code or asset name
                - fallback_text: description if visualization fails
        """
        try:
            logger.info(f"Generating visual for topic: {topic}")
            
            # Step 1: Check for pre-made assets
            premade = self._match_premade_asset(topic, description)
            if premade:
                logger.info(f"Using pre-made asset: {premade}")
                return {
                    "type": "premade",
                    "data": premade,
                    "fallback_text": description
                }
            
            # Step 2: Determine best visualization type
            viz_type = self._determine_visualization_type(description, content)
            logger.info(f"Determined visualization type: {viz_type}")
            
            # Step 3: Generate based on type
            if viz_type == "mermaid":
                return self._generate_mermaid(description, topic, content)
            elif viz_type == "svg":
                return self._generate_svg(description, topic)
            else:
                return {
                    "type": "none",
                    "data": None,
                    "fallback_text": description
                }
                
        except Exception as e:
            logger.error(f"Error generating visual: {str(e)}", exc_info=True)
            return {
                "type": "none",
                "data": None,
                "fallback_text": description
            }
    
    def _match_premade_asset(self, topic: str, description: str) -> str | None:
        """Match topic/description to pre-made asset."""
        topic_lower = topic.lower()
        desc_lower = description.lower()
        
        # Check topic first
        for keyword, asset in self.PREMADE_ASSETS.items():
            if keyword in topic_lower or keyword in desc_lower:
                return asset
        
        return None
    
    def _determine_visualization_type(self, description: str, content: str) -> str:
        """
        Determine best visualization type using LLM.
        
        Returns: "mermaid", "svg", or "none"
        """
        try:
            prompt = f"""Analyze this visual description and determine the best way to visualize it.

                Visual Description: {description}

                Content Context: {content[:500]}

                Choose ONE:
                - "mermaid" if it involves: processes, flows, cycles, steps, hierarchies, relationships, sequences, comparisons
                - "svg" if it involves: simple shapes, basic structures, labeled diagrams, spatial layouts, geometric forms
                - "none" if it's too complex or better left as text description

                Respond with ONLY one word: mermaid, svg, or none"""

            response = self.model.invoke([SystemMessage(content=prompt)])
            viz_type = response.content.strip().lower()
            
            # Validate response
            if viz_type not in ["mermaid", "svg", "none"]:
                logger.warning(f"Invalid viz_type '{viz_type}', defaulting to 'none'")
                return "none"
            
            return viz_type
            
        except Exception as e:
            logger.error(f"Error determining viz type: {str(e)}")
            return "none"
    
    def _generate_mermaid(self, description: str, topic: str, content: str) -> dict:
        """Generate Mermaid diagram code from description."""
        try:
            prompt = f"""Convert this into a Mermaid diagram.

                Topic: {topic}
                Visual Description: {description}
                Content: {content[:500]}

                Generate VALID Mermaid syntax. Choose the most appropriate diagram type:
                - flowchart TD/LR for processes, flows
                - graph TD/LR for relationships
                - sequenceDiagram for interactions
                - classDiagram for structures

                Rules:
                1. Use simple, short node IDs (A, B, C1, etc)
                2. Keep labels concise (max 20 chars)
                3. Use emojis sparingly
                4. Ensure all syntax is valid
                5. Add styling if appropriate

                Return ONLY the Mermaid code, no explanation, no markdown fences."""

            response = self.model.invoke([SystemMessage(content=prompt)])
            mermaid_code = response.content.strip()
            
            # Clean up common issues
            mermaid_code = self._clean_mermaid_code(mermaid_code)
            
            logger.info(f"Generated Mermaid diagram: {len(mermaid_code)} chars")
            return {
                "type": "mermaid",
                "data": mermaid_code,
                "fallback_text": description
            }
            
        except Exception as e:
            logger.error(f"Error generating Mermaid: {str(e)}")
            return {
                "type": "none",
                "data": None,
                "fallback_text": description
            }
    
    def _clean_mermaid_code(self, code: str) -> str:
        """Clean and validate Mermaid code."""
        # Remove markdown code fences if present
        code = re.sub(r'^```mermaid\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _generate_svg(self, description: str, topic: str) -> dict:
        """Generate SVG instructions from description."""
        try:
            prompt = f"""Create instructions for a simple SVG diagram.

            Topic: {topic}
            Description: {description}

            Generate a JSON object with this structure:
            {{
                "shapes": [
                    {{"type": "rect|circle|ellipse", "x": 0-400, "y": 0-300, "width": 0-200, "height": 0-200, "fill": "color", "label": "text", "labelX": 0-400, "labelY": 0-300}},
                    ...
                ],
                "arrows": [
                    {{"from": [x1, y1], "to": [x2, y2], "label": "text"}},
                    ...
                ]
            }}

            Rules:
            1. Keep it simple (max 5 shapes)
            2. Use colors: #4CAF50 (green), #2196F3 (blue), #FF9800 (orange), #9C27B0 (purple)
            3. Canvas size: 450x350
            4. Make labels short
            5. Ensure shapes don't overlap

            Return ONLY valid JSON, no markdown."""

            response = self.model.invoke([SystemMessage(content=prompt)])
            
            # Try to parse as JSON
            import json
            svg_data = json.loads(response.content.strip())
            
            logger.info(f"Generated SVG instructions: {len(svg_data.get('shapes', []))} shapes")
            return {
                "type": "svg",
                "data": svg_data,
                "fallback_text": description
            }
            
        except Exception as e:
            logger.error(f"Error generating SVG: {str(e)}")
            return {
                "type": "none",
                "data": None,
                "fallback_text": description
            }


# Singleton instance
_visual_generator = None

def get_visual_generator() -> VisualGenerator:
    """Get or create the visual generator instance."""
    global _visual_generator
    if _visual_generator is None:
        _visual_generator = VisualGenerator()
    return _visual_generator