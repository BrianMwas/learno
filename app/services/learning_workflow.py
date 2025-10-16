"""
Simplified LangGraph workflow - let the LLM handle conversation intelligence.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.learning_state import LearningState
from app.core.config import get_settings
from app.core.course_config import get_curriculum
from app.services.visual_generator import get_visual_generator
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class LearningWorkflow:
    """Simplified workflow - LLM guides the conversation naturally."""

    def __init__(self):
        self.model = init_chat_model(
            f"openai:{settings.OPENAI_MODEL}",
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        self.curriculum = get_curriculum(settings.COURSE_TOPIC)
        self.memory = MemorySaver()
        self.visual_generator = get_visual_generator()
        self.graph = self._build_graph()

    def initialize_state(self) -> LearningState:
        """Initialize minimal state - LLM will handle the rest."""
        return LearningState(
            messages=[],
            current_stage="introduction",
            topics_remaining=self.curriculum.copy(),
            topics_covered=[],
            slides=[],
            current_slide_index=0,
        )

    def _build_graph(self) -> StateGraph:
        """Simple graph with just main teaching node."""
        graph_builder = StateGraph(LearningState)
        
        # Single main node that handles everything
        graph_builder.add_node("teach", self.teaching_node)
        graph_builder.add_node("end", self.ending_node)
        
        # Simple routing
        graph_builder.add_edge(START, "teach")
        graph_builder.add_conditional_edges(
            "teach",
            self._should_continue,
            {"continue": "teach", "end": "end"}
        )
        graph_builder.add_edge("end", END)
        
        return graph_builder.compile(checkpointer=self.memory)

    def teaching_node(self, state: LearningState) -> LearningState:
        """Main teaching node - LLM handles all conversation intelligence."""
        
        # Build system prompt with curriculum context
        topics_covered = state.get("topics_covered", [])
        topics_remaining = state.get("topics_remaining", [])
        current_topic = topics_remaining[0] if topics_remaining else None
        
        system_prompt = self._build_system_prompt(
            topics_covered=topics_covered,
            topics_remaining=topics_remaining,
            current_topic=current_topic
        )
        
        # Let the LLM decide what to do based on conversation history
        messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = self.model.invoke(messages)
        
        state["messages"].append(response)
        
        # Generate slide for this response
        slide = self._generate_slide(
            content=response.content,
            title=self._extract_title(response.content),
            context=current_topic or "learning",
            slide_number=len(state["slides"])
        )
        state["slides"].append(slide)
        state["current_slide_index"] = len(state["slides"]) - 1
        
        # Simple topic progression detection
        if self._detect_topic_completion(response.content, state):
            self._advance_topic(state)
        
        return state

    def ending_node(self, state: LearningState) -> LearningState:
        """Final node to wrap up conversation."""
        completion_message = (
            "🎉 Congratulations on completing the course! "
            "You’ve learned all the topics in this curriculum.\n\n"
            "Would you like me to summarize everything or give you a short quiz? 😊"
        )
        
        state["messages"].append(AIMessage(content=completion_message))
        
        # Optionally, generate a final slide
        final_slide = {
            "slide_number": len(state["slides"]),
            "title": "Course Completed 🎓",
            "content": completion_message,
            "topic": "Completion",
            "visual_type": "celebration",
            "visual_data": None,
            "fallback_text": "Congratulations on completing the course!"
        }
        state["slides"].append(final_slide)
        state["current_slide_index"] = len(state["slides"]) - 1
        
        logger.info("🏁 Course completed.")
        return state

    def _build_system_prompt(self, topics_covered, topics_remaining, current_topic):
        """Build intelligent system prompt with curriculum awareness."""
        return f"""You are Meemo, an AI learning companion teaching {settings.COURSE_TOPIC}.

**Your Teaching Philosophy:**
- Guide students step-by-step through the curriculum
- Use conversational, warm tone with markdown and emojis
- Teach → Check understanding → Move forward naturally

**Curriculum Progress:**
- Completed: {', '.join(topics_covered) if topics_covered else 'None yet'}
- Current Topic: {current_topic or 'Introduction'}
- Upcoming: {', '.join(topics_remaining[1:3]) if len(topics_remaining) > 1 else 'Course nearing completion'}

**Conversation Flow (handle naturally):**
1. **Introduction** (if first message):
   - Greet warmly
   - Ask their name
   - Ask what they want to learn (or say "skip")
   - Start teaching first topic

2. **Teaching** (explain current topic):
   - Clear explanation with examples
   - Visual descriptions for diagrams
   - 3-4 paragraphs with markdown formatting

3. **Check Understanding**:
   - Ask ONE clear question about what you just taught
   - Wait for answer
   - Give warm feedback on their response
   - If they understand: move to next topic
   - If unclear: review and try again

4. **Answer Questions**:
   - If they ask something, answer helpfully
   - Then continue teaching flow

**Important Rules:**
- Remember everything from conversation history (names, goals, answers)
- Don't ask for information you already have
- Move forward naturally when they understand
- Celebrate progress and encourage learning
- Signal topic completion by saying "Great! Let's move to [next topic]"

**Current Focus:** {current_topic or 'Getting started'}
"""

    def _should_continue(self, state: LearningState) -> str:
        """Determine if learning should continue."""
        topics_remaining = state.get("topics_remaining", [])
        
        # Check if course is complete
        if not topics_remaining:
            return "end"
        
        # Otherwise continue teaching
        return "continue"

    def _detect_topic_completion(self, response_content: str, state: LearningState) -> bool:
        """Detect if LLM signaled topic completion."""
        # Simple detection: LLM mentions moving to next topic
        completion_phrases = [
            "let's move to",
            "let's explore",
            "next topic",
            "now let's learn",
            "moving on to"
        ]
        
        content_lower = response_content.lower()
        return any(phrase in content_lower for phrase in completion_phrases)

    def _advance_topic(self, state: LearningState):
        """Move to next topic in curriculum."""
        topics_remaining = state.get("topics_remaining", [])
        
        if topics_remaining:
            completed = topics_remaining[0]
            state["topics_covered"].append(completed)
            state["topics_remaining"] = topics_remaining[1:]
            
            logger.info(f"✅ Completed: {completed}")
            if state["topics_remaining"]:
                logger.info(f"➡️  Next: {state['topics_remaining'][0]}")

    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        import re
        match = re.search(r'^#+ (.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Learning Content"

    def _generate_slide(self, content: str, title: str, context: str, slide_number: int) -> dict:
        """Generate slide with visual."""
        try:
            visual_result = self.visual_generator.generate_visual(
                description=f"Educational diagram about {context}",
                topic=title,
                content=content
            )
            
            return {
                "slide_number": slide_number,
                "title": title,
                "content": content[:300],
                "full_content": content,
                "topic": context,
                "visual_type": visual_result.get("type", "none"),
                "visual_data": visual_result.get("data"),
                "fallback_text": visual_result.get("fallback_text", f"About {context}")
            }
        except Exception as e:
            logger.error(f"Slide generation error: {e}")
            return {
                "slide_number": slide_number,
                "title": title,
                "content": content[:300],
                "full_content": content,
                "topic": context,
                "visual_type": "none",
                "visual_data": None,
                "fallback_text": f"About {context}"
            }


