"""
LangGraph workflow for managing the learning progression.
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.learning_state import LearningState
from app.models.schemas import SlideContent
from langgraph.types import RetryPolicy
from app.core.config import get_settings
from app.core.course_config import get_curriculum

settings = get_settings()




class LearningWorkflow:
    """Manages the learning workflow using LangGraph."""

    def __init__(self):
        # Set up rate limiter to prevent rapid consecutive requests
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.RATE_LIMIT_REQUESTS_PER_SECOND,
            check_every_n_seconds=settings.RATE_LIMIT_CHECK_INTERVAL,
            max_bucket_size=settings.RATE_LIMIT_MAX_BURST,
        )

        # Initialize model with rate limiter
        self.model = init_chat_model(
            f"openai:{settings.OPENAI_MODEL}",
            temperature=0.7,
            rate_limiter=rate_limiter
        )

        self.curriculum = get_curriculum(settings.COURSE_TOPIC)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the learning workflow graph."""
        graph_builder = StateGraph(LearningState)

        # Create retry policy for resilience against transient failures
        retry_policy = RetryPolicy(
            max_attempts=settings.RETRY_MAX_ATTEMPTS,
            backoff_factor=settings.RETRY_BACKOFF_FACTOR,
            initial_interval=settings.RETRY_INITIAL_INTERVAL,
            max_interval=settings.RETRY_MAX_INTERVAL,
        )

        # Add nodes for each stage with retry policy
        graph_builder.add_node("introduction", self.introduction_node, retry=retry_policy)
        graph_builder.add_node("teaching", self.teaching_node, retry=retry_policy)
        graph_builder.add_node("assessment", self.assessment_node, retry=retry_policy)
        graph_builder.add_node("question_answering", self.question_answering_node, retry=retry_policy)

        # Define the flow
        graph_builder.add_edge(START, "introduction")

        # Conditional routing based on current stage
        graph_builder.add_conditional_edges(
            "introduction",
            self.route_from_introduction,
            {
                "teaching": "teaching",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "teaching",
            self.route_from_teaching,
            {
                "assessment": "assessment",
                "teaching": "teaching",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "assessment",
            self.route_from_assessment,
            {
                "teaching": "teaching",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "question_answering",
            self.route_from_question,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "question_answering": "question_answering",
                "end": END
            }
        )

        # Compile with memory checkpointer for conversation persistence
        return graph_builder.compile(checkpointer=self.memory)

    # ========== WORKFLOW NODES ==========

    def introduction_node(self, state: LearningState) -> LearningState:
        """
        Introduction node - welcomes student and gives course overview.
        """
        system_prompt = f"""You are an expert biology teacher introducing a course on {settings.COURSE_TOPIC}.

        Warmly welcome the student and provide:
        1. A brief overview of what they'll learn
        2. The structure of the course (we'll go from outer to inner: membrane → organelles)
        3. Encourage them to ask questions anytime

        Keep it engaging and motivating. Mention that each topic will have visual slides to help them learn.

        Course topics: {', '.join(self.curriculum[:5])} and more..."""

        # Get the user's first message or create a default one
        user_msg = "Hello! I'm ready to learn about cells."
        if state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                user_msg = last_msg.content

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])

        # Generate slide for introduction
        slide = self._generate_slide(
            response.content,
            "Welcome to Cell Biology",
            "Course introduction and overview",
            slide_number=0
        )

        # Update state - add slide to collection
        state["messages"].append(response)
        state["current_stage"] = "teaching"
        state["current_topic"] = self.curriculum[0]
        state["topics_remaining"] = self.curriculum[1:]
        state["topics_covered"] = []
        state["slides"].append(slide)
        state["current_slide_index"] = 0

        return state

    def teaching_node(self, state: LearningState) -> LearningState:
        """
        Teaching node - teaches the current topic with visual descriptions.
        """
        current_topic = state["current_topic"]

        system_prompt = f"""You are teaching {current_topic} in a Cell Biology course.

Explain this topic clearly with:
1. Clear definition and function
2. Key characteristics
3. Visual description (describe what it looks like, its structure, colors to use in diagrams)
4. How it relates to the cell's overall function

Keep it concise (3-4 paragraphs). Use analogies when helpful.
Remember: this is for visual slides, so be descriptive about the structure and appearance."""

        # Get user message
        user_msg = state["messages"][-1].content if state["messages"] else f"Teach me about {current_topic}"

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])

        # Generate slide - each topic gets its own slide
        slide_number = len(state["slides"])
        slide = self._generate_slide(
            response.content,
            current_topic,
            f"Learning about {current_topic}",
            slide_number=slide_number
        )

        # Update state - append new slide
        state["messages"].append(response)
        state["slides"].append(slide)
        state["current_slide_index"] = slide_number

        return state

    def assessment_node(self, state: LearningState) -> LearningState:
        """
        Assessment node - checks understanding with a question.
        """
        current_topic = state["current_topic"]

        system_prompt = f"""You just taught about {current_topic}.

Ask the student ONE simple question to check their understanding.
Make it engaging and not too difficult. After they answer, provide:
1. Encouragement
2. Correct information if needed
3. Move to next topic if they understood

Keep responses brief and supportive."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please check my understanding.")
        ])

        # Assessment doesn't create a new slide, just uses current slide
        state["messages"].append(response)
        state["current_stage"] = "assessment"
        # Keep current_slide_index pointing to the topic slide

        return state

    def question_answering_node(self, state: LearningState) -> LearningState:
        """
        Question answering node - handles student questions.
        """
        system_prompt = f"""You are a helpful biology teacher. A student has a question about {settings.COURSE_TOPIC}.

Answer their question:
1. Clearly and accurately
2. With relevant examples
3. Relate it back to what they've learned
4. Encourage further questions

Be patient and thorough."""

        user_question = state["messages"][-1].content

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ])

        # Questions don't create new slides, answer stays with current topic slide
        state["messages"].append(response)
        state["questions_asked"] = state.get("questions_asked", 0) + 1
        # Keep current_slide_index unchanged - no new slide for questions

        return state

    # ========== ROUTING FUNCTIONS ==========

    def route_from_introduction(self, state: LearningState) -> Literal["teaching", "question_answering", "end"]:
        """Route from introduction based on user response."""
        if not state["messages"]:
            return "teaching"

        last_message = state["messages"][-1].content.lower()

        # Check if user has a question
        if any(q in last_message for q in ["?", "what", "why", "how", "question"]):
            return "question_answering"

        # Otherwise start teaching
        return "teaching"

    def route_from_teaching(self, state: LearningState) -> Literal["assessment", "teaching", "question_answering", "end"]:
        """Route from teaching based on context."""
        if not state["messages"]:
            return "teaching"

        last_message = state["messages"][-1].content.lower()

        # If user asks a question
        if "?" in last_message:
            state["current_stage"] = "question_answering"
            return "question_answering"

        # If topic is taught, go to assessment
        if state["current_topic"] in state.get("topics_covered", []):
            return "assessment"

        # Move topic to covered and check if more topics
        state["topics_covered"].append(state["current_topic"])

        if state["topics_remaining"]:
            # More topics to cover
            return "teaching"
        else:
            # Course complete
            return "end"

    def route_from_assessment(self, state: LearningState) -> Literal["teaching", "question_answering", "end"]:
        """
        Route from assessment based on understanding.
        This moves to the NEXT SLIDE (next topic).
        """
        last_message = state["messages"][-1].content.lower() if state["messages"] else ""

        # If user asks a question during assessment
        if "?" in last_message:
            return "question_answering"

        # Move to next topic and next slide
        if state["topics_remaining"]:
            state["current_topic"] = state["topics_remaining"].pop(0)
            state["current_stage"] = "teaching"
            # Next slide will be created in teaching_node
            return "teaching"

        # Course completed
        return "end"

    def route_from_question(self, state: LearningState) -> Literal["teaching", "assessment", "question_answering", "end"]:
        """
        Route from question answering.
        Questions DON'T move slides - they stay on current slide.
        """
        if not state["messages"]:
            return "teaching"

        last_message = state["messages"][-1].content.lower()

        # If another question, stay in Q&A (same slide)
        if "?" in last_message:
            return "question_answering"

        # Question answered, return to assessment to check if ready for next slide
        return "assessment"

    # ========== HELPER FUNCTIONS ==========

    def _generate_slide(self, content: str, title: str, context: str, slide_number: int) -> dict:
        """Generate slide content from AI response."""
        # Use the model to extract visual description
        slide_prompt = f"""Based on this teaching content, create a visual slide description:

Content: {content}

Extract:
1. Key points (2-3 bullet points)
2. Visual description: Describe the diagram/illustration that should be shown
   - What structures to show
   - What colors to use
   - Labels and annotations
   - Spatial relationships

Format as JSON."""

        try:
            slide_response = self.model.invoke([SystemMessage(content=slide_prompt)])
            visual_desc = slide_response.content
        except Exception:
            visual_desc = "Illustration showing " + context

        return {
            "slide_number": slide_number,
            "title": title,
            "content": content[:300],  # First 300 chars for slide
            "visual_description": visual_desc,
            "full_content": content,
            "topic": title
        }

    def initialize_state(self, user_name: str | None = None) -> LearningState:
        """Initialize a new learning state."""
        return LearningState(
            messages=[],
            current_stage="introduction",
            topics_covered=[],
            current_topic=None,
            topics_remaining=self.curriculum.copy(),
            understanding_level="beginner",
            questions_asked=0,
            assessments_passed=0,
            slides=[],  # Empty list to collect all slides
            current_slide_index=0,  # Start at slide 0
            user_name=user_name,
            learning_goal=None
        )
