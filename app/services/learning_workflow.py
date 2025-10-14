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
from langgraph.errors import GraphInterrupt
from langgraph.types import RetryPolicy, interrupt, Command
from app.core.config import get_settings
from app.core.course_config import get_curriculum
import logging

logger = logging.getLogger(__name__)
settings = get_settings()




class LearningWorkflow:
    """Manages the learning workflow using LangGraph."""

    def __init__(self):
        try:
            logger.info("Initializing LearningWorkflow")

            # Set up rate limiter to prevent rapid consecutive requests
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.RATE_LIMIT_REQUESTS_PER_SECOND,
                check_every_n_seconds=settings.RATE_LIMIT_CHECK_INTERVAL,
                max_bucket_size=settings.RATE_LIMIT_MAX_BURST,
            )

            # Initialize model with rate limiter and API key
            self.model = init_chat_model(
                f"openai:{settings.OPENAI_MODEL}",
                temperature=0.7,
                rate_limiter=rate_limiter,
                api_key=settings.OPENAI_API_KEY
            )

            self.curriculum = get_curriculum(settings.COURSE_TOPIC)
            self.memory = MemorySaver()
            self.graph = self._build_graph()

            logger.info("LearningWorkflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LearningWorkflow: {str(e)}", exc_info=True)
            raise Exception(f"Workflow initialization failed: {str(e)}")

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
        graph_builder.add_node("router", self.router_node)
        graph_builder.add_node("introduction", self.introduction_node, retry=retry_policy)
        graph_builder.add_node("teaching", self.teaching_node, retry=retry_policy)
        graph_builder.add_node("assessment", self.assessment_node, retry=retry_policy)
        graph_builder.add_node("question_answering", self.question_answering_node, retry=retry_policy)

        # Start with router to decide which node to execute
        graph_builder.add_edge(START, "router")

        # Router decides which node to go to
        graph_builder.add_conditional_edges(
            "router",
            self.route_to_node,
            {
                "introduction": "introduction",
                "teaching": "teaching",
                "assessment": "assessment",
                "question_answering": "question_answering",
                "end": END
            }
        )

        # All nodes return to END after processing
        graph_builder.add_edge("introduction", END)
        graph_builder.add_edge("teaching", END)
        graph_builder.add_edge("assessment", END)
        graph_builder.add_edge("question_answering", END)

        # Compile with memory checkpointer for conversation persistence
        return graph_builder.compile(checkpointer=self.memory)

    # ========== WORKFLOW NODES ==========

    def router_node(self, state: LearningState) -> Command:
        """
        Router node - routes flow based on current state.
        """
        if not state.get("user_name"):
            return Command(goto="introduction")  # will trigger your introduction_node
        elif not state.get("learning_goal"):
            return Command(goto="introduction")  # same node handles both
        elif not state.get("slides"):
            return Command(goto="introduction")  # if you have curriculum generation
        else:
            return Command(goto="teaching")  # continue normal flow

    def introduction_node(self, state: LearningState) -> LearningState:
        """
        Introduction node - Meemo introduces itself, then collects user info.

        Flow:
        1. Meemo introduces itself (automatic greeting)
        2. Ask for user's name via interrupt
        3. Ask for learning goal via interrupt
        4. Proceed with personalized welcome

        Raises:
            Exception: If introduction generation fails
        """
        try:
            logger.info("Processing introduction node")

            # Step 1: Meemo's introduction (first time only)
            if not state["messages"]:
                logger.info("Meemo introducing itself")

                meemo_intro = f"""Hi there! 👋 I'm Meemo, your friendly AI learning companion!
                    I'm here to guide you through an exciting journey into {settings.COURSE_TOPIC}.
                    I'll be using visual slides, interactive examples, and assessments to help you master this topic. You can ask me questions anytime - 
                    I'm here to make learning fun and effective!
                    Before we start, I'd love to get to know you better..."""

                intro_message = AIMessage(content=meemo_intro)
                state["messages"].append(intro_message)

                # Create welcome slide
                welcome_slide = {
                    "slide_number": 0,
                    "title": "Meet Meemo!",
                    "content": meemo_intro,
                    "visual_description": "Friendly robot character waving hello",
                    "full_content": meemo_intro,
                    "topic": "Introduction"
                }
                state["slides"].append(welcome_slide)
                state["current_slide_index"] = 0

                logger.info("Meemo's introduction added to state")

            if not state.get("user_name"):
                user_name = interrupt("What's your name? 😊")

                # When resumed, validate the input
                if not isinstance(user_name, str) or not user_name.strip():
                    # Instead of looping, raise a new interrupt asking again
                    return interrupt("Please provide a valid name")

                state["user_name"] = user_name.strip()
                logger.info(f"User name received: {state['user_name']}")

                # Ask for optional learning goal
                learning_goal_prompt = "What brings you here today? What's your goal for learning about cells? (You can skip this if you'd like)"
                learning_goal = interrupt(learning_goal_prompt)

                # Store learning goal (can be empty)
                if isinstance(learning_goal, str) and learning_goal.strip():
                    state["learning_goal"] = learning_goal.strip()
                else:
                    state["learning_goal"] = None

            # Step 3: Personalized welcome after collecting info
            user_name = state.get("user_name", "Student")
            learning_goal = state.get("learning_goal")

            system_prompt = f"""You are Meemo, a friendly and enthusiastic AI learning companion. You've just met {user_name}.

Give {user_name} a warm, personalized welcome and provide:
1. Express excitement to work with them by name
2. A brief overview of the {settings.COURSE_TOPIC} course structure
3. Mention that you'll start with: {self.curriculum[0] if self.curriculum else "the basics"}
4. {"Acknowledge their goal: " + learning_goal if learning_goal else ""}

Keep it warm, encouraging, and conversational. You're Meemo - be friendly and a bit geeky!

Course topics we'll cover: {', '.join(self.curriculum[:5])}"""

            response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"(Meemo should welcome {user_name} and get ready to start teaching)")
            ])

            # Generate personalized welcome slide
            slide = self._generate_slide(
                response.content,
                f"Welcome, {user_name}!",
                "Personalized course welcome",
                slide_number=1
            )

            # Update state
            state["messages"].append(response)
            state["current_stage"] = "teaching"
            state["current_topic"] = self.curriculum[0] if self.curriculum else "Introduction"
            state["topics_remaining"] = self.curriculum[1:] if len(self.curriculum) > 1 else []
            state["topics_covered"] = []
            state["slides"].append(slide)
            state["current_slide_index"] = 1

            logger.info("Introduction node completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in introduction_node: {str(e)}", exc_info=True)
            # Return state with error message
            state["messages"].append(AIMessage(content=f"I apologize, but I encountered an error during introduction. Please try again."))
            raise Exception(f"Introduction node failed: {str(e)}")

    def teaching_node(self, state: LearningState) -> LearningState:
        """
        Teaching node - teaches the current topic with visual descriptions.

        Raises:
            Exception: If teaching content generation fails
        """
        try:
            current_topic = state.get("current_topic", "Biology")
            logger.info(f"Processing teaching node for topic: {current_topic}")

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

            # Mark topic as covered
            if current_topic and current_topic not in state.get("topics_covered", []):
                state["topics_covered"].append(current_topic)
                logger.info(f"Marked topic as covered: {current_topic}")

            # Move to assessment stage
            state["current_stage"] = "assessment"

            logger.info(f"Teaching node completed successfully for topic: {current_topic}")
            return state

        except Exception as e:
            logger.error(f"Error in teaching_node: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(content=f"I apologize, but I encountered an error while teaching this topic. Please try again."))
            raise Exception(f"Teaching node failed: {str(e)}")

    def assessment_node(self, state: LearningState) -> LearningState:
        """
        Assessment node - checks understanding with a question.

        Raises:
            Exception: If assessment generation fails
        """
        try:
            current_topic = state.get("current_topic", "Biology")
            logger.info(f"Processing assessment node for topic: {current_topic}")

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

            logger.info(f"Assessment node completed successfully for topic: {current_topic}")
            return state

        except Exception as e:
            logger.error(f"Error in assessment_node: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(content="Let's continue with the next topic."))
            raise Exception(f"Assessment node failed: {str(e)}")

    def question_answering_node(self, state: LearningState) -> LearningState:
        """
        Question answering node - handles student questions.

        Raises:
            Exception: If question answering fails
        """
        try:
            logger.info("Processing question answering node")

            system_prompt = f"""You are a helpful biology teacher. A student has a question about {settings.COURSE_TOPIC}.

                Answer their question:
                1. Clearly and accurately
                2. With relevant examples
                3. Relate it back to what they've learned
                4. Encourage further questions

                Be patient and thorough."""

            user_question = state["messages"][-1].content if state["messages"] else "Can you help me understand this better?"

            response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ])

            # Questions don't create new slides, answer stays with current topic slide
            state["messages"].append(response)
            state["questions_asked"] = state.get("questions_asked", 0) + 1
            # Keep current_slide_index unchanged - no new slide for questions

            logger.info("Question answering node completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in question_answering_node: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(content="I apologize, but I had trouble answering your question. Could you rephrase it?"))
            raise Exception(f"Question answering node failed: {str(e)}")

    # ========== ROUTING FUNCTIONS ==========

    def route_to_node(self, state: LearningState) -> Literal["introduction", "teaching", "assessment", "question_answering", "end"]:
        """
        Main router that decides which node to execute based on current state.
        This prevents infinite loops by ensuring each invocation does exactly one action.
        """
        try:
            current_stage = state.get("current_stage", "introduction")
            messages = state.get("messages", [])
            logger.info(f"Routing decision - current_stage: {current_stage}, message_count: {len(messages)}")

            # Check if user just asked a question (last message from user contains ?)
            if messages and len(messages) >= 2:
                # Check if last message is from user (HumanMessage) and contains a question
                last_msg = messages[-1]
                if isinstance(last_msg, HumanMessage) and "?" in last_msg.content:
                    logger.info("User asked a question - routing to question_answering")
                    return "question_answering"

            # Route based on current stage
            if current_stage == "introduction":
                # First time or re-introduction
                if not messages or len(messages) == 0:
                    logger.info("No messages yet - routing to introduction")
                    return "introduction"
                elif len(messages) == 1:
                    # Just introduced, now teach first topic
                    logger.info("Introduction complete - routing to teaching")
                    state["current_stage"] = "teaching"
                    return "teaching"
                else:
                    # Already past introduction
                    logger.info("Past introduction - routing to teaching")
                    state["current_stage"] = "teaching"
                    return "teaching"

            elif current_stage == "teaching":
                # Check if topic was just covered
                current_topic = state.get("current_topic")
                topics_covered = state.get("topics_covered", [])

                if current_topic and current_topic in topics_covered:
                    # Topic already taught, go to assessment
                    logger.info(f"Topic {current_topic} already covered - routing to assessment")
                    state["current_stage"] = "assessment"
                    return "assessment"
                else:
                    # Teach the topic
                    logger.info(f"Teaching topic: {current_topic}")
                    return "teaching"

            elif current_stage == "assessment":
                # Check if we need to move to next topic
                topics_remaining = state.get("topics_remaining", [])

                if topics_remaining:
                    # Move to next topic
                    next_topic = topics_remaining.pop(0)
                    state["current_topic"] = next_topic
                    state["current_stage"] = "teaching"
                    logger.info(f"Assessment done - moving to next topic: {next_topic}")
                    return "teaching"
                else:
                    # Course complete
                    logger.info("Course completed - no more topics")
                    return "end"

            elif current_stage == "question_answering":
                # After answering question, go back to teaching
                logger.info("Question answered - routing back to teaching")
                state["current_stage"] = "teaching"
                return "teaching"

            # Default fallback
            logger.warning(f"Unknown stage: {current_stage} - routing to teaching")
            return "teaching"

        except Exception as e:
            logger.error(f"Error in route_to_node: {str(e)}", exc_info=True)
            return "end"

    # ========== HELPER FUNCTIONS ==========

    def _generate_slide(self, content: str, title: str, context: str, slide_number: int) -> dict:
        """
        Generate slide content from AI response.
        """
        try:
            logger.debug(f"Generating slide {slide_number}: {title}")

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
            except Exception as e:
                logger.warning(f"Failed to generate visual description, using fallback: {str(e)}")
                visual_desc = "Illustration showing " + context

            slide_data = {
                "slide_number": slide_number,
                "title": title or "Learning Slide",  # ADD THIS
                "content": content[:300] if content else "Content not available",
                "visual_description": visual_desc,
                "full_content": content or "",  # ADD THIS
                "topic": title or "Topic"  # ADD THIS
            }

            logger.debug(f"Slide {slide_number} generated successfully")
            return slide_data

        except Exception as e:
            logger.error(f"Error generating slide: {str(e)}", exc_info=True)
            return {
                "slide_number": slide_number,
                "title": title or "Learning Slide",
                "content": content[:300] if content else "Content unavailable",
                "visual_description": f"Illustration showing {context}",
                "full_content": content or "",
                "topic": title or "Topic"
            }
        
    def initialize_state(self) -> LearningState:
        """
        Initialize a new learning state.

        Returns:
            New LearningState instance

        Raises:
            Exception: If state initialization fails
        """
        try:
            logger.info("Initializing new learning state")

            state = LearningState(
                messages=[],
                current_stage="introduction",
                topics_covered=[],
                current_topic=None,
                topics_remaining=self.curriculum.copy() if self.curriculum else [],
                understanding_level="beginner",
                questions_asked=0,
                assessments_passed=0,
                slides=[],  # Empty list to collect all slides
                current_slide_index=0,  # Start at slide 0
                user_name=None,  # Will be collected by introduction node
                learning_goal=None
            )
            print("completed")
            logger.info("Learning state initialized successfully")
            return state

        except Exception as e:
            logger.error(f"Error initializing learning state: {str(e)}", exc_info=True)
            raise Exception(f"Failed to initialize learning state: {str(e)}")
