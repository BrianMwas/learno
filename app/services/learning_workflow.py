"""
LangGraph workflow for managing the learning progression.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.learning_state import LearningState
from langgraph.types import RetryPolicy, interrupt
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
        graph_builder.add_node("introduction", self.introduction_node, retry=retry_policy)
        graph_builder.add_node("teaching", self.teaching_node, retry=retry_policy)
        graph_builder.add_node("assessment", self.assessment_node, retry=retry_policy)
        graph_builder.add_node("evaluate_answer", self.evaluate_answer_node, retry=retry_policy)
        graph_builder.add_node("question_answering", self.question_answering_node, retry=retry_policy)

        # Entry point: always check where to start
        def entry_router(state: LearningState) -> str:
            """Determine starting point based on state."""
            logger.info(f"Entry router - user_name: {state.get('user_name')}, stage: {state.get('current_stage')}")

            # If no user info, start with introduction
            if not state.get("user_name"):
                logger.info("No user_name - starting with introduction")
                return "introduction"

            # If we have partial info, continue introduction
            if state.get("learning_goal") is None and state.get("current_stage") == "introduction":
                logger.info("Partial user info - continuing introduction")
                return "introduction"

            # Check if we're waiting for an assessment answer
            stage = state.get("current_stage", "introduction")
            if stage == "assessment" and state.get("current_assessment_question"):
                logger.info("Resuming with user's assessment answer - routing to evaluate_answer")
                return "evaluate_answer"

            logger.info(f"Using current_stage: {stage}")

            # Map stages to node names
            stage_map = {
                "introduction": "introduction",
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluation_complete": "teaching",  # After eval, continue teaching
                "needs_hint": "evaluate_answer",
                "needs_retry": "evaluate_answer",
                "needs_review": "teaching",
                "question_answering": "question_answering"
            }

            return stage_map.get(stage, "teaching")

        # Start routes to the appropriate node
        graph_builder.add_conditional_edges(
            START,
            entry_router,
            {
                "introduction": "introduction",
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluate_answer": "evaluate_answer",
                "question_answering": "question_answering"
            }
        )

        # After each node, decide what to do next
        def post_node_router(state: LearningState) -> str:
            """Route after a node completes."""
            current_stage = state.get("current_stage", "teaching")
            messages = state.get("messages", [])

            logger.info(f"Post-node router - stage: {current_stage}, messages: {len(messages)}")

            # Check for questions (but not during assessment)
            if current_stage not in ["assessment", "needs_retry", "needs_hint", "needs_review"]:
                if messages and len(messages) >= 2:
                    last_msg = messages[-1]
                    if isinstance(last_msg, HumanMessage) and "?" in last_msg.content:
                        logger.info("Question detected - routing to question_answering")
                        return "question_answering"

            # Route based on stage
            if current_stage == "introduction":
                # After introduction, start teaching
                logger.info("Introduction done - starting teaching")
                return "teaching"

            elif current_stage == "teaching":
                # After teaching, do assessment
                logger.info("Teaching done - moving to assessment")
                return "assessment"

            elif current_stage == "assessment":
                # After assessment question is asked, end this cycle
                # The next user input will resume and evaluate their answer
                logger.info("Assessment question asked - ending to wait for user answer")
                return "end"

            elif current_stage == "evaluation_complete":
                # Assessment passed, move to next topic or end
                topics_remaining = state.get("topics_remaining", [])
                if topics_remaining:
                    # Move to next topic
                    next_topic = topics_remaining[0]
                    state["current_topic"] = next_topic
                    state["topics_remaining"] = topics_remaining[1:]
                    state["current_stage"] = "teaching"
                    logger.info(f"Assessment passed - moving to topic: {next_topic}")
                    return "teaching"
                else:
                    logger.info("No more topics - course complete")
                    return "end"

            elif current_stage == "needs_hint":
                # Give a hint and wait for another attempt
                logger.info("Providing hint - re-evaluating on next answer")
                return "evaluate_answer"

            elif current_stage == "needs_retry":
                # Ask to try again - evaluate next answer
                logger.info("Asking to retry - re-evaluating on next answer")
                return "evaluate_answer"

            elif current_stage == "needs_review":
                # Review the topic again
                logger.info("Review needed - going back to teaching")
                state["current_stage"] = "teaching"
                return "teaching"

            elif current_stage == "question_answering":
                # After answering, continue with current flow
                logger.info("Question answered - continuing teaching")
                return "teaching"

            # Default
            return "end"

        # All nodes route through the post-node router
        graph_builder.add_conditional_edges(
            "introduction",
            post_node_router,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "teaching",
            post_node_router,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "assessment",
            post_node_router,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluate_answer": "evaluate_answer",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "evaluate_answer",
            post_node_router,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluate_answer": "evaluate_answer",
                "question_answering": "question_answering",
                "end": END
            }
        )

        graph_builder.add_conditional_edges(
            "question_answering",
            post_node_router,
            {
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluate_answer": "evaluate_answer",
                "question_answering": "question_answering",
                "end": END
            }
        )

        # Compile with memory checkpointer for conversation persistence
        return graph_builder.compile(checkpointer=self.memory)

    # ========== WORKFLOW NODES ==========

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
            logger.info("Processing introduction node")

            # Step 1: Meemo's introduction (first time only)
            # Check if we've already introduced Meemo (slides will be empty on first run)
            if len(state.get("slides", [])) == 0:
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

            # Step 2: Collect user name (interrupt will pause execution)
            if not state.get("user_name"):
                logger.info("Requesting user name via interrupt")
                # On first call: interrupt() pauses and returns None
                # On resume: interrupt() returns the value from Command(resume=value)
                user_name = interrupt("What's your name? 😊")
                state["user_name"] = user_name.strip() if user_name else None
                logger.info(f"User name received: {state['user_name']}")

            # Step 3: Ask for learning goal (optional)
            if not state.get("learning_goal"):
                logger.info("Requesting learning goal via interrupt")
                learning_goal_prompt = f"What brings you here today? What's your goal for learning about {settings.COURSE_TOPIC}? (You can skip this if you'd like)"
                # On first call: interrupt() pauses and returns None
                # On resume: interrupt() returns the value from Command(resume=value)
                learning_goal = interrupt(learning_goal_prompt)
                state["learning_goal"] = learning_goal.strip() if learning_goal and learning_goal.strip() else None
                logger.info(f"Learning goal set: {state['learning_goal']}")

            # Step 4: Personalized welcome after collecting info
            try:
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
                logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
                # Return state with error message
                state["messages"].append(AIMessage(content=f"I apologize, but I encountered an error during introduction. Please try again."))
                raise Exception(f"Introduction node failed during welcome generation: {str(e)}")
    
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
        Assessment node - generates a question to check understanding.

        Raises:
            Exception: If assessment generation fails
        """
        try:
            current_topic = state.get("current_topic", "Biology")
            logger.info(f"Processing assessment node for topic: {current_topic}")

            # Check if we already have an assessment question for this topic
            if state.get("current_assessment_question"):
                logger.info("Assessment question already exists, waiting for answer")
                return state

            system_prompt = f"""You just taught about {current_topic}.

Ask the student ONE clear question to check their understanding of the key concepts.
Make it:
- Specific to what was just taught
- Not too difficult (appropriate for a beginner)
- Open-ended enough to assess understanding
- Engaging and friendly

Just ask the question - don't provide answers or hints yet."""

            response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please check my understanding.")
            ])

            # Store the assessment question
            state["current_assessment_question"] = response.content
            state["assessment_attempts"] = 0
            state["messages"].append(response)
            state["current_stage"] = "assessment"

            logger.info(f"Assessment question generated for topic: {current_topic}")
            return state

        except Exception as e:
            logger.error(f"Error in assessment_node: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(content="Let's continue with the next topic."))
            raise Exception(f"Assessment node failed: {str(e)}")

    def evaluate_answer_node(self, state: LearningState) -> LearningState:
        """
        Evaluate student's answer to assessment question.

        Analyzes the answer for understanding, provides feedback,
        and updates understanding metrics.

        Raises:
            Exception: If evaluation fails
        """
        try:
            current_topic = state.get("current_topic", "Biology")
            assessment_question = state.get("current_assessment_question")

            logger.info(f"Evaluating answer for topic: {current_topic}")

            # Get the student's answer (last message)
            student_answer = state["messages"][-1].content if state["messages"] else ""

            # Increment attempts
            state["assessment_attempts"] = state.get("assessment_attempts", 0) + 1
            attempts = state["assessment_attempts"]

            system_prompt = f"""You are evaluating a student's understanding of {current_topic}.

The assessment question was: {assessment_question}

The student's answer is: {student_answer}

Evaluate their answer and provide:
1. A judgment: "correct", "partial", or "incorrect"
2. Specific feedback on what they got right or wrong
3. Encouragement and next steps

Be supportive and constructive. If they're close but not quite right, acknowledge what they understand.

Respond in this format:
JUDGMENT: [correct/partial/incorrect]
FEEDBACK: [Your detailed feedback here]"""

            evaluation_response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please evaluate this answer.")
            ])

            # Parse the evaluation
            evaluation_text = evaluation_response.content
            logger.info(f"Evaluation response: {evaluation_text[:200]}")

            # Determine the judgment
            if "JUDGMENT: correct" in evaluation_text.lower():
                judgment = "correct"
            elif "JUDGMENT: partial" in evaluation_text.lower():
                judgment = "partial"
            else:
                judgment = "incorrect"

            logger.info(f"Judgment: {judgment}, Attempts: {attempts}")

            # Update state based on judgment
            if judgment == "correct":
                state["assessments_passed"] = state.get("assessments_passed", 0) + 1
                state["current_assessment_question"] = None  # Clear for next topic
                state["assessment_attempts"] = 0

                # Update understanding level if doing well
                assessments_passed = state["assessments_passed"]
                if assessments_passed >= 3 and state["understanding_level"] == "beginner":
                    state["understanding_level"] = "intermediate"
                    logger.info("Student promoted to intermediate level")
                elif assessments_passed >= 6 and state["understanding_level"] == "intermediate":
                    state["understanding_level"] = "advanced"
                    logger.info("Student promoted to advanced level")

                # Move to next stage
                state["current_stage"] = "evaluation_complete"

            elif judgment == "partial" and attempts < 2:
                # Give them another chance with a hint
                state["current_stage"] = "needs_hint"

            elif attempts >= 2 or judgment == "incorrect":
                # After 2 attempts or if completely incorrect, review the topic
                state["current_stage"] = "needs_review"
            else:
                # First incorrect attempt - give another try
                state["current_stage"] = "needs_retry"

            # Add the evaluation feedback to messages
            state["messages"].append(AIMessage(content=evaluation_text))

            logger.info(f"Evaluation complete - new stage: {state['current_stage']}")
            return state

        except Exception as e:
            logger.error(f"Error in evaluate_answer_node: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(content="I had trouble evaluating your answer. Let's move on."))
            state["current_stage"] = "evaluation_complete"
            raise Exception(f"Evaluation node failed: {str(e)}")

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
                current_assessment_question=None,
                assessment_attempts=0,
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
