"""
LangGraph workflow for managing the learning progression with agentic tools and visual generation.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.models.learning_state import LearningState
from langgraph.types import RetryPolicy
from app.core.config import get_settings
from app.core.course_config import get_curriculum
from app.models.schemas import AssessmentEvaluation, ConversationAnalysis, GoalExtraction, NameExtraction
from app.utils.error_messages import get_stage_error_message
from app.services.visual_generator import get_visual_generator 
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()


class LearningWorkflow:
    """Manages the learning workflow using LangGraph with agentic tools."""

    def __init__(self):
        try:
            logger.info("Initializing LearningWorkflow")

            rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.RATE_LIMIT_REQUESTS_PER_SECOND,
                check_every_n_seconds=settings.RATE_LIMIT_CHECK_INTERVAL,
                max_bucket_size=settings.RATE_LIMIT_MAX_BURST,
            )

            self.model = init_chat_model(
                f"openai:{settings.OPENAI_MODEL}",
                temperature=0.7,
                rate_limiter=rate_limiter,
                api_key=settings.OPENAI_API_KEY
            )

            self.curriculum = get_curriculum(settings.COURSE_TOPIC)
            self.memory = MemorySaver()
            self.visual_generator = get_visual_generator()  # ✨ NEW: Initialize visual generator
            self.graph = self._build_graph()

            logger.info("LearningWorkflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LearningWorkflow: {str(e)}", exc_info=True)
            raise Exception(f"Workflow initialization failed: {str(e)}")

    def _build_graph(self) -> StateGraph:
        """Build the learning workflow graph."""
        graph_builder = StateGraph(LearningState)

        retry_policy = RetryPolicy(
            max_attempts=settings.RETRY_MAX_ATTEMPTS,
            backoff_factor=settings.RETRY_BACKOFF_FACTOR,
            initial_interval=settings.RETRY_INITIAL_INTERVAL,
            max_interval=settings.RETRY_MAX_INTERVAL,
        )

        # Add nodes
        graph_builder.add_node("introduction", self.introduction_node, retry=retry_policy)
        graph_builder.add_node("teaching", self.teaching_node, retry=retry_policy)
        graph_builder.add_node("assessment", self.assessment_node, retry=retry_policy)
        graph_builder.add_node("evaluate_answer", self.evaluate_answer_node, retry=retry_policy)
        graph_builder.add_node("question_answering", self.question_answering_node, retry=retry_policy)

        # Entry routing
        graph_builder.add_conditional_edges(
            START,
            self._entry_router,
            {
                "introduction": "introduction",
                "teaching": "teaching",
                "assessment": "assessment",
                "evaluate_answer": "evaluate_answer",
                "question_answering": "question_answering"
            }
        )

        # Post-node routing
        for node in ["introduction", "teaching", "assessment", "evaluate_answer", "question_answering"]:
            graph_builder.add_conditional_edges(
                node,
                self._post_node_router,
                {
                    "teaching": "teaching",
                    "assessment": "assessment",
                    "evaluate_answer": "evaluate_answer",
                    "question_answering": "question_answering",
                    "end": END
                }
            )

        return graph_builder.compile(checkpointer=self.memory)

    # ========== ROUTING METHODS ==========

    def _entry_router(self, state: LearningState) -> str:
        """Determine starting point based on state."""
        logger.info(f"Entry router - stage: {state.get('current_stage')}")

        if not state.get("user_name"):
            return "introduction"

        if state.get("learning_goal") is None and state.get("current_stage") == "introduction":
            return "introduction"

        stage = state.get("current_stage", "introduction")
        if stage == "assessment" and state.get("current_assessment_question"):
            return "evaluate_answer"

        stage_map = {
            "introduction": "introduction",
            "teaching": "teaching",
            "assessment": "assessment",
            "evaluation_complete": "teaching",
            "needs_hint": "evaluate_answer",
            "needs_retry": "evaluate_answer",
            "needs_review": "teaching",
            "question_answering": "question_answering"
        }

        return stage_map.get(stage, "teaching")

    def _post_node_router(self, state: LearningState) -> str:
        """Route after node execution using agentic analysis."""
        current_stage = state.get("current_stage", "teaching")
        messages = state.get("messages", [])

        logger.info(f"Post-node router - stage: {current_stage}")

        # Analyze last user message for intelligent routing
        if messages and isinstance(messages[-1], HumanMessage) and current_stage not in ["assessment"]:
            try:
                analyzer = self.model.with_structured_output(ConversationAnalysis)
                analysis = analyzer.invoke([
                    SystemMessage(content=f"""Analyze this message in learning context.

Current stage: {current_stage}
Current topic: {state.get('current_topic', 'Unknown')}

Determine the routing."""),
                    HumanMessage(content=f"User message: {messages[-1].content}")
                ])

                if analysis.is_question and current_stage not in ["needs_retry", "needs_hint"]:
                    logger.info("Detected question - routing to Q&A")
                    return "question_answering"
            except Exception as e:
                logger.warning(f"Analysis failed, using stage-based routing: {e}")

        # Stage-based routing
        if current_stage == "introduction":
            if state.get("user_name") and state.get("learning_goal") is not None:
                return "teaching"
            return "end"

        elif current_stage == "teaching":
            return "assessment"

        elif current_stage == "assessment":
            return "end"

        elif current_stage == "evaluation_complete":
            topics_remaining = state.get("topics_remaining", [])
            if topics_remaining:
                state["current_topic"] = topics_remaining[0]
                state["topics_remaining"] = topics_remaining[1:]
                state["current_stage"] = "teaching"
                return "teaching"
            return "end"

        elif current_stage in ["needs_hint", "needs_retry"]:
            return "evaluate_answer"

        elif current_stage == "needs_review":
            state["current_stage"] = "teaching"
            return "teaching"

        elif current_stage == "question_answering":
            return "teaching"

        return "end"

    # ========== WORKFLOW NODES ==========

    def introduction_node(self, state: LearningState) -> LearningState:
        """Introduction with agentic name/goal extraction."""
        logger.info("Processing introduction node with agentic tools")

        messages = state.get("messages", [])
        user_message_count = sum(1 for m in messages if isinstance(m, HumanMessage))

        # STEP 1: Initial greeting
        if user_message_count == 0:
            logger.info("Generating initial greeting")

            intro_response = self.model.invoke([
                SystemMessage(content=f"""You are Meemo, a friendly AI learning companion for {settings.COURSE_TOPIC}.

Introduce yourself warmly (3-4 sentences):
- Greet the user
- Explain what you'll do together
- Ask them to say hello

**Use markdown formatting with emojis.**"""),
                HumanMessage(content="(Generate greeting)")
            ])

            state["messages"].append(intro_response)
            welcome_slide = self._generate_slide(
                intro_response.content,
                "Meet Meemo! 👋",
                "Welcome",
                slide_number=0
            )
            state["slides"].append(welcome_slide)
            state["current_slide_index"] = 0
            state["current_stage"] = "introduction"

            return state

        # STEP 2+: Extract information using agentic tools
        has_name = bool(state.get("user_name"))
        has_goal = state.get("learning_goal") is not None
        last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""

        # Extract name if needed
        if not has_name and last_user_message and user_message_count >= 1:
            try:
                name_extractor = self.model.with_structured_output(NameExtraction)
                extraction = name_extractor.invoke([
                    SystemMessage(content="""Extract name from message. Look for patterns like:
- "I'm [Name]" or "My name is [Name]"
- Just "[Name]" if short message after being asked"""),
                    HumanMessage(content=f"User: {last_user_message}")
                ])

                if extraction.name and extraction.confidence in ["high", "medium"]:
                    state["user_name"] = extraction.name
                    has_name = True
                    logger.info(f"✅ Name extracted: {extraction.name}")
            except Exception as e:
                logger.warning(f"Name extraction failed: {e}")

        # Extract goal if needed
        if has_name and not has_goal and user_message_count >= 3:
            try:
                goal_extractor = self.model.with_structured_output(GoalExtraction)
                extraction = goal_extractor.invoke([
                    SystemMessage(content="""Extract learning goal or detect if user wants to skip.
Look for goals like "I want to learn X" or skip words like "skip", "let's start"."""),
                    HumanMessage(content=f"User: {last_user_message}")
                ])

                if extraction.wants_to_skip:
                    state["learning_goal"] = None
                    has_goal = True
                elif extraction.goal:
                    state["learning_goal"] = extraction.goal
                    has_goal = True
                logger.info(f"✅ Goal: {state.get('learning_goal', 'skipped')}")
            except Exception as e:
                logger.warning(f"Goal extraction failed: {e}")

        # Generate appropriate response
        if not has_name:
            response = self.model.invoke([
                SystemMessage(content="You are Meemo. Ask for their name warmly. **Use markdown.**"),
                *messages[-2:]
            ])
            state["messages"].append(response)

        elif not has_goal:
            user_name = state.get("user_name", "friend")
            response = self.model.invoke([
                SystemMessage(content=f"""You are Meemo. Ask {user_name} about their learning goal for {settings.COURSE_TOPIC}.
Mention they can say 'skip'. **Use markdown with emojis.**"""),
                HumanMessage(content=f"Name: {user_name}")
            ])
            state["messages"].append(response)

        else:
            # Complete introduction
            user_name = state.get("user_name", "Student")
            learning_goal = state.get("learning_goal")

            response = self.model.invoke([
                SystemMessage(content=f"""Generate personalized welcome for {user_name}.
Goal: {learning_goal or 'general learning'}
First topic: {self.curriculum[0] if self.curriculum else 'basics'}
Topics: {', '.join(self.curriculum[:5])}

**Use markdown with headings, lists, emojis.**"""),
                HumanMessage(content="(Generate welcome)")
            ])

            slide = self._generate_slide(
                response.content,
                f"Welcome, {user_name}! 🎉",
                "Course welcome",
                slide_number=1
            )

            state["messages"].append(response)
            state["current_stage"] = "teaching"
            state["current_topic"] = self.curriculum[0] if self.curriculum else "Introduction"
            state["topics_remaining"] = self.curriculum[1:] if len(self.curriculum) > 1 else []
            state["topics_covered"] = []
            state["slides"].append(slide)
            state["current_slide_index"] = 1

        return state

    def teaching_node(self, state: LearningState) -> LearningState:
        """Teach current topic."""
        try:
            current_topic = state.get("current_topic", "Biology")
            logger.info(f"Teaching: {current_topic}")

            response = self.model.invoke([
                SystemMessage(content=f"""Teach {current_topic} in Cell Biology.

Include:
1. Clear definition
2. Key characteristics
3. Visual description for diagrams
4. Relevance to cells

3-4 paragraphs. **Use markdown: headings, bold, lists, emojis.**"""),
                HumanMessage(content=f"Teach {current_topic}")
            ])

            slide_number = len(state["slides"])
            slide = self._generate_slide(
                response.content,
                current_topic,
                f"Learning {current_topic}",
                slide_number=slide_number
            )

            state["messages"].append(response)
            state["slides"].append(slide)
            state["current_slide_index"] = slide_number

            if current_topic not in state.get("topics_covered", []):
                state["topics_covered"].append(current_topic)

            state["current_stage"] = "assessment"
            return state

        except Exception as e:
            logger.error(f"Teaching error: {e}")
            state["messages"].append(AIMessage(content=get_stage_error_message("teaching")))
            raise

    def assessment_node(self, state: LearningState) -> LearningState:
        """Generate assessment question."""
        try:
            current_topic = state.get("current_topic")

            if state.get("current_assessment_question"):
                return state

            response = self.model.invoke([
                SystemMessage(content=f"""Ask ONE clear question about {current_topic}.
Make it beginner-friendly and open-ended.
**Use markdown with bold and emojis.**"""),
                HumanMessage(content="Check understanding")
            ])

            state["current_assessment_question"] = response.content
            state["assessment_attempts"] = 0
            state["messages"].append(response)
            state["current_stage"] = "assessment"

            return state

        except Exception as e:
            logger.error(f"Assessment error: {e}")
            state["messages"].append(AIMessage(content=get_stage_error_message("assessment")))
            raise

    def evaluate_answer_node(self, state: LearningState) -> LearningState:
        """Evaluate answer using structured output."""
        try:
            current_topic = state.get("current_topic")
            question = state.get("current_assessment_question")
            answer = state["messages"][-1].content if state["messages"] else ""

            state["assessment_attempts"] = state.get("assessment_attempts", 0) + 1
            attempts = state["assessment_attempts"]

            # Structured evaluation
            evaluator = self.model.with_structured_output(AssessmentEvaluation)
            evaluation = evaluator.invoke([
                SystemMessage(content=f"""Evaluate understanding of {current_topic}.
Question: {question}
Answer: {answer}
Attempt: {attempts}"""),
                HumanMessage(content="Evaluate answer")
            ])

            # Generate friendly feedback
            feedback = self.model.invoke([
                SystemMessage(content=f"""Generate warm feedback:
Judgment: {evaluation.judgment}
Correct: {evaluation.what_was_correct}
Missing: {evaluation.what_was_missing}

**Use markdown with emojis.**""")
            ])

            state["messages"].append(feedback)

            # Update state based on evaluation
            if evaluation.should_pass:
                state["assessments_passed"] = state.get("assessments_passed", 0) + 1
                state["current_assessment_question"] = None
                state["assessment_attempts"] = 0

                # Level up
                passed = state["assessments_passed"]
                if passed >= 3 and state["understanding_level"] == "beginner":
                    state["understanding_level"] = "intermediate"
                elif passed >= 6 and state["understanding_level"] == "intermediate":
                    state["understanding_level"] = "advanced"

                state["current_stage"] = "evaluation_complete"

            elif evaluation.needs_review or attempts >= 2:
                state["current_stage"] = "needs_review"
                state["assessment_attempts"] = 0
                state["current_assessment_question"] = None

            else:
                state["current_stage"] = "needs_retry"

            return state

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            state["messages"].append(AIMessage(content=get_stage_error_message("evaluation")))
            state["current_stage"] = "evaluation_complete"
            raise

    def question_answering_node(self, state: LearningState) -> LearningState:
        """Answer student questions."""
        try:
            question = state["messages"][-1].content if state["messages"] else ""

            response = self.model.invoke([
                SystemMessage(content=f"""Answer biology question clearly.
Use examples and relate to {settings.COURSE_TOPIC}.
**Use markdown: headings, bold, lists, emojis.**"""),
                HumanMessage(content=question)
            ])

            state["messages"].append(response)
            state["questions_asked"] = state.get("questions_asked", 0) + 1

            return state

        except Exception as e:
            logger.error(f"Q&A error: {e}")
            state["messages"].append(AIMessage(content=get_stage_error_message("question_answering")))
            raise

    # ========== HELPER METHODS ==========

    def _generate_slide(self, content: str, title: str, context: str, slide_number: int) -> dict:
        """Generate slide with visual description and actual visual data."""
        try:
            slide_prompt = f"""Extract key points and visual description:
Content: {content}

Return JSON: {{"key_points": ["point1", "point2"], "visual_description": "diagram description"}}"""

            try:
                slide_response = self.model.invoke([SystemMessage(content=slide_prompt)])
                slide_json = json.loads(slide_response.content.strip())
                visual_desc = slide_json.get("visual_description", f"Illustration of {context}")
                key_points = slide_json.get("key_points", [])
            except:
                visual_desc = f"Illustration showing {context}"
                key_points = []

            # ✨ NEW: Generate actual visual using VisualGenerator
            visual_result = None
            visual_type = "none"
            visual_data = None
            
            try:
                logger.info(f"Generating visual for slide: {title}")
                visual_result = self.visual_generator.generate_visual(
                    description=visual_desc,
                    topic=title,
                    content=content
                )
                visual_type = visual_result.get("type", "none")
                visual_data = visual_result.get("data")
                logger.info(f"Visual generated: type={visual_type}")
            except Exception as ve:
                logger.warning(f"Visual generation failed, using fallback: {ve}")

            return {
                "slide_number": slide_number,
                "title": title or "Learning Slide",
                "content": content[:300] if content else "Content unavailable",
                "visual_description": visual_desc,
                "full_content": content or "",
                "topic": title or "Topic",
                "key_points": key_points,
                # ✨ NEW: Visual data fields
                "visual_type": visual_type,  # "mermaid" | "svg" | "premade" | "none"
                "visual_data": visual_data   # Mermaid code, SVG instructions, or asset name
            }

        except Exception as e:
            logger.error(f"Slide generation error: {e}")
            return {
                "slide_number": slide_number,
                "title": title or "Learning Slide",
                "content": content[:300] if content else "Content unavailable",
                "visual_description": f"Illustration showing {context}",
                "full_content": content or "",
                "topic": title or "Topic",
                "key_points": [],
                "visual_type": "none",
                "visual_data": None
            }

    def initialize_state(self) -> LearningState:
        """Initialize new learning state."""
        try:
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
                slides=[],
                current_slide_index=0,
                user_name=None,
                learning_goal=None
            )
            return state

        except Exception as e:
            logger.error(f"State initialization error: {e}")
            raise