"""
Dynamical systems agent powered by kiff.

Wraps kiff.agents.MemoryAgent with a context-aware signature that knows
about the current system state so the user can ask questions about what
they're looking at.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DynSysAgent:
    """
    A conversational agent that understands the current dynamical system context.

    Wraps kiff.agents.MemoryAgent with a DSPy signature that receives:
      - The user's question
      - Current system context (system type, parameters, equations)

    and returns a helpful, technically accurate response.
    """

    # DSPy signature: context + query -> response
    SIGNATURE = (
        "system_context:str, conversation_history:str, query:str -> response:str"
    )

    def __init__(
        self, base_model: str = "gemini/gemini-2.5-flash", memory_size: int = 8
    ):
        """
        Initialize the agent.

        Parameters
        ----------
        base_model : str
            LLM backend passed to kiff (default: Gemini 2.5 Flash)
        memory_size : int
            Number of turns kept in rolling memory
        """
        try:
            from kiff.agents import MemoryAgent

            self._agent = MemoryAgent(base_model=base_model, memory_size=memory_size)
            self._available = True
            logger.info("DynSysAgent initialized successfully.")
        except Exception as exc:
            logger.warning(f"Could not initialize kiff agent: {exc}")
            self._agent = None
            self._available = False

        self._history: list[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the underlying LLM agent is ready."""
        return self._available

    def chat(self, user_message: str, system_context: str) -> str:
        """
        Send a message and get a response.

        Parameters
        ----------
        user_message : str
            The user's question or statement.
        system_context : str
            A human-readable description of the current system state,
            e.g. "System: Hopf | mu=1.2 | cfreq=3.0 | w=0.5".

        Returns
        -------
        str
            The agent's response, or an error string if unavailable.
        """
        if not self._available or self._agent is None:
            return "[Agent unavailable - check API key in .env]"

        # Build rolling conversation history string for context
        history_str = self._format_history()

        try:
            result = self._agent.ping(
                user_input={
                    "system_context": (
                        "You are an expert in dynamical systems and nonlinear dynamics. "
                        "The user is interacting with an interactive flow-field visualizer. "
                        f"Current state: {system_context}. "
                        "Answer the user's question concisely and accurately. "
                        "You may suggest parameter changes, explain bifurcations, "
                        "describe attractors, or interpret what is visible in the plot."
                    ),
                    "conversation_history": history_str,
                    "query": user_message,
                },
                signature=self.SIGNATURE,
            )
            response_text = result.response

        except Exception as exc:
            logger.error(f"Agent ping failed: {exc}")
            response_text = f"[Error: {exc}]"

        # Update memory
        interaction = {"user": user_message, "assistant": response_text}
        self._history.append(interaction)
        if self._agent is not None:
            self._agent.add_to_memory(interaction)

        return response_text

    def reset_history(self):
        """Clear conversation history."""
        self._history = []
        if self._agent is not None:
            self._agent.memory = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_history(self, max_turns: int = 4) -> str:
        """Format recent history as a compact string for the prompt."""
        recent = self._history[-max_turns:]
        if not recent:
            return "(no prior conversation)"
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)
