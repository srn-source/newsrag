"""Wire the nodes into a LangGraph graph.

    START -> plan -> retrieve --(grade)--> generate -> validate -> END
                                   \\
                                    `--> refuse ---------------> END

The graph receives its three dependencies instead of creating them:

    retriever  HybridRetriever (production) | a stub (tests)
    llm        Gemini on Vertex AI          | FakeListChatModel
    planner    Gemini structured output     | PassThroughPlanner | a stub

That is dependency injection: the same graph code runs in the API, in evaluation
and in tests — only what is passed in changes.

Print the diagram for ARCHITECTURE.md:
    print(build_graph(...).get_graph().draw_mermaid())
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.nodes import Planner, Retriever
from app.graph.state import RAGState


def build_graph(
    retriever: Retriever, llm: BaseChatModel, planner: Planner, max_per_article: int = 0
):
    # Small wrappers give each node the dependency it needs, while LangGraph only
    # ever calls them with the state.
    async def plan(state: RAGState) -> RAGState:
        return await nodes.plan(state, planner)

    async def retrieve(state: RAGState) -> RAGState:
        return await nodes.retrieve(state, retriever, max_per_article)

    async def generate(state: RAGState) -> RAGState:
        return await nodes.generate(state, llm)

    graph = StateGraph(RAGState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("validate", nodes.validate)
    graph.add_node("refuse", nodes.refuse)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    # After retrieve, call nodes.grade(state); it returns the name of the next node.
    graph.add_conditional_edges("retrieve", nodes.grade, ["generate", "refuse"])
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    graph.add_edge("refuse", END)

    return graph.compile()
