from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from chainlit import user_session, langchain_factory

from os import environ


@langchain_factory(use_async=False)
def factory():
    chat = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=user_session.get("env")["OPENAI_API_KEY"],
    )
    tools = load_tools(
        ("google-search-results-json", "llm-math"),
        llm=chat,
        google_api_key=user_session.get("env")["GOOGLE_API_KEY"],
        google_cse_id=user_session.get("env")["GOOGLE_CSE_ID"],
    )
    agent = initialize_agent(
        tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return agent
