import chainlit as cl
from chainlit import user_session
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


@cl.on_chat_start
def main():
    env = user_session.get("env")
    assert env is not None
    llm = ChatOpenAI(
        temperature=0,
        # We don't have access to GPT-4-32k, so we use GPT-4 instead.
        # model="gpt-4-32k",
        model="gpt-4",
        openai_api_key=env["OPENAI_API_KEY"],
    )
    tools = load_tools(
        [
            "google-search-results-json",
            "llm-math",
            # requests_all can be used to browse the link, however it would easily exceed GPT-4's context length limit.
            # "requests_all"
        ],
        llm=llm,
        google_api_key=env["GOOGLE_API_KEY"],
        google_cse_id=env["GOOGLE_CSE_ID"],
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )

    # Store the chain in the user session
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: str):
    # Retrieve the chain from the user session
    agent = cl.user_session.get("agent")
    assert isinstance(agent, AgentExecutor)

    # Call the chain asynchronously
    res = await agent.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["output"]).send()
