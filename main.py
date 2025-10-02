from session.session import Session
from agent.state import build_graph


def main():
    # NOTE: Graph build may be app level
    graph = build_graph()
    llm = "gpt-4o"
    thread_id = "2"
    ses = Session(llm, thread_id, graph)
    while True:
        user_input = input("USER: ")
        if user_input.lower() == ".exit":
            break
        ai_msg = ses.invoke(user_input)
        print(f"AI: {ai_msg}")


if __name__ == "__main__":
    main()
