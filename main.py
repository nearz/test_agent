from agent.agent import Agent


def main():
    agent = Agent("gpt-4o")
    while True:
        user_input = input("USER: ")
        if user_input.lower() == ".exit":
            break
        ai_msg = agent.invoke(user_input)
        print(f"AI: {ai_msg}")


if __name__ == "__main__":
    main()
