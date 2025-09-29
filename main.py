from session.session import Session


def main():
    ses = Session("gpt-4o")
    while True:
        user_input = input("USER: ")
        if user_input.lower() == ".exit":
            break
        ai_msg = ses.invoke(user_input)
        print(f"AI: {ai_msg}")


if __name__ == "__main__":
    main()
