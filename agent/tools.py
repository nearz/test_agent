import os
from typing import List
from langchain_core.tools import tool, BaseTool


@tool
def list_files() -> List[str]:
    """Function that lists the files of the current working directory"""
    return os.listdir(".")


@tool
def read_file(filename: str) -> str:
    """Read the contents of a file from the current working directory.

    Args:
        filename: The name of the file to read

    Returns:
        The contents of the file as a string, or an error message if the file doesn't exist
    """
    if not os.path.exists(filename):
        return f"Error: File '{filename}' does not exist in the current directory."

    if not os.path.isfile(filename):
        return f"Error: '{filename}' is not a file."

    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


def get_tools() -> List[BaseTool]:
    return [list_files, read_file]
