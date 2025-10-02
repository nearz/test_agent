import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpoints.db")
saver = SqliteSaver(conn)

# Get the latest checkpoint
cursor = conn.cursor()
cursor.execute(
    "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
)
row = cursor.fetchone()

if row:
    thread_id, checkpoint_ns, checkpoint_id = row
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = saver.get_tuple(config)
    if checkpoint and checkpoint.checkpoint:
        messages = checkpoint.checkpoint["channel_values"].get("messages", [])
        for i, msg in enumerate(messages[:3]):
            print(f"Message {i}:")
            print(f"\tType: {msg.__class__.__name__}")
            print(f"\tHas 'id' attribute: {hasattr(msg, 'id')}")
            if hasattr(msg, "id"):
                print(f"\tID value: {msg.id}")
            print(f"\tContent preview: {str(msg.content)[:100]}")
            print()
