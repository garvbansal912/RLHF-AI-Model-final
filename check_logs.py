import sqlite3

# Connect to the local data repository in your Garv folder
conn = sqlite3.connect("Garv/chat_telemetry.db")
cursor = conn.cursor()

# Retrieve all logged chat conversations
try:
    cursor.execute("SELECT id, prompt, response FROM conversation_logs")
    records = cursor.fetchall()

    print(f"=== Total Logged Chats: {len(records)} ===")
    for row in records:
        print(f"\n[ID: {row[0]}] User Sent: '{row[1]}'")
        print(f"       AI Responded: '{row[2][:60]}...'")
except sqlite3.OperationalError:
    print("Database table is empty or hasn't been created yet. Send a message in your app first!")

conn.close()
