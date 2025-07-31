import sqlite3

def check_database():
    conn = sqlite3.connect('music_recommendations.db')
    cursor = conn.cursor()
    
    print("=== DATABASE CONTENTS ===\n")
    
    # Check user sessions
    cursor.execute('SELECT COUNT(*) FROM user_sessions')
    session_count = cursor.fetchone()[0]
    print(f"User sessions: {session_count}")
    
    # Check recommendation history
    cursor.execute('SELECT COUNT(*) FROM recommendation_history')
    history_count = cursor.fetchone()[0]
    print(f"Recommendation history: {history_count}")
    
    # Show recent recommendations
    cursor.execute('SELECT user_id, recommendation_type, user_context, created_at FROM recommendation_history ORDER BY created_at DESC LIMIT 5')
    print("\nRecent recommendations:")
    for row in cursor.fetchall():
        print(f"  User: {row[0]}, Type: {row[1]}, Context: {row[2][:50]}..., Created: {row[3]}")
    
    # Show user sessions
    cursor.execute('SELECT user_id, session_start, user_country FROM user_sessions ORDER BY session_start DESC LIMIT 3')
    print("\nRecent user sessions:")
    for row in cursor.fetchall():
        print(f"  User: {row[0]}, Country: {row[2]}, Started: {row[1]}")
    
    conn.close()

if __name__ == "__main__":
    check_database() 