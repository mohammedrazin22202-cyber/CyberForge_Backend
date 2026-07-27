# Benchmark Chatbot Matching Speed
import time
import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
import app

queries = [
    "who are you",
    "tell me about yourself",
    "what is logisense",
    "tell me about LogiSense 360",
    "how does logisense work",
    "what technologies were used in logisense",
    "what is vconnect",
    "what is cyberforge",
    "are you looking for a job",
    "what is your experience level",
    "what roles are you interested in",
    "download resume",
    "contact information",
    "show projects",
    "what are your achievements"
]

print("Benchmarking query scoring engine...")
start_time = time.time()
iterations = 200

for _ in range(iterations):
    for q in queries:
        app.score_input(q)

end_time = time.time()
total_queries = iterations * len(queries)
duration = end_time - start_time
print(f"Processed {total_queries} queries in {duration:.4f} seconds.")
print(f"Throughput: {total_queries / duration:.2f} queries per second.")
