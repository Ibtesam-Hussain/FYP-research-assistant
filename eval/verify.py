# # verify_and_fix_testset.py
# # Run this, then manually update relevant_chunk_ids in your testset
# from src.pipeline import RAGPipeline
# import json

# pipeline = RAGPipeline(llm_client=None)

# with open("data/eval/qa_testset.json") as f:
#     questions = json.load(f)

# answerable = [q for q in questions if q.get("answerable")]

# print("="*70)
# print("CHUNK ID VERIFICATION — update your testset based on this output")
# print("="*70)

# for item in answerable:
#     print(f"\n[{item['question_id']}] {item['question'][:70]}...")
    
#     result = pipeline.retrieve(item["question"])
    
#     print(f"  Top-5 retrieved chunks:")
#     for r in result["final_results"]:
#         print(f"    chunk_id : {r['chunk_id']}")
#         print(f"    preview  : {r['text'][:120].strip()}...")
#         print()
    
#     print(f"  Current testset relevant_chunk_ids: {item['relevant_chunk_ids']}")
#     print("-"*70)


# paste into a quick python script or terminal
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(llm_client=None)

failures = [
    ("q001", "What is the main limitation of monocular depth estimation compared to stereo methods?"),
    ("q006", "What network architecture does martins2018 use for monocular depth estimation?"),
    ("q013", "What is the main motivation for developing unsupervised monocular depth estimation?"),
]

for qid, query in failures:
    result = pipeline.retrieve(query)
    print(f"\n{qid}: {query[:60]}...")
    for r in result["final_results"]:
        print(f"  {r['chunk_id']}  score={r['rerank_score']:.3f}")
        print(f"  {r['text'][:100]}...")