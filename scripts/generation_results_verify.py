import json

with open("eval/results/generation_results.json", encoding="utf-8") as f:
    data = json.load(f)

per_q = data["per_question"]
total = len(per_q)

null_f = [q for q in per_q if q.get("faithfulness") is None]
null_r = [q for q in per_q if q.get("response_relevancy") is None and q.get("answer_relevancy") is None]
scored = [q for q in per_q if q.get("faithfulness") is not None]

print(f"Total questions:        {total}")
print(f"Faithfulness scored:    {total - len(null_f)}")
print(f"Faithfulness null:      {len(null_f)}")
print(f"Relevancy scored:       {total - len(null_r)}")
print(f"Relevancy null:         {len(null_r)}")
print(f"\nNulls:")
for q in null_f:
    print(f"  {q['question'][:70]}...")