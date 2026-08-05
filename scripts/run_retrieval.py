"""
run_retrieval.py — CLI nhận 1 query, chạy qua task_modules tương ứng, in kết quả.

Chạy:
  python scripts/run_retrieval.py --type KIS --query "người mặc áo đỏ trên sân khấu"
  python scripts/run_retrieval.py --type QA  --query "..." --question "Màu gì?"
  python scripts/run_retrieval.py --type TRAKE --query "..."
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Run single query through pipeline")
    p.add_argument("--type",     choices=["KIS", "QA", "TRAKE"], required=True)
    p.add_argument("--query",    required=True)
    p.add_argument("--question", default="")
    p.add_argument("--top_k",   type=int, default=100)
    p.add_argument("--output",   default="")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[*] Query type: {args.type}")
    print(f"[*] Query: {args.query}")

    try:
        from main import build_pipeline
        pipeline = build_pipeline()
    except Exception as e:
        print(f"[!] Pipeline không khởi tạo được: {e}")
        sys.exit(1)

    results = []
    if args.type == "KIS":
        results = pipeline.search_kis(args.query, top_k=args.top_k)
    elif args.type == "QA":
        results = pipeline.search_qa(args.query, args.question, top_k=args.top_k)
    elif args.type == "TRAKE":
        results = pipeline.search_trake(args.query, top_k=args.top_k)

    print(f"\n[+] {len(results)} results:")
    for i, r in enumerate(results[:10]):
        print(f"  [{i+1:3d}] {r.get('video_id')} @ {r.get('frame_id')} | score={r.get('score', 0):.4f}", end="")
        if args.type == "QA":
            print(f" | ans={r.get('answer', '')}", end="")
        if args.type == "TRAKE":
            print(f" | frames={r.get('frame_ids', [])}", end="")
        print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[+] Results saved: {args.output}")


if __name__ == "__main__":
    main()
