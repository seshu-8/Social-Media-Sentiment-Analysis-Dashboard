"""
main.py
--------
Central entry point for the Social Media Sentiment Analysis project.

Usage:
    python main.py --generate     → Create synthetic dataset
    python main.py --preprocess   → Clean & preprocess text
    python main.py --train        → Train ML models
    python main.py --evaluate     → Generate charts & insights
    python main.py --predict      → Demo predictions
    python main.py --all          → Run entire pipeline
    python main.py --dashboard    → Launch Streamlit dashboard
"""

import argparse
import os
import sys
import subprocess

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_generate():
    print("\n🔄 Step 1: Generating synthetic dataset...")
    from data.generate_dataset import generate_dataset
    generate_dataset()


def cmd_preprocess():
    print("\n🔄 Step 2: Running preprocessing pipeline...")
    import pandas as pd
    from src.preprocess import preprocess_dataframe

    df = pd.read_csv("data/social_media_data.csv")
    df = preprocess_dataframe(df)

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/preprocessed_data.csv"
    df.to_csv(out, index=False)
    print(f"✅ Preprocessed data saved → {out}")
    print(df[["text", "cleaned_text", "processed_text"]].head(5).to_string())


def cmd_train():
    print("\n🔄 Step 3: Training ML models...")
    from src.train_model import run_training
    run_training()


def cmd_evaluate():
    print("\n🔄 Step 4: Generating evaluation charts...")
    from src.evaluate import run_all_charts
    run_all_charts()


def cmd_predict():
    print("\n🔄 Step 5: Running demo predictions...")
    from src.predict import predict_vader, predict_ml, load_artifacts

    sample_texts = [
        "Zomato delivery was super fast today! Loved the packaging!",
        "Netflix keeps buffering. Very frustrating experience!",
        "Placed my Zomato order. Waiting for delivery.",
        "Amazon delivered a damaged product. Refund is a nightmare!",
        "Really happy with my purchase. Exactly as described online.",
        "Swiggy cancelled my order without any notice. Terrible!",
        "The new Netflix series is absolutely mind-blowing!",
        "Received a broken item. Customer service not responding.",
    ]

    print("\n" + "=" * 65)
    print("  VADER Rule-Based Predictions")
    print("=" * 65)
    for r in predict_vader(sample_texts):
        e = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}.get(r["sentiment"])
        print(f"{e} [{r['sentiment']:8s}] (score={r['compound']:+.3f}) → {r['text'][:55]}")

    try:
        model, le, vectorizer = load_artifacts()
        print("\n" + "=" * 65)
        print("  ML Model Predictions")
        print("=" * 65)
        for r in predict_ml(sample_texts, model, le, vectorizer):
            e   = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}.get(r["sentiment"])
            c   = f"(conf={r['confidence']}%)" if r["confidence"] else ""
            print(f"{e} [{r['sentiment']:8s}] {c:12s} → {r['text'][:55]}")
    except FileNotFoundError:
        print("\n⚠️  Trained model not found. Run --train first.")


def cmd_dashboard():
    print("\n🚀 Launching Streamlit dashboard...")
    print("   Open your browser at: http://localhost:8501\n")
    subprocess.run(["streamlit", "run", "app/dashboard.py"], check=True)


def cmd_all():
    cmd_generate()
    cmd_preprocess()
    cmd_train()
    cmd_evaluate()
    cmd_predict()
    print("\n✅ Full pipeline complete!")
    print("   Run `python main.py --dashboard` to launch the dashboard.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media Sentiment Analysis – Pipeline Runner"
    )
    parser.add_argument("--generate",   action="store_true", help="Generate dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess text")
    parser.add_argument("--train",      action="store_true", help="Train ML models")
    parser.add_argument("--evaluate",   action="store_true", help="Generate charts")
    parser.add_argument("--predict",    action="store_true", help="Demo predictions")
    parser.add_argument("--dashboard",  action="store_true", help="Launch dashboard")
    parser.add_argument("--all",        action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    if args.all:        cmd_all()
    elif args.generate:   cmd_generate()
    elif args.preprocess: cmd_preprocess()
    elif args.train:      cmd_train()
    elif args.evaluate:   cmd_evaluate()
    elif args.predict:    cmd_predict()
    elif args.dashboard:  cmd_dashboard()
    else:
        parser.print_help()
        print("\n💡 Tip: Run `python main.py --all` to execute the complete pipeline.")
