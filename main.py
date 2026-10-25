"""
TruthLens AI — Primary Project Entrypoint (main.py)
Unified launcher for the web application, training pipeline, and test suite.

Usage:
    python main.py               # Launches the Streamlit Web Application
    python main.py --train       # Executes dataset validation and model training
    python main.py --test        # Runs automated unit tests
    python main.py --cli "text"  # Runs instant command-line prediction
"""

import sys
import os
import subprocess
import argparse


# Guarantee UTF-8 output encoding across Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


VENV_PYTHON = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
PYTHON_EXEC = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


def launch_app():
    """Launches the Streamlit interactive dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print("\n" + "=" * 60)
    print("[*] Launching TruthLens AI News Intelligence Dashboard...")
    print("[i] Localhost URL: http://localhost:8501")
    print("[i] Press Ctrl + C to stop the server.")
    print("=" * 60 + "\n")

    cmd = [PYTHON_EXEC, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[TruthLens AI] Application stopped gracefully.")


def run_training():
    """Executes the training and model comparison pipeline."""
    print("\n" + "=" * 60)
    print("[*] Initiating Model Training & Evaluation Pipeline...")
    print("=" * 60 + "\n")
    from train import run_training_pipeline
    run_training_pipeline()


def run_tests():
    """Executes pytest unit test suite."""
    print("\n" + "=" * 60)
    print("[*] Running TruthLens AI Automated Tests...")
    print("=" * 60 + "\n")
    tests_path = os.path.join(os.path.dirname(__file__), "tests")
    cmd = [PYTHON_EXEC, "-m", "pytest", tests_path, "-v", "-p", "no:anyio"]
    subprocess.run(cmd)


def run_cli_prediction(text: str):
    """Executes prediction on given text via terminal."""
    print("\n" + "=" * 60)
    print("[*] Analyzing Text via CLI...")
    print("=" * 60)
    from predict import TruthLensPredictor
    predictor = TruthLensPredictor()
    res = predictor.predict(text)
    if not res["success"]:
        print(f"[!] Error: {res['error']}")
    else:
        print(f"Prediction : {res['label']}")
        if res['confidence'] is not None:
            print(f"Score      : {res['confidence'] * 100:.1f}% ({res['confidence_type']})")
        print(f"Features   : {[f['token'] for f in res['influential_features'][:5]]}")
        print(f"\nNotice     : {res['disclaimer']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TruthLens AI - News Intelligence Platform Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train baseline models and evaluate performance metrics"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pytest automated test suite"
    )
    parser.add_argument(
        "--cli",
        type=str,
        metavar="TEXT",
        help="Classify a news snippet directly in the console"
    )

    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.test:
        run_tests()
    elif args.cli:
        run_cli_prediction(args.cli)
    else:
        # Default action: launch Streamlit Web App
        launch_app()


if __name__ == "__main__":
    main()
