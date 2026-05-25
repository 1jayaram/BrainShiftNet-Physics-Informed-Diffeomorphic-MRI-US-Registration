#!/usr/bin/env python
"""
Quick reference: Run evaluation on test/val/train set and display metrics

Usage:
  # Evaluate on test set
  python scripts/quick_eval.py test
  
  # Evaluate on validation set
  python scripts/quick_eval.py val
  
  # Evaluate on training set
  python scripts/quick_eval.py train
  
  # With custom options
  python scripts/quick_eval.py test --checkpoint outputs/run_01/checkpoint_epoch0040.pth
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_model import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation script - Run model on dataset split and display metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quick_eval.py test
  python scripts/quick_eval.py val --checkpoint outputs/run_01/best_model.pth
  python scripts/quick_eval.py train --device cpu
        """
    )
    
    parser.add_argument("split", nargs="?", default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("-c", "--checkpoint", 
                        default="outputs/run_01/best_model.pth",
                        help="Model checkpoint path")
    parser.add_argument("-d", "--data_dir",
                        default="data/processed2",
                        help="Processed data directory")
    parser.add_argument("-o", "--output_dir",
                        default="results/evaluation",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Computation device")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BRAIN SHIFT MODEL EVALUATION")
    print("="*80)
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    print("\n" + "="*80)
    print("ℹ️  Results saved to: results/evaluation/")
    print("   - evaluation_{split}.json (machine-readable)")
    print("   - evaluation_{split}.md (formatted report)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
