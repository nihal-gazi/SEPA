"""
run_all.py — GSCv3 Master Execution Pipeline

Automatically executes the entire GSCv3 training and inference pipeline from scratch.
Ensures memory isolation between phases by using subprocesses.
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Runs a Python script as a subprocess and streams its output."""
    print(f"\n{'='*60}")
    print(f"🚀 LAUNCHING: {script_name}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(script_name):
        print(f"❌ Error: Could not find '{script_name}' in the current directory.")
        sys.exit(1)
        
    try:
        # Uses the exact same Python executable running this master script
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FATAL: '{script_name}' crashed with exit code {e.returncode}. Pipeline stopped.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⏸️ Pipeline interrupted by user during '{script_name}'.")
        sys.exit(1)

def main():
    print("============================================================")
    print("           GSCv3 AUTOMATED PIPELINE INITIATED               ")
    print("============================================================")
    
    # Define the strict sequence of operations
    pipeline = [
        "train_phase1.py",          # 1. Train Autoencoder
        "valid_pair_extractor.py",   # 2. Map Valid Codebook Space
        "train_phase2.py",          # 3. Train AR Latent Generator
        "infer.py"                  # 4. Generate Output
    ]
    
    for script in pipeline:
        run_script(script)
        
    print("\n" + "="*60)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)

if __name__ == "__main__":
    main()