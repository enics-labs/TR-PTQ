import os
import subprocess
import sys
import argparse

def get_config(block_name):
    # This dictionary dynamically maps your block arguments to their specific .f files
    configs = {
        "exp": {"f_file": "../tb_soc/tr_exp/tr_exp.f"},
        "ln":  {"f_file": "../tb_soc/tr_ln/tr_ln.f"}
        # Add future blocks here! (e.g., "gelu": {"f_file": "../tb_soc/tr_gelu/tr_gelu.f"})
    }
    return configs.get(block_name.lower())

def run_verification():
    parser = argparse.ArgumentParser(description="Run HW vs SW Verification for a specific block.")
    parser.add_argument("block", type=str, help="The block to verify (e.g., 'exp' or 'ln')")
    args = parser.parse_args()
    
    config = get_config(args.block)
    if not config:
        print(f"[ERROR] Unknown block '{args.block}'. Check your mapping in verify_block.py.")
        sys.exit(1)

    print(f"\n====================================================")
    print(f" 🚀 STARTING HW vs SW VERIFICATION: {args.block.upper()}")
    print(f"====================================================\n")

    if not os.getcwd().endswith("workspace"):
        print("[WARNING] Please cd into 'workspace/' and run: python ../verification/verify_block.py <block>")
        sys.exit(1)

    workspace_dir = os.getcwd()
    
    # 1. Setup mock CUDA header
    mock_cuda_dir = os.path.abspath(os.path.join(workspace_dir, "../emulation/mock_cuda"))
    os.makedirs(mock_cuda_dir, exist_ok=True)
    with open(os.path.join(mock_cuda_dir, "cuda_runtime.h"), "w") as f:
        f.write("// Auto-generated dummy file\n")
    
    math_include_dir = os.path.abspath(os.path.join(
        workspace_dir, "../emulation/stable_code/mrcp_quant/optimized_layers/common"
    ))
    
    cpp_source_file = os.path.abspath(os.path.join(
        workspace_dir, "../emulation/cpu_math_model.cpp"
    ))

    # 2. Compile C++ Golden Model
    print(">> [1/4] Compiling Unified Golden C++ Model...")
    cpp_compile = subprocess.run([
        "g++", "-O3", f"-I{math_include_dir}", f"-I{mock_cuda_dir}", cpp_source_file, "-o", "cpu_model"
    ], capture_output=True, text=True)

    if cpp_compile.returncode != 0:
        print("[FATAL] C++ Compilation failed:\n", cpp_compile.stderr)
        sys.exit(1)

    # 3. Run C++ Model (Pass the block mode argument!)
    print(f">> [2/4] Generating Vectors for {args.block.upper()}...")
    subprocess.run(["./cpu_model", args.block])

    # 4. Run RTL Simulation using the mapped .f file
    print(f">> [3/4] Running RTL Simulation for {args.block.upper()}...")
    rtl_compile = subprocess.run(
        ["xrun", "-f", config["f_file"]], 
        capture_output=True, text=True
    )
    
    if rtl_compile.returncode != 0:
        print("[FATAL] RTL Simulation failed:\n", rtl_compile.stderr)
        print(rtl_compile.stdout)
        sys.exit(1)

    # 5. Bit-True Comparison (Now using standard generic filenames)
    print("\n>> [4/4] Analyzing Bit-True Compliance...")
    try:
        with open("expected.txt", "r") as f_exp, open("hdl_out.txt", "r") as f_hdl:
            exp_lines = f_exp.readlines()
            hdl_lines = f_hdl.readlines()
    except FileNotFoundError as e:
        print(f"[FATAL] Missing output logs: {e}")
        sys.exit(1)

    if len(exp_lines) != len(hdl_lines):
        print(f"[ERROR] Line count mismatch! C++: {len(exp_lines)} | HDL: {len(hdl_lines)}")
        sys.exit(1)

    mismatches = 0
    total = len(exp_lines)

    for exp, hdl in zip(exp_lines, hdl_lines):
        if exp.split() != hdl.split():
            mismatches += 1
            if mismatches <= 10:
                print(f"  [MISMATCH] Input: {exp.split()[0]} | CUDA expected: {exp.split()[1]} | HDL got: {hdl.split()[1]}")

    match_rate = ((total - mismatches) / total) * 100
    
    print("\n----------------------------------------------------")
    print(f" Total Checked Elements : {total}")
    print(f" Mismatch Count         : {mismatches}")
    print(f" Bit-True Match Rate    : {match_rate:.2f}%")
    print("====================================================")
    
    if mismatches > 0:
        print(" ❌ VERIFICATION FAILED. Fix RTL and re-run.")
    else:
        print(f" ✅ VERIFICATION PASSED. Hardware {args.block.upper()} matches math!")

if __name__ == "__main__":
    run_verification()