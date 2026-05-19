import os
import subprocess
import sys

def run_verification():
    print("\n====================================================")
    print(" 🚀 STARTING HW vs SW VERIFICATION: EXP_ALU")
    print("====================================================\n")

    if not os.getcwd().endswith("workspace"):
        print("[WARNING] You are not in the 'workspace' directory.")
        print("Please cd into 'workspace/' and run: python ../verification/verify_flow.py")
        sys.exit(1)

    workspace_dir = os.getcwd()
    
    # --- FIX 1: Create a dummy CUDA header to bypass the missing include error ---
    mock_cuda_dir = os.path.abspath(os.path.join(workspace_dir, "../emulation/mock_cuda"))
    os.makedirs(mock_cuda_dir, exist_ok=True)
    with open(os.path.join(mock_cuda_dir, "cuda_runtime.h"), "w") as f:
        f.write("// Auto-generated dummy file to allow CPU compilation of CUDA headers\n")
    
    # --- FIX 2: Correctly point to the 'common/' directory ---
    math_include_dir = os.path.abspath(os.path.join(
        workspace_dir, 
        "../emulation/stable_code/mrcp_quant/optimized_layers/common"
    ))
    
    cpp_source_file = os.path.abspath(os.path.join(
        workspace_dir, 
        "../emulation/cpu_test_harness.cpp"
    ))

    print(f">> [1/4] Compiling Golden C++ Model...")
    
    # Pass BOTH include paths to g++
    cpp_compile = subprocess.run([
        "g++", "-O3", 
        f"-I{math_include_dir}",  # Lets it find tr_math.cuh
        f"-I{mock_cuda_dir}",     # Lets it find our dummy cuda_runtime.h
        cpp_source_file, 
        "-o", "cpu_model"
    ], capture_output=True, text=True)

    if cpp_compile.returncode != 0:
        print("[FATAL] C++ Compilation failed:\n", cpp_compile.stderr)
        sys.exit(1)

    # Step 2: Run C++ Model to generate vectors
    print(">> [2/4] Generating Vectors & Expected Outputs...")
    subprocess.run(["./cpu_model"])

    # Step 3: Run RTL Simulation
    # Executes xrun using the .f file. xrun will drop its logs and waves into workspace/
    print(">> [3/4] Running RTL Simulation (xrun)...")
    rtl_compile = subprocess.run(
        ["xrun", "-f", "../tb_soc/tr_exp/tr_exp.f"], 
        capture_output=True, text=True
    )
    
    # xrun returns non-zero on compilation or severe runtime errors
    if rtl_compile.returncode != 0:
        print("[FATAL] RTL Simulation failed:\n", rtl_compile.stderr)
        # Still print stdout as xrun often dumps errors there
        print(rtl_compile.stdout)
        sys.exit(1)

    # Step 4: Bit-True Comparison
    print("\n>> [4/4] Analyzing Bit-True Compliance...")
    try:
        with open("expected_exp_out.txt", "r") as f_exp, open("hdl_exp_out.txt", "r") as f_hdl:
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

    for idx, (exp, hdl) in enumerate(zip(exp_lines, hdl_lines)):
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
        print(" ✅ VERIFICATION PASSED. Hardware matches math!")

if __name__ == "__main__":
    run_verification()