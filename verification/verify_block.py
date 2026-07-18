import os
import subprocess
import sys
import argparse

def get_config(block_name):
    configs = {
        "exp":     {"f_file": "../tb_soc/tr_exp/tr_exp.f",         "max_error": 0},
        "ln":      {"f_file": "../tb_soc/tr_ln/tr_ln.f",           "max_error": 0},
        "softmax": {"f_file": "../tb_soc/tr_softmax/tr_softmax.f", "max_error": 0},
        # gelu: tr_gelu_int.f drives the REAL production GL_P1..GL_P3 sequence
        # via tr_soc_top_int (CMD=0x02) -- tr_gelu.sv (tb_soc/tr_gelu/, the
        # prior target) is dead code outside tr_swiglu.sv, unused in the
        # production ViT path. Same trap tr_rmsnorm.sv turned out to be.
        "gelu":    {"f_file": "../tb_soc/tr_gelu_int/tr_gelu_int.f",  "max_error": 0},
        "gelu_dead_module": {"f_file": "../tb_soc/tr_gelu/tr_gelu.f", "max_error": 0},
        "swiglu":  {"f_file": "../tb_soc/tr_swiglu/tr_swiglu.f",   "max_error": 0},
        "quant":   {"f_file": "../tb_soc/tr_quant/tr_quant.f",       "max_error": 0},
        "matmul":  {"f_file": "../tb_soc/tr_matmul/tr_matmul.f",     "max_error": 0},
        "rmsnorm": {"f_file": "../tb_soc/tr_rmsnorm_int/tr_rmsnorm_int.f", "max_error": 0},
        "gelu_fused": {"f_file": "../tb_soc/tr_gelu_fused_int/tr_gelu_fused_int.f", "max_error": 0},
    }
    return configs.get(block_name.lower())

def run_verification():
    parser = argparse.ArgumentParser(description="Run HW vs SW Verification for a specific block.")
    parser.add_argument("block", type=str, help="The block to verify (e.g., 'exp', 'ln', 'softmax')")
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

    print(">> [1/4] Compiling Unified Golden C++ Model...")
    cpp_compile = subprocess.run([
        "g++", "-O3", f"-I{math_include_dir}", f"-I{mock_cuda_dir}", cpp_source_file, "-o", "cpu_model"
    ], capture_output=False, text=True)

    if cpp_compile.returncode != 0:
        print("[FATAL] C++ Compilation failed:\n", cpp_compile.stderr)
        sys.exit(1)

    print(f">> [2/4] Generating Vectors for {args.block.upper()}...")
    subprocess.run(["./cpu_model", args.block])

    print(f">> [3/4] Running RTL Simulation for {args.block.upper()}...")
    
    current_env = os.environ.copy()
    current_env["CDS_LIC_FILE"] = "5280@enicsw01"
    current_env["DISPLAY"] = "" 
    
    xrun_cmd = f"xrun -f {config['f_file']}"
    rtl_compile = subprocess.run(
        ["tcsh", "-i", "-c", xrun_cmd], 
        env=current_env,
        capture_output=False, 
        text=True
    )
    
    if rtl_compile.returncode != 0:
        print("[FATAL] RTL Simulation failed:\n", rtl_compile.stderr)
        print(rtl_compile.stdout)
        sys.exit(1)

    print("\n>> [4/4] Analyzing Error Tolerance...")
    try:
        # Load all three files
        with open("inputs.txt", "r") as f_in, open("expected.txt", "r") as f_exp, open("hdl_out.txt", "r") as f_hdl:
            in_lines = f_in.readlines()
            exp_lines = f_exp.readlines()
            hdl_lines = f_hdl.readlines()
    except FileNotFoundError as e:
        print(f"[FATAL] Missing output logs: {e}")
        sys.exit(1)

    # Strip the header line from inputs.txt if it exists
    if len(in_lines) > len(exp_lines):
        in_lines = in_lines[1:]

    if len(exp_lines) != len(hdl_lines):
        print(f"[ERROR] Line count mismatch! C++: {len(exp_lines)} | HDL: {len(hdl_lines)}")
        sys.exit(1)

    total_elements = 0
    mismatches = 0
    max_delta = 0
    sum_delta = 0
    failed_rows_logged = 0

    # Parse and compare every single element
    for idx, (in_line, exp_line, hdl_line) in enumerate(zip(in_lines, exp_lines, hdl_lines)):
        try:
            # For EXP and LN, the expected/hdl files include the input as the first token.
            # We skip parsing that token for the delta calculation if we are in those modes.
            if args.block in ["exp", "ln"]:
                exp_vals = [int(exp_line.split()[1])]
                hdl_vals = [int(hdl_line.split()[1])]
            else:
                exp_vals = [int(x) for x in exp_line.split()]
                hdl_vals = [int(x) for x in hdl_line.split()]
        except ValueError:
            print(f"[ERROR] Could not parse integers on line {idx+1}.")
            continue

        row_has_error = False
        row_max_delta = 0
        
        for col_idx, (e_val, h_val) in enumerate(zip(exp_vals, hdl_vals)):
            total_elements += 1
            delta = abs(e_val - h_val)
            
            if delta > 0:
                mismatches += 1
                sum_delta += delta
                if delta > max_delta:
                    max_delta = delta
                if delta > config["max_error"]:
                    row_has_error = True
                    if delta > row_max_delta:
                        row_max_delta = delta
                        
        # Print the full vector comparison for the first 10 failures
        if row_has_error and failed_rows_logged < 10:
            print(f"  [MISMATCH] Row {idx+1} (Max Error in row: {row_max_delta})")
            print(f"    Input   : {in_line.strip()}")
            if args.block in ["softmax", "gelu", "swiglu", "quant", "matmul", "rmsnorm", "gelu_fused"]:
                print(f"    Expected: {exp_line.strip()}")
                print(f"    HDL Got : {hdl_line.strip()}\n")
            else:
                # For Exp/Ln we strip the input token for cleaner display
                print(f"    Expected: {exp_line.split()[1]}")
                print(f"    HDL Got : {hdl_line.split()[1]}\n")
            failed_rows_logged += 1

    if failed_rows_logged == 10:
        print("  ... (additional threshold breaches hidden)")

    avg_error = (sum_delta / total_elements) if total_elements > 0 else 0
    match_rate = ((total_elements - mismatches) / total_elements) * 100
    
    print("\n----------------------------------------------------")
    print(f" Total Elements Checked : {total_elements}")
    print(f" Exact Matches          : {total_elements - mismatches} ({match_rate:.2f}%)")
    print(f" Average Error          : {avg_error:.4f}")
    print(f" Maximum Error Detected : {max_delta}")
    print(f" Allowed Error Threshold: <= {config['max_error']}")
    print("====================================================")
    
    if max_delta > config["max_error"]:
        print(f" ❌ VERIFICATION FAILED. Max error ({max_delta}) exceeds allowed threshold ({config['max_error']}).")
        sys.exit(1)
    else:
        print(f" ✅ VERIFICATION PASSED. All approximation errors are within bounds!")

if __name__ == "__main__":
    run_verification()