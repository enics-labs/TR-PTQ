import torch
import sys
import os

# Append TR-PTQ source so the copied files can still resolve their dependencies 
# (e.g. MinMaxObserver, TaylorExponent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tr_ptq/TR-PTQ/src')))
# Append verification folder to easily import newly copied components
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../verification')))

# Import the copied model
from int_softmax import IntSoftmaxTS
from q_layernorm import QLayerNorm
from int_gelu import IntGeluTS

def verify_softmax():
    print("\n======================================================")
    print("  STARTING DETERMINISTIC SWEEP: SOFTMAX")
    print("======================================================")
    print("\n>>> RUNNING VERIFICATION: SoftMax Regression Profile")
    
    # Initialize SoftMax (Matches W=8)
    # We use nof_bits=8 to reflect the 8-bit Q4.4 format in the SV hardware.
    model = IntSoftmaxTS(nof_bits=8, quant=True)
    
    # SV Hardware Array: '{8'd0, -8'd16, -8'd32, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128};
    # The HW uses Q4.4, which means divide by 16 to get the real equivalent.
    # Q4.4 to Float: 0/16 = 0.0, -16/16 = -1.0, -32/16 = -2.0, -128/16 = -8.0
    test_in_1 = torch.tensor([[0.0, -1.0, -2.0, -8.0, -8.0, -8.0, -8.0, -8.0]])
    
    # Since IntSoftmaxTS uses MinMaxObserver to determine scaling, 
    # we calibrate it quickly with the input to lock the observer max/min bounds
    model.set_calibration_flag()
    model(test_in_1)
    model.unset_calibration_flag()
    
    model.set_quant()
    out_1 = model(test_in_1)
    print(f"Input floats: {test_in_1.tolist()[0]}")
    print(f"Output SoftMax: {out_1.tolist()[0]}")
    
    
    print("\n>>> RUNNING VERIFICATION: SoftMax Uniform Distribution")
    # SV Hardware Array: '{8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16};
    # Q4.4 to Float: 16/16 = 1.0
    test_in_2 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    
    # Re-calibrate for the new limits
    model.set_calibration_flag()
    model(test_in_2)
    model.unset_calibration_flag()
    
    model.set_quant()
    out_2 = model(test_in_2)
    print(f"Input floats: {test_in_2.tolist()[0]}")
    print(f"Output SoftMax: {out_2.tolist()[0]}")


def verify_layernorm():
    print("\n======================================================")
    print("  STARTING DETERMINISTIC SWEEP: LAYERNORM")
    print("======================================================")
    
    # Initialize LayerNorm (Matches normalized_shape=8, in1_bits=8 for Q4.4 format)
    # We use elementwise_affine=True (the default), but force weight=1.0 and bias=0.0
    # because the hardware simply normalizes the value without applying an affine shift
    model = QLayerNorm(8, in1_bits=8, in2_bits=8, quant=True, elementwise_affine=True)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(0.0)
    
    print("\n>>> RUNNING VERIFICATION: LayerNorm Zero-Mean")
    # SV Array: '{8'd32, 8'd32, 8'd32, 8'd32, -8'd32, -8'd32, -8'd32, -8'd32} -> Float: 2.0 / -2.0
    test_in_1 = torch.tensor([[2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0]])
    model.set_calibration_flag()
    model(test_in_1)
    model.unset_calibration_flag()
    model.set_quant()
    out_1 = model(test_in_1)
    print(f"Input floats: {test_in_1.tolist()[0]}")
    print(f"Output LayerNorm: {out_1.tolist()[0]}")
    
    print("\n>>> RUNNING VERIFICATION: LayerNorm Positive Skew")
    # SV Array: '{8'd64, 8'd32, 8'd32, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0} -> Float: 4.0 / 2.0 / 0.0
    test_in_2 = torch.tensor([[4.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model.set_calibration_flag()
    model(test_in_2)
    model.unset_calibration_flag()
    model.set_quant()
    out_2 = model(test_in_2)
    print(f"Input floats: {test_in_2.tolist()[0]}")
    print(f"Output LayerNorm: {out_2.tolist()[0]}")


def verify_gelu():
    print("\n======================================================")
    print("  STARTING DETERMINISTIC SWEEP: GELU")
    print("======================================================")
    model = IntGeluTS(nof_bits=8, quant=True)
    
    print("\n>>> RUNNING VERIFICATION: GELU Regression Profile")
    test_in_1 = torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model.set_calibration_flag()
    model(test_in_1)
    model.unset_calibration_flag()
    model.set_quant()
    out_1 = model(test_in_1)
    print(f"Input floats: {test_in_1.tolist()[0]}")
    print(f"Output GELU: {out_1.tolist()[0]}")

    print("\n>>> RUNNING VERIFICATION: GELU Positive Upper Bounds")
    test_in_2 = torch.tensor([[2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model.set_calibration_flag()
    model(test_in_2)
    model.unset_calibration_flag()
    model.set_quant()
    out_2 = model(test_in_2)
    print(f"Input floats: {test_in_2.tolist()[0]}")
    print(f"Output GELU: {out_2.tolist()[0]}")

if __name__ == "__main__":
    verify_softmax()
    verify_layernorm()
    verify_gelu()