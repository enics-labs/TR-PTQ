# Objectives
Unlike expected, the golden reference model would be dot-product unit with standard quantization unit. This because the design exploit a share unit structure. We want to see if we can compute the exponent sum (the reciprocol) on the dotproduct unit without causing major implications.

1. Calculate PPA for dot-product unit.
2. Calculate PPA for the exponent unit (standalone).
3. Calculate the combine unit and explore how the exponent affect the design in terms of PPA.
4. Report the buttlenecks and issues that found in each of the above.

