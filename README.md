# Work Plan
1. run a clean soc environment:
* change the compilation chain - it's too complicated
* seperate the software and the hardware.
* the environment contains the following files:
- verification
- sw
- soc
- rtl
- sim
- scripts
- docs

2. UVM environment integration:
* extension tested first in unit-level and just then in soc-level
* the unit level environmet use daniel UVM framework.
* random tests should be used in the unit level - functional tests.
* after getting satisfying coverage, run test vectors. - use the eqf (when available) to generate input and output files straight from chosen layers.

3. software environment - relevent for soc level only
* BM os - [EXPLORATION]. For now, udi's bm environment is enough.
* eqf - eli quantization framework. pytorch code -> torch.fx graph -> onnx optimization -> TVM compiler.
* the test should run have two modes: gcc, simulation. the same code should support both using macro defines.

4. soc:
* for now use the standard hamsa arc, but check the kunts5 and other open-source SOCs.
* peripheral extensions are integrated via gpp registers.
* consider using a simpler environmemt becuase the current takes too much memory and space.

5. rtl - ai extension:
* implementation of TR-vit.
* each TR-vit extension should have it's own interface.
* consider implementing one multi-function unit.

6. running environment:
* files should be compiled and exucute from everywhere.
* a dump directory called sim is the only place where generated files (.exe, logs, hex, etc.) should be exist!
* each time executing run, a directory with the name <prefix>_<day>_<month>_<years> should be open. If exists, override the file.

7. scripts
* fun factor is a key factor
* messy environment is not fun - only one place for products.
* hook must be deleted and formalized.
* Too much python for rtl running doesn't make you a good hardware designer.
* less (scripts) is more!!!.
* every script must be documented with help flag or else it would be deleted.
* if the user ask you to write too complicated script, tell him that he is doing a bad job.
* the first to get fired is the one who build the script!!!
* scripts are opened to extend but not for change.
* the less you use system instruction the better.
* be aware from write scripts that crashes simulation.

8. docs:
* if someone ask you what to run or how to run, it must be documented!
* diagram is the best documentation.
* you do not submit anything before update the documentation (if needed).

9. postlog:
* those istructions are the product of working with truely idiotic peoples.
* smart people also helped.
