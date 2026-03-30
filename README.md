# ptq_tr

Repository structure for migrating the notebook-based PTQ transformer work into a maintainable codebase.

## Important note

The `quantization/` directory simulates hardware behavior. During the migration from notebook to package layout, code in this directory must not be changed semantically. Refactoring around it is allowed, but its quantization behavior must remain identical.

## Repository tree

```text
ptq_tr/
  README.md
  pyproject.toml

  docs/
    architecture.md
    workflow.md
    diagrams/
      vision_pipeline.drawio
      repo_structure.drawio

  notebooks/
    vision/
      SWIN_(FLOAT)_VERSION.ipynb
    nlp/

  configs/
    vision/
      deit.yaml
      swin.yaml
    nlp/
      glue.yaml

  src/
    ptq_tr/
      __init__.py

      quantization/
        __init__.py
        observers/
          __init__.py
          base.py
          minmax.py
        modules/
          __init__.py
          quant_linear.py
          quant_matmul.py
          int_softmax.py
          int_gelu.py
          q_layernorm.py
          q_hadamard.py
        utils.py
        qparams.py

      models/
        __init__.py

        vision/
          __init__.py
          factories.py
          base.py

          vit/
            __init__.py
            mlp.py
            attention.py
            block.py

            deit/
              __init__.py
              model.py

            swin/
              __init__.py
              patch_embed.py
              window_attention.py
              swin_block.py
              patch_merging.py
              basic_layer.py
              model.py

        nlp/
          __init__.py
          factories.py
          base.py

      preprocessing/
        __init__.py

        vision/
          __init__.py
          transforms.py
          image_processing.py

        nlp/
          __init__.py
          tokenization.py
          text_processing.py

      dataloaders/
        __init__.py

        vision/
          __init__.py
          imagenet.py
          builder.py

        nlp/
          __init__.py
          glue.py
          builder.py

      workflows/
        __init__.py
        calibration/
          __init__.py
          run.py
        optimization/
          __init__.py
          run.py
        validation/
          __init__.py
          run.py

      metrics/
        __init__.py
        classification.py
        similarity.py

      common/
        __init__.py
        logging.py
        seed.py
        device.py

  scripts/
    vision/
      calibrate.py
      optimize.py
      validate.py
    nlp/
      calibrate.py
      optimize.py
      validate.py
```

## Run code location

The code a user runs should always be easy to find:

- `scripts/vision/` contains the runnable vision entrypoints.
- `scripts/nlp/` contains the runnable NLP entrypoints.
- `src/ptq_tr/workflows/` contains the end-to-end logic invoked by those entrypoints.

The intended flow is:

1. Run a script from `scripts/`.
2. The script calls a workflow under `src/ptq_tr/workflows/`.
3. The workflow uses `models/`, `quantization/`, `preprocessing/`, `dataloaders/`, and `metrics/`.
