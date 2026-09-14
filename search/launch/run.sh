#!/usr/bin/env bash

## Model: replace this with any model supported by WECO, for example:
##   --model gpt-5.5
##   --model claude-opus-4-7
##   --model gemini-3-pro-preview
##
## Proxy model size for the search. The loop trains this model once per step to
## score a candidate selector, so it is deliberately smaller than the model you
## actually want to pre-train. Rule of thumb: 4-5x fewer parameters than the
## target.
##   target d24 (1.3 B params)  -> proxy d12 (286.3 M)   ~4.5x smaller
##   target d12 (286.3 M)       -> proxy d8  (125.8 M)   ~2.3x smaller
## A larger proxy transfers better to the target but costs proportionally more
## per step, and there are 200 steps. Only depths 8 and 12 have proxy recipes
## (search/weco_eval.py PROXY_RECIPE); 8 is what every published run used.
export AUTODATA_EVAL_DEPTH=8

## Instructions/objective: switch these options together:
##   CORE:    --metric core --goal maximize
##            --additional-instructions search/instructions/full_core.md
##   val-bpb: --metric val_bpb --goal minimize
##            --additional-instructions search/instructions/full.md

weco run \
    --sources search/data_select_template.py \
    --eval-command 'bash search/weco_eval.sh' \
    --metric core --goal maximize --steps 200 \
    --model gemini-3-pro-preview \
    --eval-timeout 2400 --save-logs --output plain \
    --additional-instructions search/instructions/full_core.md
