#!/bin/bash
# WECO eval wrapper — invokes search/weco_eval.py from the AutoData repo root.
# Required env vars are set by search/launch/run.sh.
set -u
exec python -m search.weco_eval
