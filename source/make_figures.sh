#!/bin/zsh
# all the figures of tex/tesi.tex, one after the other (each run already uses all cores),
# then recompile tesi.tex. run from anywhere: ./make_figures.sh
set -e
cd "$(dirname "$0")"

PY=../venv/tesi-clean/bin/python
OUT=../tex/outputs
# constants of the thesis: union, Rademacher-Massart, noise bound
C=(--C_union 0.0006 --C_radmass 0.125 --C_noise 0.01)

$PY experiments.py $C --curves train_err test_err                --out $OUT/train_test.png
$PY experiments.py $C --curves gap bound_union                   --out $OUT/union_bound.png
$PY experiments.py $C --curves test_err train_bound_union        --out $OUT/train_union_bound.png
$PY experiments.py $C --curves gap bound_radmass                 --out $OUT/radmass_bound.png
$PY experiments.py $C --curves test_err train_bound_radmass      --out $OUT/train_radmass_bound.png
$PY experiments.py $C --curves bound_union bound_radmass         --out $OUT/union_bound_radmass_bound.png
$PY experiments.py $C --curves gap --noise_level 5 20 40         --out $OUT/gap_many_noise.png
$PY experiments.py $C --curves test_err --noise_level 5 20 40    --out $OUT/test_many_noise.png
$PY experiments.py $C --curves gap bound_radmass --noise_level 5 20 40 --out $OUT/radmass_bound_many_noise.png
$PY experiments.py $C --curves gap bound_noise                   --out $OUT/noise_bound.png
$PY experiments.py $C --curves bound_noise --noise_level 5 20 40 --out $OUT/noise_bound_only_many_noise.png
$PY experiments.py --best_risk --noise_level 5 20 40 --out $OUT/best_risk_many_noise.png

cd ../tex && latexmk -pdf -interaction=nonstopmode tesi.tex > /dev/null && echo "tesi.pdf updated"
