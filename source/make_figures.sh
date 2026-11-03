#!/bin/zsh
# all the figures of tex/final/martedi.tex, one after the other (each run already uses all cores),
# then recompile martedi.tex. run from anywhere: ./make_figures.sh
set -e
cd "$(dirname "$0")"

PY=../venv/tesi-clean/bin/python
OUT=../tex/final/outputs
# constants of the thesis: union, Rademacher-Massart, noise bound
C=(--C_union 0.0006 --C_radmass 0.125 --C_noise 0.01)
# three times the ones above, for the comparisons at many noise levels
C3=(--C_radmass 0.375 --C_noise 0.03)

$PY experiments.py $C --curves train_err test_err                --out $OUT/train_test.png
$PY experiments.py $C --curves gap bound_union                   --out $OUT/union_bound.png
$PY experiments.py $C --curves test_err train_bound_union        --out $OUT/train_union_bound.png
$PY experiments.py $C --curves gap bound_radmass                 --out $OUT/radmass_bound.png
$PY experiments.py $C --curves test_err train_bound_radmass      --out $OUT/train_radmass_bound.png
$PY experiments.py $C --curves bound_union bound_radmass         --out $OUT/union_bound_radmass_bound.png
$PY experiments.py $C --curves gap bound_radmass --noise_level 5 20 40 --out $OUT/radmass_bound_many_noise.png
$PY experiments.py $C --curves gap bound_noise                   --out $OUT/noise_bound.png
$PY experiments.py $C --curves test_err train_bound_noise        --out $OUT/train_noise_bound.png
$PY experiments.py $C3 --curves bound_noise bound_radmass --noise_level 5 20 40 --out $OUT/noise_bound_radmass_bound_many_noise.png
$PY experiments.py $C3 --curves bound_noise bound_radmass --noise_level 1 20 49 --out $OUT/noise_bound_radmass_bound_many_noise_2.png

cd ../tex/final && latexmk -pdf -interaction=nonstopmode martedi.tex > /dev/null && echo "martedi.pdf updated"
