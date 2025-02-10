# bash scripts/bash.sh

echo "-------------------------------- Run examples.main --------------------------------"
python -m examples.main

wait
echo "-------------------------------- Compute EX stats --------------------------------"
python -m src.evaluation.compute_EX

