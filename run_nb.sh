find /Users/tristanchaang/MIT/UROP-Traffic/mpc-traffic-flow \
  -type d -name __pycache__ -prune -exec rm -r {} +

python experiments/cc_plot.py
python experiments/table_plot.py
python experiments/tsd_plot.py
python experiments/constr_plot.py

git fetch origin

python src/analyze_refactor.py