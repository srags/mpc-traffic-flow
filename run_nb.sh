python experiments/synthetic_analysis.py
python experiments/table_plot.py
python experiments/tsd_plot.py
python experiments/constr_plot.py

# How many characters saved
git fetch origin
python src/analyze_refactor.py

# Delete __pycache__ directories
find . -type d -name __pycache__ -prune -exec rm -r {} +
