import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cc_analysis import run_static_analysis, run_static_dynamic_analysis, static_results_to_latex

# ── Dates to sweep ───────────────────────────────────────────────────────────
#STATIC_DATES = ["11_30"]
STATIC_DATES = [
    "11_21", "11_22", "11_23", "11_24", "11_25",
    "11_28", "11_29", "11_30", "12_01", "12_02"
]

DYNAMIC_DATES = ["11_30"]
# DYNAMIC_DATES = [
#     "11_21", "11_22", "11_23", "11_24", "11_25",
#     "11_28", "11_29", "11_30", "12_01", "12_02"
# ]

if __name__ == "__main__":

    # # Available columns for plot_static_table: key -> (header label, formatter).
    # STATIC_COLUMNS = {
    #     'date':                    ('Date',                       lambda r: r['date'].replace('_', '/')),
    #     'sim_mape':                ('Sim MAPE',                   lambda r: f"{r['sim_mape']:.2f}%"),
    #     'tts_mape':                ('TTS MAPE',                   lambda r: f"{r['tts_mape']:.2f}%"),
    #     'cc':                      ('CC (%)',                     lambda r: f"{r['cc']:.2f}%"),
    #     'uncontrolled_tts':        ('Uncontrolled TTT (veh-hr)',  lambda r: f"{r['uncontrolled_tts']:.2f}"),
    #     'controlled_tts':          ('Controlled TTT (veh-hr)',    lambda r: f"{r['controlled_tts']:.2f}"),
    #     'avg_tt_reduced':          ('Avg TT Reduced/Veh (min)',   lambda r: f"{r['avg_tt_reduced']:.2f}"),
    #     'avg_delay_uncontrolled':  ('Avg Uncontrolled Delay/Veh (min)', lambda r: f"{r['avg_delay_uncontrolled']:.2f}"),
    #     'avg_delay_controlled':    ('Avg Controlled Delay/Veh (min)',   lambda r: f"{r['avg_delay_controlled']:.2f}"),
    # }

    print("── Static-only CC sweep ──────────────────────────────────")
    STATIC_TABLE_COLUMNS = ["date", "tts_mape", "uncontrolled_tts",
                            "controlled_tts", "cc", "avg_tt_reduced"]
    static_results = run_static_analysis(STATIC_DATES, columns=STATIC_TABLE_COLUMNS,
                                         save_path="figs/i24_cc_table.png")

    latex_table = static_results_to_latex(static_results, columns=STATIC_TABLE_COLUMNS)
    with open("figs/i24_cc_table.tex", "w") as f:
        f.write(latex_table)
    print("\nLaTeX table saved to: figs/i24_cc_table.tex\n")
    print(latex_table)

    # print("\n── Static + dynamic CC sweep ─────────────────────────────")
    # run_static_dynamic_analysis(DYNAMIC_DATES, save_path="figs/i24_cc_table_withdyn.png")
