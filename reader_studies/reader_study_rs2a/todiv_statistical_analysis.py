"""
TODIV Score Statistical Analysis (Reader Study 2A)

Compares DermFM-Zero zero-shot performance against clinicians on the TODIV
(Test Of Dermoscopy for International Validation) platform.

Input:  todiv_scores.xlsx (sheets: Humans, Baseline, DermFM-Zero)
Output: todiv_analysis_results.txt, todiv_boxplots.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
import argparse
import os
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================
# Configuration
# ==============================
parser = argparse.ArgumentParser(description='TODIV Statistical Analysis')
parser.add_argument('--real', action='store_const', const='real', dest='mode', help='Use real data (real_data/)')
parser.add_argument('--demo', action='store_const', const='demo', dest='mode', help='Use demo data (demo_data/)')
parser.set_defaults(mode='demo')
args = parser.parse_args()

if args.mode == 'real':
    FILE_PATH = 'real_data/todiv_scores.xlsx'
    OUTPUT_DIR = 'real_output'
else:
    FILE_PATH = 'demo_data/todiv_scores.xlsx'
    OUTPUT_DIR = 'demo_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_TXT = os.path.join(OUTPUT_DIR, 'todiv_analysis_results.txt')

# ==============================
# Dual output (terminal + file)
# ==============================
class Tee:
    """Redirect print to both terminal and file."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


log_file = open(OUTPUT_TXT, 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

print("=" * 80)
print("TODIV Score Statistical Analysis Report")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Source file: {FILE_PATH}")
print("=" * 80)
print()

# ==============================
# Load data
# ==============================
# Sheet name mapping: supports both release names and internal codenames
SHEET_ALIASES = {
    'Humans': ['Humans'],
    'Baseline': ['Baseline', 'Ypsono'],
    'DermFM-Zero': ['DermFM-Zero', 'Milkk10 Zero Shot', 'Milk10 Zero Shot'],
}


def load_sheet(filepath, canonical_name):
    """Load a sheet by trying canonical name first, then known aliases."""
    for alias in SHEET_ALIASES[canonical_name]:
        try:
            return pd.read_excel(filepath, sheet_name=alias)
        except ValueError:
            continue
    raise ValueError(f"No matching sheet found for '{canonical_name}'. "
                     f"Tried: {SHEET_ALIASES[canonical_name]}")


print("Reading Excel file...")
try:
    df_humans = load_sheet(FILE_PATH, 'Humans')
    df_baseline = load_sheet(FILE_PATH, 'Baseline')
    df_dermfm = load_sheet(FILE_PATH, 'DermFM-Zero')
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File '{FILE_PATH}' not found!")
    sys.stdout = original_stdout
    log_file.close()
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.stdout = original_stdout
    log_file.close()
    exit(1)


# ==============================
# Helper functions
# ==============================
def calculate_ci(data, confidence=0.95):
    """Calculate mean and 95% confidence interval using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - ci, mean + ci


def perform_ttest(group1, group2):
    """Two-sided independent samples t-test."""
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value


# ==============================
# Prepare data groups
# ==============================
humans_all = df_humans['score'].values
gps = df_humans[df_humans['Job'] == 'General Practitioner']['score'].values
dermatologists = df_humans[df_humans['Job'] == 'Dermatologist']['score'].values

exp_less1 = df_humans[df_humans['Since when'] == '< 1 year']['score'].values
exp_1to3 = df_humans[df_humans['Since when'] == '1-3 years']['score'].values
exp_3to10 = df_humans[df_humans['Since when'] == '3-10 years']['score'].values
exp_more10 = df_humans[df_humans['Since when'] == '> 10 years']['score'].values

baseline_scores = df_baseline['Tableau 1'].values
dermfm_scores = df_dermfm['Tableau 1'].values

# ==============================
# Statistical analysis
# ==============================
print("\n" + "=" * 80)
print("DETAILED STATISTICAL ANALYSIS")
print("=" * 80)

print("\n" + "-" * 80)
print("1. PERFORMANCE BY JOB CATEGORY")
print("-" * 80)

groups = {
    'Humans (All)': humans_all,
    'General Practitioners': gps,
    'Dermatologists': dermatologists,
    'Baseline (AI)': baseline_scores,
    'DermFM-Zero': dermfm_scores
}

groups_stats = {}
for name, scores in groups.items():
    mean, ci_low, ci_high = calculate_ci(scores)
    groups_stats[name] = {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': len(scores)}
    print(f"{name:25s}: {mean:6.2f}%  (95% CI: {ci_low:6.2f} - {ci_high:6.2f})  n={len(scores)}")

print("\n" + "-" * 80)
print("2. PERFORMANCE BY EXPERIENCE LEVEL")
print("-" * 80)

experience_groups = {
    'Humans (All)': humans_all,
    '<1 year': exp_less1,
    '1-3 years': exp_1to3,
    '3-10 years': exp_3to10,
    '>10 years': exp_more10,
    'Baseline (AI)': baseline_scores,
    'DermFM-Zero': dermfm_scores
}

for name, scores in experience_groups.items():
    mean, ci_low, ci_high = calculate_ci(scores)
    print(f"{name:25s}: {mean:6.2f}%  (95% CI: {ci_low:6.2f} - {ci_high:6.2f})  n={len(scores)}")

print("\n" + "-" * 80)
print("3. P-VALUES (DermFM-Zero vs Others, two-sided t-test)")
print("-" * 80)

comparisons = [
    ('Humans (All)', humans_all),
    ('General Practitioners', gps),
    ('Dermatologists', dermatologists),
    ('<1 year experience', exp_less1),
    ('1-3 years experience', exp_1to3),
    ('3-10 years experience', exp_3to10),
    ('>10 years experience', exp_more10),
    ('Baseline (AI)', baseline_scores)
]

for name, scores in comparisons:
    t_stat, p_val = perform_ttest(dermfm_scores, scores)
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"DermFM-Zero vs {name:25s}: t = {t_stat:7.3f},  p = {p_val:.6e}  {significance}")

print("\n" + "-" * 80)
print("4. KEY FINDINGS")
print("-" * 80)
print(f"  DermFM-Zero achieves {groups_stats['DermFM-Zero']['mean']:.2f}% accuracy")
print(f"  vs Dermatologists: {groups_stats['DermFM-Zero']['mean'] - groups_stats['Dermatologists']['mean']:+.2f} pp")
print(f"  vs Humans (All):   {groups_stats['DermFM-Zero']['mean'] - groups_stats['Humans (All)']['mean']:+.2f} pp")

print("\n" + "-" * 80)
print("5. SUMMARY TABLE")
print("-" * 80)
print(f"{'Group':<25} {'Accuracy (%)':<15} {'95% CI':<20} {'N':<10} {'p vs DermFM-Zero'}")
print("-" * 80)
for name in ['DermFM-Zero', 'Dermatologists', 'Humans (All)', 'General Practitioners', 'Baseline (AI)']:
    s = groups_stats[name]
    if name == 'DermFM-Zero':
        p_str = "—"
    else:
        _, p_val = perform_ttest(dermfm_scores, groups[name])
        p_str = f"p < 0.001***" if p_val < 0.001 else f"p = {p_val:.4f}"
    print(f"{name:<25} {s['mean']:<15.2f} {s['ci_low']:.2f}-{s['ci_high']:.2f}{'':>5} {s['n']:<10} {p_str}")

print("\n" + "=" * 80)
print("End of Report")
print("=" * 80)

# ==============================
# Generate figure
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel A: Performance by Group
box_data_a = [humans_all, gps, dermatologists, baseline_scores, dermfm_scores]
labels_a = ['Humans\n(All)', 'General\nPractitioners', 'Dermatologists', 'Baseline\n(AI)', 'DermFM-Zero']
colors_a = ['#2E86AB', '#A23B72', '#F18F01', '#8B8680', '#3F7D20']

bp1 = axes[0].boxplot(box_data_a, labels=labels_a, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp1['boxes'], colors_a):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('TODIV Score (%)', fontsize=12)
axes[0].set_title('A. Performance by Group', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Panel B: Performance by Experience
box_data_b = [humans_all, exp_less1, exp_1to3, exp_3to10, exp_more10, baseline_scores, dermfm_scores]
labels_b = ['Humans\n(All)', '<1 yr', '1-3 yr', '3-10 yr', '>10 yr', 'Baseline\n(AI)', 'DermFM-Zero']
colors_b = ['#2E86AB', '#C73E1D', '#C73E1D', '#C73E1D', '#C73E1D', '#8B8680', '#3F7D20']

bp2 = axes[1].boxplot(box_data_b, labels=labels_b, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp2['boxes'], colors_b):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('TODIV Score (%)', fontsize=12)
axes[1].set_title('B. Performance by Experience', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'todiv_boxplots.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")

# Restore stdout
sys.stdout = original_stdout
log_file.close()
print(f"Results saved to: {OUTPUT_TXT}")
