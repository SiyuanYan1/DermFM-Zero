"""
Reader Study 1: GP AI-Assisted Diagnostic Accuracy
Unified analysis for CN and EN cohorts (30 readers total).
Produces: comprehensive statistics CSV, Nature-style 6-panel figure.

Figure layout (matching original):
  Row 1: (a) Dx Rubric | (b) Mgmt Rubric | (c) Top-3 Violin
  Row 2: (d) Dx Score Violin | (e) Mgmt Distribution (stacked) | (f) Confidence
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(description='RS1 Statistical Analysis')
parser.add_argument('--real', action='store_const', const='real', dest='mode',
                    help='Use real data (real_data/)')
parser.add_argument('--demo', action='store_const', const='demo', dest='mode',
                    help='Use demo data (demo_data/)')
parser.add_argument('--exclude_cases', metavar='CSV', default=None,
                    help='Sensitivity analysis: CSV with a case_id column; '
                         'those cases are excluded and outputs go to '
                         '{OUTPUT_DIR}/sensitivity/')
parser.set_defaults(mode='real')
args = parser.parse_args()

if args.mode == 'real':
    DATA_DIR = 'real_data'
    OUTPUT_DIR = 'real_output'
else:
    DATA_DIR = 'demo_data'
    OUTPUT_DIR = 'demo_output'

if args.exclude_cases:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'sensitivity')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION — Nature Style
# ============================================================================
C_UNAIDED = "#4E79A7"
C_ASSISTED = "#E15759"
COLOR_UNAIDED_LIGHT = '#A8C5DD'
COLOR_ASSISTED_LIGHT = '#F4A8AA'

# Management category colors
COLOR_DANGER = '#E15759'
COLOR_HARMLESS = '#F4A8AA'
COLOR_ADEQUATE = '#A8C5DD'
COLOR_EXCELLENT = '#2E5A87'

# Rubric color scales
COLORS_DX = ['#E15759', '#F4A8AA', '#CBCBCB', '#A8C5DD', '#2E5A87']
COLORS_MGMT = [COLOR_DANGER, COLOR_HARMLESS, COLOR_ADEQUATE, COLOR_EXCELLENT]

MGMT_MAP = {
    'Inadequate and dangerous': 1,
    'Inadequate but harmless': 2,
    'Adequate': 3,
    'Perfect': 4
}

from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ============================================================================
# DATA LOADING
# ============================================================================
print(f"{'='*70}")
print(f"RS1 ANALYSIS — mode: {args.mode}")
print(f"{'='*70}")

_all = pd.read_csv(os.path.join(DATA_DIR, 'rs1_reader_data.csv'))
if args.exclude_cases:
    _excl = set(pd.read_csv(args.exclude_cases)['case_id'])
    _n0 = len(_all)
    _all = _all[~_all['Case ID'].isin(_excl)].copy()
    print(f"Sensitivity analysis: excluded {sorted(_excl)} "
          f"({_n0} -> {len(_all)} observations)")
cn = _all[_all['Cohort'] == 'CN'].copy()
en = _all[_all['Cohort'] == 'EN'].copy()

# Prefix Responder_ID to avoid collisions between sites
cn['Responder_ID'] = 'CN_' + cn['Responder_ID'].astype(str)
en['Responder_ID'] = 'EN_' + en['Responder_ID'].astype(str)

# Combine on shared columns
shared_cols = list(set(cn.columns) & set(en.columns))

df = pd.concat([cn[shared_cols], en[shared_cols]], ignore_index=True)

print(f"Loaded: {len(df)} cases from {df['Responder_ID'].nunique()} readers")
print(f"  CN: {cn['Responder_ID'].nunique()} readers, {len(cn)} cases")
print(f"  EN: {en['Responder_ID'].nunique()} readers, {len(en)} cases")

# ============================================================================
# DATA PREPARATION
# ============================================================================
df['Unaided_Mgmt_Grade'] = df['Unaided_Mgmt_Grade'].astype(str).str.strip().replace('adequate', 'Adequate')
df['Assisted_Mgmt_Grade'] = df['Assisted_Mgmt_Grade'].astype(str).str.strip().replace('adequate', 'Adequate')
df['Unaided_Mgmt_Score'] = df['Unaided_Mgmt_Grade'].map(MGMT_MAP)
df['Assisted_Mgmt_Score'] = df['Assisted_Mgmt_Grade'].map(MGMT_MAP)

# Ensure confidence columns are numeric
for col in ['Unaided_Dx_Confidence', 'Assisted_Dx_Confidence',
            'Unaided_Mgmt_Confidence', 'Assisted_Mgmt_Confidence']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


def get_harm_rate(x):
    return (x == 1).sum() / len(x)


def get_success_rate(x):
    return (x >= 3).sum() / len(x)


# Reader-level aggregation (including confidence)
reader_stats = df.groupby('Responder_ID').agg({
    'Unaided_Dx_Score': 'mean',
    'Assisted_Dx_Score': 'mean',
    'Unaided_Top3_SpotOn': 'mean',
    'Assisted_Top3_SpotOn': 'mean',
    'Unaided_Mgmt_Score': ['mean', get_harm_rate, get_success_rate],
    'Assisted_Mgmt_Score': ['mean', get_harm_rate, get_success_rate],
    'Unaided_Dx_Confidence': 'mean',
    'Assisted_Dx_Confidence': 'mean',
    'Unaided_Mgmt_Confidence': 'mean',
    'Assisted_Mgmt_Confidence': 'mean',
}).reset_index()

reader_stats.columns = [
    'ID', 'Dx_U', 'Dx_A', 'Top3_U', 'Top3_A',
    'MgmtScore_U', 'Harm_U', 'Success_U',
    'MgmtScore_A', 'Harm_A', 'Success_A',
    'DxConf_U', 'DxConf_A', 'MgmtConf_U', 'MgmtConf_A',
]

print(f"Reader-level stats: {len(reader_stats)} readers")


# ============================================================================
# COMPREHENSIVE STATISTICS
# ============================================================================
def calculate_statistics(data, col_unaided, col_assisted, metric_name, lower_is_better=False):
    """Comprehensive paired statistics with CI, effect size, and improvement.

    Inference: two-sided Wilcoxon signed-rank test (no directional assumption).
    `lower_is_better` is retained for downstream visual cues only and does not
    affect the statistical test.
    """
    data_clean = data[[col_unaided, col_assisted]].dropna()
    if len(data_clean) < 2:
        return {'Metric': metric_name, 'n': 0}

    unaided = data_clean[col_unaided].values
    assisted = data_clean[col_assisted].values
    n = len(unaided)

    unaided_mean = np.mean(unaided)
    unaided_std = np.std(unaided, ddof=1)
    unaided_median = np.median(unaided)
    unaided_iqr = np.percentile(unaided, 75) - np.percentile(unaided, 25)
    unaided_ci = stats.t.interval(0.95, n - 1, loc=unaided_mean, scale=stats.sem(unaided))

    assisted_mean = np.mean(assisted)
    assisted_std = np.std(assisted, ddof=1)
    assisted_median = np.median(assisted)
    assisted_iqr = np.percentile(assisted, 75) - np.percentile(assisted, 25)
    assisted_ci = stats.t.interval(0.95, n - 1, loc=assisted_mean, scale=stats.sem(assisted))

    mean_diff = assisted_mean - unaided_mean
    improvement_pct = (mean_diff / unaided_mean) * 100 if unaided_mean != 0 else np.nan

    # Two-sided Wilcoxon signed-rank test.
    try:
        stat_two, p_two = stats.wilcoxon(unaided, assisted, alternative='two-sided')
        p_val = p_two
        # |z| corresponding to two-sided p; effect-size r = |z| / sqrt(n)
        z_score = stats.norm.ppf(1 - p_two / 2) if p_two > 0 else np.inf
        effect_size_r = z_score / np.sqrt(n)
        p_text = '***' if p_two < 0.001 else ('**' if p_two < 0.01 else ('*' if p_two < 0.05 else 'ns'))
    except Exception:
        p_two, p_val, effect_size_r, p_text = np.nan, np.nan, np.nan, 'Error'

    return {
        'Metric': metric_name, 'n': n,
        'Unaided_Mean': unaided_mean, 'Unaided_SD': unaided_std,
        'Unaided_Median': unaided_median, 'Unaided_IQR': unaided_iqr,
        'Unaided_95CI_Lower': unaided_ci[0], 'Unaided_95CI_Upper': unaided_ci[1],
        'Assisted_Mean': assisted_mean, 'Assisted_SD': assisted_std,
        'Assisted_Median': assisted_median, 'Assisted_IQR': assisted_iqr,
        'Assisted_95CI_Lower': assisted_ci[0], 'Assisted_95CI_Upper': assisted_ci[1],
        'Mean_Difference': mean_diff, 'P_Value': p_val,
        'P_TwoSided_Wilcoxon': p_two,
        'P_Value_Text': p_text, 'Effect_Size_r': effect_size_r,
        'Improvement_Pct': improvement_pct,
    }


# 4th element: lower_is_better (True for Harm Rate; False for everything else).
metrics = [
    ('Dx_U', 'Dx_A', 'Diagnostic Accuracy Score', False),
    ('MgmtScore_U', 'MgmtScore_A', 'Management Quality Score', False),
    ('Top3_U', 'Top3_A', 'Top-3 Diagnostic Utility', False),
    ('Harm_U', 'Harm_A', 'Harm Rate (Dangerous Management)', True),
    ('Success_U', 'Success_A', 'Success Rate (Adequate or Perfect)', False),
    ('DxConf_U', 'DxConf_A', 'Diagnosis Confidence', False),
    ('MgmtConf_U', 'MgmtConf_A', 'Management Confidence', False),
]

results = []
for col_u, col_a, name, lower_is_better in metrics:
    r = calculate_statistics(reader_stats, col_u, col_a, name, lower_is_better=lower_is_better)
    results.append(r)
    sig = r.get('P_Value_Text', '')
    print(f"  {name}: {r.get('Unaided_Mean', 0):.3f} -> {r.get('Assisted_Mean', 0):.3f}, "
          f"p={r.get('P_Value', float('nan')):.4f} {sig}")

stats_df = pd.DataFrame(results)
stats_path = os.path.join(OUTPUT_DIR, 'rs1_statistics.csv')
stats_df.to_csv(stats_path, index=False, float_format='%.4f')
print(f"\nStatistics CSV saved: {stats_path}")
print(stats_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# ============================================================================
# CASE-LEVEL McNEMAR for Harm Rate & Success Rate
# (proportions match Figure 3e; p-values populate Extended Data Table 10 †rows)
# ============================================================================
def mcnemar_paired(u_bin, a_bin):
    """
    Paired McNemar test on binary outcomes (two-sided).
    Returns dict with b, c, n_disc, p_two.
    Continuity-corrected chi-square for n_disc >= 25; exact binomial otherwise.
    """
    b = int(((u_bin == True) & (a_bin == False)).sum())   # only unaided positive
    c = int(((u_bin == False) & (a_bin == True)).sum())   # only assisted positive
    n_disc = b + c
    if n_disc == 0:
        return {'b': b, 'c': c, 'n_disc': 0, 'p_two': 1.0}

    if n_disc >= 25:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc            # continuity-corrected
        p_two = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        p_two = min(2 * stats.binom.cdf(min(b, c), n_disc, 0.5), 1.0)

    return {'b': b, 'c': c, 'n_disc': n_disc, 'p_two': p_two}


print(f"\n{'='*70}\nCASE-LEVEL McNEMAR (matches Figure 3e proportions)\n{'='*70}")

sub = df[['Unaided_Mgmt_Score', 'Assisted_Mgmt_Score']].dropna()
n_cases = len(sub)
harm_u = (sub['Unaided_Mgmt_Score'] == 1)
harm_a = (sub['Assisted_Mgmt_Score'] == 1)
succ_u = (sub['Unaided_Mgmt_Score'] >= 3)
succ_a = (sub['Assisted_Mgmt_Score'] >= 3)

print(f"  n cases = {n_cases}")
print(f"  Harm    (case-level): {harm_u.mean():.1%} -> {harm_a.mean():.1%}")
print(f"  Success (case-level): {succ_u.mean():.1%} -> {succ_a.mean():.1%}")

mn_h = mcnemar_paired(harm_u, harm_a)      # Harm rate (two-sided)
mn_s = mcnemar_paired(succ_u, succ_a)      # Success rate (two-sided)

print(f"\n  Harm Rate    McNemar (two-sided): b={mn_h['b']}  c={mn_h['c']}  "
      f"n_disc={mn_h['n_disc']}  p_two={mn_h['p_two']:.4f}")
print(f"  Success Rate McNemar (two-sided): b={mn_s['b']}  c={mn_s['c']}  "
      f"n_disc={mn_s['n_disc']}  p_two={mn_s['p_two']:.4f}")

case_level_path = os.path.join(OUTPUT_DIR, 'rs1_caselevel_mcnemar.csv')
pd.DataFrame([
    {'Metric': 'Harm Rate', 'n_cases': n_cases,
     'Unaided_%': harm_u.mean() * 100, 'Assisted_%': harm_a.mean() * 100,
     'McNemar_b_only_unaided': mn_h['b'], 'McNemar_c_only_assisted': mn_h['c'],
     'n_discordant': mn_h['n_disc'],
     'P_TwoSided_McNemar': mn_h['p_two']},
    {'Metric': 'Success Rate', 'n_cases': n_cases,
     'Unaided_%': succ_u.mean() * 100, 'Assisted_%': succ_a.mean() * 100,
     'McNemar_b_only_unaided': mn_s['b'], 'McNemar_c_only_assisted': mn_s['c'],
     'n_discordant': mn_s['n_disc'],
     'P_TwoSided_McNemar': mn_s['p_two']},
]).to_csv(case_level_path, index=False, float_format='%.4f')
print(f"\nCase-level McNemar saved: {case_level_path}")


# ============================================================================
# VISUALIZATION FUNCTIONS (matching original Nature style)
# ============================================================================

def draw_rubric(ax, labels, colors, title, scores=None):
    """Rubric ladder diagram (panels a, b)."""
    ax.axis('off')
    n = len(labels)

    for i, (label, color) in enumerate(zip(labels, colors)):
        y_pos = i * 0.16
        rect = patches.Rectangle(
            (0.1, y_pos), 0.8, 0.14,
            facecolor=color, edgecolor='white',
            linewidth=2, alpha=0.95
        )
        ax.add_patch(rect)

        ax.text(0.5, y_pos + 0.07, label,
                ha='center', va='center', fontsize=13, color='white')

        score_num = scores[i] if scores else (i + 1)
        ax.text(0.06, y_pos + 0.07, str(score_num),
                ha='center', va='center', fontsize=15, color='#333')

    ax.set_title(title, fontsize=15, loc='left', pad=10)
    ax.set_ylim(-0.02, n * 0.16)
    ax.set_xlim(0, 1)


def create_violin_plot(ax, data, col_u, col_a, title, ylabel, ylim,
                       is_proportion=False, lower_is_better=False):
    """Violin + box + paired lines panel (matching original res1.py style).

    P-value computed via two-sided Wilcoxon signed-rank test.
    `lower_is_better` is retained for visual cues only and does not affect the test.
    """
    y_u = data[col_u].dropna().values
    y_a = data[col_a].dropna().values
    min_len = min(len(y_u), len(y_a))
    y_u, y_a = y_u[:min_len], y_a[:min_len]

    positions = [1, 2]

    # 1. Violin plots
    parts = ax.violinplot([y_u, y_a], positions=positions, widths=0.6,
                          showmeans=False, showextrema=False, showmedians=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLOR_UNAIDED_LIGHT if i == 0 else COLOR_ASSISTED_LIGHT)
        pc.set_alpha(0.7)
        pc.set_edgecolor('none')

    # 2. Paired individual lines (subtle)
    for i in range(len(y_u)):
        ax.plot([1, 2], [y_u[i], y_a[i]],
                color='#999', alpha=0.08, linewidth=0.6, zorder=1)

    # 3. Box plots
    bp = ax.boxplot([y_u, y_a], positions=positions, widths=0.25,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    boxprops=dict(linewidth=1.5),
                    capprops=dict(color='#444', linewidth=1.5),
                    whiskerprops=dict(color='#444', linewidth=1.5))
    for patch, color in zip(bp['boxes'], [C_UNAIDED, C_ASSISTED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # 4. Mean markers and connecting line
    m_u, m_a = np.mean(y_u), np.mean(y_a)
    ax.plot([1, 2], [m_u, m_a], color='black', linewidth=3, zorder=10)
    ax.scatter([1, 2], [m_u, m_a],
               color='white', edgecolor='black',
               s=100, zorder=11, linewidth=2.5)

    # 5. Mean value labels with background box
    fmt = "{:.1%}" if is_proportion else "{:.2f}"
    offset_u = (ylim[1] - ylim[0]) * 0.08

    ax.text(1, m_u - offset_u, fmt.format(m_u),
            ha='center', va='top', fontsize=12, color='#222',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='none', alpha=0.9))
    ax.text(2, m_a - offset_u, fmt.format(m_a),
            ha='center', va='top', fontsize=12, color=C_ASSISTED,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='none', alpha=0.9))

    # 6. P-value with significance bracket (two-sided Wilcoxon signed-rank)
    try:
        stat, p = stats.wilcoxon(y_u, y_a, alternative='two-sided')
        p_text = 'P<0.001' if p < 0.001 else f'P={p:.3f}'

        bar_h = ylim[1] - (ylim[1] - ylim[0]) * 0.08
        ax.plot([1.1, 1.9], [bar_h, bar_h], c='#222', lw=2)
        ax.text(1.5, bar_h + (ylim[1] - ylim[0]) * 0.02, p_text,
                ha='center', va='bottom', fontsize=12)
    except Exception:
        pass

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(['Unaided', 'AI-Assisted'], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, loc='left', fontsize=15, pad=10)
    ax.set_ylim(ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Sample size annotation
    ax.text(0.98, 0.02, f'n={len(y_u)}',
            transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom', color='#555')


def plot_management_distribution(ax, df_raw):
    """Stacked bar chart for management grade distribution (panel e)."""
    cats = ['Inadequate and dangerous', 'Inadequate but harmless', 'Adequate', 'Perfect']

    counts_u = df_raw['Unaided_Mgmt_Grade'].value_counts(normalize=True).reindex(cats, fill_value=0)
    counts_a = df_raw['Assisted_Mgmt_Grade'].value_counts(normalize=True).reindex(cats, fill_value=0)

    # Group into 3 categories for cleaner visualization
    def get_grouped(series):
        danger = series['Inadequate and dangerous']
        harmless = series['Inadequate but harmless']
        adequate = series['Adequate'] + series['Perfect']
        return [danger, harmless, adequate]

    u_vals = get_grouped(counts_u)
    a_vals = get_grouped(counts_a)

    bar_width = 0.55
    indices = [1, 2]
    colors = [COLOR_DANGER, COLOR_HARMLESS, COLOR_ADEQUATE]
    labels = ['Inadequate/Dangerous', 'Inadequate/Harmless', 'Adequate or Perfect']

    # Stacked bars
    bottoms_u = [0, u_vals[0], u_vals[0] + u_vals[1]]
    bottoms_a = [0, a_vals[0], a_vals[0] + a_vals[1]]

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.bar(indices, [u_vals[i], a_vals[i]], width=bar_width,
               bottom=[bottoms_u[i], bottoms_a[i]],
               color=color, alpha=0.9, label=label,
               edgecolor='white', linewidth=2)

    # Percentage labels inside bars
    for x, vals, bottoms in zip(indices, [u_vals, a_vals], [bottoms_u, bottoms_a]):
        for i, (val, bottom) in enumerate(zip(vals, bottoms)):
            if val > 0.04:
                text_color = 'white' if i != 1 else '#222'
                ax.text(x, bottom + val / 2, f'{val:.1%}',
                        ha='center', va='center',
                        color=text_color, fontsize=12)

    # Formatting
    ax.set_xticks(indices)
    ax.set_xticklabels(['Unaided', 'AI-Assisted'], fontsize=12)
    ax.set_ylabel('Proportion of Decisions', fontsize=13)
    ax.set_title('e. Management Quality Distribution', loc='left', fontsize=15, pad=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Legend
    ax.legend(loc='upper center', frameon=False, fontsize=10.5,
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              columnspacing=1.5, handlelength=1.5)

    # Harm reduction arrow
    if u_vals[0] > a_vals[0]:
        arrow_x = 1.5
        y_u = u_vals[0] / 2
        y_a = a_vals[0] / 2
        ax.annotate('', xy=(arrow_x, y_a), xytext=(arrow_x, y_u),
                    arrowprops=dict(arrowstyle='->', color=COLOR_DANGER,
                                    lw=3, shrinkA=2, shrinkB=2))
        ax.text(arrow_x + 0.15, (y_u + y_a) / 2, 'Harm\nReduced',
                color=COLOR_DANGER, va='center', ha='left', fontsize=10.5)


def plot_confidence_comparison(ax, data, title, lower_is_better=False):
    """Dual-group confidence violin panel (panel f).
    P-values computed via two-sided Wilcoxon signed-rank test.
    """
    dx_u = data['DxConf_U'].values
    dx_a = data['DxConf_A'].values
    mgmt_u = data['MgmtConf_U'].values
    mgmt_a = data['MgmtConf_A'].values

    positions = [[1, 2], [3.5, 4.5]]
    width = 0.6

    all_data = [[dx_u, dx_a], [mgmt_u, mgmt_a]]
    group_labels = ['Diagnosis', 'Management']

    for group_idx, (group_pos, group_data) in enumerate(zip(positions, all_data)):
        # Violins
        parts = ax.violinplot(group_data, positions=group_pos, widths=width,
                              showmeans=False, showextrema=False, showmedians=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLOR_UNAIDED_LIGHT if i == 0 else COLOR_ASSISTED_LIGHT)
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')

        # Box plots
        bp = ax.boxplot(group_data, positions=group_pos, widths=width * 0.4,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2.5),
                        boxprops=dict(linewidth=1.5),
                        capprops=dict(color='#444', linewidth=1.5),
                        whiskerprops=dict(color='#444', linewidth=1.5))
        for patch, color in zip(bp['boxes'], [C_UNAIDED, C_ASSISTED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Mean line and markers
        m_u, m_a = np.mean(group_data[0]), np.mean(group_data[1])
        ax.plot(group_pos, [m_u, m_a], color='black', linewidth=3, zorder=10)
        ax.scatter(group_pos, [m_u, m_a], color='white', edgecolor='black',
                   s=100, zorder=11, linewidth=2.5)

        # Mean labels with background
        offset = 0.25
        ax.text(group_pos[0], m_u - offset, f'{m_u:.2f}',
                ha='center', va='top', fontsize=12, color='#222',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='none', alpha=0.9))
        ax.text(group_pos[1], m_a - offset, f'{m_a:.2f}',
                ha='center', va='top', fontsize=12, color=C_ASSISTED,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='none', alpha=0.9))

        # P-value with bracket (two-sided Wilcoxon signed-rank)
        try:
            stat, p = stats.wilcoxon(group_data[0], group_data[1], alternative='two-sided')
            p_text = 'P<0.001' if p < 0.001 else f'P={p:.3f}'
            bar_h = 4.2
            mid_x = np.mean(group_pos)
            ax.plot([group_pos[0] + 0.15, group_pos[1] - 0.15], [bar_h, bar_h],
                    c='#222', lw=2)
            ax.text(mid_x, bar_h + 0.1, p_text, ha='center', va='bottom', fontsize=12)
        except Exception:
            pass

    # Formatting
    all_positions = [pos for group in positions for pos in group]
    group_centers = [np.mean(pos) for pos in positions]

    ax.set_xticks(all_positions)
    ax.set_xticklabels(['Unaided', 'AI-Assisted'] * 2, fontsize=11)

    for center, label in zip(group_centers, group_labels):
        ax.text(center, -0.5, label, ha='center', va='top', fontsize=12)

    ax.set_ylabel('Confidence Score', fontsize=13)
    ax.set_title(title, loc='left', fontsize=15, pad=10)
    ax.set_ylim(1, 4.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.text(0.98, 0.02, f'n={len(dx_u)}',
            transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom', color='#555')


# ============================================================================
# GENERATE FIGURE — 6-panel Nature layout (original structure)
# ============================================================================
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 3,
                      width_ratios=[1.2, 1.2, 1.0],
                      wspace=0.28, hspace=0.38,
                      left=0.06, right=0.98, top=0.94, bottom=0.06)

# Row 1: Rubrics + Top-3
ax_a = fig.add_subplot(gs[0, 0])
draw_rubric(ax_a,
            labels=['Potential Harm', 'Different Class', 'Same Class',
                    'Generic Correct', 'Spot On'],
            colors=COLORS_DX,
            title='a. Diagnostic Accuracy Rubric')

ax_b = fig.add_subplot(gs[0, 1])
draw_rubric(ax_b,
            labels=['Inadequate/Dangerous', 'Inadequate/Harmless',
                    'Adequate', 'Perfect'],
            colors=COLORS_MGMT,
            title='b. Management Rubric')

ax_c = fig.add_subplot(gs[0, 2])
create_violin_plot(ax_c, reader_stats, 'Top3_U', 'Top3_A',
                   'c. Top-3 Diagnostic Utility',
                   'Proportion', (0, 0.85), is_proportion=True)

# Row 2: Dx Score + Mgmt Distribution + Confidence
ax_d = fig.add_subplot(gs[1, 0])
create_violin_plot(ax_d, reader_stats, 'Dx_U', 'Dx_A',
                   'd. Diagnostic Accuracy Distribution',
                   'Diagnostic Score', (1.0, 4.5), is_proportion=False)

ax_e = fig.add_subplot(gs[1, 1])
plot_management_distribution(ax_e, df)

ax_f = fig.add_subplot(gs[1, 2])
plot_confidence_comparison(ax_f, reader_stats, 'f. Confidence Enhancement')

# Save (PNG only)
fig_path = os.path.join(OUTPUT_DIR, 'rs1_figure.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
