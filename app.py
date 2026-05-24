"""
ChurnZero Dashboard - Flask Web Application
Serves predictions, metrics, and visualizations on localhost.
"""

import os
import io
import base64
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
_cache = {}

def get_train_data():
    if 'train' not in _cache:
        _cache['train'] = pd.read_csv('ChurnZero_dataset_v1.csv')
    return _cache['train']

def get_test_data():
    if 'test' not in _cache:
        _cache['test'] = pd.read_csv('ChurnZero_test_v1.csv')
    return _cache['test']

def get_predictions():
    if 'preds' not in _cache:
        _cache['preds'] = pd.read_csv('ChurnZero_Antigravity_Predictions.csv')
    return _cache['preds']

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------
DARK_BG   = '#0F172A'
CARD_BG   = '#1E293B'
NAVY      = '#1D3557'
CORAL     = '#E63946'
TEAL      = '#10B981'
SLATE400  = '#94A3B8'
SLATE200  = '#E2E8F0'
WHITE     = '#F8FAFC'

def _style_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SLATE400)
    ax.spines['bottom'].set_color(SLATE400)
    ax.tick_params(colors=SLATE200, labelsize=9)
    ax.xaxis.label.set_color(SLATE200)
    ax.yaxis.label.set_color(SLATE200)
    ax.title.set_color(WHITE)

def make_class_dist_chart():
    df = get_train_data()
    counts = df['churn'].value_counts().sort_index()
    pcts = df['churn'].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    _style_ax(ax)

    bars = ax.bar(['Active (0)', 'Churned (1)'], counts, color=[NAVY, CORAL], width=0.55, edgecolor='none')
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.annotate(f'{cnt:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=WHITE)
    ax.set_title('Training Set Class Distribution', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_ylim(0, max(counts)*1.18)
    return fig_to_base64(fig)

def make_prediction_dist_chart():
    preds = get_predictions()
    counts = preds['churn_prediction'].value_counts().sort_index()
    pcts = preds['churn_prediction'].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    _style_ax(ax)

    bars = ax.bar(['Active (0)', 'Churned (1)'], counts, color=[TEAL, CORAL], width=0.55, edgecolor='none')
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.annotate(f'{cnt:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=WHITE)
    ax.set_title('Test Set Predicted Distribution', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_ylim(0, max(counts)*1.18)
    return fig_to_base64(fig)

def make_probability_hist():
    preds = get_predictions()

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    _style_ax(ax)

    ax.hist(preds['churn_probability'], bins=50, color=NAVY, edgecolor=CARD_BG, linewidth=0.5)
    ax.axvline(x=0.031, color=CORAL, linestyle='--', linewidth=1.5, label='Threshold = 0.031')
    ax.set_title('Predicted Churn Probability Distribution', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Probability', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(facecolor=CARD_BG, edgecolor=SLATE400, labelcolor=SLATE200, fontsize=8)
    return fig_to_base64(fig)

def make_feature_importance_chart():
    # Hardcoded from our trained model results (top 10 features)
    features = [
        ('Unresolved Complaints', 0.2402),
        ('Balance Decline %', 0.1548),
        ('Total Digital Logins', 0.0726),
        ('RM Interactions', 0.0405),
        ('Total Trans Count', 0.0394),
        ('Escalation Count', 0.0386),
        ('Avg Monthly Balance', 0.0368),
        ('Campaign Responses', 0.0329),
        ('Cash Withdrawals', 0.0316),
        ('EMI Payment Delays', 0.0266),
    ]
    names = [f[0] for f in features][::-1]
    vals  = [f[1] for f in features][::-1]
    colors = [CORAL if v >= 0.10 else NAVY for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor(CARD_BG)
    _style_ax(ax)

    bars = ax.barh(names, vals, color=colors, height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.1%}', xy=(w, bar.get_y()+bar.get_height()/2),
                    xytext=(4, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, color=SLATE200)
    ax.set_title('Top 10 Churn Drivers (XGBoost)', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Relative Importance', fontsize=9)
    ax.set_xlim(0, max(vals)*1.15)
    return fig_to_base64(fig)

def make_cost_comparison_chart():
    labels = ['Default\n(t=0.50)', 'Optimized\n(t=0.031)']
    costs = [440500, 15000]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    _style_ax(ax)

    bars = ax.bar(labels, costs, color=[CORAL, TEAL], width=0.5, edgecolor='none')
    for bar, c in zip(bars, costs):
        ax.annotate(f'INR {c:,}',
                    xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=WHITE)
    ax.set_title('Business Cost Comparison', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Total Cost (INR)', fontsize=9)
    ax.set_ylim(0, max(costs)*1.18)
    return fig_to_base64(fig)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def api_summary():
    preds = get_predictions()
    train = get_train_data()
    return jsonify({
        'train_rows': len(train),
        'train_features': train.shape[1] - 2,  # minus customer_id and churn
        'test_rows': len(get_test_data()),
        'pred_rows': len(preds),
        'churn_rate_train': f"{train['churn'].mean()*100:.1f}%",
        'churn_rate_pred': f"{preds['churn_prediction'].mean()*100:.1f}%",
        'pred_churners': int(preds['churn_prediction'].sum()),
        'pred_active': int((preds['churn_prediction'] == 0).sum()),
        'pr_auc': '0.99995',
        'f1_default': '0.99537',
        'f1_optimized': '0.98861',
        'threshold': '0.0310',
        'cost_default': 'INR 440,500',
        'cost_optimized': 'INR 15,000',
        'cost_savings': 'INR 425,500',
        'cost_savings_pct': '96.6%',
        'top_driver': 'Unresolved Complaints (24.0%)',
        'null_count': int(preds.isnull().sum().sum()),
    })

@app.route('/api/charts')
def api_charts():
    return jsonify({
        'class_dist': make_class_dist_chart(),
        'pred_dist': make_prediction_dist_chart(),
        'prob_hist': make_probability_hist(),
        'feature_imp': make_feature_importance_chart(),
        'cost_compare': make_cost_comparison_chart(),
    })

@app.route('/api/predictions')
def api_predictions():
    preds = get_predictions()
    # Return first 100 rows for the table
    rows = preds.head(100).to_dict(orient='records')
    return jsonify(rows)

if __name__ == '__main__':
    print("\n  ChurnZero Dashboard starting...")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
