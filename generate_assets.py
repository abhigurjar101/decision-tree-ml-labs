import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

assets_dir = str(Path(__file__).resolve().parent / "assets")
os.makedirs(assets_dir, exist_ok=True)

# 1. Entropy vs Gini vs Misclassification Error Curve
def plot_impurity_curves():
    p = np.linspace(0.0001, 0.9999, 500)
    gini = 2 * p * (1 - p)  # For binary: 1 - (p^2 + (1-p)^2) = 2p(1-p)
    entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    scaled_entropy = 0.5 * entropy
    misclass = 1 - np.maximum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(p, entropy, label=r'Entropy $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$', color='#ef4444', lw=2.5)
    ax.plot(p, scaled_entropy, label=r'Scaled Entropy $\frac{1}{2} H(p)$', color='#f59e0b', lw=2, linestyle='--')
    ax.plot(p, gini, label=r'Gini Impurity $2p(1-p)$ (Max = 0.5)', color='#3b82f6', lw=2.5)
    ax.plot(p, misclass, label=r'Misclassification Error $1 - \max(p, 1-p)$', color='#10b981', lw=2, linestyle=':')

    ax.set_title('Comparison of Impurity Measures (Binary Classification)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Probability of Class 1 ($p$)', fontsize=12, labelpad=10)
    ax.set_ylabel('Impurity Value', fontsize=12, labelpad=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.axvline(0.5, color='#94a3b8', linestyle='--', alpha=0.7)
    ax.text(0.51, 0.95, 'Max Uncertainty\n(p = 0.5)', color='#64748b', fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='lower center')
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "impurity_comparison.png"))
    plt.close()
    print("Saved impurity_comparison.png")

# 2. Overfitting Curve (Depth vs Train/Test Accuracy)
def plot_overfitting_curve():
    np.random.seed(42)
    # Generate non-linear classification dataset
    X = np.random.randn(400, 4)
    y = ((X[:, 0]**2 + X[:, 1] > 1.2) ^ (X[:, 2] > 0.5)).astype(int)
    # Add 10% label noise
    noise_idx = np.random.choice(len(y), size=int(0.1 * len(y)), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    depths = list(range(1, 16))
    train_accs = []
    test_accs = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        test_accs.append(accuracy_score(y_test, clf.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax.plot(depths, train_accs, marker='o', label='Training Accuracy', color='#3b82f6', lw=2.5)
    ax.plot(depths, test_accs, marker='s', label='Testing / Validation Accuracy', color='#ef4444', lw=2.5)

    best_depth = depths[np.argmax(test_accs)]
    best_acc = max(test_accs)
    ax.axvline(best_depth, color='#10b981', linestyle='--', lw=2, label=f'Optimal Depth (Depth={best_depth})')

    # Shading regions
    ax.axvspan(1, best_depth-0.5, color='#fef3c7', alpha=0.4, label='Underfitting Zone')
    ax.axvspan(best_depth+0.5, 15, color='#fee2e2', alpha=0.4, label='Overfitting Zone (High Variance)')

    ax.set_title('Decision Tree Complexity: Underfitting vs. Overfitting', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Tree Depth (max_depth)', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy Score', fontsize=12, labelpad=10)
    ax.set_xticks(depths)
    ax.set_ylim([0.55, 1.02])
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "overfitting_depth.png"))
    plt.close()
    print("Saved overfitting_depth.png")

# 3. Bank Loan Approval Tree Visualization
def plot_loan_approval_tree():
    np.random.seed(42)
    n = 600
    income = np.random.normal(55000, 18000, n).clip(20000, 120000)
    credit_score = np.random.normal(680, 70, n).clip(450, 850)
    debt = np.random.normal(15000, 8000, n).clip(0, 50000)
    loan_amount = np.random.normal(25000, 10000, n).clip(5000, 70000)

    # Probability of approval
    z = (credit_score - 640)/60 + (income - 45000)/25000 - (debt/20000) - (loan_amount/35000)
    p_approve = 1 / (1 + np.exp(-z))
    approved = (np.random.rand(n) < p_approve).astype(int)

    df = pd.DataFrame({
        'Income ($)': income.round(-2),
        'Credit_Score': credit_score.round().astype(int),
        'Existing_Debt ($)': debt.round(-2),
        'Loan_Amount ($)': loan_amount.round(-2),
        'Approved': approved
    })
    df.to_csv(str(Path(__file__).resolve().parent / "data" / "bank_loan_data.csv"), index=False)

    clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
    feature_cols = ['Income ($)', 'Credit_Score', 'Existing_Debt ($)', 'Loan_Amount ($)']
    clf.fit(df[feature_cols], df['Approved'])

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    plot_tree(clf, feature_names=feature_cols, class_names=['Denied', 'Approved'],
              filled=True, rounded=True, precision=2, fontsize=10, ax=ax,
              impurity=True, proportion=False)
    ax.set_title('Loan Approval Decision Tree (Max Depth = 3)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "loan_tree_viz.png"))
    plt.close()
    print("Saved loan_tree_viz.png")

    # Feature Importance
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()
    importances.plot(kind='barh', color='#3b82f6', ax=ax)
    ax.set_title('Feature Importance (Gini Impurity Decrease)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Relative Importance Score', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "feature_importance_loan.png"))
    plt.close()
    print("Saved feature_importance_loan.png")

# 4. House Price Regression Tree Visualization
def plot_house_price_tree():
    np.random.seed(42)
    n = 500
    area = np.random.normal(1800, 500, n).clip(700, 4000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    dist_center = np.random.uniform(2, 25, n)
    age = np.random.uniform(1, 40, n)

    # House price in $1,000s
    price = 120 + 0.14 * area + 25 * bedrooms - 4.5 * dist_center - 1.2 * age + np.random.normal(0, 25, n)
    price = price.clip(80, 800)

    df_reg = pd.DataFrame({
        'Area_sqft': area.round().astype(int),
        'Bedrooms': bedrooms,
        'Distance_KM': dist_center.round(1),
        'Property_Age': age.round().astype(int),
        'Price_k$': price.round(1)
    })
    df_reg.to_csv(str(Path(__file__).resolve().parent / "data" / "house_prices_data.csv"), index=False)

    reg = DecisionTreeRegressor(max_depth=3, criterion='squared_error', random_state=42)
    features = ['Area_sqft', 'Bedrooms', 'Distance_KM', 'Property_Age']
    reg.fit(df_reg[features], df_reg['Price_k$'])

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    plot_tree(reg, feature_names=features, filled=True, rounded=True, precision=1, fontsize=9, ax=ax)
    ax.set_title('House Price Regression Tree (Max Depth = 3, Criterion = MSE)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "house_regression_tree.png"))
    plt.close()
    print("Saved house_regression_tree.png")

# 5. Cost-Complexity Pruning Path (ccp_alpha)
def plot_pruning_path():
    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = ((X[:, 0] + X[:, 1]**2 - X[:, 2] > 0.8)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude maximum alpha which prunes entire tree

    clfs = []
    for alpha in ccp_alphas:
        c = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        c.fit(X_train, y_train)
        clfs.append(c)

    train_scores = [accuracy_score(y_train, c.predict(X_train)) for c in clfs]
    test_scores = [accuracy_score(y_test, c.predict(X_test)) for c in clfs]
    node_counts = [c.tree_.node_count for c in clfs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    ax1.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy', color='#3b82f6', drawstyle="steps-post")
    ax1.plot(ccp_alphas, test_scores, marker='s', label='Test Accuracy', color='#ef4444', drawstyle="steps-post")
    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    ax1.axvline(best_alpha, color='#10b981', linestyle='--', label=f'Best alpha ({best_alpha:.4f})')
    ax1.set_xlabel('Effective Alpha (ccp_alpha)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs. Alpha for Training & Testing', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)

    ax2.plot(ccp_alphas, node_counts, marker='d', color='#8b5cf6', drawstyle="steps-post")
    ax2.set_xlabel('Effective Alpha (ccp_alpha)', fontsize=11)
    ax2.set_ylabel('Total Number of Nodes', fontsize=11)
    ax2.set_title('Tree Complexity (Node Count) vs. Alpha', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "ccp_pruning_path.png"))
    plt.close()
    print("Saved ccp_pruning_path.png")

if __name__ == "__main__":
    plot_impurity_curves()
    plot_overfitting_curve()
    plot_loan_approval_tree()
    plot_house_price_tree()
    plot_pruning_path()
    print("All visual assets successfully generated!")
