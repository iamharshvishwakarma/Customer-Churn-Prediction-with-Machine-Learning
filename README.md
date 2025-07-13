<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-left: 4px solid #3498db;
            padding-left: 10px;
            margin-top: 30px;
        }
        .badge {
            display: inline-block;
            padding: 3px 7px;
            background: #3498db;
            color: white;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .winner {
            background-color: #e6f7e6;
            font-weight: bold;
        }
        code {
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }
        .highlight {
            background-color: #fffde7;
            padding: 15px;
            border-left: 4px solid #ffd600;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Customer Churn Prediction</h1>

    <div class="highlight">
        <strong>📌 Project Overview:</strong> Predictive modeling to identify telecom customers likely to cancel subscriptions using machine learning.
    </div>

    <h2>🎯 Business Value</h2>
    <ul>
        <li>Reduce customer churn by identifying at-risk customers early</li>
        <li>Improve retention strategies with data-driven insights</li>
        <li>Increase revenue by retaining high-value customers</li>
        <li>Optimize marketing spend by targeted interventions</li>
    </ul>

    <h2>📂 Dataset</h2>
    <p><strong>Source:</strong> <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">IBM Telco Customer Churn Dataset</a></p>
    <p><strong>Features:</strong></p>
    <ul>
        <li><span class="badge">Demographics</span> Gender, senior citizen status, partner/dependents</li>
        <li><span class="badge">Services</span> Phone, internet, streaming subscriptions</li>
        <li><span class="badge">Account</span> Tenure, contract type, payment method</li>
        <li><span class="badge">Target</span> Churn status (Yes/No)</li>
    </ul>
    <p><strong>Size:</strong> 7,043 customers × 21 features</p>

    <h2>🛠️ Models Implemented</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
            </tr>
        </thead>
        <tbody>
            <tr class="winner">
                <td>Random Forest (Tuned) 🏆</td>
                <td>80.12%</td>
                <td>72.35%</td>
                <td>79.21%</td>
                <td>75.62%</td>
                <td>85.04%</td>
            </tr>
            <tr>
                <td>Gradient Boosting</td>
                <td>79.35%</td>
                <td>71.82%</td>
                <td>78.45%</td>
                <td>74.98%</td>
                <td>84.12%</td>
            </tr>
            <tr>
                <td>Random Forest (Default)</td>
                <td>78.94%</td>
                <td>71.25%</td>
                <td>77.89%</td>
                <td>74.42%</td>
                <td>83.76%</td>
            </tr>
            <tr>
                <td>Logistic Regression</td>
                <td>77.42%</td>
                <td>69.87%</td>
                <td>75.32%</td>
                <td>72.48%</td>
                <td>81.95%</td>
            </tr>
            <tr>
                <td>Decision Tree</td>
                <td>74.82%</td>
                <td>67.45%</td>
                <td>72.18%</td>
                <td>69.71%</td>
                <td>78.23%</td>
            </tr>
            <tr>
                <td>KNN</td>
                <td>74.41%</td>
                <td>66.92%</td>
                <td>71.54%</td>
                <td>69.12%</td>
                <td>77.65%</td>
            </tr>
            <tr>
                <td>SVM</td>
                <td>73.68%</td>
                <td>66.34%</td>
                <td>70.89%</td>
                <td>68.52%</td>
                <td>76.87%</td>
            </tr>
        </tbody>
    </table>

    <h3>🔍 Best Model: Tuned Random Forest</h3>
    <ul>
        <li>Highest Accuracy (80.12%)</li>
        <li>Best ROC-AUC (85.04%)</li>
        <li>Balanced Precision & Recall</li>
        <li>Provides feature importance for explainability</li>
        <li>Robust against overfitting</li>
    </ul>

    <h2>📊 Key Insights</h2>
    <h3>Top Factors Influencing Churn</h3>
    <ol>
        <li><strong>Contract Type:</strong> Month-to-month customers churn 3× more</li>
        <li><strong>Internet Service:</strong> Fiber optic users churn more than DSL</li>
        <li><strong>Tenure:</strong> Newer customers are more likely to churn</li>
        <li><strong>Monthly Charges:</strong> Higher charges correlate with churn</li>
        <li><strong>Tech Support:</strong> Lack of support increases churn risk</li>
    </ol>

    <h2>🚀 How to Run the Code</h2>
    <h3>Prerequisites</h3>
    <pre><code>pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn</code></pre>

    <h3>Steps</h3>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/yourusername/customer-churn-prediction.git</code></pre>
        </li>
        <li>Download dataset from <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">Kaggle</a></li>
        <li>Run the script:
            <pre><code>python churn_prediction.py</code></pre>
        </li>
    </ol>

    <h2>📜 License</h2>
    <p>MIT License - Open Source</p>

    <hr>
    <p><strong>👨‍💻 Author:</strong> Your Name<br>
    <strong>📧 Contact:</strong> your.email@example.com<br>
    <strong>🔗 GitHub:</strong> <a href="https://github.com/yourusername" target="_blank">github.com/yourusername</a></p>
</body>
</html>
