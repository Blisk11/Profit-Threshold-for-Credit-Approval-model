import streamlit as st
import psutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Update the page config
st.set_page_config(
    page_title="Maximizing Profitability in Credit Approval Models 💸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# replace sklearn.metrics
def confusion_matrix(y_true, y_pred):
    # Initialize the counts
    tn, fp, fn, tp = 0, 0, 0, 0
    
    # Loop through the true and predicted labels
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1  # True Negative
        elif true == 0 and pred == 1:
            fp += 1  # False Positive
        elif true == 1 and pred == 0:
            fn += 1  # False Negative
        elif true == 1 and pred == 1:
            tp += 1  # True Positive
            
    return np.array([[tn, fp], [fn, tp]])

def roc_curve(y_true, y_prob):
    # Sort the probabilities and corresponding true values
    thresholds = np.linspace(0, 1, 250)
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        # Apply the threshold to get binary predictions
        y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
        
        # Compute confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate TPR and FPR
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    return fpr, tpr, thresholds

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Function to display memory usage in the sidebar
def display_memory_usage():
    st.sidebar.markdown("### Memory Usage")
    memory_usage = get_memory_usage()
    st.sidebar.markdown(f"**Current Memory Usage:** {memory_usage:.2f} MB")

# Function to add definitions to the sidebar
def add_definitions():
    st.sidebar.title("Definitions 📚")
    st.sidebar.markdown("""
    <span style="font-size: 16px;">
    <b>Accuracy</b>: The ratio of correctly predicted instances to the total instances.<br>
    <b>Precision</b>: The ratio of correctly predicted positive observations to the total predicted positives.<br>
    <b>ROI (Return on Investment)</b>: A measure of the profitability of an investment.<br>
    <b>Confusion Matrix</b>: A table used to evaluate the performance of a classification model.<br>
    <b>Threshold</b>: A value used to determine the cutoff point for classifying instances.<br>
    <b>Delinquency</b>: The failure to repay a loan according to the terms agreed upon.
    </span>
    """, unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    return joblib.load('data/data2.joblib')

# Function to compute confusion matrices
@st.cache_data
def compute_confusion_matrices(y_test, test_predictions, ACCURACY_threshold, PRECISION_threshold):
    accuracy_conf_matrix = confusion_matrix(y_test, test_predictions >= ACCURACY_threshold)
    precision_conf_matrix = confusion_matrix(y_test, test_predictions >= PRECISION_threshold)
    return accuracy_conf_matrix, precision_conf_matrix

# Function to compute ROC curve
@st.cache_data
def compute_roc_curve(y_test, test_predictions):
    return roc_curve(y_test, test_predictions)

# Function to compute profit sums
@st.cache_data
def compute_profit_sums(y_train_combined, y_test_combined, train_predictions, test_predictions, thresholds):
    train_profit_sums = [y_train_combined[train_predictions <= t]['net_profit_loss'].sum() for t in thresholds]
    test_profit_sums = [y_test_combined[test_predictions <= t]['net_profit_loss'].sum() for t in thresholds]
    return train_profit_sums, test_profit_sums

# Function to plot confusion matrix
def plot_confusion_matrix_v1(conf_matrix, title):
    fig = plt.figure()
    group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    labels = np.asarray(group_counts).reshape(2,2)
    categories = ['Repaid', 'Delinquencies']
    colors = np.array([['beige', 'red'], ['beige', 'red']])
    
    sns.heatmap(conf_matrix, annot=labels, xticklabels=categories, yticklabels=categories, 
                fmt='', cbar=False, cmap=None, center=0, mask=False, square=True, linewidths=1,
                annot_kws={'size': 20, 'color': 'black'}, linecolor='black')
    
    heatmap = plt.gca().collections
    if heatmap:  # Check if heatmap is created
        for path, color in zip(heatmap[0].get_paths(), colors.flatten()):
            patch = plt.matplotlib.patches.PathPatch(path, facecolor=color)
            plt.gca().add_patch(patch)
    
    plt.title(title)
    plt.ylabel('Actual Event', rotation=0)
    plt.xlabel('Model Prediction', fontweight='bold', color='red')
    plt.yticks(rotation=0)
    return fig

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, thresholds, PROFIT_threshold, AUC_threshold, ACCURACY_threshold, PRECISION_threshold, custom_threshold):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fpr, tpr, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    roi_idx = np.abs(thresholds - PROFIT_threshold).argmin()
    roc_idx = np.abs(thresholds - AUC_threshold).argmin()
    custom_idx = np.abs(thresholds - custom_threshold).argmin()
    accuracy_idx = np.abs(thresholds - ACCURACY_threshold).argmin()
    precision_idx = np.abs(thresholds - PRECISION_threshold).argmin()
    
    ax.plot(fpr[roi_idx], tpr[roi_idx], 'ro', label=f'ROI Threshold ({PROFIT_threshold:.3f})')
    ax.plot(fpr[roc_idx], tpr[roc_idx], 'go', label=f'AUC Threshold ({AUC_threshold:.3f})')
    ax.plot(fpr[custom_idx], tpr[custom_idx], 'o', color='#615ef3', label=f'Custom Threshold ({custom_threshold:.3f})')
    ax.plot(fpr[accuracy_idx], tpr[accuracy_idx], 'yo', label=f'Accuracy Threshold ({ACCURACY_threshold:.3f})')
    ax.plot(fpr[precision_idx], tpr[precision_idx], 'mo', label=f'Precision Threshold ({PRECISION_threshold:.3f})')
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with Different Thresholds')
    ax.legend()
    ax.grid(True)
    return fig

# Function to plot distribution of profits and losses
def plot_distribution_of_profits_losses(y_combined):
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    
    sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Repaid']['net_profit_loss'], bins=30, kde=True, ax=ax[0, 0], color='green')
    ax[0, 0].set_title('Distribution of Profits for Repaid Loans')
    ax[0, 0].set_xlabel('Profits')
    ax[0, 0].set_ylabel('Frequency')
    
    sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Delinquencies']['net_profit_loss'], bins=30, kde=True, ax=ax[0, 1], color='red')
    ax[0, 1].set_title('Distribution of Profits / Losses for Delinquent Loans')
    ax[0, 1].set_xlabel('Profits / Losses')
    ax[0, 1].set_ylabel('Frequency')
    
    sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Repaid']['roi_percentage'], bins=30, kde=True, ax=ax[1, 0], color='green')
    ax[1, 0].set_title('Distribution of ROI for Repaid Loans')
    ax[1, 0].set_xlabel('ROI (%)')
    ax[1, 0].set_ylabel('Frequency')
    
    sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Delinquencies']['roi_percentage'], bins=30, kde=True, ax=ax[1, 1], color='red')
    ax[1, 1].set_title('Distribution of ROI for Delinquent Loans')
    ax[1, 1].set_xlabel('ROI (%)')
    ax[1, 1].set_ylabel('Frequency')
    
    for bar in ax[0, 1].patches + ax[1, 1].patches:
        if bar.get_x() >= 0:
            bar.set_color('#7fbf7f')
            bar.set_edgecolor('black')
    
    for axis in ax.flatten():
        axis.xaxis.set_major_locator(plt.MaxNLocator(5))
        if axis in [ax[0, 0], ax[0, 1]]:
            axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        else:
            axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.0%}'))
    
    return fig

# Function to highlight max profit
def highlight_max_profit(s):
    is_max = s.str.replace('M$', '').astype(float) == s.str.replace('M$', '').astype(float).max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

# Function to highlight negative profits
def highlight_negative_profits(s):
    return ['background-color: red; color: white' if '-' in str(v) else '' for v in s]

# Function to compare performance
def compare_performance(threshold_name, threshold, y_test, test_predictions, y_add_test):
    test_binary_predictions = (test_predictions >= threshold).astype(int)
    approved_mask = test_binary_predictions == 0
    approved_count = np.sum(approved_mask)
    filtered_data = y_add_test.loc[y_test[approved_mask].index]
    total_profits = filtered_data['net_profit_loss'].sum()
    weight_avg_roi = filtered_data['net_profit_loss'].sum() / filtered_data['loan_amnt'].sum()
    actual_defaults = y_test[approved_mask].sum()
    default_rate = actual_defaults / approved_count if approved_count > 0 else 0
    original_default_rate = y_test.mean()
    
    st.markdown(f"##### {threshold_name} Threshold ({threshold:.1%}) Analysis")
    st.markdown(f"**Approved Loans**: {approved_count:,} ({approved_count/len(y_test):.1%} of total)")
    st.markdown(f"**Total Profits**: ${total_profits / 1e6:.1f}M")
    st.markdown(f"**ROI**: {weight_avg_roi:.1%}")
    st.markdown(f"**Original Delinquency Rate**: {original_default_rate * 100:.1f} %")
    st.markdown(f"**New Delinquency Rate**: {default_rate* 100:.1f} %")
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#e1e0fe', '#615ef3'])
    cm = confusion_matrix(y_test, test_binary_predictions)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, xticklabels=['Repaid', 'Delinquencies'], yticklabels=['Repaid', 'Delinquencies'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix {threshold_name} Threshold')
    st.pyplot(fig)

# Function to clear the cache
def clear_cache():
    st.cache_data.clear()

# Main function
def main():
    # Add a button to clear the cache manually
    if st.sidebar.button("Clear Cache"):
        clear_cache()
        st.rerun()
    data = load_data()
    y_test = data['y_test']
    y_add_test = data['y_add_test']
    y_train = data['y_train']
    y_add_train = data['y_add_train']
    test_predictions = data['test_predictions']
    train_predictions = data['train_predictions']
    PROFIT_threshold = data['PROFIT_threshold']
    test_PROFIT_threshold = data['test_PROFIT_threshold']
    AUC_threshold = data['AUC_threshold']
    ACCURACY_threshold = data['ACCURACY_threshold']
    PRECISION_threshold = data['PRECISION_threshold']
    
    display_memory_usage()
    
    st.markdown("""### Maximizing Profitability in Credit Approval Models: Prioritizing ROI Over Sensitivity and Specificity 💰""")
    st.markdown("""
    The following app is a demonstration of my master's thesis in which I showed examples of the importance of focusing on the business problem.
    In this project, I emphasize the significance of optimizing credit approval models not just for accuracy or precision, but for maximizing profitability.
    By adjusting the thresholds for loan approvals, we can significantly increase the return on investment while managing the risk of defaults.
    This approach ensures that the model aligns with the financial goals of the business, ultimately leading to better decision-making and higher profits.
    """)
    st.markdown("""#### The Data Science Rite of Passage: Unbalanced Classification Models 📉""")
    st.markdown("""The credit default problem is a classic because the often used metrics like accuracy and precision scores often give very poor results.""")
    
    accuracy_conf_matrix, precision_conf_matrix = compute_confusion_matrices(y_test, test_predictions, ACCURACY_threshold, PRECISION_threshold)
    st.write('')
    fig_accuracy = plot_confusion_matrix_v1(accuracy_conf_matrix, "Test Confusion Matrix with Accuracy threshold")
    fig_precision = plot_confusion_matrix_v1(precision_conf_matrix, "Test Confusion Matrix with Precision threshold")
    
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        st.pyplot(fig_accuracy)
    with col3:
        st.pyplot(fig_precision)
    
    col1, col2, col3 = st.columns([.5, 1, .5])
    with col2:
        st.markdown("""We can see that unless you coerce your model, most people get a loan!! 😅💸:""")
        st.image("data/oprah_meme.jpg", caption="Oprah you get a car meme", use_container_width=False, width=500)
    
    st.write("")
    st.subheader('Traditional Threshold vs New Profit Method 📊', anchor='profit-method')
    st.markdown("""The classic method and the metric I used to train the model is called [Area Under the Curve](https://www.geeksforgeeks.org/auc-roc-curve/).
    Compared to the precision and accuracy, AUC-ROC aims does a better job of denying bad loans, even if that means denying good ones.
    But instead of doing a binary classification model, we have the output as a confidence interval, and customise our threshold for approval/ denial.
    **Use the slider in the sidebar to change the custom threshold.**""")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    if 'custom_threshold' not in st.session_state:
        st.session_state.custom_threshold = 0.45
    
    with st.sidebar:
        st.markdown("---")
        custom_threshold = st.number_input(
            "[Adjust **Custom** Threshold for Comparison](#profit-method)",
            min_value= float(0) , 
            max_value= float(1), 
            value= 0.45,
            step = 0.02,
            format="%.2f",
        )
        st.write('Enter', round(test_PROFIT_threshold, 2), 'to beat my profit threshold')
    
    if custom_threshold != st.session_state.custom_threshold:
        st.session_state.custom_threshold = custom_threshold
        st.rerun()
    
    custom_threshold = st.session_state.custom_threshold

    add_definitions()
    
    fpr, tpr, thresholds = compute_roc_curve(y_test, test_predictions)
    st.pyplot(plot_roc_curve(fpr, tpr, thresholds, PROFIT_threshold, AUC_threshold, ACCURACY_threshold, PRECISION_threshold, custom_threshold))
    
    st.markdown("""#### Why is there a better way? 🤔""")
    st.markdown("""In traditional credit models, the issue lies in treating each default and each repaid loan as having the same impact, as you can see below, that is clearly not the case.""")
    
    y_train_renamed = y_train.replace({0: 'Repaid', 1: 'Delinquencies'})
    y_train_combined = pd.concat([y_add_train, y_train_renamed], axis=1)
    y_test_renamed = y_test.replace({0: 'Repaid', 1: 'Delinquencies'})
    y_test_combined = pd.concat([y_add_test, y_test_renamed], axis=1)
    y_combined = pd.concat([y_train_combined, y_test_combined])
    y_combined = y_combined[~((y_combined['loan_status_binary'] == 'Repaid') & (y_combined['net_profit_loss'] < 0))]
    
    st.pyplot(plot_distribution_of_profits_losses(y_combined))
    
    st.markdown("""#### Profit Threshold 💸:""")
    st.markdown("""While determining our threshold for denial and approval, we can calculate, on our training data, the threshold that maximizes our profit.
    **This drastically increases our expected profits on our test data.**""")
    
    train_results_df = pd.DataFrame({
        'Threshold': ['No Threshold', f"{PROFIT_threshold:.0%}", f"{AUC_threshold:.0%}", f"{custom_threshold:.0%}"],
        'Approved Loans (%)': [
            100.0,
            (train_predictions <= PROFIT_threshold).mean() * 100,
            (train_predictions <= AUC_threshold).mean() * 100,
            (train_predictions <= custom_threshold).mean() * 100
        ],
        'Net Profit/Loss (M$)': [
            y_train_combined['net_profit_loss'].sum() / 1e6,
            y_train_combined[train_predictions <= PROFIT_threshold]['net_profit_loss'].sum() / 1e6,
            y_train_combined[train_predictions <= AUC_threshold]['net_profit_loss'].sum() / 1e6,
            y_train_combined[train_predictions <= custom_threshold]['net_profit_loss'].sum() / 1e6
        ]
    }, index=['No threshold', 'Profit threshold', 'AUC threshold', 'Custom threshold'])
    
    test_results_df = pd.DataFrame({
        'Threshold': ['No Threshold', f"{PROFIT_threshold:.0%}", f"{AUC_threshold:.0%}", f"{custom_threshold:.0%}"],
        'Approved Loans (%)': [
            100.0,
            (train_predictions <= PROFIT_threshold).mean() * 100,
            (train_predictions <= AUC_threshold).mean() * 100,
            (train_predictions <= custom_threshold).mean() * 100
        ],
        'Net Profit/Loss (M$)': [
            y_test_combined['net_profit_loss'].sum() / 1e6,
            y_test_combined[test_predictions <= PROFIT_threshold]['net_profit_loss'].sum() / 1e6,
            y_test_combined[test_predictions <= AUC_threshold]['net_profit_loss'].sum() / 1e6,
            y_test_combined[test_predictions <= custom_threshold]['net_profit_loss'].sum() / 1e6
        ]
    }, index=['No threshold', 'Profit threshold', 'AUC threshold', 'Custom threshold'])
    
    train_max_profit = train_results_df['Net Profit/Loss (M$)'].max()
    test_max_profit = test_results_df['Net Profit/Loss (M$)'].max()
    
    train_results_df['Lost Profits (%)'] = (((train_results_df['Net Profit/Loss (M$)'] / train_max_profit)-1) * 100).round(1).astype(int).astype(str) + '%'
    test_results_df['Lost Profits (%)'] = ((( test_results_df['Net Profit/Loss (M$)'] / test_max_profit)-1) * 100).round(1).astype(int).astype(str) + '%'
    
    train_results_df['Approved Loans (%)'] = train_results_df['Approved Loans (%)'].map('{:.0f}%'.format)
    test_results_df['Approved Loans (%)'] = test_results_df['Approved Loans (%)'].map('{:.0f}%'.format)
    
    train_results_df['Net Profit/Loss (M$)'] = train_results_df['Net Profit/Loss (M$)'].map('{:.1f}M$'.format)
    test_results_df['Net Profit/Loss (M$)'] = test_results_df['Net Profit/Loss (M$)'].map('{:.1f}M$'.format)
    
    train_results_df_styled = train_results_df.style.apply(highlight_max_profit, subset=['Net Profit/Loss (M$)']).apply(highlight_negative_profits, subset=['Lost Profits (%)'])
    test_results_df_styled = test_results_df.style.apply(highlight_max_profit, subset=['Net Profit/Loss (M$)']).apply(highlight_negative_profits, subset=['Lost Profits (%)'])
    
    st.markdown(
        """
        <style>
        table {
            text-align: center !important;
        }
        th, td {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.table(train_results_df_styled)
    with col2:
        st.table(test_results_df_styled)
    
    train_profit_sums, test_profit_sums = compute_profit_sums(y_train_combined, y_test_combined, train_predictions, test_predictions, thresholds)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].plot(thresholds, train_profit_sums, color='black', linewidth=0.6)
    ax[0].set_title('Net Profit vs. Threshold (Training Data)')
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylabel('Profit $')
    ax[0].axvline(PROFIT_threshold, color='green', linestyle='--', label='Profit threshold')
    ax[0].axvline(AUC_threshold, color='red', linestyle='--', label='AUC threshold')
    ax[0].axvline(custom_threshold, color='#615ef3',linestyle= '-.', linewidth=2.5, label='Custom Threshold')
    ax[0].legend()
    fig.subplots_adjust(wspace=.3)
    
    ax[1].plot(thresholds, test_profit_sums, color='black', linewidth=0.6)
    ax[1].set_title('Net Profit/Loss vs. Threshold (Test Data)')
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('Profit $')
    ax[1].axvline(PROFIT_threshold, color='green', linestyle='--', label='Profit threshold')
    ax[1].axvline(AUC_threshold, color='red', linestyle='--', label='AUC threshold')
    ax[1].axvline(custom_threshold, color='#615ef3', linestyle= '-.', linewidth=2.5, label='Custom Threshold')
    ax[1].legend()
    
    with st.sidebar:
        st.markdown("### Formulas")
        st.markdown('''
            <style>
            .katex-html {
                text-align: left;
                font-size: 15px;
            }
            </style>''',
            unsafe_allow_html=True
        )
    
        st.latex(r"\text{ROI} = \frac{\text{Net Profit}}{\text{Investment}}")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{AUC} = \int_{0}^{1} TPR(FPR) \, d(FPR)")
    
    for axis in ax:
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    st.pyplot(fig)
    
    custom_threshold_returns = y_test_combined[test_predictions <= custom_threshold]['net_profit_loss'].sum()
    PROFIT_threshold_returns = y_test_combined[test_predictions <= PROFIT_threshold]['net_profit_loss'].sum()
    
    value = custom_threshold_returns - PROFIT_threshold_returns
    delta = custom_threshold_returns / PROFIT_threshold_returns
    lost_profit_AUC = test_results_df.loc['AUC threshold']['Lost Profits (%)']
    
    if custom_threshold_returns < PROFIT_threshold_returns:
        st.caption(f"""
            ### As you can see, the lost profits associated with the AUC threshold are substantial <span style='color:red'>{lost_profit_AUC}</span>.
            #### Move the slider on the left to beat my *Profit Threshold*, try to aim for the highest point on the right lineplot.
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label = 'ROI Threshold VS Custom Threshold', value=f"${value:,.2f}", delta=f"{(delta - 1) * 100:.2f}%", border = True)
    else:
        st.caption("""
        #### Congragulations, let's see how much more money you made :dollar:.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label = 'ROI Threshold VS Custom Threshold', value=f"${value:,.2f}", delta=f"{(delta - 1) * 100:.2f}%", border = True)
        st.caption("""
            #### You can go back up to see how much better you did than the AUC threshold.
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Let's see how the different thresholds affect the number of approved loans the expected returns and the reduction in delinquencies.")
    
    col1, col2, col3, col4, col5 = st.columns([1, 0.1, 1, 0.1, 1])
    with col1:
        compare_performance('Profit', PROFIT_threshold, y_test, test_predictions, y_add_test)
    with col3:
        compare_performance('AUC', AUC_threshold, y_test, test_predictions, y_add_test)
    with col5:
        compare_performance('Custom', custom_threshold, y_test, test_predictions, y_add_test)
    
    st.markdown("""
    ### Conclusion
    This analysis presents a novel approach to a common problem and highlights the importance of considering the business application of machine learning models. 
    To comply with Streamlit's Free tier limits, I reduced the dataset. However, increasing the data volume brings the training profit threshold closer to its optimal value on test data. 
    In some cases, I have observed the profit threshold outperforming the AUC-based one by more than 25%.
    
    ### Considerations and comments:
    - **Market conditions**: The data spans 9 years, including the subprime mortgage crisis. Considering macroeconomic conditions and understanding how Lending Club determines its interest rates would be a necessary step.  
    - **Model-Induced Data Shift**: We did not use the available refused loans data for this project. This profit threshold does not solve _Model-Induced Data Shift_, but I don't think it increases its impact.  
    - **Changing Y target**: I experimented with changing the y value to ROI or profit/loss, but in my opinion, this reduces the model's interpretability and compresses the distribution of predictions.  

    ### Data Source
    The data used in this analysis is sourced from [Kaggle's Lending Club dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The main focus of this project is to highlight the _Profit threshold_ a lot of optimisations are still possible.
    
    ### Github Link Source
    Want fork this project? You can have a look at the notebook used to create the model and data, as well as the app. [GitHub Repo](https://github.com/Blisk11/Profit-Threshold-for-Credit-Approval-model).
                
    """)

if __name__ == "__main__":
    main()
