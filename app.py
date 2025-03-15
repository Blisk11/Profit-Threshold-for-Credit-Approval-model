import streamlit as st
import psutil
# Update the page config
st.set_page_config(
    page_title="Maximizing Profitability in Credit Approval Models 💸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# """
# This Streamlit application analyzes loan default ROI and compares different credit approval models.

# **Modules:**
# - `streamlit`: For creating the web application.
# - `joblib`: For loading the saved test data.
# - `pandas`: For data manipulation and analysis.
# - `numpy`: For numerical operations.
# - `sklearn.metrics`: For calculating confusion matrices.
# - `matplotlib.pyplot`: For plotting graphs.
# - `seaborn`: For creating heatmaps.

# **Functions:**
# - `plot_confusion_matrix_v1(conf_matrix, title)`: Plots a confusion matrix with custom colors and labels.

# **Data:**
# - `data`: Loaded test data from 'data/data2.joblib'.
# - `X_test`: Test features.
# - `y_test`: Test labels.
# - `y_add_test`: Additional test data.
# - `X_train`: Training features.
# - `y_train`: Training labels.
# - `y_add_train`: Additional training data.
# - `test_predictions`: Predictions on the test set.
# - `train_predictions`: Predictions on the training set.
# - `PROFIT_threshold`: Threshold for maximizing profit.
# - `AUC_threshold`: Threshold for ROC.
# - `ACCURACY_threshold`: Threshold for accuracy.
# - `PRECISION_threshold`: Threshold for precision.

# **Variables:**
# - `accuracy_conf_matrix`: Confusion matrix for accuracy model.
# - `precision_conf_matrix`: Confusion matrix for precision model.
# - `y_train_renamed`: Training labels renamed as 'Repaid' and 'Delinquencies'.
# - `y_train_combined`: Combined dataframe of additional training data and renamed training labels.
# - `y_test_renamed`: Test labels renamed as 'Repaid' and 'Delinquencies'.
# - `y_test_combined`: Combined dataframe of additional test data and renamed test labels.
# - `custom_threshold`: Custom threshold selected by the user.
# - `train_results_df`: Dataframe for training results.
# - `test_results_df`: Dataframe for test results.
# - `train_results_df_styled`: Styled dataframe for training results.
# - `test_results_df_styled`: Styled dataframe for test results.
# - `train_profit_sums`: List of net profit/loss for training data at different thresholds.
# - `test_profit_sums`: List of net profit/loss for test data at different thresholds.
# """

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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

# Cache the loading of data
@st.cache_data
def load_data():
    return joblib.load('data/data2.joblib')

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

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Display memory usage in the sidebar
st.sidebar.markdown("### Memory Usage")
memory_usage = get_memory_usage()
st.sidebar.markdown(f"**Current Memory Usage:** {memory_usage:.2f} MB")

# Add a sidebar with definitions
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

st.markdown("""### Maximizing Profitability in Credit Approval Models: Prioritizing ROI Over Sensitivity and Specificity 💰""")
# Description of Standard Credit Risk Assessment
st.markdown("""
#### The Data Science Rite of Passage: Unbalanced Classification Models 📉
The credit default problem is a classic because the often used metrics like accuracy and precision scores often give very poor results.
""")

# Cache the computation of confusion matrices
@st.cache_data
def compute_confusion_matrices(y_test, test_predictions, ACCURACY_threshold, PRECISION_threshold):
    accuracy_conf_matrix = confusion_matrix(y_test, test_predictions >= ACCURACY_threshold)
    precision_conf_matrix = confusion_matrix(y_test, test_predictions >= PRECISION_threshold)
    return accuracy_conf_matrix, precision_conf_matrix

accuracy_conf_matrix, precision_conf_matrix = compute_confusion_matrices(y_test, test_predictions, ACCURACY_threshold, PRECISION_threshold)
st.write('')
def plot_confusion_matrix_v1(conf_matrix, title):
    fig = plt.figure()

    group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    labels = np.asarray(group_counts).reshape(2,2)
    categories = ['Repaid', 'Delinquencies']

    # Create a custom color mask
    colors = np.array([['beige', 'red'],
                      ['beige', 'red']])
    
    sns.heatmap(conf_matrix, annot = labels, xticklabels=categories, yticklabels=categories, 
                fmt='', cbar=False, cmap=None, center=0,
                mask=False, square=True, linewidths=1,
                annot_kws={'size': 20, 'color': 'black'},
                linecolor='black')

    # Color each cell individually
    heatmap = plt.gca().collections[0]
    for path, color in zip(heatmap.get_paths(), colors.flatten()):
        patch = plt.matplotlib.patches.PathPatch(path, facecolor=color)
        plt.gca().add_patch(patch)

    plt.title(title)
    plt.ylabel('Actual Event', rotation=0)
    plt.xlabel('Model Prediction', fontweight='bold', color='red')
    plt.yticks(rotation=0)
    
    # Highlight "Delinquencies" labels
    #plt.gca().get_xticklabels()[1].set_color('red')   
    
    return fig
st.write('')
# Plot confusion matrices
fig_accuracy = plot_confusion_matrix_v1(accuracy_conf_matrix, "Test Confusion Matrix with Accuracy threshold")
fig_precision = plot_confusion_matrix_v1(precision_conf_matrix, "Test Confusion Matrix with Precision threshold")

# Display the plots side by side with a separator
col1, col2, col3 = st.columns([1, 0.2, 1])  # Adjust column widths to add space for separator
with col1:
    st.pyplot(fig_accuracy)
with col3:
    st.pyplot(fig_precision)

col1, col2, col3 = st.columns([.5, 1, .5])  # Create two equal-width columns for the image and markdown
with col2:
    st.markdown("""
    We can see that unless you coerce your model, most people get a loan!! 😅💸:
    """)
    st.image("data/oprah_meme.jpg", caption="Oprah you get a car meme", use_container_width=False, width=500)


st.write("")
st.subheader('Traditional Threshold vs New Profit Method 📊', anchor='profit-method')
st.markdown("""
The classic method and the metric I used to train the model is called [Area Under the Curve](https://www.geeksforgeeks.org/auc-roc-curve/).
Compared to the precision and accuracy, AUC-ROC aims does a better job of denying bad loans, even if that means denying good ones.
But instead of doing a binary classification model, we have the output as a confidence interval, and customise our threshold for approval/ denial.
**Use the slider in the sidebar to change the custom threshold.**
""")

col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths to center the slider
# Initialize session state for the slider
if 'custom_threshold' not in st.session_state:
    st.session_state.custom_threshold = 0.45

# Create a slider for selecting a custom threshold
with st.sidebar:
    st.markdown("---")
    custom_threshold = st.number_input(
        "[Adjust **Custom** Threshold for Comparison](#profit-method)",
        min_value= float(0) , 
        max_value= float(1), 
        value= 0.45,
        step = 0.05,
        format="%.2f",
       # on_change = update_custom_threshold,
    )
    st.write(round(test_PROFIT_threshold, 2), 'to beat my profit threshold')

# Update session state if slider value changes
if custom_threshold != st.session_state.custom_threshold:
    st.session_state.custom_threshold = custom_threshold
    st.rerun()  # Updated from st.experimental_rerun()

# Use the updated custom_threshold for calculations
custom_threshold = st.session_state.custom_threshold

# Cache the computation of ROC curve
@st.cache_data
def compute_roc_curve(y_test, test_predictions):
    return roc_curve(y_test, test_predictions)

fpr, tpr, thresholds = compute_roc_curve(y_test, test_predictions)
# Plot ROC curve with both thresholds marked
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fpr, tpr, label='ROC Curve')
ax.plot([0, 1], [0, 1], 'k--', label='Random')

# Mark thresholds on the curve
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

st.pyplot(fig)
st.markdown("""
#### Why is there a better way? 🤔
Traditional credit models focus on minimizing defaults, **each delinquency does not yield the same loss, and each repaid loan does not generate the same return.**
""")

# Rename y_train binary values as 'Repaid' and 'Delinquencies'
y_train_renamed = y_train.replace({0: 'Repaid', 1: 'Delinquencies'})
# Combine the dataframes
y_train_combined = pd.concat([y_add_train, y_train_renamed], axis=1)

# Rename y_train binary values as 'Repaid' and 'Delinquencies'
y_test_renamed = y_test.replace({0: 'Repaid', 1: 'Delinquencies'})
# Combine the dataframes
y_test_combined = pd.concat([y_add_test, y_test_renamed], axis=1)

y_combined = pd.concat([y_train_combined, y_test_combined])
# Remove all values where 'loan_status_binary' == 'Repaid' and 'net_profit_loss' is negative
y_combined = y_combined[~((y_combined['loan_status_binary'] == 'Repaid') & (y_combined['net_profit_loss'] < 0))]
# Calculate and display the distribution of net profit/loss and ROI for repaid and delinquent loans
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Distribution of net profit/loss for repaid loans
sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Repaid']['net_profit_loss'], bins=30, kde=True, ax=ax[0, 0], color='green')
ax[0, 0].set_title('Distribution of Profits for Repaid Loans')
ax[0, 0].set_xlabel('Profits')
ax[0, 0].set_ylabel('Frequency')

# Distribution of net profit/loss for delinquent loans
sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Delinquencies']['net_profit_loss'], bins=30, kde=True, ax=ax[0, 1], color='red')
ax[0, 1].set_title('Distribution of Profits / Losses for Delinquent Loans')
ax[0, 1].set_xlabel('Profits / Losses')
ax[0, 1].set_ylabel('Frequency')

# Distribution of ROI for repaid loans
sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Repaid']['roi_percentage'], bins=30, kde=True, ax=ax[1, 0], color='green')
ax[1, 0].set_title('Distribution of ROI for Repaid Loans')
ax[1, 0].set_xlabel('ROI (%)')
ax[1, 0].set_ylabel('Frequency')

# Distribution of ROI for delinquent loans
sns.histplot(y_combined[y_combined['loan_status_binary'] == 'Delinquencies']['roi_percentage'], bins=30, kde=True, ax=ax[1, 1], color='red')
ax[1, 1].set_title('Distribution of ROI for Delinquent Loans')
ax[1, 1].set_xlabel('ROI (%)')
ax[1, 1].set_ylabel('Frequency')

# Highlight positive values in green for net profit/loss
for bar in ax[0, 1].patches + ax[1, 1].patches:
    if bar.get_x() >= 0:
        bar.set_color('#7fbf7f')
        bar.set_edgecolor('black')

# Reduce number of xticks and format them
for axis in ax.flatten():
    axis.xaxis.set_major_locator(plt.MaxNLocator(5))
    if axis in [ax[0, 0], ax[0, 1]]:
        axis.set_xticklabels([f"${x:,.0f}" for x in axis.get_xticks()])
    else:
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.0%}'))
    

st.pyplot(fig)

st.markdown("""
#### Profit Threshold 💸:
While determining our threshold for denial and approval, we can calculate, on our training data, the threshold that maximizes our profit.
**This drastically increases our profit.**
""")
st.caption("Use the custom slider on the left to view the impact")
# Create dataframes to store the results for training and test data
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

# Calculate the new column values
train_max_profit = train_results_df['Net Profit/Loss (M$)'].max()
test_max_profit = test_results_df['Net Profit/Loss (M$)'].max()

train_results_df['Lost Profits (%)'] = (((train_results_df['Net Profit/Loss (M$)'] / train_max_profit)-1) * 100).round(1).astype(int).astype(str) + '%'
test_results_df['Lost Profits (%)'] = ((( test_results_df['Net Profit/Loss (M$)'] / test_max_profit)-1) * 100).round(1).astype(int).astype(str) + '%'

# Format the 'Approved Loans (%)' column as :.0f}%
train_results_df['Approved Loans (%)'] = train_results_df['Approved Loans (%)'].map('{:.0f}%'.format)
test_results_df['Approved Loans (%)'] = test_results_df['Approved Loans (%)'].map('{:.0f}%'.format)


def highlight_max_profit(s):
    is_max = s.str.replace('M$', '').astype(float) == s.str.replace('M$', '').astype(float).max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

# Format 'Net Profit/Loss (M$)' column to have only 1 digit
train_results_df['Net Profit/Loss (M$)'] = train_results_df['Net Profit/Loss (M$)'].map('{:.1f}M$'.format)
test_results_df['Net Profit/Loss (M$)'] = test_results_df['Net Profit/Loss (M$)'].map('{:.1f}M$'.format)

def highlight_negative_profits(s):
    return ['background-color: red; color: white' if '-' in str(v) else '' for v in s]

train_results_df_styled = train_results_df.style.apply(highlight_max_profit, subset=['Net Profit/Loss (M$)']).apply(highlight_negative_profits, subset=['Lost Profits (%)'])
test_results_df_styled = test_results_df.style.apply(highlight_max_profit, subset=['Net Profit/Loss (M$)']).apply(highlight_negative_profits, subset=['Lost Profits (%)'])

# CSS to center table values
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
# Display the styled dataframes as tables
col1, col2 = st.columns([1, 1])  # Create two equal-width columns for the tables
with col1:
    st.table(train_results_df_styled)
with col2:
    st.table(test_results_df_styled)


# Cache the computation of profit sums
@st.cache_data
def compute_profit_sums(y_train_combined, y_test_combined, train_predictions, test_predictions, thresholds):
    train_profit_sums = [y_train_combined[train_predictions <= t]['net_profit_loss'].sum() for t in thresholds]
    test_profit_sums = [y_test_combined[test_predictions <= t]['net_profit_loss'].sum() for t in thresholds]
    return train_profit_sums, test_profit_sums

train_profit_sums, test_profit_sums = compute_profit_sums(y_train_combined, y_test_combined, train_predictions, test_predictions, thresholds)

# Create the line plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training data plot
ax[0].plot(thresholds, train_profit_sums, color='black', linewidth=0.6)
ax[0].set_title('Net Profit vs. Threshold (Training Data)')
ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Profit $')
ax[0].axvline(PROFIT_threshold, color='green', linestyle='--', label='Profit threshold')
ax[0].axvline(AUC_threshold, color='red', linestyle='--', label='AUC threshold')
ax[0].axvline(custom_threshold, color='#615ef3',linestyle= '-.', linewidth=2.5, label='Custom Threshold')
ax[0].legend()
fig.subplots_adjust(wspace=.3)  # Adjust the width space between the subplots
# Test data plot
ax[1].plot(thresholds, test_profit_sums, color='black', linewidth=0.6)
ax[1].set_title('Net Profit/Loss vs. Threshold (Test Data)')
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Profit $')
ax[1].axvline(PROFIT_threshold, color='green', linestyle='--', label='Profit threshold')
ax[1].axvline(AUC_threshold, color='red', linestyle='--', label='AUC threshold')
ax[1].axvline(custom_threshold, color='#615ef3', linestyle= '-.', linewidth=2.5, label='Custom Threshold')
ax[1].legend()

# Add formulas to the sidebar
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


# Format y-axis ticks as currency
for axis in ax:
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

st.pyplot(fig)
# Display the results

custom_threshold_returns = y_test_combined[test_predictions <= custom_threshold]['net_profit_loss'].sum()
PROFIT_threshold_returns = y_test_combined[test_predictions <= PROFIT_threshold]['net_profit_loss'].sum()

value = custom_threshold_returns - PROFIT_threshold_returns
delta = custom_threshold_returns / PROFIT_threshold_returns
lost_profit_AUC = test_results_df.loc['AUC threshold']['Lost Profits (%)']

# Format the value with color based on positivity/negativity

if custom_threshold_returns < PROFIT_threshold_returns:
# Display the caption with dynamic value insertion
    st.caption(f"""
        ### As you can see, the lost profits associated with the AUC threshold are substantial <span style='color:red'>{lost_profit_AUC}</span>.
        #### Move the slider on the left to beat my *Profit Threshold*, look for the highest point on the right lineplot.
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])  # Create two equal-width columns for the tables
    with col1:
        st.metric(label = 'ROI Threshold VS Custom Threshold', value=f"${value:,.2f}", delta=f"{(delta - 1) * 100:.2f}%", border = True)
else:
    st.caption("""
    #### Congragulations, let's see how much more money you made :dollar:.
    """)
    col1, col2 = st.columns([1, 2])  # Create two equal-width columns for the tables
    with col1:
        st.metric(label = 'ROI Threshold VS Custom Threshold', value=f"${value:,.2f}", delta=f"{(delta - 1) * 100:.2f}%", border = True)
    st.caption("""
        #### You can go back up to see how much better you did than the AUC threshold.
        """)

st.markdown("<br>", unsafe_allow_html=True)
st.write("Let's see how the different thresholds affect the number of approved loans the expected returns and the reduction in delinquencies.")
# Calculate confusion matrices for different thresholds
def compare_performance( threshold_name, threshold):
    # Make predictions using current threshold
    test_binary_predictions = (test_predictions >= threshold).astype(int)
    
    # Calculate metrics for approved loans (predicted 0)
    approved_mask = test_binary_predictions == 0
    approved_count = np.sum(approved_mask)
    filtered_data = y_add_test.loc[y_test[approved_mask].index]
    total_profits = filtered_data['net_profit_loss'].sum()
    weight_avg_roi = filtered_data['net_profit_loss'].sum() / filtered_data['loan_amnt'].sum()
    
    # Calculate default rate for approved loans
    actual_defaults = y_test[approved_mask].sum()
    default_rate = actual_defaults / approved_count if approved_count > 0 else 0
    original_default_rate = y_test.mean()
    #delinquency_reduction = (original_default_rate - default_rate) * 100
    
    st.markdown(f"##### {threshold_name} Threshold ({threshold:.1%}) Analysis")
    st.markdown(f"**Approved Loans**: {approved_count:,} ({approved_count/len(y_test):.1%} of total)")
    st.markdown(f"**Total Profits**: ${total_profits / 1e6:.1f}M")
    st.markdown(f"**ROI**: {weight_avg_roi:.1%}")
    st.markdown(f"**Original Delinquency Rate**: {original_default_rate * 100:.1f} %")
    st.markdown(f"**New Delinquency Rate**: {default_rate* 100:.1f} %")

    # Create a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#e1e0fe', '#615ef3'])

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, test_binary_predictions)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap= custom_cmap, xticklabels=['Repaid', 'Delinquencies'], yticklabels=['Repaid', 'Delinquencies'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix {threshold_name} Threshold')
    st.pyplot(fig)

# Display the plots side by side with a separator
col1, col2, col3, col4, col5 = st.columns([1, 0.1, 1, 0.1, 1])  # Adjust column widths to add space for separator
with col1:
    compare_performance('Profit', PROFIT_threshold)
with col3:
    compare_performance('AUC', AUC_threshold)
with col5:
    compare_performance('Custom', custom_threshold)

# Display the results
st.markdown("""
### Conclusion
This analysis showcases a novel approach on a common problem and demonstrates that it is critical to always keep in mind the business application of our machine learning models.

### Considerations and comments:
- **Market conditions**: The data spans 9 years including the subprime mortgage crisis, considering macroeconomic conditions, knowing how Lending Club determines their interest rates would be a necessary step.
- **Model-Induced Data Shift**: We did not use the available refused loans data for this project. This profit threshold does not solve _Model-Induced Data Shift_, but I don't think it increases it's impact.

### Data Source
The data used in this analysis is sourced from [Kaggle's Lending Club dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The main focus of this project is to highlight the _Profit threshold_ a lot of optimisations are still possible.

### Github Link Source
Want fork this project? You can have a look at the notebook used to create the model and data, as well as the app. [GitHub Repo](https://github.com/Blisk11/Profit-Threshold-for-Credit-Approval-model).
            
""")
