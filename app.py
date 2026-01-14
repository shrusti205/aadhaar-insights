import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
import requests
from urllib.request import urlopen
import time
from datetime import datetime, timedelta
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards

# Set page config with wide layout
st.set_page_config(
    page_title="Aadhaar Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
    }
    
    /* Responsive text */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.3rem !important; }
        
        /* Stack columns on mobile */
        .st-eb {
            flex-direction: column !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("aadhaar_data.csv")
        required_columns = ['date', 'district', 'enrolments', 'updates']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m', errors='coerce')
        
        if df['date'].isna().any():
            st.warning(f"Dropped {df['date'].isna().sum()} rows with invalid dates")
            df = df.dropna(subset=['date'])
        
        df = df.drop_duplicates(subset=['date', 'district'], keep='first')
        all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')
        all_districts = df['district'].unique()
        
        complete_index = pd.MultiIndex.from_product(
            [all_dates, all_districts],
            names=['date', 'district']
        )
        complete_df = pd.DataFrame(index=complete_index).reset_index()
        
        result_df = pd.merge(
            complete_df,
            df,
            on=['date', 'district'],
            how='left'
        )
        
        cat_cols = ['state', 'update_type', 'age_group', 'gender', 'center_type']
        for col in cat_cols:
            if col in result_df.columns:
                result_df[col] = result_df.groupby('district')[col].ffill().bfill()
        
        num_cols = ['enrolments', 'updates']
        for col in num_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0).astype(int)
        
        result_df['month'] = result_df['date'].dt.month_name()
        result_df['year'] = result_df['date'].dt.year
        result_df['quarter'] = result_df['date'].dt.quarter
        result_df['year_quarter'] = result_df['year'].astype(str) + ' Q' + result_df['quarter'].astype(str)
        
        for col in cat_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype('category')
        
        return result_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame(columns=['date', 'district', 'enrolments', 'updates'])
# Load data
with st.spinner('🔄 Loading data...'):
    df = load_data()
if df.empty or 'date' not in df.columns:
    st.error("❌ Failed to load data. Please check the data file and try again.")
    st.stop()
# ====================
# SIDEBAR FILTERS
# ====================
st.sidebar.title('🎛️ Filters')
# Date Range Filter
st.sidebar.subheader('📅 Date Range')
min_date = df['date'].min().date()
max_date = df['date'].max().date()
start_date = st.sidebar.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End Date', max_date, min_value=min_date, max_value=max_date)
# State and District Filter
st.sidebar.subheader('📍 Location')
all_states = ['All'] + sorted(df['state'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    'Select States',
    options=all_states[1:],
    default=all_states[1:],
    key='state_filter'
)
# District filter (updates based on selected states)
if 'All' in selected_states or not selected_states:
    districts = df['district'].unique()
else:
    districts = df[df['state'].isin(selected_states)]['district'].unique()
all_districts = ['All'] + sorted(districts.tolist())
selected_districts = st.sidebar.multiselect(
    'Select Districts',
    options=all_districts[1:],
    default=all_districts[1:],
    key='district_filter'
)
# Update Type Filter
st.sidebar.subheader('🔄 Update Types')
update_types = df['update_type'].dropna().unique().tolist()
selected_update_types = st.sidebar.multiselect(
    'Select Update Types',
    options=update_types,
    default=update_types,  # Select all by default
    key='update_type_filter'
)

# Age Group Filter
st.sidebar.subheader('👥 Demographics')
age_groups = sorted(df['age_group'].dropna().unique().tolist())
selected_age_groups = st.sidebar.multiselect(
    'Age Groups',
    options=age_groups,
    default=age_groups,  # Select all by default
    key='age_group_filter'
)

# Gender Filter
genders = df['gender'].dropna().unique().tolist()
selected_genders = st.sidebar.multiselect(
    'Gender',
    options=genders,
    default=genders,  # Select all by default
    key='gender_filter'
)

# Center Type Filter
center_types = df['center_type'].dropna().unique().tolist()
selected_center_types = st.sidebar.multiselect(
    'Center Type',
    options=center_types,
    default=center_types,  # Select all by default
    key='center_type_filter'
)

# Apply Filters Button
if st.sidebar.button('🔍 Apply Filters', use_container_width=True):
    st.rerun()

# Reset Filters Button
if st.sidebar.button('🔄 Reset Filters', use_container_width=True):
    selected_states = all_states[1:]
    selected_districts = all_districts[1:]
    selected_update_types = update_types
    selected_age_groups = age_groups
    selected_genders = genders
    selected_center_types = center_types
    st.rerun()

# ====================
# APPLY FILTERS TO DATA
# ====================
filtered_df = df.copy()

# Apply date filter
filtered_df = filtered_df[
    (filtered_df['date'].dt.date >= start_date) & 
    (filtered_df['date'].dt.date <= end_date)
]

# Apply state and district filters
if selected_states and 'All' not in selected_states:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    
if selected_districts and 'All' not in selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Apply update type filter
if selected_update_types:
    filtered_df = filtered_df[filtered_df['update_type'].isin(selected_update_types)]

# Apply demographic filters
if selected_age_groups:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age_groups)]

if selected_genders:
    filtered_df = filtered_df[filtered_df['gender'].isin(selected_genders)]

# Apply center type filter
if selected_center_types:
    filtered_df = filtered_df[filtered_df['center_type'].isin(selected_center_types)]

# Show active filters in the sidebar
st.sidebar.markdown("---")
# Get date range for display
date_min = filtered_df['date'].min()
date_max = filtered_df['date'].max()
    
# Format the date range string
if pd.isna(date_min) or pd.isna(date_max):
    date_range_str = "No date data available"
else:
    date_range_str = f"Showing data from {date_min.strftime('%b %d, %Y')} to {date_max.strftime('%b %d, %Y')}"
    
st.sidebar.caption(date_range_str)
st.sidebar.caption(f"{len(filtered_df):,} records match your filters")

# Title and header
st.markdown("""
<div style="text-align: center;">
    <h1 style="margin-bottom: 0.2em;">📊 Aadhaar Analytics Dashboard</h1>
    <p style="color: #666; margin-top: 0;">Comprehensive insights into Aadhaar enrolments and updates across India</p>
</div>
""", unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# Enrollment Trend Analysis
st.subheader("📊 State-wise Enrollment Trends")

# Calculate enrollment changes by state
if not df.empty:
    # Get the latest two months of data
    latest_dates = df['date'].nlargest(2)
    recent_data = df[df['date'].isin(latest_dates)].copy()
    
    # Calculate total enrollments by state for the two most recent months
    state_trends = recent_data.groupby(['state', 'date'])['enrolments'].sum().unstack().reset_index()
    
    # Calculate month-over-month change
    if len(state_trends.columns) > 2:  # Ensure we have at least two months of data
        months = sorted(state_trends.columns[1:], reverse=True)  # Get months in descending order
        state_trends['previous_month'] = state_trends[months[1]]
        state_trends['current_month'] = state_trends[months[0]]
        
        # Calculate change and percentage change
        state_trends['change'] = state_trends['current_month'] - state_trends['previous_month']
        state_trends['pct_change'] = (state_trends['change'] / state_trends['previous_month']) * 100
        
        # Categorize the trend
        def get_trend_status(row):
            # Calculate updates to enrollment ratio for the current month
            current_month_data = recent_data[recent_data['date'] == months[0]]
            state_data = current_month_data[current_month_data['state'] == row['state']]
            
            total_updates = state_data['updates'].sum()
            total_enrollments = state_data['enrolments'].sum()
            
            # Check if updates exceed enrollments
            if total_updates > total_enrollments and total_enrollments > 0:
                return '⚠️ High Updates'
                
            # Otherwise, use the original trend logic
            if row['pct_change'] < -10:  # Significant decrease
                return '🔴 Critical Decrease'
            elif -10 <= row['pct_change'] < 0:  # Slight decrease
                return '🟠 Slight Decrease'
            elif row['pct_change'] == 0:  # No change
                return '⚪ No Change'
            elif 0 < row['pct_change'] <= 10:  # Slight increase
                return '🟢 Slight Increase'
            else:  # Significant increase
                return '🟢 Significant Increase'
        
        state_trends['status'] = state_trends.apply(get_trend_status, axis=1)
        
        # Sort by percentage change (most critical first)
        state_trends = state_trends.sort_values('pct_change')
        
        # Display the trend analysis
        st.write("### State-wise Enrollment Change (Latest Month vs Previous Month)")
        
        # Create a styled table
        def color_status(val):
            if 'High Updates' in val:
                return 'background-color: #fff3cd'  # Light yellow for high updates warning
            elif 'Critical' in val:
                return 'background-color: #ffcccc'  # Light red
            elif 'Slight Decrease' in val:
                return 'background-color: #ffe6cc'  # Light orange
            elif 'No Change' in val:
                return 'background-color: #f0f0f0'  # Light gray
            else:
                return 'background-color: #e6ffe6'  # Light green
        
        # Format the table
        display_cols = ['state', 'previous_month', 'current_month', 'change', 'pct_change', 'status']
        display_df = state_trends[display_cols].copy()
        display_df.columns = ['State', 'Previous Month', 'Current Month', 'Change', '% Change', 'Status']
        
        # Apply styling
        styled_df = display_df.style.applymap(
            lambda x: 'color: red' if 'Decrease' in str(x) else 'color: green' if 'Increase' in str(x) else '', 
            subset=['Status']
        ).format({
            'Previous Month': '{:,.0f}',
            'Current Month': '{:,.0f}',
            'Change': '{:+,.0f}',
            '% Change': '{:+.1f}%'
        })
        
        # Display the styled table
        st.dataframe(
            styled_df,
            column_config={
                'State': st.column_config.TextColumn("State"),
                'Previous Month': st.column_config.NumberColumn("Previous Month", format="%d"),
                'Current Month': st.column_config.NumberColumn("Current Month", format="%d"),
                'Change': st.column_config.NumberColumn("Change", format="%+d"),
                '% Change': st.column_config.NumberColumn("% Change", format="%+.1f%%"),
                'Status': st.column_config.TextColumn("Status")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Add some insights based on the data
        st.write("### Key Insights")
        
        # Get top and bottom states
        if not state_trends.empty:
            top_state = state_trends.nlargest(1, 'pct_change').iloc[0]
            bottom_state = state_trends.nsmallest(1, 'pct_change').iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🚀 Top Performer",
                    f"{top_state['state']}",
                    f"{top_state['pct_change']:+.1f}%"
                )
            
            with col2:
                st.metric(
                    "⚠️ Needs Attention",
                    f"{bottom_state['state']}",
                    f"{bottom_state['pct_change']:+.1f}%"
                )
    else:
        st.warning("Insufficient data for trend analysis. Need at least two months of data.")

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2em;">
    <p>Aadhaar Insights Dashboard | Data Source: UIDAI | Last Updated: {}</p>
    <p>For official use only | Contact: support@aadhaarinsights.gov.in</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics with NaN handling
total_enrolments = int(filtered_df.enrolments.fillna(0).sum())
total_updates = int(filtered_df.updates.fillna(0).sum())

# Calculate average updates with NaN handling
updates_by_month = filtered_df.groupby(['year', 'month'])['updates'].sum()
avg_updates = int(updates_by_month.mean()) if not updates_by_month.empty else 0

# Calculate update growth with NaN handling
update_growth = 0
if len(filtered_df) > 1:
    updates_by_date = filtered_df.groupby('date')['updates'].sum()
    if len(updates_by_date) > 1 and updates_by_date.iloc[0] != 0:
        update_growth = int(((updates_by_date.iloc[-1] / updates_by_date.iloc[0]) - 1) * 100)

# Display metrics with delta values
col1.metric("Total Enrolments", f"{total_enrolments:,}")
col2.metric("Total Updates", f"{total_updates:,}")
col3.metric("Avg Monthly Updates", f"{avg_updates:,}")
col4.metric("Update Growth", f"{update_growth}%" if update_growth != 0 else "N/A")

# Trend Analysis
st.subheader("📈 Trends Over Time")

# Time series trends
trend = filtered_df.groupby("date")[["enrolments", "updates"]].sum().reset_index()

# Create figure with secondary y-axis
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Scatter(
        x=trend['date'], 
        y=trend['enrolments'],
        name="Enrolments",
        line=dict(color='#1f77b4')
    )
)

fig.add_trace(
    go.Scatter(
        x=trend['date'],
        y=trend['updates'],
        name="Updates",
        line=dict(color='#ff7f0e')
    )
)

# Update layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Count",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400
)

# Add range slider
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig, config={"displayModeBar": True, "responsive": True})

# Add insights for the time series chart
with st.expander("📊 Insights", expanded=True):
    st.markdown("""
    - The time series shows the trend of Aadhaar enrolments and updates over time
    - Look for seasonal patterns or significant changes in the data
    - Compare the relative growth rates of enrolments vs updates
    - Hover over points to see exact values and dates
    """)

# Monthly patterns
st.subheader("📅 Monthly Patterns")

# Create monthly heatmap
if not filtered_df.empty:
    monthly = filtered_df.copy()
    monthly['month_name'] = monthly['date'].dt.month_name()
    monthly_pivot = monthly.pivot_table(
        index='month_name',
        columns='year',
        values='updates',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder months
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    monthly_pivot = monthly_pivot.reindex(month_order)
    
    fig_heatmap = px.imshow(
        monthly_pivot,
        labels=dict(x="Year", y="Month", color="Updates"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    fig_heatmap.update_xaxes(side="bottom")
    st.plotly_chart(fig_heatmap, config={"displayModeBar": True, "responsive": True})

# Add insights for the monthly heatmap
with st.expander("📊 Insights", expanded=True):
    st.markdown("""
    - The heatmap reveals monthly patterns in Aadhaar updates
    - Darker colors indicate higher update volumes
    - Look for consistent monthly patterns or anomalies
    - Compare year-over-year changes for the same month
    """)

# Update Type Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Update Type Distribution")
    if 'update_type' in filtered_df.columns and not filtered_df.empty:
        update_type_data = filtered_df.groupby('update_type')['updates'].sum().reset_index()
        if not update_type_data.empty:
            fig_pie = px.pie(
                update_type_data, 
                names="update_type", 
                values="updates",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_pie, config={"displayModeBar": True, "responsive": True})
        
            # Add insights for the update type distribution
            with st.expander("📊 Insights", expanded=True):
                st.markdown("""
                - The pie chart shows the distribution of different types of Aadhaar updates
                - The largest segments represent the most common update types
                - Consider focusing improvement efforts on frequently updated areas
                """)
        else:
            st.warning("No update type data available for the selected filters.")
    else:
        st.warning("Update type data not available.")

with col2:
    st.subheader("📊 Top Districts")
    if not filtered_df.empty:
        top_districts = filtered_df.groupby(['state', 'district'])['updates'].sum().nlargest(5).reset_index()
        if not top_districts.empty:
            fig_bar = px.bar(
                top_districts,
                x='updates',
                y='district',
                orientation='h',
                color='state',
                title='Top 5 Districts by Updates',
                labels={'updates': 'Total Updates', 'district': 'District'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, config={"displayModeBar": True, "responsive": True})
        
            # Add insights for the top districts
            with st.expander("📊 Insights", expanded=True):
                st.markdown("""
                - The horizontal bar chart shows the top performing districts
                - Color coding indicates different states
                - Consider analyzing what makes these districts perform well
                - Look for opportunities to replicate successful strategies in other areas
                """)
        else:
            st.warning("No district data available.")
    else:
        st.warning("No data available for the selected filters.")

# Anomaly Detection
st.subheader("🚨 Anomaly Detection")
st.markdown("""
This section identifies unusual patterns in the data that may require further investigation. 
Anomalies are detected using z-scores, highlighting values that are more than 2 standard 
deviations from the mean for each district.
""")

if not filtered_df.empty:
    # Calculate z-scores for updates by district
    filtered_df['update_zscore'] = filtered_df.groupby('district')['updates'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Identify anomalies (z-score > 2 or < -2)
    anomalies = filtered_df[
        (filtered_df['update_zscore'].abs() > 2) & 
        (filtered_df['updates'] > 0)  # Ignore zero values
    ]
    
    if not anomalies.empty:
        st.warning(f"🚨 {len(anomalies)} potential anomalies detected!")
        
        # Display top anomalies
        top_anomalies = anomalies.nlargest(5, 'update_zscore')
        for _, row in top_anomalies.iterrows():
            st.markdown(
                f"""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{row['date'].strftime('%Y-%m-%d')}</strong> | {row['district']}, {row['state']}<br>
                    Updates: <strong>{int(row['updates']):,}</strong> (Z-Score: {row['update_zscore']:.1f})
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show all anomalies in a collapsible section
        with st.expander("View all anomalies"):
            st.dataframe(
                anomalies[['date', 'state', 'district', 'updates', 'update_zscore']]
                .sort_values('update_zscore', ascending=False)
                .round(2),
                use_container_width=True
            )
    else:
        st.success("✅ No significant anomalies detected in the selected data range.")
        
        # Show data quality summary
        st.subheader("🔍 Data Quality Summary")
        
        # Calculate data completeness
        total_expected = len(filtered_df)
        missing_data = filtered_df[['enrolments', 'updates']].isnull().sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{total_expected:,}")
        col2.metric("Missing Enrolments", f"{missing_data['enrolments']:,}")
        col3.metric("Missing Updates", f"{missing_data['updates']:,}")
        
        # Data recency
        latest_date = filtered_df['date'].max()
        days_since_update = (pd.Timestamp.now() - latest_date).days
        
        st.metric(
            "Latest Data Available",
            latest_date.strftime("%B %Y"),
            delta=f"{days_since_update} days ago"
        )

# Insights
st.subheader("💡 Auto Insights")

if df.updates.mean() > 5000:
    st.write("• High update demand detected → recommend self-service update kiosks")

if anomalies.shape[0] > 0:
    st.write("• Sudden spikes indicate migration or data quality issues")

if df.enrolments.sum() < 10000:
    st.write("• Low enrolment region → deploy mobile Aadhaar vans")

# Basic Insights
st.subheader("📊 Key Insights")

# Calculate basic insights
total_enrolments = df.enrolments.sum()
total_updates = df.updates.sum()
avg_updates = df.groupby(['year', 'month'])['updates'].sum().mean()

# State analysis
state_analysis = df.groupby('state')['updates'].sum().reset_index()
top_state = state_analysis.loc[state_analysis['updates'].idxmax()] if not state_analysis.empty else {'state': 'N/A', 'updates': 0}

# Update type analysis
if 'update_type' in df.columns:
    update_type_analysis = df.groupby('update_type')['updates'].sum()
    if not update_type_analysis.empty:
        top_update_type = update_type_analysis.idxmax()
        top_update_percent = (update_type_analysis.max() / total_updates) * 100 if total_updates > 0 else 0
    else:
        top_update_type = "N/A"
        top_update_percent = 0
else:
    top_update_type = "N/A"
    top_update_percent = 0

# Display insights
st.markdown(f"""
- 📈 **Total Enrolments:** {total_enrolments:,}
- 🔄 **Total Updates:** {total_updates:,}
- 📅 **Avg. Monthly Updates:** {avg_updates:,.0f}
- 🏆 **Top State:** {top_state['state']} ({top_state['updates']:,} updates)
- 🏷️ **Most Common Update Type:** {top_update_type} ({top_update_percent:.1f}% of total)
""")
st.subheader("🗺️ State-wise Aadhaar Updates")

# Load India GeoJSON data
@st.cache_data(ttl=3600)
def load_geojson():
    try:
        with open('india_states.geojson', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {str(e)}")
        return None
def prepare_map_data(df, selected_states, selected_districts, selected_update_types, 
                   selected_age_groups, selected_genders, selected_center_types,
                   start_date, end_date):
    # Apply all filters
    filtered_df = df.copy()
    
    # Date filter
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= start_date) & 
        (filtered_df['date'].dt.date <= end_date)
    ]
    
    # State filter
    if 'All' not in selected_states and selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    
    # District filter
    if 'All' not in selected_districts and selected_districts:
        filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]
    
    # Update type filter
    if selected_update_types:
        filtered_df = filtered_df[filtered_df['update_type'].isin(selected_update_types)]
    
    # Age group filter
    if selected_age_groups:
        filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age_groups)]
    
    # Gender filter
    if selected_genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(selected_genders)]
    
    # Center type filter
    if selected_center_types:
        filtered_df = filtered_df[filtered_df['center_type'].isin(selected_center_types)]
    
    # Group by state for the map
    state_wise = filtered_df.groupby('state').agg({
        'enrolments': 'sum',
        'updates': 'sum',
        'date': 'count'
    }).reset_index()
    
    # Standardize state names to match GeoJSON
    state_wise['state'] = state_wise['state'].str.title()
    
    # Calculate updates per enrolment ratio
    state_wise['updates_per_enrolment'] = (state_wise['updates'] / state_wise['enrolments'].replace(0, 1)).round(2)
    
    return state_wise, filtered_df

# Map visualization section
st.subheader("🗺️ State-wise Aadhaar Data Analysis")

# Add a description that updates based on filters
filter_description = "Showing data"
if 'All' not in selected_states and selected_states:
    filter_description += f" for {', '.join(selected_states)}"
if 'All' not in selected_districts and selected_districts:
    filter_description += f" in districts: {', '.join(selected_districts)}"
if selected_update_types:
    filter_description += f" | Update types: {', '.join(selected_update_types)}"
if selected_age_groups:
    filter_description += f" | Age groups: {', '.join(selected_age_groups)}"
if selected_genders:
    filter_description += f" | Genders: {', '.join(selected_genders)}"
if selected_center_types:
    filter_description += f" | Center types: {', '.join(selected_center_types)}"

st.markdown(f"*{filter_description}*")

# Add map type selector with better descriptions
map_type = st.radio(
    "Select Map View:",
    ["Updates Count", "Enrolments", "Update Rate"],
    help="Choose what data to visualize on the map. 'Update Rate' shows updates per enrolment.",
    horizontal=True,
    key="map_view_selector"
)

try:
    # Load the GeoJSON data
    india_geojson = load_geojson()
    if india_geojson is None:
        raise Exception("Could not load GeoJSON data")
    
    # Add a spinner while processing map data
    with st.spinner('Updating map with current filters...'):
        # Apply filters and prepare data
        map_data, filtered_data = prepare_map_data(
            df,
            selected_states,
            selected_districts,
            selected_update_types,
            selected_age_groups,
            selected_genders,
            selected_center_types,
            start_date,
            end_date
        )
        
    if map_data.empty:
        st.warning("No data available for the selected filters. Try adjusting your filters.")
        st.stop()
    
    # Create choropleth map with better visualization
    if map_type == "Updates Count":
        color_col = 'updates'
        title = f'Total Aadhaar Updates by State\n({filter_description})'
        color_scale = 'YlOrRd'
        hover_data = {
            'state': True,
            'updates': ':,',
            'enrolments': ':,',
            'updates_per_enrolment': ':.2f'
        }
        hover_name = 'state'
        
    elif map_type == "Enrolments":
        color_col = 'enrolments'
        title = f'Total Aadhaar Enrolments by State\n({filter_description})'
        color_scale = 'YlGnBu'
        hover_data = {
            'state': True,
            'enrolments': ':,',
            'updates': ':,',
            'updates_per_enrolment': ':.2f'
        }
        hover_name = 'state'
        
    else:  # Update Rate
        color_col = 'updates_per_enrolment'
        title = f'Update Rate (Updates per Enrolment)\n({filter_description})'
        color_scale = 'RdYlGn'
        hover_data = {
            'state': True,
            'updates_per_enrolment': ':.2f',
            'updates': ':,',
            'enrolments': ':'
        }
        hover_name = 'state'
    
    # Set color label based on map type
    if map_type == "Updates Count":
        color_label = "Updates Count"
    elif map_type == "Enrolments":
        color_label = "Enrolments"
    else:  # Update Rate
        color_label = "Update Rate"
    
    # Define hover template
    hover_template = (
        "<b>%{location}</b><br>" +
        "Updates: %{customdata[0]:,}<br>" +
        "Enrolments: %{customdata[1]:,}<br>" +
        "Update Rate: %{customdata[2]:.2f}<br>" +
        "<extra></extra>"
    )
    
    # Create the choropleth map with enhanced features
    fig = px.choropleth(
        map_data,
        geojson=india_geojson,
        locations='state',
        featureidkey='properties.NAME_1',
        color=color_col,
        color_continuous_scale=color_scale
    )
    
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="Black",
        showocean=True,
        oceancolor="LightBlue",
        showlakes=True,
        lakecolor="LightBlue",
        showland=True,
        landcolor="WhiteSmoke",
        visible=False,
        center=dict(lat=20.5, lon=78.9),
        projection_scale=4.5
    )
    
    fig.update_layout(
        margin={"r":0, "t":40, "l":0, "b":0},
        height=650,
        geo=dict(
            scope='asia',
            showframe=False
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=color_label,
                font=dict(size=12)
            ),
            thickness=15,
            len=0.75,
            y=0.5,
            yanchor="middle",
            x=0.9
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "responsive": True})
    
    # Add summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌍 States Covered", len(map_data))
    with col2:
        st.metric("📊 Total Updates", f"{map_data['updates'].sum():,}")
    with col3:
        st.metric("👥 Total Enrolments", f"{map_data['enrolments'].sum():,}")
    
    # Add data table
    with st.expander("📋 View Detailed Data", expanded=False):
        st.dataframe(
            map_data[['state', 'updates', 'enrolments', 'updates_per_enrolment', 'date']]
            .rename(columns={
                'state': 'State',
                'updates': 'Total Updates',
                'enrolments': 'Total Enrolments',
                'updates_per_enrolment': 'Update Rate',
                'date': 'Records Count'
            })
            .sort_values('Total Updates', ascending=False)
            .style.format({
                'Total Updates': '{:,}',
                'Total Enrolments': '{:,}',
                'Update Rate': '{:.2f}'
            })
            .background_gradient(
                cmap='Blues',
                subset=['Total Updates', 'Total Enrolments']
            )
            .background_gradient(
                cmap='YlGnBu',
                subset=['Update Rate']
            ),
            use_container_width=True,
            height=400
        )
        
except Exception as e:
    st.error(f"Error creating map: {str(e)}")
    st.warning("Displaying filtered data table instead")
    if 'filtered_data' in locals():
        st.dataframe(filtered_data)
    else:
        st.dataframe(df)