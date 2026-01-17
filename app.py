import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import glob
import os
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No runtime found, using MemoryCacheStorageManager")
warnings.filterwarnings("ignore", message="The default of observed=False is deprecated")

# Page Config
st.set_page_config(
    page_title="Aadhaar Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-radius: 10px; margin-top: 1rem; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #eee; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("aadhaar_data.csv")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m', errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Ensure categorical columns
        cat_cols = ['state', 'update_type', 'age_group', 'gender', 'center_type']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Derived time columns
        df['month'] = df['date'].dt.month_name()
        df['year'] = df['date'].dt.year
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Loading detailed data...")
def load_detailed_data():
    """
    Loads and merges split CSVs from api_data folders with optimized performance
    and better error handling.
    """
    detailed_data = {}
    
    def load_folder(folder_name, progress_bar=None):
        """Helper to load all CSV files in a folder with progress tracking"""
        path = os.path.join(os.getcwd(), folder_name, "*.csv")
        files = glob.glob(path)
        
        if not files:
            if progress_bar:
                progress_bar.progress(100, text=f"No files found in {folder_name}")
            return pd.DataFrame()
            
        dfs = []
        total_files = len(files)
        
        for i, f in enumerate(files, 1):
            try:
                # Use low_memory=False to avoid mixed type warnings
                # Use chunks for very large files
                chunk_size = 10000
                chunks = []
                for chunk in pd.read_csv(f, low_memory=False, chunksize=chunk_size):
                    chunks.append(chunk)
                
                if chunks:
                    dfs.append(pd.concat(chunks, ignore_index=True))
                
                if progress_bar:
                    progress = int((i / total_files) * 100)
                    progress_bar.progress(progress, text=f"Loading {folder_name}... ({i}/{total_files} files)")
                    
            except Exception as e:
                print(f"Error reading {os.path.basename(f)}: {str(e)[:200]}...")
                continue
                
        if not dfs:
            return pd.DataFrame()
            
        # Concatenate all dataframes at once
        return pd.concat(dfs, ignore_index=True)

    # Create a progress container
    progress_container = st.empty()
    
    try:
        # Initialize progress bar
        with st.spinner("Loading detailed data..."):
            progress_bar = progress_container.progress(0, text="Starting data load...")
            
            # Load Enrolment Data
            enrol_df = load_folder("api_data_aadhar_enrolment", progress_bar)
            if not enrol_df.empty:
                enrol_df['date'] = pd.to_datetime(enrol_df['date'], format='%d-%m-%Y', errors='coerce')
                enrol_df.rename(columns={
                    'age_18_greater': 'age_18_plus',
                    'age_0_5': 'age_0_5',
                    'age_5_17': 'age_5_17'
                }, inplace=True)
                detailed_data['enrolment'] = enrol_df
                progress_bar.progress(33, text=f"✅ Loaded {len(enrol_df):,} enrolment records")
            
            # Load Biometric Data
            bio_df = load_folder("api_data_aadhar_biometric", progress_bar)
            if not bio_df.empty:
                bio_df['date'] = pd.to_datetime(bio_df['date'], format='%d-%m-%Y', errors='coerce')
                bio_df.rename(columns={
                    'bio_age_17_': 'bio_age_18_plus',
                    'bio_age_0_5': 'bio_age_0_5',
                    'bio_age_5_17': 'bio_age_5_17'
                }, inplace=True)
                detailed_data['biometric'] = bio_df
                progress_bar.progress(66, text=f"✅ Loaded {len(bio_df):,} biometric records")
            
            # Load Demographic Data
            demo_df = load_folder("api_data_aadhar_demographic", progress_bar)
            if not demo_df.empty:
                demo_df['date'] = pd.to_datetime(demo_df['date'], format='%d-%m-%Y', errors='coerce')
                demo_df.rename(columns={
                    'demo_age_17_': 'demo_age_18_plus',
                    'demo_age_0_5': 'demo_age_0_5',
                    'demo_age_5_17': 'demo_age_5_17'
                }, inplace=True)
                detailed_data['demographic'] = demo_df
                progress_bar.progress(100, text=f"✅ Loaded {len(demo_df):,} demographic records")
                
            # Brief pause to show completion
            time.sleep(0.3)
            
    except Exception as e:
        st.error(f"❌ Error loading detailed data: {str(e)}")
        st.exception(e)  # Show full traceback in the UI for debugging
    finally:
        # Clear the progress bar when done
        progress_container.empty()
    
    return detailed_data

@st.cache_data(ttl=3600)
def load_geojson():
    try:
        with open('india_states.geojson', 'r') as f:
            return json.load(f)
    except Exception:
        return None

df = load_data()
if df.empty:
    st.error("Main data file (aadhaar_data.csv) not found or empty.")
    st.stop()

# Pre-load detailed data for all tabs
detailed = load_detailed_data()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.title('🎛️ Analytics Controls')

# Date Range
min_date = df['date'].min().date()
max_date = df['date'].max().date()
st.sidebar.subheader('📅 Period')
start_date = st.sidebar.date_input('Start', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End', max_date, min_value=min_date, max_value=max_date)

# Location Filters
st.sidebar.subheader('📍 Location')
all_states = ['All'] + sorted(df['state'].unique().tolist())
selected_states = st.sidebar.multiselect('States', all_states[1:], default=all_states[1:])
# Filter districts based on state
if 'All' in selected_states or not selected_states:
    districts = df['district'].unique()
else:
    districts = df[df['state'].isin(selected_states)]['district'].unique()
selected_districts = st.sidebar.multiselect('Districts', ['All'] + sorted(districts.tolist()), default=['All'])

# Category Filters - Accordion
with st.sidebar.expander("📝 Advanced Filters"):
    selected_update_types = st.multiselect('Update Types', df['update_type'].unique(), default=df['update_type'].unique())
    selected_age_groups = st.multiselect('Age Groups', df['age_group'].unique(), default=df['age_group'].unique())
    selected_genders = st.multiselect('Gender', df['gender'].unique(), default=df['gender'].unique())

if st.sidebar.button('Reset Filters', use_container_width=True):
    st.rerun()

# ----------------- FILTER LOGIC -----------------
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)]

if selected_states and 'All' not in selected_states:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
if selected_districts and 'All' not in selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]
if selected_update_types:
    filtered_df = filtered_df[filtered_df['update_type'].isin(selected_update_types)]
if selected_age_groups:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age_groups)]
if selected_genders:
    filtered_df = filtered_df[filtered_df['gender'].isin(selected_genders)]

# ----------------- MAIN DASHBOARD -----------------
st.title("📊 Aadhaar Insights Dashboard")
st.markdown("Comprehensive analytics on enrolment trends, demographics, and social impact.")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", 
    "👥 Demographics", 
    "🔍 Deep Dive", 
    "🔮 Forecast & Impact"
])

# Use helper function for common charts to avoid repetition if needed, but keeping separate for clarity for now.

# --- TAB 1: OVERVIEW ---
with tab1:
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_enrolments = filtered_df['enrolments'].sum()
    total_updates = filtered_df['updates'].sum()
    avg_monthly = filtered_df.groupby('date')['updates'].sum().mean()
    
    # Growth Calculation
    growth = 0
    updates_by_date = filtered_df.groupby('date')['updates'].sum()
    if len(updates_by_date) > 1 and updates_by_date.iloc[0] > 0:
        growth = ((updates_by_date.iloc[-1] / updates_by_date.iloc[0]) - 1) * 100
    
    kpi1.metric("Total Enrolments", f"{total_enrolments:,}")
    kpi2.metric("Total Updates", f"{total_updates:,}")
    kpi3.metric("Avg Monthly Updates", f"{avg_monthly:,.0f}")
    kpi4.metric("Growth Rate", f"{growth:+.1f}%")

    st.divider()

    # Map and Time Series
    col_map, col_trend = st.columns([1, 1])
    
    with col_map:
        st.subheader("🗺️ Geographic Distribution")
        # Reuse map logic but simplified with Aggregation
        state_agg = filtered_df.groupby('state')[['enrolments', 'updates']].sum().reset_index()
        india_geojson = load_geojson()
        
        if india_geojson:
            # Extract Lat/Lon from GeoJSON Points
            try:
                state_coords = []
                for feature in india_geojson['features']:
                    st_nm = feature['properties']['ST_NM']
                    coords = feature['geometry']['coordinates']
                    state_coords.append({'state': st_nm, 'lon': coords[0], 'lat': coords[1]})
                
                coords_df = pd.DataFrame(state_coords)
                # Merge with data
                map_data = pd.merge(state_agg, coords_df, on='state', how='inner')
                
                if not map_data.empty:
                    fig_map = px.scatter_geo(
                        map_data,
                        lon='lon',
                        lat='lat',
                        text='state',
                        size='updates',
                        color='updates',
                        hover_name='state',
                        hover_data=['enrolments'],
                        color_continuous_scale='Viridis',
                        title='Total Updates by State (Bubble Map)',
                        projection='natural earth'
                    )
                    fig_map.update_geos(
                        visible=True, resolution=50,
                        showcountries=True, countrycolor="Black",
                        showsubunits=True, subunitcolor="Gray",
                        fitbounds="locations"
                    )
                    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("No matching state data found for map.")
            except Exception as e:
                st.error(f"Error processing map data: {e}")
        else:
            st.warning("GeoJSON not loaded")

    with col_trend:
        st.subheader("📈 Trends Over Time")
        trend_data = filtered_df.groupby('date')[['enrolments', 'updates']].sum().reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['enrolments'], name='Enrolments', fill='tozeroy'))
        fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['updates'], name='Updates', fill='tozeroy'))
        fig_trend.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: DEMOGRAPHICS ---
with tab2:
    st.subheader("👥 Demographic Analysis")
    
    col_sun, col_pyr = st.columns(2)
    
    with col_sun:
        st.markdown("##### 📍 Distribution Hierarchy (State -> District)")
        # Limit to top 5 states for performance if too many
        top_states = filtered_df.groupby('state')['updates'].sum().nlargest(5).index
        sunburst_df = filtered_df[filtered_df['state'].isin(top_states)]
        # Aggregation for sunburst
        sun_agg = sunburst_df.groupby(['state', 'district'])['updates'].sum().reset_index()
        if not sun_agg.empty:
            fig_sun = px.sunburst(sun_agg, path=['state', 'district'], values='updates', 
                                 title="Updates Distribution (Top 5 States)")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("Insufficient data for Sunburst")

    with col_pyr:
        st.markdown("##### 🚻 Gender Split")
        gender_agg = filtered_df.groupby(['age_group', 'gender'])[['enrolments', 'updates']].sum().reset_index()
        if not gender_agg.empty and 'Male' in gender_agg['gender'].unique() and 'Female' in gender_agg['gender'].unique():
             # Creating a population pyramid style chart
             male_data = gender_agg[gender_agg['gender'] == 'Male'].copy()
             female_data = gender_agg[gender_agg['gender'] == 'Female'].copy()
             
             male_data['updates'] = male_data['updates'] * -1 # Negative for left side
             
             fig_pyr = go.Figure()
             fig_pyr.add_trace(go.Bar(y=male_data['age_group'], x=male_data['updates'], name='Male', orientation='h', marker_color='blue'))
             fig_pyr.add_trace(go.Bar(y=female_data['age_group'], x=female_data['updates'], name='Female', orientation='h', marker_color='pink'))
             
             fig_pyr.update_layout(title="Updates by Age & Gender (Pyramid)", barmode='overlay', 
                                  xaxis=dict(tickvals=[-10000, -5000, 0, 5000, 10000], title="Updates (Male | Female)"))
             st.plotly_chart(fig_pyr, use_container_width=True)
        else:
            st.info("Insufficient gender data")

    # update types
    st.subheader("🔄 Update Types Overview")
    if 'update_type' in filtered_df.columns:
        type_cnt = filtered_df.groupby('update_type')['updates'].sum().reset_index()
        fig_pie = px.pie(type_cnt, names='update_type', values='updates', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- NEW: Detailed Age Group Analysis ---
    st.markdown("---")
    st.subheader("📊 Detailed Age Group Analytics (New Data)")
    
    detailed = load_detailed_data()
    
    if 'enrolment' in detailed and not detailed['enrolment'].empty:
        e_df = detailed['enrolment']
        if 'All' not in selected_states:
            e_df = e_df[e_df['state'].isin(selected_states)]
        
        # Aggregate by age group with defensive check
        age_cols = [c for c in ['age_0_5', 'age_5_17', 'age_18_plus'] if c in e_df.columns]
        if age_cols:
            age_sums = e_df[age_cols].sum().reset_index()
            age_sums.columns = ['Age Group', 'Count']
            
            fig_age = px.bar(age_sums, x='Age Group', y='Count', color='Age Group', 
                             title='Total Enrolments by Age Group', text_auto=True)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age group columns missing in enrolment data.")
    else:
        st.info("Detailed enrolment data not available for age analysis.")

    # --- NEW: Updates Comparison ---
    col_bio, col_demo = st.columns(2)
    with col_bio:
        if 'biometric' in detailed and not detailed['biometric'].empty:
            b_df = detailed['biometric']
            if 'All' not in selected_states:
                b_df = b_df[b_df['state'].isin(selected_states)]
            total_bio = b_df[['bio_age_5_17', 'bio_age_18_plus']].sum().sum()
            st.metric("Total Biometric Updates", f"{total_bio:,}")
    
    with col_demo:
        if 'demographic' in detailed and not detailed['demographic'].empty:
            d_df = detailed['demographic']
            if 'All' not in selected_states:
                d_df = d_df[d_df['state'].isin(selected_states)]
            total_demo = d_df[['demo_age_5_17', 'demo_age_18_plus']].sum().sum()
            st.metric("Total Demographic Updates", f"{total_demo:,}")

# --- TAB 3: DEEP DIVE ---
with tab3:
    col_sca, col_cor = st.columns([2, 1])
    
    with col_sca:
        st.subheader("🔍 District Performance Matrix")
        st.caption("Scatter plot showing relationship between Enrolments and Updates. Outliers indicate high activity.")
        dist_agg = filtered_df.groupby(['state', 'district'])[['enrolments', 'updates']].sum().reset_index()
        
        if not dist_agg.empty:
            fig_sca = px.scatter(
                dist_agg, x='enrolments', y='updates', 
                color='state', hover_data=['district'], size='updates',
                log_x=True, log_y=True, 
                title="Enrolments vs Updates (Log Scale)"
            )
            st.plotly_chart(fig_sca, use_container_width=True)
            
    with col_cor:
        st.subheader("🔗 Correlation")
        st.caption("Correlation between key metrics")
        corr = filtered_df[['enrolments', 'updates']].corr()
        fig_hm = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig_hm, use_container_width=True)

    # --- NEW: Pincode Analysis ---
    if 'All' not in selected_districts and selected_districts:
        st.subheader(f"📍 Pincode Level Analysis")
        if 'enrolment' in detailed and not detailed['enrolment'].empty:
            pin_df = detailed['enrolment']
            pin_df = pin_df[pin_df['district'].isin(selected_districts)]
            
            if not pin_df.empty:
                # Top 15 Pincodes by Enrolment
                cols_to_sum = [c for c in ['age_0_5', 'age_5_17', 'age_18_plus'] if c in pin_df.columns]
                if cols_to_sum:
                    top_pins = pin_df.groupby('pincode')[cols_to_sum].sum()
                    top_pins['Total'] = top_pins.sum(axis=1)
                    top_pins = top_pins.sort_values('Total', ascending=False).head(15).reset_index()
                    top_pins['pincode'] = top_pins['pincode'].astype(str)
                    
                    fig_pin = px.bar(top_pins, x='pincode', y='Total', color='Total',
                                    title="Top 15 Pincodes by Enrolment Activity",
                                    labels={'Total': 'Enrolments', 'pincode': 'Pincode'})
                    st.plotly_chart(fig_pin, use_container_width=True)
                else:
                    st.info("Required age columns missing for pincode analysis.")
            else:
                st.info("No pincode data found for selected districts.")
    else:
        st.info("Select specific District(s) in sidebar to see Pincode-level analysis.")

    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection (Statistical Z-Score)")
    filtered_df['z_score'] = (filtered_df['updates'] - filtered_df['updates'].mean()) / filtered_df['updates'].std()
    anomalies = filtered_df[filtered_df['z_score'].abs() > 3]
    
    if not anomalies.empty:
        st.warning(f"Detected {len(anomalies)} anomalies (Z-Score > 3)")
        st.dataframe(anomalies[['date', 'state', 'district', 'updates', 'enrolments', 'z_score']].head(10))
    else:
        st.success("No significant anomalies detected.")

# --- TAB 4: FORECAST & IMPACT ---
with tab4:
    st.subheader("🔮 Predictive Analytics & Impact")
    
    col_fore, col_impact = st.columns(2)
    
    with col_fore:
        st.markdown("#### 📈 3-Month Forecast (Simple Moving Average)")
        monthly_trend = filtered_df.groupby('date')['updates'].sum()
        if len(monthly_trend) > 3:
            # Simple forecast
            last_3_avg = monthly_trend.rolling(window=3).mean().iloc[-1]
            future_dates = [monthly_trend.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
            forecast_values = [last_3_avg * (1 + 0.02 * i) for i in range(1, 4)] # Assuming 2% growth
            
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_values})
            
            fig_fore = go.Figure()
            fig_fore.add_trace(go.Scatter(x=monthly_trend.index, y=monthly_trend.values, name='Historical'))
            fig_fore.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_fore, use_container_width=True)
        else:
            st.info("Need more data for forecasting")
            
    with col_impact:
        st.markdown("#### 🌱 Digital Inclusion Impact")
        
        # Calculate a hypothetical "Inclusion Score"
        # Logic: Higher updates + Higher Enrollments per capita (proxy) = Better Inclusion
        total_recs = len(filtered_df)
        high_activity = len(filtered_df[filtered_df['updates'] > filtered_df['updates'].median()])
        inclusion_score = min(100, (high_activity / total_recs) * 100 + 50) if total_recs > 0 else 0
        
        st.metric("Digital Inclusion Index (DII)", f"{inclusion_score:.1f}/100", delta="Aggregated Metric")
        st.progress(inclusion_score / 100)
        
        st.info("""
        **DII Explanation**:  
        This synthetic index measures the reach of Aadhaar services based on consistent update activity across districts.
        """)

    # Recommendations
    st.subheader("💡 Policy Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.success("✅ **Infrastructure**: Expand centers in districts with High Z-Scores (See Deep Dive).")
    with rec_col2:
        st.warning("⚠️ **Demographics**: Targeted camps needed for under-represented age groups (See Demographics).")

st.markdown("---")
st.caption(f"Aadhaar Insights | Generated on {datetime.now().strftime('%Y-%m-%d')}")