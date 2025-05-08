import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.font_manager as fm
import ssl
import psycopg2
from psycopg2.extras import RealDictCursor
import json

import matplotlib as mpl
mpl.font_manager.fontManager.addfont('fonts/THSarabunNew.ttf') # Ensuring matplotlib recognizes the font
mpl.rc('font', family='TH Sarabun New') # Setting the default font to TH Sarabun New

# Fix for the SSL certificate issue with NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Only import wordcloud and nltk if needed - with error handling
try:
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk_available = True
except ImportError:
    nltk_available = False
    st.warning("WordCloud and/or NLTK not available. Text analysis features will be limited.")

# Only import pythainlp if needed - with error handling
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    pythainlp_available = True
except ImportError:
    pythainlp_available = False
    st.warning("PythaiNLP not available. Thai language processing will be limited.")

# Thai font setup
try:
    thai_font_path = os.path.join("fonts", "THSarabunNew.ttf")
    if os.path.exists(thai_font_path):
        font_prop = fm.FontProperties(fname=thai_font_path)
    else:
        font_prop = None
except Exception:
    font_prop = None

# Set page configuration
st.set_page_config(
    page_title="Traffy Fondue Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("📊 Traffy Fondue Data Analysis Dashboard")
st.markdown("""
This dashboard analyzes data from Traffy Fondue, a platform for citizens to report urban issues in Bangkok.
Explore the patterns, response times, and common problems reported by citizens.
""")

# Database configuration - with environment variable support
DB_HOST = os.getenv("DB_HOST", "localhost")  # Default to localhost instead of postgres
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "airflow")
DB_USER = os.getenv("DB_USER", "airflow")
DB_PASSWORD = os.getenv("DB_PASSWORD", "airflow")

# Function to connect to PostgreSQL
def connect_to_db():
    try:
        st.info(f"Connecting to database at {DB_HOST}:{DB_PORT}...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        st.success("Database connection established!")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.info("""
        **Database Connection Troubleshooting:**
        1. Make sure PostgreSQL is running
        2. Check your connection settings
        3. If running locally, try setting DB_HOST to 'localhost'
        4. If running in Docker, make sure the network is properly configured
        
        You can set database connection parameters using environment variables:
        - DB_HOST: Database hostname (default: localhost)
        - DB_PORT: Database port (default: 5432)
        - DB_NAME: Database name (default: airflow)
        - DB_USER: Database username (default: airflow)
        - DB_PASSWORD: Database password (default: airflow)
        """)
        return None

# Function to load and preprocess data from PostgreSQL
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def load_data():
    conn = connect_to_db()
    if conn is None:
        # Add option for demo data when DB connection fails
        use_demo = st.checkbox("Use demo data instead?", value=True)
        if use_demo:
            st.info("Using demo data. Note that this is not real-time data.")
            # Try to load demo data from a local file if available
            try:
                demo_file = "data_and_rain.csv"
                if os.path.exists(demo_file):
                    # Load demo data
                    df = pd.read_csv(demo_file, nrows=1000)
                    
                    # Convert timestamp and last_activity to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if 'last_activity' in df.columns:
                        df['last_activity'] = pd.to_datetime(df['last_activity'])
                    
                    # Calculate resolution time in hours (if needed)
                    if 'timestamp' in df.columns and 'last_activity' in df.columns:
                        df['resolution_time_hours'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 3600
                        df['resolution_time_days'] = df['resolution_time_hours'] / 24
                    
                    # Process array fields if they exist in the demo data
                    # In CSV, these might be stored as strings that look like lists
                    if 'type' in df.columns:
                        try:
                            df['type_list'] = df['type'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
                            df['type_list_str'] = df['type_list'].apply(lambda x: str(x))
                        except:
                            df['type_list'] = [[] for _ in range(len(df))]
                            df['type_list_str'] = df['type_list'].apply(lambda x: str(x))
                    
                    if 'organization' in df.columns:
                        try:
                            df['organization_list'] = df['organization'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
                            df['organization_list_str'] = df['organization_list'].apply(lambda x: str(x))
                        except:
                            df['organization_list'] = [[] for _ in range(len(df))]
                            df['organization_list_str'] = df['organization_list'].apply(lambda x: str(x))
                    
                    # Extract coordinates if stored in a single field
                    if 'coords' in df.columns:
                        try:
                            # Try to extract coordinates if they're in "lat,long" format
                            df[['latitude', 'longitude']] = df['coords'].str.split(',', expand=True).astype(float)
                        except:
                            st.warning("Could not parse coordinates from the coords column.")
                    
                    # Fill missing district information
                    if 'district' in df.columns:
                        df['district'] = df['district'].fillna('ไม่ระบุ')
                    
                    st.success(f"Successfully loaded demo data with {len(df)} records")
                    return df
                else:
                    st.error("Demo data file not found")
            except Exception as e:
                st.error(f"Error loading demo data: {e}")
        return pd.DataFrame()
    
    try:
        # Use RealDictCursor to get column names
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to get all issues
        query = """
        SELECT 
            id, 
            message_id, 
            type,
            ST_X(coordinates::geometry) as longitude,
            ST_Y(coordinates::geometry) as latitude,
            problem_type_fondue,
            org,
            org_action,
            description,
            photo_url,
            address,
            subdistrict,
            district,
            province,
            timestamp,
            state,
            last_activity
        FROM issues;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Close connection
        cursor.close()
        conn.close()
        
        # If DataFrame is empty, return it
        if df.empty:
            st.warning("No data found in the database. The table might be empty.")
            return df
        
        # Convert timestamp and last_activity to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['last_activity'] = pd.to_datetime(df['last_activity'])
        
        # Calculate resolution time in hours
        df['resolution_time_hours'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 3600
        df['resolution_time_days'] = df['resolution_time_hours'] / 24
        
        # Process array fields (problem_type_fondue, org, org_action)
        # Convert PostgreSQL arrays to Python lists
        if 'problem_type_fondue' in df.columns:
            df['type_list'] = df['problem_type_fondue'].apply(lambda x: [] if x is None else (x if isinstance(x, list) else json.loads(x.replace('{', '[').replace('}', ']'))))
            df['type_list_str'] = df['type_list'].apply(lambda x: str(x))
        else:
            df['type_list'] = [[] for _ in range(len(df))]
            df['type_list_str'] = df['type_list'].apply(lambda x: str(x))
        
        if 'org' in df.columns:
            df['organization_list'] = df['org'].apply(lambda x: [] if x is None else (x if isinstance(x, list) else json.loads(x.replace('{', '[').replace('}', ']'))))
            df['organization_list_str'] = df['organization_list'].apply(lambda x: str(x))
        else:
            df['organization_list'] = [[] for _ in range(len(df))]
            df['organization_list_str'] = df['organization_list'].apply(lambda x: str(x))
        
        # Fill missing district information
        df['district'] = df['district'].fillna('ไม่ระบุ')
        
        return df
        
    except Exception as e:
        st.error(f"Error querying database: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

# Function to load rainfall data
@st.cache_data(ttl=300)
def load_rainfall_data():
    try:
        # Try to load rainfall data
        rainfall_file = "data_and_rain.csv"
        if os.path.exists(rainfall_file):
            rain_df = pd.read_csv(rainfall_file, nrows=1000)
            return rain_df
        else:
            st.warning("Rainfall data file not found.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading rainfall data: {e}")
        return pd.DataFrame()

# Load the data
try:
    df = load_data()
    rain_df = load_rainfall_data()
    
    if df.empty:
        st.error("No data found in the database. Please make sure the Airflow DAG to fetch data has been run.")
        st.stop()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    # Create empty dataframe for demonstration
    df = pd.DataFrame()
    st.stop()

# Display raw data with toggle
with st.expander("Show raw data"):
    st.dataframe(df)

# Create tabs for different analyses
tab1, tab_map, tab2, tab3, tab4, tab_rain, tab_model = st.tabs(["Overview", "Map Visualization", "Issue Analysis", "Response Time", "Text Analysis", "Rainfall Analysis", "Model Prediction"])

with tab1:
    st.header("Overview of Reported Issues")
    
    # col1, col2 = st.columns(2)
    
   
        # Count issues by status
    status_counts = df['state'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.pie(status_counts, values='Count', names='Status', 
                title='Distribution of Issue Status',
                color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # with col2:
    #     # Issues over time
    #     df_time = df.copy()
    #     df_time['date'] = df_time['timestamp'].dt.date  # แสดงเฉพาะวันที่ ไม่รวมเวลา
    #     daily_counts = df_time.groupby('date').size().reset_index(name='count')

    #     fig = px.line(daily_counts, x='date', y='count', 
    #                 title='Number of Reported Issues by Day',
    #                 labels={'count': 'Number of Issues', 'date': 'Date'})
    #     fig.update_layout(xaxis_tickangle=-45)
    #     # บังคับให้แกน Y เริ่มจาก 0
    #     fig.update_yaxes(rangemode="tozero")
    #     st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution
    st.subheader("Geographic Distribution of Issues")
    
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['District', 'Count']
    district_counts = district_counts.sort_values('Count', ascending=False)
    
    fig = px.bar(district_counts.head(15), x='District', y='Count',
                title='Top 15 Districts by Number of Issues',
                color='Count', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analysis of Issue Types")
    
    # Extract all unique issue types
    all_types = [item for sublist in df['type_list'] for item in sublist]
    type_counts = Counter(all_types)
    
    # Prepare data for visualization
    issue_df = pd.DataFrame.from_dict(type_counts, orient='index').reset_index()
    issue_df.columns = ['Issue Type', 'Count']
    issue_df = issue_df.sort_values('Count', ascending=False)
    
    # Display issues by type
    st.subheader("Most Common Issue Types")
    
    fig = px.bar(issue_df.head(10), x='Issue Type', y='Count',
                title='Top 10 Issue Types Reported',
                color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
    
    # Organizations responsible for issues
    st.subheader("Organizations Handling Issues")
    
    all_orgs = [item for sublist in df['organization_list'] for item in sublist if item != 'ไม่ระบุ']
    org_counts = Counter(all_orgs)
    
    org_df = pd.DataFrame.from_dict(org_counts, orient='index').reset_index()
    org_df.columns = ['Organization', 'Count']
    org_df = org_df.sort_values('Count', ascending=False)
    
    fig = px.bar(org_df.head(10), x='Organization', y='Count',
                title='Top 10 Organizations Handling Issues',
                color='Count', color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Type and Organization relationship
    st.subheader("Relationship Between Issue Type and Organization")
    
    # Get top 5 issue types
    top_types = issue_df.head(5)['Issue Type'].tolist()
    
    # Filter for these top types
    filtered_data = []
    for i, row in df.iterrows():
        for issue_type in row['type_list']:
            if issue_type in top_types:
                for org in row['organization_list']:
                    filtered_data.append({'Issue Type': issue_type, 'Organization': org})
    
    if filtered_data:
        type_org_df = pd.DataFrame(filtered_data)
        type_org_counts = type_org_df.groupby(['Issue Type', 'Organization']).size().reset_index(name='Count')
        type_org_counts = type_org_counts.sort_values(['Issue Type', 'Count'], ascending=[True, False])
        
        # Keep top 3 organizations for each issue type
        top_orgs_per_type = []
        for issue_type in top_types:
            top_for_type = type_org_counts[type_org_counts['Issue Type'] == issue_type].head(3)
            top_orgs_per_type.append(top_for_type)
        
        top_org_df = pd.concat(top_orgs_per_type)
        
        fig = px.bar(top_org_df, x='Issue Type', y='Count', color='Organization',
                    title='Top Organizations Handling the Most Common Issue Types',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Response Time Analysis")
    
    # Basic statistics on resolution time
    st.subheader("Resolution Time Statistics (Days)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{df['resolution_time_days'].mean():.1f} days")
    with col2:
        st.metric("Median", f"{df['resolution_time_days'].median():.1f} days")
    with col3:
        st.metric("Minimum", f"{df['resolution_time_days'].min():.1f} days")
    with col4:
        st.metric("Maximum", f"{df['resolution_time_days'].max():.1f} days")
    
    # Distribution of resolution times
    st.subheader("Distribution of Resolution Times")
    
    fig = px.histogram(df, x='resolution_time_days', nbins=30,
                      title='Distribution of Resolution Times in Days',
                      labels={'resolution_time_days': 'Resolution Time (Days)'},
                      color_discrete_sequence=['darkblue'])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Resolution time by issue type
    st.subheader("Average Resolution Time by Issue Type")
    
    # Prepare data for this visualization
    issue_resolution_data = []
    for i, row in df.iterrows():
        for issue_type in row['type_list']:
            issue_resolution_data.append({
                'Issue Type': issue_type,
                'Resolution Time (Days)': row['resolution_time_days']
            })
    
    issue_resolution_df = pd.DataFrame(issue_resolution_data)
    avg_resolution_by_type = issue_resolution_df.groupby('Issue Type')['Resolution Time (Days)'].mean().reset_index()
    avg_resolution_by_type = avg_resolution_by_type.sort_values('Resolution Time (Days)', ascending=False)
    
    fig = px.bar(avg_resolution_by_type.head(10), x='Issue Type', y='Resolution Time (Days)',
                title='Average Resolution Time by Issue Type (Top 10 Slowest)',
                color='Resolution Time (Days)', color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Resolution time by organization
    st.subheader("Average Resolution Time by Organization")
    
    org_resolution_data = []
    for i, row in df.iterrows():
        for org in row['organization_list']:
            if org != 'ไม่ระบุ':
                org_resolution_data.append({
                    'Organization': org,
                    'Resolution Time (Days)': row['resolution_time_days']
                })
    
    org_resolution_df = pd.DataFrame(org_resolution_data)
    avg_resolution_by_org = org_resolution_df.groupby('Organization')['Resolution Time (Days)'].mean().reset_index()
    avg_resolution_by_org = avg_resolution_by_org.sort_values('Resolution Time (Days)', ascending=False)
    
    fig = px.bar(avg_resolution_by_org.head(10), x='Organization', y='Resolution Time (Days)',
                title='Average Resolution Time by Organization (Top 10 Slowest)',
                color='Resolution Time (Days)', color_continuous_scale='RdYlGn_r')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Text Analysis of Issue Comments")
    
    # Word cloud of comments - only if libraries are available
    st.subheader("Word Cloud of Issue Comments")
    
    # Check if required libraries are available
    if not nltk_available or not pythainlp_available:
        st.warning("Cannot create word cloud: Required libraries (WordCloud, NLTK, or PythaiNLP) are not available.")
        st.info("To enable full text analysis features, please install the required libraries:")
        st.code("pip install wordcloud nltk pythainlp")
    
    # Check if comments are available
    elif 'description' in df.columns and not df['description'].dropna().empty:
        # Combine all comments
        all_comments = ' '.join(df['description'].dropna().astype(str))
        
        try:
            # Use pythainlp for Thai word tokenization
            tokens = word_tokenize(all_comments, engine='newmm')
            
            # Get Thai stopwords
            try:
                thai_stops = list(thai_stopwords())
            except:
                thai_stops = []
            
            # Add custom stopwords
            custom_stops = [
                'ไม่', 'ให้', 'แล้ว', 'เป็น', 'มี', 'การ', 'ของ', 'ก็', 'ที่', 'ได้', 'ว่า', 'จะ',
                'ใน', 'แต่', 'และ', 'หรือ', 'มาก', 'กับ', 'จาก', 'ถ้า', 'อยู่', 'อย่าง', 'ซึ่ง',
                'ต้อง', 'ตาม', 'หาก', 'เพื่อ', 'โดย', 'เมื่อ', 'เพราะ', 'นี้', 'นั้น', 'จึง',
                'ยัง', 'แบบ', 'ทั้ง', 'เคย', 'กว่า', 'อีก', 'ต่อ', 'ๆ', '1', '2', '3', '4', '5',
                'ครับ', 'ค่ะ', 'น่า', 'มัน', 'กทม', 'กรุงเทพมหานคร'
            ]
            stopwords_list = set(thai_stops + custom_stops)
            
            # Filter out stopwords
            filtered_tokens = [token for token in tokens if token not in stopwords_list and len(token) > 1]
            
            # Create new text after filtering
            filtered_text = ' '.join(filtered_tokens)
            
            # Let user choose word cloud type
            cloud_type = st.radio(
                "Choose Word Cloud Type:",
                ["Classic", "Treemap (Rectangle)"]
            )
            
            if cloud_type == "Classic":
                # Create classic word cloud
                try:
                    # Find Thai font
                    thai_font = None
                    for font in fm.fontManager.ttflist:
                        if any(name in font.name.lower() for name in ['thai', 'tahoma', 'sarabun', 'angsana']):
                            thai_font = fm.findfont(fm.FontProperties(family=font.name))
                            break
                    
                    # Create word cloud
                    wordcloud = WordCloud(
                        font_path=thai_font,
                        width=800, 
                        height=400,
                        background_color='white',
                        stopwords=stopwords_list,
                        max_words=100,
                        contour_width=3,
                        contour_color='steelblue',
                        regexp=r"[^\s]+"  # Help support Thai language
                    ).generate(filtered_text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating word cloud: {e}")
                
            else:
                # Create treemap word cloud
                st.subheader("Word Treemap")
                
                # Count frequency of each word
                word_counts = Counter(filtered_tokens)
                
                try:
                    # Import squarify for treemap visualization
                    import squarify
                    
                    # Select top 30 most frequent words
                    top_words = dict(word_counts.most_common(30))
                    
                    # Create treemap
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Set colors
                    colors = plt.cm.viridis(np.linspace(0, 1, len(top_words)))
                    
                    # Create treemap
                    squarify.plot(
                        sizes=list(top_words.values()),
                        label=list(top_words.keys()),
                        alpha=0.8,
                        color=colors,
                        ax=ax,
                        text_kwargs={'fontproperties': font_prop} if font_prop else {}
                    )
                    
                    # Customize graph
                    plt.axis('off')
                    plt.title('Most Frequent Words in Issue Comments')
                    
                    # Show treemap
                    st.pyplot(fig)
                except ImportError:
                    st.error("squarify library not found. Please install with: pip install squarify")
                except Exception as e:
                    st.error(f"Error creating treemap: {e}")
        except Exception as e:
            st.error(f"Error in text processing: {e}")
    
    else:
        st.warning("No comment data found in the dataset")
    
    # Show most common words as bar chart
    st.subheader("Most Common Words in Issue Comments")
    
    try:
        if pythainlp_available and 'description' in df.columns and not df['description'].dropna().empty:
            # Tokenize and count words
            all_tokens = []
            for comment in df['description'].dropna():
                try:
                    tokens = word_tokenize(str(comment), engine='newmm')
                    all_tokens.extend([token for token in tokens if token not in stopwords_list and len(token) > 1])
                except:
                    continue
            
            word_counts = Counter(all_tokens)
            
            # Convert to DataFrame
            word_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
            
            fig = px.bar(word_df, x='Word', y='Count',
                        title='Most Frequent Words in Issue Comments',
                        color='Count', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error analyzing frequent words: {e}")

with tab_map:
    st.header("Geographic Distribution of Issues")
    
    # Filter out rows with missing coordinates
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    if not map_df.empty:
        # Create a column for hover information
        map_df['hover_text'] = map_df.apply(
            lambda row: f"ID: {row['message_id']}<br>" +
                       f"Type: {', '.join(row['type_list'])}<br>" +
                       f"District: {row['district']}<br>" +
                       f"Status: {row['state']}<br>" +
                       f"Reported: {row['timestamp'].strftime('%Y-%m-%d')}", 
            axis=1
        )
        
        # Let user filter by issue type
        all_types = list(set([item for sublist in map_df['type_list'].tolist() for item in sublist]))
        selected_types = st.multiselect(
            "Filter by issue type:", 
            options=['All'] + sorted(all_types),
            default=['All']
        )
        
        # Filter data based on selection
        if selected_types and 'All' not in selected_types:
            filtered_map_df = map_df[map_df['type_list'].apply(lambda x: any(item in selected_types for item in x))]
        else:
            filtered_map_df = map_df
        
        # Color by status
        st.subheader("Issues Mapped by Location")
        
        fig = px.scatter_mapbox(
            filtered_map_df,
            lat="latitude", 
            lon="longitude",
            color="state",
            hover_name="message_id",
            hover_data=["district", "timestamp"],
            custom_data=["hover_text"],
            size_max=15,
            zoom=10,
            height=600,
            mapbox_style="carto-positron"
        )
        
        # Improve hover information
        fig.update_traces(
            hovertemplate="%{customdata[0]}"
        )
        
        # Update layout
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            legend_title_text='Issue Status'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a heatmap view option
        st.subheader("Issue Density Heatmap")
        
        fig2 = px.density_mapbox(
            filtered_map_df, 
            lat='latitude', 
            lon='longitude', 
            z='resolution_time_days',
            radius=20,
            center=dict(lat=13.75, lon=100.5),
            zoom=10,
            mapbox_style="carto-positron",
            height=600,
            title="Heatmap of Issue Density and Resolution Time",
            opacity=0.8,
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add district-based choropleth map
        st.subheader("Issues by District")
        
        # Count issues by district
        district_counts = filtered_map_df.groupby('district').size().reset_index(name='count')
        
        # Create a simple bar chart of districts
        fig3 = px.bar(
            district_counts.sort_values('count', ascending=False),
            x='district',
            y='count',
            title="Issue Count by District",
            color='count',
            color_continuous_scale="Reds"
        )
        
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.error("No geographic coordinates found in the data. Cannot display map.")

# New tab for rainfall analysis
with tab_rain:
    st.header("Rainfall Data Analysis")
    
    if rain_df.empty:
        st.error("No rainfall data available.")
    else:
        st.success(f"Loaded rainfall data with {len(rain_df)} records")
        
        with st.expander("Show rainfall data"):
            st.dataframe(rain_df)
        
        st.subheader("Rainfall Statistics")
        
        # Display basic rainfall statistics
        rainfall_stats = pd.DataFrame({
            'Statistic': ['Min', 'Max', 'Mean', 'Median'],
            'MinRain': [rain_df['MinRain'].min(), rain_df['MinRain'].max(), 
                        rain_df['MinRain'].mean(), rain_df['MinRain'].median()],
            'MaxRain': [rain_df['MaxRain'].min(), rain_df['MaxRain'].max(), 
                        rain_df['MaxRain'].mean(), rain_df['MaxRain'].median()],
            'AvgRain': [rain_df['AvgRain'].min(), rain_df['AvgRain'].max(), 
                        rain_df['AvgRain'].mean(), rain_df['AvgRain'].median()]
        })
        
        st.table(rainfall_stats)
        
        # Visualization of rainfall data
        st.subheader("Rainfall Distribution by Month")
        
        # if 'MONTH' in rain_df.columns:
        #     # Create a month-based rainfall visualization
        #     monthly_rain = rain_df.groupby('MONTH')['AvgRain'].mean().reset_index()
            
        #     fig = px.line(monthly_rain, x='MONTH', y='AvgRain',
        #                  title='Average Rainfall by Month',
        #                  labels={'AvgRain': 'Average Rainfall', 'MONTH': 'Month'},
        #                  markers=True)
        #     fig.update_layout(xaxis=dict(tickmode='linear'))
        #     st.plotly_chart(fig, use_container_width=True)
        
        # Compare rainfall with issue frequency
        st.subheader("Relationship Between Rainfall and Issue Reporting")
        
        # Check if we can merge data based on district/province
        if 'district' in df.columns and 'PROV_T' in rain_df.columns:
            try:
                # Convert columns to same case for easier matching
                df_copy = df.copy()
                rain_copy = rain_df.copy()
                
                df_copy['district_lower'] = df_copy['district'].str.lower()
                rain_copy['PROV_T_lower'] = rain_copy['PROV_T'].str.lower()
                
                # Count issues by district
                district_issues = df_copy.groupby('district_lower').size().reset_index(name='issue_count')
                
                # Merge with rainfall data
                merged_data = pd.merge(
                    district_issues, 
                    rain_copy,
                    left_on='district_lower',
                    right_on='PROV_T_lower',
                    how='inner'
                )
                
                if not merged_data.empty:
                    # Create a scatter plot
                    fig = px.scatter(
                        merged_data, 
                        x='AvgRain', 
                        y='issue_count',
                        size='issue_count',
                        color='AvgRain',
                        hover_name='PROV_T',
                        labels={
                            'AvgRain': 'Average Rainfall',
                            'issue_count': 'Number of Issues Reported'
                        },
                        title='Relationship Between Average Rainfall and Number of Issues Reported'
                    )
                    
                    # Add trendline
                    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGray')))
                    fig.update_layout(coloraxis_colorbar=dict(title='Avg Rainfall'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    correlation = merged_data['AvgRain'].corr(merged_data['issue_count'])
                    st.metric("Correlation between Rainfall and Issues", f"{correlation:.3f}")
                    
                    if correlation > 0.3:
                        st.info("There appears to be a positive correlation between rainfall and issue reporting.")
                    elif correlation < -0.3:
                        st.info("There appears to be a negative correlation between rainfall and issue reporting.")
                    else:
                        st.info("There does not appear to be a strong correlation between rainfall and issue reporting.")
                else:
                    st.warning("Could not match districts between datasets. Check that names are consistent.")
            except Exception as e:
                st.error(f"Error analyzing rainfall correlation: {e}")
                
        # Rainfall by province/district
        # if 'PROV_T' in rain_df.columns:
        #     st.subheader("Average Rainfall by Province")
            
        #     # Group by province
        #     province_rain = rain_df.groupby('PROV_T')['AvgRain'].mean().reset_index()
        #     province_rain = province_rain.sort_values('AvgRain', ascending=False)
            
        #     fig = px.bar(
        #         province_rain,
        #         x='PROV_T',
        #         y='AvgRain',
        #         title='Average Rainfall by Province',
        #         color='AvgRain',
        #         color_continuous_scale='Blues'
        #     )
            
        #     fig.update_layout(xaxis_tickangle=-45)
        #     st.plotly_chart(fig, use_container_width=True)
        
        # Analyze peak rainfall months
        if 'MONTH' in rain_df.columns and 'AvgRain' in rain_df.columns:
            st.subheader("Peak Rainfall Months")
            
            # Get average rainfall by month
            monthly_avg = rain_df.groupby('MONTH')['AvgRain'].mean().reset_index()
            monthly_avg = monthly_avg.sort_values('AvgRain', ascending=False)
            
            # Map month numbers to names
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            monthly_avg['Month Name'] = monthly_avg['MONTH'].map(month_names)
            
            # Show peak rainfall months
            st.markdown("#### Months with Highest Average Rainfall")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display top 3 months with highest rainfall
                st.write(f"จำนวนเดือนทั้งหมดในข้อมูล: {len(monthly_avg)}")
                monthly_by_month = monthly_avg.sort_values('MONTH')
                st.table(monthly_by_month[['Month Name', 'AvgRain']])
            
            with col2:
                # Create a simple visualization for peak months
                fig = px.bar(
                    monthly_avg.head(12),
                    x='Month Name',
                    y='AvgRain',
                    color='AvgRain',
                    color_continuous_scale='Blues',
                    title='Months with Highest Rainfall'
                )
                st.plotly_chart(fig, use_container_width=True)
# เพิ่มในส่วน tab_rain

        # ปรับปรุงโค้ดส่วนแผนที่ปริมาณฝนใน tab_rain
        st.subheader("แผนที่ปริมาณฝน (ใช้พิกัดจากคอลัมน์ coords โดยตรง)")

        # ตรวจสอบว่ามีคอลัมน์ coords หรือไม่
        if 'coords' in rain_df.columns:
            # แยกพิกัดจากคอลัมน์ coords
            rain_df_with_coords = rain_df.copy()
            
            try:
                # แยกค่า longitude และ latitude จากคอลัมน์ coords
                # รูปแบบตัวอย่าง: "100.53084,13.81865"
                rain_df_with_coords[['longitude', 'latitude']] = rain_df_with_coords['coords'].str.split(',', expand=True).astype(float)
                
                # แสดงข้อมูลพิกัดที่สกัดได้
                # st.write("ข้อมูลฝนที่มีพิกัด:", len(rain_df_with_coords), "แถว")
                # st.dataframe(rain_df_with_coords[['coords', 'longitude', 'latitude', 'AvgRain']].head())
                
                # สร้าง heatmap จากข้อมูลพิกัดโดยตรง
                fig = px.density_mapbox(
                    rain_df_with_coords,
                    lat="latitude", 
                    lon="longitude", 
                    z="AvgRain",
                    radius=15,  # ปรับรัศมีตามความเหมาะสม
                    center=dict(lat=13.75, lon=100.5),  # พิกัดกลางของกรุงเทพฯ
                    zoom=9,
                    mapbox_style="carto-positron",
                    height=600,
                    opacity=0.7,
                    color_continuous_scale="RdBu_r",
                    title="แผนที่ความเข้มของฝนตกโดยใช้พิกัดจริง"
                )
                
                # ปรับแต่ง layout
                fig.update_layout(
                    margin={"r":0, "t":30, "l":0, "b":0},
                    coloraxis_colorbar=dict(
                        title="ปริมาณฝนเฉลี่ย"
                    )
                )
                
                # แสดงแผนที่
                st.plotly_chart(fig, use_container_width=True)
                
                # เพิ่มการวิเคราะห์ตามเขต - คำนวณค่าเฉลี่ยของแต่ละเขต
                st.subheader("พื้นที่ที่มีปริมาณฝนมากที่สุด/น้อยที่สุด (เฉลี่ยตามเขต)")
                
                # ตรวจสอบว่ามีคอลัมน์เขตหรือไม่
                district_column = None
                if 'district' in rain_df_with_coords.columns:
                    district_column = 'district'
                elif 'เขต' in rain_df_with_coords.columns:
                    district_column = 'เขต'
                
                if district_column:
                    # คำนวณค่าเฉลี่ยของแต่ละเขต
                    district_avg_rain = rain_df_with_coords.groupby(district_column)['AvgRain'].mean().reset_index()
                    district_avg_rain.columns = ['เขต', 'ปริมาณฝนเฉลี่ย']
                    
                    # คำนวณจำนวนข้อมูลในแต่ละเขต
                    district_count = rain_df_with_coords.groupby(district_column).size().reset_index()
                    district_count.columns = ['เขต', 'จำนวนข้อมูล']
                    
                    # รวมข้อมูลเข้าด้วยกัน
                    district_stats = pd.merge(district_avg_rain, district_count, on='เขต')
                    
                    # เรียงลำดับตามปริมาณฝนเฉลี่ย
                    district_stats_desc = district_stats.sort_values('ปริมาณฝนเฉลี่ย', ascending=False)
                    district_stats_asc = district_stats.sort_values('ปริมาณฝนเฉลี่ย')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # แสดงเขตที่มีฝนมากที่สุด 10 อันดับแรก
                        st.write("เขตที่มีฝนมากที่สุด 10 อันดับแรก:")
                        st.dataframe(district_stats_desc.head(10))
                    
                    with col2:
                        # แสดงเขตที่มีฝนน้อยที่สุด 10 อันดับแรก
                        st.write("เขตที่มีฝนน้อยที่สุด 10 อันดับแรก:")
                        st.dataframe(district_stats_asc.head(10))
                    
                    # สร้างกราฟแท่งแสดงปริมาณฝนเฉลี่ยของแต่ละเขต (Top 15)
                    st.subheader("ปริมาณฝนเฉลี่ยของแต่ละเขต (15 อันดับแรก)")
                    
                    fig_bar = px.bar(
                        district_stats_desc.head(15),
                        x='เขต',
                        y='ปริมาณฝนเฉลี่ย',
                        color='ปริมาณฝนเฉลี่ย',
                        text='จำนวนข้อมูล',  # แสดงจำนวนข้อมูลบนแต่ละแท่ง
                        color_continuous_scale="Blues",
                        height=500,
                        title="เขตที่มีปริมาณฝนเฉลี่ยสูงที่สุด 15 อันดับแรก"
                    )
                    
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    fig_bar.update_traces(texttemplate='%{text} ข้อมูล', textposition='outside')
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                else:
                    st.warning("ไม่พบคอลัมน์เขต (district) ในข้อมูล จึงไม่สามารถคำนวณค่าเฉลี่ยตามเขตได้")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการแยกพิกัด: {e}")
                st.write("ตัวอย่างค่าในคอลัมน์ coords:")
                st.write(rain_df['coords'].head())
        else:
            st.error("ไม่พบคอลัมน์ 'coords' ในข้อมูลฝน")

        # วิเคราะห์ความสัมพันธ์ระหว่างปริมาณน้ำฝนกับจำนวนปัญหาตามประเภท
        st.header("ความสัมพันธ์ระหว่างปริมาณน้ำฝนกับจำนวนปัญหา")

        if not rain_df.empty and not df.empty:
            # ตรวจสอบว่ามีคอลัมน์ district ในทั้งสองชุดข้อมูลหรือไม่
            if 'district' in df.columns and 'district' in rain_df.columns:
                # ปรับให้ชื่อเขตเป็นมาตรฐานเดียวกัน
                df_copy = df.copy()
                rain_copy = rain_df.copy()
                
                # แปลงชื่อเขตให้เป็นพิมพ์เล็กและตัดช่องว่างเพื่อให้จับคู่ได้ง่ายขึ้น
                df_copy['district_lower'] = df_copy['district'].str.lower().str.strip()
                rain_copy['district_lower'] = rain_copy['district'].str.lower().str.strip()
                
                # สร้างข้อมูลจำนวนปัญหาแต่ละประเภทในแต่ละเขต
                issue_counts_by_district = []
                
                # นับจำนวนปัญหาแต่ละประเภทในแต่ละเขต
                for i, row in df_copy.iterrows():
                    district = row['district_lower']
                    for issue_type in row['type_list']:
                        issue_counts_by_district.append({
                            'district': district,
                            'issue_type': issue_type
                        })
                
                # แปลงเป็น DataFrame
                issue_df = pd.DataFrame(issue_counts_by_district)
                
                # นับจำนวนปัญหาแต่ละประเภทในแต่ละเขต
                district_issue_counts = issue_df.groupby(['district', 'issue_type']).size().reset_index(name='count')
                
                # เชื่อมข้อมูลปัญหากับข้อมูลฝน
                merged_data = pd.merge(
                    district_issue_counts,
                    rain_copy,
                    left_on='district',
                    right_on='district_lower',
                    how='inner'
                )
                
                # ตรวจสอบว่าเชื่อมข้อมูลได้หรือไม่
                if merged_data.empty:
                    st.warning("ไม่สามารถเชื่อมข้อมูลได้ ลองตรวจสอบข้อมูลเขตในทั้งสองชุดข้อมูล")
                    
                    # แสดงตัวอย่างข้อมูลเพื่อดีบัก
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("ชื่อเขตในข้อมูลปัญหา (5 อันดับแรก):")
                        st.write(df_copy['district_lower'].value_counts().head())
                    with col2:
                        st.write("ชื่อเขตในข้อมูลฝน (5 อันดับแรก):")
                        st.write(rain_copy['district_lower'].value_counts().head())
                else:
                    st.success(f"เชื่อมข้อมูลสำเร็จ! มีข้อมูลที่ตรงกัน {len(merged_data)} รายการ")
                    
                    # หาคอลัมน์ที่เกี่ยวกับฝน
                    rain_columns = [col for col in merged_data.columns if any(word in col.lower() for word in ['rain', 'ฝน', 'precipitation'])]
                    
                    if not rain_columns:
                        st.warning("ไม่พบคอลัมน์ที่เกี่ยวกับปริมาณฝน ตรวจสอบชื่อคอลัมน์ในข้อมูล")
                        st.write("คอลัมน์ทั้งหมดในข้อมูลที่เชื่อมแล้ว:")
                        st.write(merged_data.columns.tolist())
                        
                        # ถ้าไม่พบคอลัมน์ฝน ให้ใช้คอลัมน์ดีฟอลต์
                        rain_columns = ['AvgRain', 'MinRain', 'MaxRain']
                    
                    # เลือกคอลัมน์ฝนที่จะวิเคราะห์
                    selected_rain_column = st.selectbox(
                        "เลือกคอลัมน์ปริมาณฝนที่ต้องการวิเคราะห์:",
                        options=rain_columns,
                        index=0 if 'AvgRain' in rain_columns else 0
                    )
                    
                    # เลือกประเภทการวิเคราะห์
                    analysis_type = st.radio(
                        "เลือกรูปแบบการวิเคราะห์:",
                        options=["แยกตามประเภทปัญหา", "ภาพรวมทั้งหมด"],
                        horizontal=True
                    )
                    
                    if analysis_type == "แยกตามประเภทปัญหา":
                        # หาประเภทปัญหาที่พบบ่อยที่สุด 5 อันดับแรก
                        top_issues = merged_data['issue_type'].value_counts().head(5).index.tolist()
                        
                        # กรองข้อมูลเฉพาะประเภทปัญหาที่พบบ่อย
                        filtered_data = merged_data[merged_data['issue_type'].isin(top_issues)]
                        
                        # สร้างกราฟความสัมพันธ์
                        st.subheader(f"ความสัมพันธ์ระหว่าง {selected_rain_column} กับจำนวนปัญหาแต่ละประเภท")
                        
                        # สร้าง Scatter plot แยกตามประเภทปัญหา
                        fig = px.scatter(
                            filtered_data,
                            x=selected_rain_column,
                            y='count',
                            color='issue_type',
                            hover_name='district_x',  # ใช้ชื่อเขตเป็น hover
                            hover_data=[selected_rain_column, 'count'],
                            labels={
                                selected_rain_column: f'ปริมาณฝน ({selected_rain_column})',
                                'count': 'จำนวนปัญหา',
                                'issue_type': 'ประเภทปัญหา'
                            },
                            title=f'ความสัมพันธ์ระหว่างปริมาณฝน ({selected_rain_column}) กับจำนวนปัญหาตามประเภท',
                            trendline='ols'  # เพิ่มเส้นแนวโน้ม
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # วิเคราะห์ค่าสหสัมพันธ์สำหรับแต่ละประเภทปัญหา
                        st.subheader("การวิเคราะห์ค่าสหสัมพันธ์")
                        
                        correlation_data = []
                        for issue in top_issues:
                            issue_data = filtered_data[filtered_data['issue_type'] == issue]
                            if len(issue_data) >= 5:  # ต้องมีข้อมูลอย่างน้อย 5 จุดจึงจะคำนวณ
                                corr = issue_data[selected_rain_column].corr(issue_data['count'])
                                correlation_data.append({
                                    'ประเภทปัญหา': issue,
                                    'ค่าสหสัมพันธ์': corr,
                                    'ระดับความสัมพันธ์': abs(corr)
                                })
                        
                        if correlation_data:
                            corr_df = pd.DataFrame(correlation_data)
                            corr_df = corr_df.sort_values('ระดับความสัมพันธ์', ascending=False)
                            
                            # แสดงผลในรูปแบบตาราง
                            st.write("ค่าสหสัมพันธ์ระหว่างปริมาณฝนกับจำนวนปัญหาแต่ละประเภท")
                            
                            # สร้าง color map สำหรับค่าสหสัมพันธ์
                            def correlaction_color(val):

                                if val > 0.7:
                                    return 'background-color: #00802b; color: white'  # สีเขียวเข้ม + ตัวอักษรขาว (สัมพันธ์บวกมาก)
                                elif val > 0.3:
                                    return 'background-color: #4CAF50; color: black'  # สีเขียวปานกลาง + ตัวอักษรดำ (สัมพันธ์บวกปานกลาง)
                                elif val < -0.7:
                                    return 'background-color: #CC0000; color: white'  # สีแดงเข้ม + ตัวอักษรขาว (สัมพันธ์ลบมาก)
                                elif val < -0.3:
                                    return 'background-color: #FF5733; color: black'  # สีแดงส้ม + ตัวอักษรดำ (สัมพันธ์ลบปานกลาง)
                                else:
                                    return 'background-color: #E8E8E8; color: black'  # สีเทาอ่อน + ตัวอักษรดำ (สัมพันธ์น้อย)   
                            
                            # แสดงตารางพร้อมระบายสี
                            st.dataframe(corr_df.style.format({
                                'ค่าสหสัมพันธ์': '{:.3f}'
                            }).applymap(correlaction_color, subset=['ค่าสหสัมพันธ์']))
                            
                            # แสดงการตีความผล
                            st.subheader("การตีความผล")
                            
                            # หาประเภทปัญหาที่มีความสัมพันธ์เชิงบวกมากที่สุด
                            positive_corr = corr_df[corr_df['ค่าสหสัมพันธ์'] > 0].sort_values('ค่าสหสัมพันธ์', ascending=False)
                            if not positive_corr.empty:
                                pos_issue = positive_corr.iloc[0]['ประเภทปัญหา']
                                pos_corr = positive_corr.iloc[0]['ค่าสหสัมพันธ์']
                                
                                if pos_corr > 0.7:
                                    st.success(f"ปัญหาประเภท '{pos_issue}' มีความสัมพันธ์เชิงบวกในระดับสูงมาก ({pos_corr:.3f}) กับปริมาณฝน นั่นคือเมื่อฝนตกมาก ปัญหาประเภทนี้มีแนวโน้มที่จะเพิ่มขึ้นมาก")
                                elif pos_corr > 0.3:
                                    st.info(f"ปัญหาประเภท '{pos_issue}' มีความสัมพันธ์เชิงบวกในระดับปานกลาง ({pos_corr:.3f}) กับปริมาณฝน นั่นคือเมื่อฝนตกมาก ปัญหาประเภทนี้มีแนวโน้มที่จะเพิ่มขึ้นบ้าง")
                            
                            # หาประเภทปัญหาที่มีความสัมพันธ์เชิงลบมากที่สุด
                            negative_corr = corr_df[corr_df['ค่าสหสัมพันธ์'] < 0].sort_values('ค่าสหสัมพันธ์', ascending=True)
                            if not negative_corr.empty:
                                neg_issue = negative_corr.iloc[0]['ประเภทปัญหา']
                                neg_corr = negative_corr.iloc[0]['ค่าสหสัมพันธ์']
                                
                                if neg_corr < -0.7:
                                    st.success(f"ปัญหาประเภท '{neg_issue}' มีความสัมพันธ์เชิงลบในระดับสูงมาก ({neg_corr:.3f}) กับปริมาณฝน นั่นคือเมื่อฝนตกมาก ปัญหาประเภทนี้มีแนวโน้มที่จะลดลงมาก")
                                elif neg_corr < -0.3:
                                    st.info(f"ปัญหาประเภท '{neg_issue}' มีความสัมพันธ์เชิงลบในระดับปานกลาง ({neg_corr:.3f}) กับปริมาณฝน นั่นคือเมื่อฝนตกมาก ปัญหาประเภทนี้มีแนวโน้มที่จะลดลงบ้าง")
                                    
                            # หากไม่พบความสัมพันธ์ที่ชัดเจน
                            if (positive_corr.empty or positive_corr.iloc[0]['ค่าสหสัมพันธ์'] <= 0.3) and (negative_corr.empty or negative_corr.iloc[0]['ค่าสหสัมพันธ์'] >= -0.3):
                                st.warning("ไม่พบความสัมพันธ์ที่ชัดเจนระหว่างปริมาณฝนกับจำนวนปัญหาทุกประเภท อาจมีปัจจัยอื่นที่มีอิทธิพลต่อจำนวนปัญหามากกว่าปริมาณฝน")
                        else:
                            st.warning("ไม่มีข้อมูลเพียงพอในการคำนวณค่าสหสัมพันธ์")
                        
                        # สร้างกราฟเส้นแสดงแนวโน้มของแต่ละประเภทปัญหา
                        # st.subheader("แนวโน้มของแต่ละประเภทปัญหาตามปริมาณฝน")
                        
                        # # จัดกลุ่มข้อมูลตามช่วงปริมาณฝน
                        # filtered_data['rain_range'] = pd.cut(
                        #     filtered_data[selected_rain_column],
                        #     bins=5,  # แบ่งเป็น 5 ช่วง
                        #     labels=["น้อยมาก", "น้อย", "ปานกลาง", "มาก", "มากที่สุด"]
                        # )
                        
                        # # หาค่าเฉลี่ยของจำนวนปัญหาแต่ละประเภทในแต่ละช่วงปริมาณฝน
                        # avg_by_rain = filtered_data.groupby(['rain_range', 'issue_type'])['count'].mean().reset_index()
                        
                        # # สร้างกราฟเส้น
                        # line_fig = px.line(
                        #     avg_by_rain,
                        #     x='rain_range',
                        #     y='count',
                        #     color='issue_type',
                        #     markers=True,
                        #     labels={
                        #         'rain_range': 'ช่วงปริมาณฝน',
                        #         'count': 'จำนวนปัญหาเฉลี่ย',
                        #         'issue_type': 'ประเภทปัญหา'
                        #     },
                        #     title=f'จำนวนปัญหาเฉลี่ยตามช่วงปริมาณฝน ({selected_rain_column})'
                        # )
                        
                        # st.plotly_chart(line_fig, use_container_width=True)
                        
                    else:  # ภาพรวมทั้งหมด
                        # รวมข้อมูลปัญหาทุกประเภทเข้าด้วยกันตามเขต
                        district_total = merged_data.groupby('district_x')[['count', selected_rain_column]].agg({
                            'count': 'sum',
                            selected_rain_column: 'mean'
                        }).reset_index()
                        
                        # สร้างกราฟความสัมพันธ์
                        st.subheader(f"ความสัมพันธ์ระหว่าง {selected_rain_column} กับจำนวนปัญหาทั้งหมด")
                        
                        # สร้าง Scatter plot ภาพรวม
                        fig = px.scatter(
                            district_total,
                            x=selected_rain_column,
                            y='count',
                            hover_name='district_x',
                            hover_data=[selected_rain_column, 'count'],
                            labels={
                                selected_rain_column: f'ปริมาณฝน ({selected_rain_column})',
                                'count': 'จำนวนปัญหาทั้งหมด'
                            },
                            title=f'ความสัมพันธ์ระหว่างปริมาณฝน ({selected_rain_column}) กับจำนวนปัญหาทั้งหมด',
                            trendline='ols'  # เพิ่มเส้นแนวโน้ม
                        )
                        
                        # ปรับแต่งกราฟ
                        fig.update_traces(marker=dict(size=10, opacity=0.7))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # คำนวณค่าสหสัมพันธ์ภาพรวม
                        overall_corr = district_total[selected_rain_column].corr(district_total['count'])
                        
                        # แสดงค่าสหสัมพันธ์ภาพรวม
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("ค่าสหสัมพันธ์ภาพรวม", f"{overall_corr:.3f}")
                        
                        with col2:
                            if abs(overall_corr) > 0.7:
                                st.success("มีความสัมพันธ์ระดับสูงมาก")
                            elif abs(overall_corr) > 0.5:
                                st.info("มีความสัมพันธ์ระดับสูง")
                            elif abs(overall_corr) > 0.3:
                                st.info("มีความสัมพันธ์ระดับปานกลาง")
                            else:
                                st.warning("มีความสัมพันธ์ระดับต่ำ")
                        
                        # อธิบายผลการวิเคราะห์
                        st.subheader("การตีความผล")
                        
                        if overall_corr > 0.7:
                            st.success(f"มีความสัมพันธ์เชิงบวกในระดับสูงมาก (r = {overall_corr:.3f}) ระหว่างปริมาณฝนกับจำนวนปัญหาทั้งหมด นั่นคือเมื่อฝนตกมาก จำนวนปัญหาทั้งหมดมีแนวโน้มเพิ่มขึ้นมาก")
                        elif overall_corr > 0.3:
                            st.info(f"มีความสัมพันธ์เชิงบวกในระดับปานกลาง (r = {overall_corr:.3f}) ระหว่างปริมาณฝนกับจำนวนปัญหาทั้งหมด นั่นคือเมื่อฝนตกมาก จำนวนปัญหาทั้งหมดมีแนวโน้มเพิ่มขึ้นบ้าง")
                        elif overall_corr < -0.7:
                            st.success(f"มีความสัมพันธ์เชิงลบในระดับสูงมาก (r = {overall_corr:.3f}) ระหว่างปริมาณฝนกับจำนวนปัญหาทั้งหมด นั่นคือเมื่อฝนตกมาก จำนวนปัญหาทั้งหมดมีแนวโน้มลดลงมาก")
                        elif overall_corr < -0.3:
                            st.info(f"มีความสัมพันธ์เชิงลบในระดับปานกลาง (r = {overall_corr:.3f}) ระหว่างปริมาณฝนกับจำนวนปัญหาทั้งหมด นั่นคือเมื่อฝนตกมาก จำนวนปัญหาทั้งหมดมีแนวโน้มลดลงบ้าง")
                        else:
                            st.warning(f"ไม่พบความสัมพันธ์ที่ชัดเจน (r = {overall_corr:.3f}) ระหว่างปริมาณฝนกับจำนวนปัญหาทั้งหมด อาจมีปัจจัยอื่นที่มีอิทธิพลต่อจำนวนปัญหามากกว่าปริมาณฝน")
                        
                        # สร้างกราฟเส้นแสดงแนวโน้มตามช่วงปริมาณฝน
                        st.subheader("แนวโน้มของจำนวนปัญหาตามปริมาณฝน")
                        
                        # จัดกลุ่มข้อมูลตามช่วงปริมาณฝน
                        district_total['rain_range'] = pd.cut(
                            district_total[selected_rain_column],
                            bins=5,  # แบ่งเป็น 5 ช่วง
                            labels=["น้อยมาก", "น้อย", "ปานกลาง", "มาก", "มากที่สุด"]
                        )
                        
                        # หาค่าเฉลี่ยของจำนวนปัญหาในแต่ละช่วงปริมาณฝน
                        avg_by_rain = district_total.groupby('rain_range')['count'].mean().reset_index()
                        
                        # สร้างกราฟเส้น
                        bar_fig = px.bar(
                            avg_by_rain,
                            x='rain_range',
                            y='count',
                            color='rain_range',
                            labels={
                                'rain_range': 'ช่วงปริมาณฝน',
                                'count': 'จำนวนปัญหาเฉลี่ย'
                            },
                            title=f'จำนวนปัญหาเฉลี่ยตามช่วงปริมาณฝน ({selected_rain_column})'
                        )
                        
                        st.plotly_chart(bar_fig, use_container_width=True)
                        
                        # แสดงเขตที่มีปริมาณฝนและจำนวนปัญหามากที่สุด/น้อยที่สุด
                        st.subheader("เขตที่มีปริมาณฝนและจำนวนปัญหาสูงสุด/ต่ำสุด")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # เขตที่มีปริมาณฝนมากที่สุด
                            max_rain_district = district_total.sort_values(selected_rain_column, ascending=False).head(5)
                            st.write(f"เขตที่มีปริมาณฝน ({selected_rain_column}) มากที่สุด:")
                            st.dataframe(max_rain_district[['district_x', selected_rain_column, 'count']])
                        
                        with col2:
                            # เขตที่มีจำนวนปัญหามากที่สุด
                            max_issue_district = district_total.sort_values('count', ascending=False).head(5)
                            st.write("เขตที่มีจำนวนปัญหามากที่สุด:")
                            st.dataframe(max_issue_district[['district_x', selected_rain_column, 'count']])
            else:
                st.warning("ไม่พบคอลัมน์ district ในชุดข้อมูลใดชุดข้อมูลหนึ่งหรือทั้งสองชุด")
                
                # แสดงคอลัมน์ทั้งหมดในทั้งสองชุดข้อมูล
                col1, col2 = st.columns(2)
                with col1:
                    st.write("คอลัมน์ในข้อมูลปัญหา:")
                    st.write(df.columns.tolist())
                with col2:
                    st.write("คอลัมน์ในข้อมูลฝน:")
                    st.write(rain_df.columns.tolist())
        else:
            st.error("ไม่มีข้อมูลพร้อมใช้งาน กรุณาตรวจสอบการโหลดข้อมูล")


with tab_model:
    # สร้างแท็บสำหรับการเปรียบเทียบการทำนาย
    st.header("การเปรียบเทียบระหว่างเวลาจริงกับเวลาที่ทำนาย")

    # ตรวจสอบว่ามีไฟล์ CSV สำหรับข้อมูลการทำนายหรือไม่
    pred_file = "y_pred_compared.csv"

    if os.path.exists(pred_file):
        # โหลดข้อมูล
        try:
            pred_df = pd.read_csv(pred_file, nrows=1000)
            st.success(f"โหลดข้อมูลการทำนายสำเร็จ! มีข้อมูลทั้งหมด {len(pred_df)} แถว")
            
            # แสดงข้อมูลตัวอย่าง
            with st.expander("ดูข้อมูลทำนายตัวอย่าง"):
                st.dataframe(pred_df[['comment', 'time_taken', 'predicted_time']].head(10))
            
            # สร้างแท็บย่อยสำหรับการวิเคราะห์ต่างๆ
            pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
                "การเปรียบเทียบโดยตรง", "การวิเคราะห์ความแม่นยำ", "การวิเคราะห์ตามเขต", "การวิเคราะห์ตามประเภทปัญหา"
            ])
            
            with pred_tab1:
                st.subheader("เปรียบเทียบเวลาจริงกับเวลาที่ทำนาย")
                
                # สร้างกราฟ Scatter plot
                fig = px.scatter(
                    pred_df,
                    x='time_taken',
                    y='predicted_time',
                    hover_data=['comment'],  # แสดงข้อความในการ hover
                    title='การเปรียบเทียบระหว่างเวลาจริงกับเวลาที่ทำนาย',
                    labels={
                        'time_taken': 'เวลาที่ใช้จริง (ชั่วโมง)',
                        'predicted_time': 'เวลาที่ทำนาย (ชั่วโมง)'
                    },
                    opacity=0.7
                )
                
                # เพิ่ม line y=x (เส้นทำนายสมบูรณ์แบบ)
                max_val = max(pred_df['time_taken'].max(), pred_df['predicted_time'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        name='ทำนายสมบูรณ์แบบ',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # เพิ่ม trend line
                fig.add_trace(
                    px.scatter(
                        pred_df, 
                        x='time_taken', 
                        y='predicted_time', 
                        trendline='ols'
                    ).data[1]
                )
                
                # ปรับแต่งกราฟ
                fig.update_layout(
                    xaxis=dict(range=[0, max_val]),
                    yaxis=dict(range=[0, max_val])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                
                # สร้างฮิสโตแกรมเปรียบเทียบการกระจายตัว
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Histogram(
                    x=pred_df['time_taken'],
                    name='เวลาที่ใช้จริง',
                    opacity=0.7,
                    marker_color='blue'
                ))
                hist_fig.add_trace(go.Histogram(
                    x=pred_df['predicted_time'],
                    name='เวลาที่ทำนาย',
                    opacity=0.7,
                    marker_color='orange'
                ))
                
                hist_fig.update_layout(
                    title="การกระจายตัวของเวลาจริงและเวลาที่ทำนาย",
                    xaxis_title="เวลา (ชั่วโมง)",
                    yaxis_title="จำนวน",
                    barmode='overlay'
                )
                
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # สร้าง Box plot เพื่อเปรียบเทียบ
                box_data = pd.DataFrame({
                    'เวลาที่ใช้จริง': pred_df['time_taken'],
                    'เวลาที่ทำนาย': pred_df['predicted_time']
                })
                
                box_fig = px.box(
                    box_data,
                    points='all',
                    title="เปรียบเทียบการกระจายตัวของเวลาจริงและเวลาที่ทำนาย"
                )
                
                st.plotly_chart(box_fig, use_container_width=True)
            
            with pred_tab2:
                st.subheader("การวิเคราะห์ความแม่นยำของการทำนาย")
                
                # คำนวณความคลาดเคลื่อน
                pred_df['error'] = pred_df['predicted_time'] - pred_df['time_taken']
                pred_df['abs_error'] = abs(pred_df['error'])
                pred_df['squared_error'] = pred_df['error'] ** 2
                
                # คำนวณ % ความคลาดเคลื่อน (ป้องกัน division by zero)
                pred_df['pct_error'] = np.where(
                    pred_df['time_taken'] > 0,
                    (pred_df['error'] / pred_df['time_taken']) * 100,
                    0
                )
                
                # คำนวณเมตริกต่างๆ
                mae = pred_df['abs_error'].mean()
                mse = pred_df['squared_error'].mean()
                rmse = np.sqrt(mse)
                mape = np.abs(pred_df['pct_error']).mean()
                
                # แสดงเมตริกความแม่นยำ
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} ชั่วโมง")
                with metrics_col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} ชั่วโมง")
                with metrics_col3:
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                with metrics_col4:
                    # คำนวณ R-Squared
                    correlation = np.corrcoef(pred_df['time_taken'], pred_df['predicted_time'])[0, 1]
                    r_squared = correlation ** 2
                    st.metric("R-Squared (R²)", f"{r_squared:.3f}")
                
                # แปลความหมายผล
                if r_squared > 0.7:
                    st.success("แบบจำลองมีประสิทธิภาพดี (R² > 0.7)")
                elif r_squared > 0.5:
                    st.info("แบบจำลองมีประสิทธิภาพปานกลาง (R² > 0.5)")
                else:
                    st.warning("แบบจำลองมีประสิทธิภาพต่ำ (R² < 0.5) อาจต้องปรับปรุง")
                
                # สร้างกราฟการกระจายของความคลาดเคลื่อน
                error_fig = px.histogram(
                    pred_df,
                    x='error',
                    nbins=30,
                    title="การกระจายตัวของความคลาดเคลื่อนในการทำนาย",
                    labels={'error': 'ความคลาดเคลื่อน (ชั่วโมง)'}
                )
                
                error_fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                st.plotly_chart(error_fig, use_container_width=True)
                
                # สร้างกราฟความคลาดเคลื่อนเทียบกับค่าจริง
                error_scatter = px.scatter(
                    pred_df,
                    x='time_taken',
                    y='error',
                    title="ความคลาดเคลื่อนเทียบกับเวลาที่ใช้จริง",
                    labels={
                        'time_taken': 'เวลาที่ใช้จริง (ชั่วโมง)',
                        'error': 'ความคลาดเคลื่อน (ชั่วโมง)'
                    }
                )
                
                error_scatter.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")
                st.plotly_chart(error_scatter, use_container_width=True)
                
                # แสดงกรณีที่ทำนายแย่ที่สุด (ความคลาดเคลื่อนมากที่สุด)
                st.subheader("กรณีที่ทำนายแย่ที่สุด (คลาดเคลื่อนมากที่สุด)")
                worst_predictions = pred_df.sort_values('abs_error', ascending=False).head(10)
                st.dataframe(worst_predictions[['comment', 'time_taken', 'predicted_time', 'error', 'abs_error']])
            
            with pred_tab3:
                st.subheader("การวิเคราะห์ความแม่นยำตามเขต")
                
                # ดึงรายชื่อคอลัมน์ที่เกี่ยวกับเขต
                district_cols = [col for col in pred_df.columns if col.startswith('organization_')]
                
                if district_cols:
                    # สร้างคอลัมน์เขตหลัก (เขตที่มีค่ามากที่สุดสำหรับแต่ละแถว)
                    pred_df['main_district'] = pred_df[district_cols].idxmax(axis=1)
                    
                    # แปลงชื่อคอลัมน์ให้เป็นชื่อเขตที่อ่านง่ายขึ้น
                    pred_df['main_district'] = pred_df['main_district'].str.replace('organization_', '')
                    
                    # คำนวณความแม่นยำเฉลี่ยตามเขต
                    district_accuracy = pred_df.groupby('main_district').agg({
                        'time_taken': 'mean',
                        'predicted_time': 'mean',
                        'abs_error': 'mean',
                        'pct_error': lambda x: np.abs(x).mean()
                    }).reset_index()
                    
                    district_accuracy = district_accuracy.rename(columns={
                        'time_taken': 'เวลาจริงเฉลี่ย',
                        'predicted_time': 'เวลาทำนายเฉลี่ย',
                        'abs_error': 'ความคลาดเคลื่อนเฉลี่ย',
                        'pct_error': 'ความคลาดเคลื่อน %'
                    })
                    
                    # เรียงตามความคลาดเคลื่อนเฉลี่ย
                    district_accuracy = district_accuracy.sort_values('ความคลาดเคลื่อนเฉลี่ย')
                    
                    # แสดงตาราง
                    st.dataframe(district_accuracy)
                    
                    # สร้างกราฟความแม่นยำตามเขต
                    accuracy_fig = px.bar(
                        district_accuracy,
                        x='main_district',
                        y='ความคลาดเคลื่อนเฉลี่ย',
                        title="ความคลาดเคลื่อนเฉลี่ยตามเขต",
                        color='ความคลาดเคลื่อนเฉลี่ย',
                        labels={
                            'main_district': 'เขต',
                            'ความคลาดเคลื่อนเฉลี่ย': 'ความคลาดเคลื่อนเฉลี่ย (ชั่วโมง)'
                        }
                    )
                    
                    accuracy_fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(accuracy_fig, use_container_width=True)
                    
                    # สร้างกราฟเปรียบเทียบเวลาจริงกับทำนายตามเขต
                    compare_fig = px.bar(
                        district_accuracy,
                        x='main_district',
                        y=['เวลาจริงเฉลี่ย', 'เวลาทำนายเฉลี่ย'],
                        barmode='group',
                        title="เปรียบเทียบเวลาจริงและเวลาทำนายเฉลี่ยตามเขต",
                        labels={'main_district': 'เขต', 'value': 'เวลา (ชั่วโมง)', 'variable': ''}
                    )
                    
                    compare_fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(compare_fig, use_container_width=True)
                    
                    # แสดงเขตที่ทำนายแม่นยำที่สุดและแย่ที่สุด
                    st.subheader("เขตที่ทำนายแม่นยำที่สุดและแย่ที่สุด")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        best_districts = district_accuracy.head(5)
                        st.write("เขตที่ทำนายแม่นยำที่สุด:")
                        st.dataframe(best_districts[['main_district', 'ความคลาดเคลื่อนเฉลี่ย']])
                    
                    with col2:
                        worst_districts = district_accuracy.sort_values('ความคลาดเคลื่อนเฉลี่ย', ascending=False).head(5)
                        st.write("เขตที่ทำนายแม่นยำน้อยที่สุด:")
                        st.dataframe(worst_districts[['main_district', 'ความคลาดเคลื่อนเฉลี่ย']])
                else:
                    st.warning("ไม่พบข้อมูลเขตในชุดข้อมูล")
            
            with pred_tab4:
                st.subheader("การวิเคราะห์ความแม่นยำตามประเภทปัญหา")
                
                # ดึงรายชื่อคอลัมน์ที่เกี่ยวกับประเภทปัญหา (ไม่รวม organization_ และ คอลัมน์อื่นๆ ที่ไม่เกี่ยวข้อง)
                issue_cols = [col for col in pred_df.columns if col not in [
                    'Unnamed: 0', 'comment', 'time_taken', 'predicted_time', 'error', 'abs_error', 
                    'squared_error', 'pct_error', 'main_district'
                ] and not col.startswith('organization_')]
                
                if issue_cols:
                    # สร้างคอลัมน์ประเภทปัญหาหลัก (ประเภทที่มีค่ามากที่สุดสำหรับแต่ละแถว)
                    if len(issue_cols) > 0:
                        pred_df['main_issue'] = pred_df[issue_cols].idxmax(axis=1)
                        
                        # คำนวณความแม่นยำเฉลี่ยตามประเภทปัญหา
                        issue_accuracy = pred_df.groupby('main_issue').agg({
                            'time_taken': 'mean',
                            'predicted_time': 'mean',
                            'abs_error': 'mean',
                            'pct_error': lambda x: np.abs(x).mean()
                        }).reset_index()
                        
                        issue_accuracy = issue_accuracy.rename(columns={
                            'time_taken': 'เวลาจริงเฉลี่ย',
                            'predicted_time': 'เวลาทำนายเฉลี่ย',
                            'abs_error': 'ความคลาดเคลื่อนเฉลี่ย',
                            'pct_error': 'ความคลาดเคลื่อน %'
                        })
                        
                        # เรียงตามความคลาดเคลื่อนเฉลี่ย
                        issue_accuracy = issue_accuracy.sort_values('ความคลาดเคลื่อนเฉลี่ย')
                        
                        # แสดงตาราง
                        st.dataframe(issue_accuracy)
                        
                        # สร้างกราฟความแม่นยำตามประเภทปัญหา
                        issue_fig = px.bar(
                            issue_accuracy,
                            x='main_issue',
                            y='ความคลาดเคลื่อนเฉลี่ย',
                            title="ความคลาดเคลื่อนเฉลี่ยตามประเภทปัญหา",
                            color='ความคลาดเคลื่อนเฉลี่ย',
                            labels={
                                'main_issue': 'ประเภทปัญหา',
                                'ความคลาดเคลื่อนเฉลี่ย': 'ความคลาดเคลื่อนเฉลี่ย (ชั่วโมง)'
                            }
                        )
                        
                        issue_fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(issue_fig, use_container_width=True)
                        
                        # สร้างกราฟเปรียบเทียบเวลาจริงกับทำนายตามประเภทปัญหา
                        issue_compare_fig = px.bar(
                            issue_accuracy,
                            x='main_issue',
                            y=['เวลาจริงเฉลี่ย', 'เวลาทำนายเฉลี่ย'],
                            barmode='group',
                            title="เปรียบเทียบเวลาจริงและเวลาทำนายเฉลี่ยตามประเภทปัญหา",
                            labels={'main_issue': 'ประเภทปัญหา', 'value': 'เวลา (ชั่วโมง)', 'variable': ''}
                        )
                        
                        issue_compare_fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(issue_compare_fig, use_container_width=True)
                        
                        # สร้างกราฟความสัมพันธ์ระหว่างความซับซ้อนของปัญหากับความแม่นยำ
                        # (สมมติว่าจำนวนประเภทปัญหาที่เกี่ยวข้องคือความซับซ้อน)
                        
                        # สร้างคอลัมน์จำนวนประเภทปัญหาที่เกี่ยวข้อง
                        pred_df['issue_count'] = pred_df[issue_cols].sum(axis=1)
                        
                        # คำนวณความแม่นยำตามจำนวนประเภทปัญหา
                        complexity_accuracy = pred_df.groupby('issue_count').agg({
                            'time_taken': 'mean',
                            'predicted_time': 'mean',
                            'abs_error': 'mean'
                        }).reset_index()
                        
                        # สร้างกราฟ
                        complexity_fig = px.line(
                            complexity_accuracy,
                            x='issue_count',
                            y='abs_error',
                            markers=True,
                            title="ความคลาดเคลื่อนตามความซับซ้อนของปัญหา",
                            labels={
                                'issue_count': 'จำนวนประเภทปัญหาที่เกี่ยวข้อง',
                                'abs_error': 'ความคลาดเคลื่อนเฉลี่ย (ชั่วโมง)'
                            }
                        )
                        
                        st.plotly_chart(complexity_fig, use_container_width=True)
                        
                        # แสดงประเภทปัญหาที่ทำนายแม่นยำที่สุดและแย่ที่สุด
                        st.subheader("ประเภทปัญหาที่ทำนายแม่นยำที่สุดและแย่ที่สุด")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            best_issues = issue_accuracy.head(5)
                            st.write("ประเภทปัญหาที่ทำนายแม่นยำที่สุด:")
                            st.dataframe(best_issues[['main_issue', 'ความคลาดเคลื่อนเฉลี่ย']])
                        
                        with col2:
                            worst_issues = issue_accuracy.sort_values('ความคลาดเคลื่อนเฉลี่ย', ascending=False).head(5)
                            st.write("ประเภทปัญหาที่ทำนายแม่นยำน้อยที่สุด:")
                            st.dataframe(worst_issues[['main_issue', 'ความคลาดเคลื่อนเฉลี่ย']])
                    else:
                        st.warning("ไม่พบคอลัมน์ที่เกี่ยวกับประเภทปัญหา")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดหรือประมวลผลข้อมูล: {e}")
    else:
        st.warning(f"ไม่พบไฟล์ {pred_file} สำหรับการวิเคราะห์ โปรดตรวจสอบว่าไฟล์อยู่ในตำแหน่งที่ถูกต้อง")
        
        # เสนอทางเลือกในการอัปโหลดไฟล์
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ที่มีข้อมูล time_taken และ predicted_time", type=["csv"])
        
        if uploaded_file is not None:
            try:
                pred_df = pd.read_csv(uploaded_file)
                
                # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นหรือไม่
                required_cols = ['time_taken', 'predicted_time']
                
                if all(col in pred_df.columns for col in required_cols):
                    st.success("โหลดข้อมูลสำเร็จ!")
                    
                    # เรียกใช้โค้ดการวิเคราะห์ที่คล้ายกับด้านบนที่นี่
                    # ... (สามารถคัดลอกโค้ดจากด้านบนมาใช้ได้)
                else:
                    st.error(f"ไฟล์ที่อัปโหลดต้องมีคอลัมน์: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")


# Footer
st.markdown("""
---
### About This Dashboard
This dashboard was created using Streamlit, Plotly, and other data analysis libraries to visualize and analyze data from Traffy Fondue.
The analysis focuses on understanding the types of issues reported, response times, textual patterns in user comments, and the relationship with rainfall data.
""")