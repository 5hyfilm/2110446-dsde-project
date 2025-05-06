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
        use_demo = st.checkbox("Use demo data instead?", value=False)
        if use_demo:
            st.info("Using demo data. Note that this is not real-time data.")
            # Try to load demo data from a local file if available
            try:
                demo_file = "demo_data.csv"
                if os.path.exists(demo_file):
                    return pd.read_csv(demo_file)
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

# Load the data
try:
    df = load_data()
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
tab1, tab_map, tab2, tab3, tab4 = st.tabs(["Overview", "Map Visualization", "Issue Analysis", "Response Time", "Text Analysis"])

with tab1:
    st.header("Overview of Reported Issues")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count issues by status
        status_counts = df['state'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(status_counts, values='Count', names='Status', 
                    title='Distribution of Issue Status',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Issues over time
        df_time = df.copy()
        df_time['date'] = df_time['timestamp'].dt.date  # แสดงเฉพาะวันที่ ไม่รวมเวลา
        daily_counts = df_time.groupby('date').size().reset_index(name='count')

        fig = px.line(daily_counts, x='date', y='count', 
                    title='Number of Reported Issues by Day',
                    labels={'count': 'Number of Issues', 'date': 'Date'})
        fig.update_layout(xaxis_tickangle=-45)
        # บังคับให้แกน Y เริ่มจาก 0
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
    
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

# Footer
st.markdown("""
---
### About This Dashboard
This dashboard was created using Streamlit, Plotly, and other data analysis libraries to visualize and analyze data from Traffy Fondue.
The analysis focuses on understanding the types of issues reported, response times, and textual patterns in user comments.
""")