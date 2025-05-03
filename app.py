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
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.font_manager as fm
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
nltk.download('stopwords')
import os
import matplotlib.font_manager as fm

thai_font_path = os.path.join("fonts", "THSarabunNew.ttf")
font_prop = fm.FontProperties(fname=thai_font_path)

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

# Function to load and preprocess data
@st.cache_data
def load_data():
    # For demonstration, we'll use the provided data
    # In a real application, you'd use st.file_uploader or load from a database
    df = pd.read_csv('first_100_rows.csv')
    
    # Convert timestamp and last_activity to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['last_activity'] = pd.to_datetime(df['last_activity'])
    
    # Calculate resolution time in hours
    df['resolution_time_hours'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 3600
    df['resolution_time_days'] = df['resolution_time_hours'] / 24
    
    # Clean up type column - remove curly braces and split into list
    df['type_list'] = df['type'].apply(lambda x: re.findall(r'{([^}]*)}', str(x)))
    df['type_list'] = df['type_list'].apply(lambda x: [item.strip() for sublist in [i.split(',') for i in x] for item in sublist] if x else ['ไม่ระบุ'])
    
    # Extract organization as list
    df['organization_list'] = df['organization'].apply(lambda x: str(x).split(',') if pd.notna(x) else ['ไม่ระบุ'])
    df['organization_list'] = df['organization_list'].apply(lambda x: [org.strip() for org in x])
    
    # Extract district information
    df['district'] = df['district'].fillna('ไม่ระบุ')
    
    return df

@st.cache_data
def extract_coordinates(df):
    """Extract latitude and longitude from coords column"""
    # Initialize columns with NaN
    df['latitude'] = np.nan
    df['longitude'] = np.nan
    
    # Extract coordinates
    for i, row in df.iterrows():
        if pd.notna(row['coords']):
            try:
                # Try to extract coordinates from the format "100.53084,13.81865"
                coords = row['coords'].split(',')
                if len(coords) == 2:
                    # Note: In coords format, longitude comes first, then latitude
                    df.at[i, 'longitude'] = float(coords[0])
                    df.at[i, 'latitude'] = float(coords[1])
            except:
                continue
    
    return df

# Load the data
try:
    df = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    # Create empty dataframe for demonstration
    df = pd.DataFrame()
    st.stop()


df = extract_coordinates(df)

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
        df_time['month_year'] = df_time['timestamp'].dt.strftime('%Y-%m')
        monthly_counts = df_time.groupby('month_year').size().reset_index(name='count')
        
        fig = px.line(monthly_counts, x='month_year', y='count', 
                    title='Number of Reported Issues Over Time',
                    labels={'count': 'Number of Issues', 'month_year': 'Month-Year'})
        fig.update_layout(xaxis_tickangle=-45)
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
    all_types = [item for sublist in df['type_list'].tolist() for item in sublist]
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
    
    all_orgs = [item for sublist in df['organization_list'].tolist() for item in sublist if item != 'ไม่ระบุ']
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
    
    # Word cloud of comments
    st.subheader("Word Cloud of Issue Comments")
    
    # ตรวจสอบว่ามีข้อความหรือไม่
    if 'comment' in df.columns and not df['comment'].dropna().empty:
        # รวมข้อความทั้งหมด
        all_comments = ' '.join(df['comment'].dropna().astype(str))
        
        try:
            # ระบุฟอนต์ไทย - ใช้ฟอนต์มาตรฐานถ้าหาฟอนต์ไทยไม่เจอ
            try:
                # ตรวจสอบฟอนต์ไทยที่มีในระบบ
                thai_fonts = [f.name for f in fm.fontManager.ttflist 
                            if any(name in f.name.lower() for name in ['thai', 'tahoma', 'sarabun', 'angsana'])]
                
                if thai_fonts:
                    st.success(f"พบฟอนต์ที่น่าจะรองรับภาษาไทย: {', '.join(thai_fonts[:5])}")
                    thai_font = fm.findfont(fm.FontProperties(family=thai_fonts[0]))
                else:
                    st.warning("ไม่พบฟอนต์ที่รองรับภาษาไทยในระบบ จะใช้ฟอนต์เริ่มต้น")
                    thai_font = None
            except Exception as e:
                st.warning(f"ไม่สามารถตรวจสอบฟอนต์ได้: {e}")
                thai_font = None
            
            # ใช้ pythainlp สำหรับตัดคำไทย
            try:
                tokens = word_tokenize(all_comments, engine='newmm')
                
                # กรองคำหยุด (stopwords) ภาษาไทย
                try:
                    thai_stops = list(thai_stopwords())
                except:
                    st.warning("ไม่สามารถโหลด thai_stopwords ได้ จะใช้ stopwords ที่กำหนดเอง")
                    thai_stops = []
                
                # เพิ่มคำหยุดที่ต้องการเพิ่มเติม
                custom_stops = [
                    'ไม่', 'ให้', 'แล้ว', 'เป็น', 'มี', 'การ', 'ของ', 'ก็', 'ที่', 'ได้', 'ว่า', 'จะ',
                    'ใน', 'แต่', 'และ', 'หรือ', 'มาก', 'กับ', 'จาก', 'ถ้า', 'อยู่', 'อย่าง', 'ซึ่ง',
                    'ต้อง', 'ตาม', 'หาก', 'เพื่อ', 'โดย', 'เมื่อ', 'เพราะ', 'นี้', 'นั้น', 'จึง',
                    'ยัง', 'แบบ', 'ทั้ง', 'เคย', 'กว่า', 'อีก', 'ต่อ', 'ๆ', '1', '2', '3', '4', '5',
                    'ครับ', 'ค่ะ', 'น่า', 'มัน', 'กทม', 'กรุงเทพมหานคร'
                ]
                stopwords_list = set(thai_stops + custom_stops)
                
                # กรองคำหยุดออก
                filtered_tokens = [token for token in tokens if token not in stopwords_list and len(token) > 1]
                
                # สร้างข้อความใหม่หลังกรอง
                filtered_text = ' '.join(filtered_tokens)
                
                # แสดงทางเลือกให้ผู้ใช้
                cloud_type = st.radio(
                    "เลือกรูปแบบ Word Cloud:",
                    ["แบบคลาสสิก", "แบบ Treemap (สี่เหลี่ยม)"]
                )
                
                if cloud_type == "แบบคลาสสิก":
                    # สร้าง Word Cloud แบบคลาสสิก
                    wordcloud = WordCloud(
                        font_path=thai_font,
                        width=800, 
                        height=400,
                        background_color='white',
                        stopwords=stopwords_list,
                        max_words=100,
                        contour_width=3,
                        contour_color='steelblue',
                        regexp=r"[^\s]+"  # ช่วยให้รองรับภาษาไทย
                    ).generate(filtered_text)
                    
                    # แสดง Word Cloud
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                else:
                    # สร้าง Word Cloud แบบ Treemap
                    st.subheader("Word Treemap")
                    
                    # นับความถี่ของแต่ละคำ
                    word_counts = Counter(filtered_tokens)
                    
                    try:
                        # เตรียมข้อมูลสำหรับ treemap
                        import squarify
                        
                        # เลือกคำที่พบบ่อยที่สุด 30 คำ
                        top_words = dict(word_counts.most_common(30))
                        
                        # สร้าง Treemap
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # กำหนดสี - เราใช้ชุดสีที่หลากหลาย
                        colors = plt.cm.viridis(np.linspace(0, 1, len(top_words)))
                        
                        # สร้าง treemap
                        squarify.plot(
                            sizes=list(top_words.values()),
                            label=list(top_words.keys()),
                            alpha=0.8,
                            color=colors,
                            ax=ax,
                            text_kwargs={'fontproperties': font_prop}
                        )

                        
                        # ปรับแต่งกราฟ
                        plt.axis('off')
                        if thai_font:
                            plt.title('คำที่พบบ่อยในข้อความแจ้งปัญหา', fontproperties=fm.FontProperties(fname=thai_font))
                        else:
                            plt.title('คำที่พบบ่อยในข้อความแจ้งปัญหา')
                        
                        # แสดง treemap
                        st.pyplot(fig)
                    except ImportError:
                        st.error("ไม่พบไลบรารี่ squarify กรุณาติดตั้งด้วย: pip install squarify")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสร้าง Treemap: {e}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการตัดคำ: {e}")
                st.info("ลองติดตั้ง pythainlp: pip install pythainlp")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้าง Word Cloud: {e}")
            
            # แนะนำการติดตั้งไลบรารี่ที่จำเป็น
            st.info("คุณอาจต้องติดตั้งไลบรารี่เพิ่มเติม:")
            st.code("pip install pythainlp squarify matplotlib")
            
            # แสดงข้อความที่จะถูกใช้ (สำหรับการแก้ไขปัญหา)
            with st.expander("ดูข้อมูลเพื่อแก้ไขปัญหา"):
                st.write("ตัวอย่างข้อความ 500 ตัวอักษรแรก:")
                st.write(all_comments[:500])
    
    else:
        st.warning("ไม่พบข้อมูลข้อความในชุดข้อมูล")
    
    # แสดงคำที่พบบ่อยในรูปแบบกราฟแท่ง
    st.subheader("คำที่พบบ่อยที่สุดในข้อความแจ้งปัญหา")
    
    try:
        if 'comment' in df.columns and not df['comment'].dropna().empty:
            # Tokenize และนับคำ
            all_tokens = []
            for comment in df['comment'].dropna():
                try:
                    tokens = word_tokenize(str(comment), engine='newmm')
                    all_tokens.extend([token for token in tokens if token not in stopwords_list and len(token) > 1])
                except:
                    continue
            
            word_counts = Counter(all_tokens)
            
            # แปลงเป็น DataFrame
            word_df = pd.DataFrame(word_counts.most_common(20), columns=['คำ', 'จำนวนครั้ง'])
            
            fig = px.bar(word_df, x='คำ', y='จำนวนครั้ง',
                        title='คำที่พบบ่อยที่สุดในข้อความแจ้งปัญหา',
                        color='จำนวนครั้ง', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์คำที่พบบ่อย: {e}")

with tab_map:
    st.header("Geographic Distribution of Issues")
    
    # Filter out rows with missing coordinates
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    if not map_df.empty:
        # Create a column for hover information
        map_df['hover_text'] = map_df.apply(
            lambda row: f"ID: {row['ticket_id']}<br>" +
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
            hover_name="ticket_id",
            hover_data=["district", "timestamp"],
            custom_data=["hover_text"],
            size_max=15,
            zoom=10,
            height=600,
            mapbox_style="carto-positron"  # You can change this to other styles like "open-street-map"
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