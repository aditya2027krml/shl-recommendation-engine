import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import logging
from recommendation_engine import AdvancedRecommendationEngine, create_recommendation_engine
from shl_api_client import load_assessments_with_fallback, SHLAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="SHL AI Assessment Recommender",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .assessment-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .difficulty-beginner { border-left: 4px solid #00CC96; }
    .difficulty-intermediate { border-left: 4px solid #FFA15A; }
    .difficulty-advanced { border-left: 4px solid #EF553B; }
</style>
""", unsafe_allow_html=True)

# --- Load data and engine ---
@st.cache_resource
def load_engine(_use_api=False):
    """Load recommendation engine with optional API integration"""
    try:
        # Try to load from API first, fallback to local data
        assessments = load_assessments_with_fallback(use_real_api=_use_api)
        
        # Load jobs locally (since we don't have API for this)
        jobs = pd.read_csv("job_descriptions.csv")
        jobs.columns = jobs.columns.str.strip()
        
        logger.info(f"Loaded {len(assessments)} assessments and {len(jobs)} jobs")
        return AdvancedRecommendationEngine(assessments, jobs)
        
    except Exception as e:
        logger.error(f"Engine loading failed: {e}")
        # Fallback to local data only
        st.sidebar.error(f"❌ API loading failed: {e}")
        assessments = pd.read_csv("assessment_catalogue.csv")
        jobs = pd.read_csv("job_descriptions.csv")
        jobs.columns = jobs.columns.str.strip()
        return AdvancedRecommendationEngine(assessments, jobs)

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'use_api' not in st.session_state:
    st.session_state.use_api = False

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 SHL AI Recommender")
    st.markdown("**Built by Aditya Kumar**  \nAI Research Intern Candidate")
    
    # API Configuration
    st.markdown("---")
    st.subheader("🔌 API Configuration")
    
    use_api = st.checkbox("Use SHL API Simulation", value=False, 
                         help="Simulate real API integration with SHL's assessment catalog")
    
    if use_api != st.session_state.use_api:
        st.session_state.use_api = use_api
        st.session_state.engine = None  # Reset engine to reload with new config
    
    # API Status
    with st.expander("📊 API Status"):
        client = SHLAPIClient()
        status = client.get_api_status()
        
        st.write(f"**Status:** {status['status'].title()}")
        st.write(f"**Version:** {status['version']}")
        st.write(f"**Rate Limit:** {status['rate_limit']['remaining']}/{status['rate_limit']['limit']}")
        
        if st.button("🔄 Test Connection"):
            with st.spinner("Testing API connection..."):
                if client.validate_api_connection():
                    st.success("✅ API Connection Successful")
                else:
                    st.error("❌ API Connection Failed")
    
    # Data Source Info
    st.markdown("---")
    st.subheader("📁 Data Source")
    if st.session_state.use_api:
        st.success("✅ Using SHL API Simulation")
        st.info("""
        **Simulating:**
        - Real API authentication
        - Rate limiting
        - Error handling
        - Professional client patterns
        """)
    else:
        st.info("🔧 Using Local Dataset")
        st.write("100+ custom assessments")
        st.write("20+ job roles")

# --- Main Interface ---
st.markdown('<div class="main-header">🎯 SHL Assessment Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown("### Intelligent assessment matching using advanced AI techniques and professional API integration")

# Initialize engine
if st.session_state.engine is None:
    with st.spinner("🚀 Initializing recommendation engine..."):
        st.session_state.engine = load_engine(st.session_state.use_api)

engine = st.session_state.engine

# Display engine stats
if engine:
    stats = engine.get_engine_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Assessments", stats["total_assessments"])
    with col2:
        st.metric("👔 Job Roles", stats["total_jobs"])
    with col3:
        st.metric("🌐 Domains", stats["assessment_domains"])
    with col4:
        st.metric("🔧 Method", "API" if st.session_state.use_api else "Local")

# Job selection and configuration
st.markdown("---")
st.subheader("🎯 Recommendation Configuration")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    jobs = engine.jobs if engine else pd.read_csv("job_descriptions.csv")
    selected_job = st.selectbox("Select Job Role:", jobs["role"].unique())
with col2:
    recommendation_method = st.selectbox(
        "Recommendation Method:",
        ["Hybrid (Recommended)", "Semantic Similarity", "Skill-Based Matching", "Role-Specific Filtering"]
    )
with col3:
    top_k = st.slider("Number of Recommendations", 3, 10, 5)

# Filters
with st.expander("🎛️ Advanced Filters", expanded=False):
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        max_duration = st.slider("⏱️ Max Duration (min)", 10, 120, 60)
    with filter_col2:
        difficulty_filter = st.multiselect(
            "📊 Difficulty", 
            ["Beginner", "Intermediate", "Advanced"],
            default=["Beginner", "Intermediate", "Advanced"]
        )
    with filter_col3:
        domains = engine.assessments["domain"].unique() if engine else []
        domain_filter = st.multiselect(
            "🌐 Domain",
            domains,
            default=domains[:3] if len(domains) > 3 else domains
        )
    with filter_col4:
        min_score = st.slider("🎯 Min Match Score", 0.0, 1.0, 0.3, 0.1)

# --- Generate Recommendations ---
generate_col1, generate_col2 = st.columns([3, 1])
with generate_col1:
    if st.button("🚀 Generate Smart Recommendations", type="primary", use_container_width=True):
        if not engine:
            st.error("❌ Engine not initialized. Please check data files.")
        else:
            with st.spinner("🤔 Analyzing job role and finding optimal assessments..."):
                try:
                    # Get recommendations based on selected method
                    if recommendation_method == "Semantic Similarity":
                        recommendations = engine.semantic_similarity(selected_job, top_k)
                    elif recommendation_method == "Skill-Based Matching":
                        recommendations = engine.skill_based_matching(selected_job, top_k)
                    elif recommendation_method == "Role-Specific Filtering":
                        recommendations = engine.role_specific_filtering(selected_job, top_k)
                    else:  # Hybrid
                        recommendations = engine.hybrid_recommendation(selected_job, top_k)
                    
                    # Apply filters
                    filtered_recommendations = [
                        rec for rec in recommendations 
                        if (rec["Duration"] <= max_duration and 
                            rec["Difficulty"] in difficulty_filter and
                            rec["Domain"] in domain_filter and
                            rec["Score"] >= min_score)
                    ]
                    
                    st.session_state.recommendations = filtered_recommendations
                    
                except Exception as e:
                    st.error(f"❌ Recommendation generation failed: {e}")
                    logger.error(f"Recommendation error: {e}")

with generate_col2:
    if st.button("🔄 Clear Results", use_container_width=True):
        st.session_state.recommendations = None
        st.rerun()

# --- Display Results ---
if st.session_state.recommendations:
    rec_df = pd.DataFrame(st.session_state.recommendations)
    
    if rec_df.empty:
        st.warning("🤷 No assessments match your current filters. Try adjusting filter settings.")
    else:
        st.success(f"🎉 Found {len(rec_df)} optimal assessments for **{selected_job}**")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Recommendations", "📊 Analytics", "🔍 Comparison", "💾 Export"])
        
        with tab1:
            # Display as enhanced cards with visual indicators
            for i, rec in enumerate(rec_df.to_dict('records')):
                # Color code difficulty
                difficulty_class = f"difficulty-{rec['Difficulty'].lower()}"
                
                with st.container():
                    st.markdown(f'<div class="assessment-card {difficulty_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.subheader(f"{i+1}. {rec['Assessment']}")
                        
                        # Visual score indicator
                        score_percent = int(rec['Score'] * 100)
                        st.progress(rec['Score'], text=f"Match Score: {score_percent}%")
                        
                        # Enhanced metadata with icons
                        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                        with meta_col1:
                            st.caption(f"🌐 **{rec['Domain']}**")
                        with meta_col2:
                            difficulty_emoji = "🟢" if rec['Difficulty'] == 'Beginner' else "🟡" if rec['Difficulty'] == 'Intermediate' else "🔴"
                            st.caption(f"{difficulty_emoji} **{rec['Difficulty']}**")
                        with meta_col3:
                            st.caption(f"⏱️ **{rec['Duration']}min**")
                        with meta_col4:
                            st.caption(f"🔧 **{rec['Method']}**")
                        
                        st.write(f"**🛠️ Skills:** {rec['Skills']}")
                        st.info(f"💡 **Explanation:** {rec['Explanation']}")
                        
                    with col2:
                        # Score visualization
                        score_color = "normal" if rec['Score'] > 0.4 else "off"
                        st.metric(
                            "Match Score", 
                            f"{rec['Score']:.3f}",
                            delta="High" if rec['Score'] > 0.7 else "Medium" if rec['Score'] > 0.4 else "Low",
                            delta_color=score_color
                        )
                    
                    with col3:
                        # Quick actions
                        if st.button("📊 Details", key=f"details_{i}", use_container_width=True):
                            st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
                        
                        if st.session_state.get(f"show_details_{i}", False):
                            with st.expander("Detailed Analysis", expanded=True):
                                st.write(f"**Method:** {rec.get('Method', 'N/A')}")
                                if 'Contributing_Methods' in rec:
                                    st.write(f"**Contributing Methods:** {rec['Contributing_Methods']}")
                                st.write(f"**Domain:** {rec['Domain']}")
                                st.write(f"**Difficulty:** {rec['Difficulty']}")
                                st.write(f"**Duration:** {rec['Duration']} minutes")
                                st.write(f"**Skills:** {rec['Skills']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.subheader("📊 Recommendation Analytics")
            
            # Interactive plot
            if not rec_df.empty:
                fig1 = px.bar(rec_df, 
                            x='Score', 
                            y='Assessment',
                            orientation='h',
                            color='Difficulty',
                            title=f"Assessment Match Scores for {selected_job}",
                            color_discrete_map={
                                'Beginner': '#00CC96', 
                                'Intermediate': '#FFA15A', 
                                'Advanced': '#EF553B'
                            })
                fig1.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig1, use_container_width=True)
                
                # Charts in columns
                col1, col2 = st.columns(2)
                with col1:
                    diff_fig = px.pie(rec_df, names='Difficulty', title='Difficulty Distribution')
                    st.plotly_chart(diff_fig, use_container_width=True)
                with col2:
                    domain_fig = px.pie(rec_df, names='Domain', title='Domain Distribution')
                    st.plotly_chart(domain_fig, use_container_width=True)
                
                # Duration analysis
                fig2 = px.scatter(rec_df, x='Duration', y='Score', color='Difficulty',
                                size='Score', hover_data=['Assessment'],
                                title='Duration vs Match Score')
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("🔍 Compare Assessments")
            if len(rec_df) >= 2:
                compare_options = st.multiselect(
                    "Select assessments to compare:",
                    rec_df["Assessment"].tolist(),
                    default=rec_df["Assessment"].head(2).tolist()
                )
                
                if compare_options:
                    compare_data = rec_df[rec_df["Assessment"].isin(compare_options)]
                    
                    # Display comparison table
                    st.dataframe(compare_data.set_index("Assessment").T, use_container_width=True)
                    
                    # Visual comparison
                    fig = px.bar(compare_data, x='Assessment', y='Score',
                               color='Difficulty', title='Comparison of Selected Assessments',
                               text='Score')
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least 2 recommendations for comparison")
        
        with tab4:
            st.subheader("💾 Export Recommendations")
            
            # Export options
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                # CSV Export
                csv = rec_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"shl_recommendations_{selected_job.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with export_col2:
                # JSON Export
                json_data = rec_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_data,
                    file_name=f"shl_recommendations_{selected_job.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Preview exported data
            with st.expander("📋 Preview Export Data"):
                st.write("CSV Preview:")
                st.dataframe(rec_df, use_container_width=True)
                
                st.write("JSON Preview:")
                st.json(json_data)

# --- Additional Tools Section ---
st.markdown("---")
st.subheader("🔧 Additional Tools")

# Assessment Browser
with st.expander("📚 Browse Assessment Catalog", expanded=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search assessments by skill, title, or domain:")
    with col2:
        browse_domain = st.selectbox("Filter by domain:", ["All"] + list(engine.assessments["domain"].unique()))
    
    # Filter assessments
    filtered_assessments = engine.assessments
    if search_term:
        filtered_assessments = filtered_assessments[
            filtered_assessments["title"].str.contains(search_term, case=False, na=False) | 
            filtered_assessments["skills_measured"].str.contains(search_term, case=False, na=False) |
            filtered_assessments["domain"].str.contains(search_term, case=False, na=False)
        ]
    if browse_domain != "All":
        filtered_assessments = filtered_assessments[filtered_assessments["domain"] == browse_domain]
    
    st.write(f"**Found {len(filtered_assessments)} assessments**")
    st.dataframe(filtered_assessments[["title", "domain", "difficulty", "duration_min", "skills_measured"]], 
                 use_container_width=True)

# Method Comparison
with st.expander("🔬 Compare Recommendation Methods", expanded=False):
    st.info("Compare how different recommendation methods perform for the same job role")
    
    compare_job = st.selectbox("Select job for comparison:", jobs["role"].unique(), key="compare_job")
    
    if st.button("🔄 Run Method Comparison"):
        with st.spinner("Running method comparison..."):
            try:
                all_methods = engine.get_all_recommendation_methods(compare_job, top_k=3)
                
                comparison_data = []
                for method_name, method_recs in all_methods.items():
                    for rec in method_recs:
                        comparison_data.append({
                            'Method': method_name,
                            'Assessment': rec['Assessment'],
                            'Score': rec['Score'],
                            'Domain': rec['Domain'],
                            'Difficulty': rec['Difficulty']
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    fig = px.box(comparison_df, x='Method', y='Score', 
                               title='Score Distribution by Recommendation Method',
                               color='Method')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top recommendations by method
                    for method_name, method_recs in all_methods.items():
                        with st.expander(f"📋 {method_name.title()} Recommendations"):
                            if method_recs:
                                method_df = pd.DataFrame(method_recs)
                                st.dataframe(method_df[['Assessment', 'Score', 'Domain', 'Difficulty']], 
                                           use_container_width=True)
                            else:
                                st.write("No recommendations found")
                
            except Exception as e:
                st.error(f"Method comparison failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💡 <em>This system uses advanced NLP, machine learning, and professional API integration patterns to match SHL assessments with job roles.</em></p>
    <p><strong>Built with ❤️ by Aditya Kumar for SHL AI Research Intern position</strong></p>
    <p> 
        <small>Features: Multi-modal recommendations • SHL API integration • Professional error handling • Enterprise-grade patterns</small>
    </p>
</div>
""", unsafe_allow_html=True)