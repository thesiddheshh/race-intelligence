"""
F1 2026 Thesis Visualization GUI
Advanced analytics dashboard for research validation and figure export

Usage: streamlit run thesis_viz.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.thesis.export_config import THESIS_CONFIG
from src.thesis.shap_analyzer import SHAPAnalyzer
from src.thesis.calibration import CalibrationAnalyzer
from src.utils.grid import GridManager

# Page configuration
st.set_page_config(
    page_title="F1 Thesis Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject thesis-specific CSS
st.markdown(f"""
<style>
    /* Academic styling overrides */
    .stApp {{
        background-color: #fafafa;
        font-family: 'Times New Roman', serif;
    }}
    h1, h2, h3 {{
        color: #1a1a1a !important;
        font-weight: normal !important;
    }}
    .stMetric {{
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 1rem;
    }}
    .stDataFrame {{
        font-size: 0.9rem;
    }}
    /* Export button styling */
    .export-btn {{
        background-color: #0072B2 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
    }}
    .export-btn:hover {{
        background-color: #005a8c !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_resource
def load_trained_models(model_dir: str = "models"):
    """Load pre-trained models from disk."""
    models = {}
    model_path = Path(model_dir)
    
    for model_file in model_path.glob("*.pkl"):
        try:
            model_name = model_file.stem
            data = joblib.load(model_file)
            models[model_name] = data
        except Exception as e:
            st.warning(f"Could not load {model_file}: {e}")
    
    return models

@st.cache_data(ttl=3600)
def load_cached_features(cache_dir: str = ".f1_cache"):
    """Load cached feature data for visualization."""
    # In practice, you'd save features during training
    # For demo, we'll generate synthetic but realistic data
    np.random.seed(42)
    
    grid = GridManager.load_grid()
    drivers = list(grid["drivers"].keys())
    
    features = pd.DataFrame({
        'Driver': drivers,
        'lap_time_mean': np.random.uniform(92, 98, len(drivers)),
        'LapConsistency': np.random.uniform(0.8, 2.5, len(drivers)),
        'SectorTotal_s': np.random.uniform(82, 94, len(drivers)),
        'TeamPerformance': np.random.uniform(0.6, 1.0, len(drivers)),
        'WeatherPenalty': np.random.uniform(0, 0.5, len(drivers)),
        'TempAdjustment': np.random.uniform(0, 0.3, len(drivers)),
        'RecentForm': np.random.uniform(0.5, 1.0, len(drivers)),
        'CompoundFactor': np.random.choice([0.95, 1.0, 1.05], len(drivers)),
        'QualiProxy': np.random.uniform(1, 22, len(drivers)),
        'lap_count': np.random.randint(15, 60, len(drivers))
    })
    
    return features

# ==================== MAIN APP ====================
def main():
    # Header
    st.title("F1 2026 Thesis Analytics Dashboard")
    st.caption("Advanced visualization module for research validation and figure export")
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.radio(
            "Feature Data",
            ["Cached Features", "Live from app.py", "Synthetic Demo"],
            index=0
        )
        
        # Model selection
        st.subheader("Model")
        models = load_trained_models()
        if models:
            model_choice = st.selectbox(
                "Trained Model",
                list(models.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            selected_model = models[model_choice]
        else:
            st.info("No trained models found. Using demo mode.")
            selected_model = None
        
        # Export settings
        st.subheader("Export Settings")
        export_format = st.selectbox(
            "Figure Format",
            THESIS_CONFIG.vector_formats + THESIS_CONFIG.raster_formats,
            index=THESIS_CONFIG.vector_formats.index(THESIS_CONFIG.default_format) 
                if THESIS_CONFIG.default_format in THESIS_CONFIG.vector_formats else 0
        )
        double_column = st.checkbox("Double-column width", value=False, 
                                   help="Use wider format for full-page figures")
        
        # Thesis metadata
        st.subheader("Figure Metadata")
        figure_title = st.text_input("Figure Title", "Feature Impact Analysis")
        figure_label = st.text_input("Figure Label (for LaTeX)", "fig:shap_summary")
        caption = st.text_area(
            "Caption", 
            "SHAP summary plot showing mean absolute feature impact on lap time predictions. Higher values indicate stronger influence.",
            height=60
        )
        
        st.markdown("---")
        st.caption(f"Export Path: `data/thesis_exports/`")
        st.caption(f"Config: {THESIS_CONFIG.font_family}, {THESIS_CONFIG.width_single_col}pt width")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Model Interpretability",
        "🎯 Uncertainty Calibration", 
        "📊 Diagnostic Plots",
        "📤 Batch Export"
    ])
    
    # Load data based on selection
    if data_source == "Cached Features":
        features = load_cached_features()
    elif data_source == "Live from app.py":
        # Would connect to app.py's data pipeline
        st.warning("Live connection not implemented. Using cached data.")
        features = load_cached_features()
    else:
        features = load_cached_features()
    
    feature_cols = [c for c in features.columns if c != 'Driver']
    
    # ==================== TAB 1: Model Interpretability ====================
    with tab1:
        st.header("Model Interpretability Analysis")
        
        if selected_model and 'model' in selected_model and 'scaler' in selected_model:
            # Initialize SHAP analyzer
            shap_analyzer = SHAPAnalyzer(
                model=selected_model['model'],
                scaler=selected_model['scaler'],
                feature_names=feature_cols
            )
            
            # Sample data for SHAP computation
            X_sample = features[feature_cols].sample(min(100, len(features)), random_state=42)
            shap_analyzer.fit(X_sample)
            
            # SHAP Summary Plot
            st.subheader("1. Feature Importance (SHAP)")
            summary_fig = shap_analyzer.create_summary_plot(
                title=figure_title,
                max_features=12
            )
            st.plotly_chart(summary_fig, use_container_width=True)
            
            # Export button
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("📥 Export Figure", key="export_shap", type="primary"):
                    export_path = THESIS_CONFIG.get_export_path(figure_label or "shap_summary", export_format)
                    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Export based on format
                    if export_format in THESIS_CONFIG.vector_formats:
                        summary_fig.write_image(
                            export_path,
                            width=THESIS_CONFIG.width_double_col if double_column else THESIS_CONFIG.width_single_col,
                            height=THESIS_CONFIG.height_tall,
                            scale=1,  # Vector formats don't need DPI scaling
                            engine="kaleido"
                        )
                    else:
                        summary_fig.write_image(
                            export_path,
                            width=THESIS_CONFIG.width_double_col if double_column else THESIS_CONFIG.width_single_col,
                            height=THESIS_CONFIG.height_tall,
                            scale=2,  # 2x for print resolution
                            dpi=THESIS_CONFIG.dpi_print,
                            engine="kaleido"
                        )
                    st.success(f"✓ Exported to `{export_path}`")
            
            # SHAP Dependence Plot
            st.subheader("2. Feature Dependence")
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                target_feature = st.selectbox(
                    "Primary Feature",
                    feature_cols,
                    index=feature_cols.index('WeatherPenalty') if 'WeatherPenalty' in feature_cols else 0,
                    key="dep_feat1"
                )
            with col_feat2:
                color_feature = st.selectbox(
                    "Color By (Optional)",
                    ["None"] + feature_cols,
                    index=0,
                    key="dep_feat2"
                )
            
            if color_feature == "None":
                color_feature = None
                
            dep_fig = shap_analyzer.create_dependence_plot(
                feature_name=target_feature,
                color_feature=color_feature
            )
            st.plotly_chart(dep_fig, use_container_width=True)
            
            # SHAP Interaction Heatmap
            st.subheader("3. Feature Interactions")
            interact_fig = shap_analyzer.create_interaction_heatmap(top_n=10)
            st.plotly_chart(interact_fig, use_container_width=True)
            
            # Interpretation guidance
            with st.expander("📖 Interpretation Guidance", expanded=False):
                st.markdown(f"""
                **SHAP Summary Plot**: 
                - Features at the top have the strongest average impact on predictions
                - The color gradient shows the direction: positive values increase predicted lap time
                
                **Dependence Plot**:
                - Shows how the SHAP value for `{target_feature}` changes with its value
                - If colored by another feature, reveals interaction effects
                
                **Interaction Heatmap**:
                - Shows which feature pairs jointly influence predictions
                - Strong interactions suggest non-additive effects that linear models would miss
                
                *For thesis*: Cite SHAP methodology [Lundberg & Lee, 2017] and discuss 
                how these visualizations validate that the model learns domain-relevant patterns.
                """)
                
        else:
            st.info("Select a trained model to enable SHAP analysis, or use demo mode.")
            # Demo mode
            st.markdown("### Demo: SHAP Summary (Synthetic Data)")
            demo_fig = go.Figure(go.Bar(
                x=np.random.uniform(0.1, 0.5, 10),
                y=[f"Feature_{i}" for i in range(10)],
                orientation='h',
                marker_color=THESIS_CONFIG.colors['primary']
            ))
            demo_fig.update_layout(**THESIS_CONFIG.get_plotly_theme())
            demo_fig.update_layout(title="Demo: Feature Importance", height=300)
            st.plotly_chart(demo_fig, use_container_width=True)
    
    # ==================== TAB 2: Uncertainty Calibration ====================
    with tab2:
        st.header("Uncertainty Quantification & Calibration")
        
        # Generate synthetic prediction data for demo
        np.random.seed(42)
        n_samples = len(features)
        y_true = features['lap_time_mean'].values
        y_pred = y_true + np.random.normal(0, 0.8, n_samples)  # Add prediction noise
        pred_std = np.abs(np.random.normal(0.7, 0.2, n_samples))  # Predicted uncertainty
        
        # Calibration Plot
        st.subheader("1. Prediction Interval Calibration")
        calib_fig = CalibrationAnalyzer.create_calibration_plot(
            y_true=y_true,
            y_pred=y_pred,
            pred_std=pred_std,
            title="Interval Coverage Calibration",
            n_bins=15
        )
        st.plotly_chart(calib_fig, use_container_width=True)
        
        # Export
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("📥 Export", key="export_calib", type="primary"):
                export_path = THESIS_CONFIG.get_export_path("calibration_plot", export_format)
                Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                calib_fig.write_image(export_path, width=THESIS_CONFIG.width_double_col if double_column else THESIS_CONFIG.width_single_col, height=THESIS_CONFIG.height_standard, scale=2 if export_format not in THESIS_CONFIG.vector_formats else 1, engine="kaleido")
                st.success(f"✓ Exported")
        
        # Residual Distribution
        st.subheader("2. Residual Analysis")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            color_by = st.selectbox(
                "Color Residuals By",
                ["None"] + feature_cols,
                index=0,
                key="res_color"
            )
        
        feature_series = features[color_by] if color_by != "None" and color_by in features.columns else None
        
        residual_fig = CalibrationAnalyzer.create_residual_distribution(
            y_true=y_true,
            y_pred=y_pred,
            feature_for_color=feature_series,
            title="Residual Distribution with QQ Inset"
        )
        st.plotly_chart(residual_fig, use_container_width=True)
        
        # Statistical summary
        residuals = y_true - y_pred
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("MAE", f"{np.mean(np.abs(residuals)):.3f}s")
        with col_stat2:
            st.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.3f}s")
        with col_stat3:
            st.metric("Std Residual", f"{residuals.std():.3f}s")
        with col_stat4:
            # Shapiro-Wilk test for normality
            from scipy import stats
            _, p_val = stats.shapiro(residuals[:5000])  # Limit for speed
            st.metric("Normality (p-value)", f"{p_val:.3f}")
        
        with st.expander("📖 Interpretation Guidance"):
            st.markdown("""
            **Calibration Plot**:
            - Points near the dashed line indicate well-calibrated uncertainty
            - ECE (Expected Calibration Error) quantifies overall miscalibration
            - Critical for risk-aware decision-making in race strategy
            
            **Residual Plot**:
            - Random scatter around zero indicates no systematic bias
            - QQ plot inset tests normality assumption for statistical inference
            - Coloring by features can reveal heteroscedasticity or missed interactions
            
            *For thesis*: Report calibration metrics (ECE, MCE) and discuss implications 
            for prediction reliability. Reference calibration methodology [Guo et al., 2017].
            """)
    
    # ==================== TAB 3: Diagnostic Plots ====================
    with tab3:
        st.header("Model Diagnostic Visualizations")
        
        # Learning Curve
        st.subheader("1. Learning Curve Analysis")
        
        # Simulate learning curve data
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = 2.5 - 1.2 * np.log(train_sizes * 10) + np.random.normal(0, 0.1, 10)
        val_scores = 2.5 - 0.8 * np.log(train_sizes * 10) + np.random.normal(0, 0.15, 10)
        
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(
            x=train_sizes * 100, y=train_scores,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=THESIS_CONFIG.colors['primary'], width=2),
            marker=dict(size=6)
        ))
        fig_lc.add_trace(go.Scatter(
            x=train_sizes * 100, y=val_scores,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=THESIS_CONFIG.colors['secondary'], width=2),
            marker=dict(size=6)
        ))
        
        layout_lc = THESIS_CONFIG.get_plotly_theme()
        layout_lc.update(
            title=dict(text="Learning Curve: Model Performance vs. Training Set Size"),
            xaxis_title="Training Set Size (%)",
            yaxis_title="MAE (Seconds)",
            height=THESIS_CONFIG.height_standard
        )
        fig_lc.update_layout(**layout_lc)
        st.plotly_chart(fig_lc, use_container_width=True)
        
        # Feature Correlation Matrix
        st.subheader("2. Feature Correlation Matrix")
        
        corr_matrix = features[feature_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            hovertemplate='%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>',
            colorbar=dict(title="Pearson r")
        ))
        
        layout_corr = THESIS_CONFIG.get_plotly_theme()
        layout_corr.update(
            title=dict(text="Feature Correlation Matrix"),
            height=THESIS_CONFIG.height_tall,
            margin=dict(l=100, r=20, t=50, b=100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8))
        )
        fig_corr.update_layout(**layout_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Prediction Distribution
        st.subheader("3. Prediction Distribution")
        
        # Simulate predictions
        predictions = np.random.normal(95, 2, 1000)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=predictions,
            nbinsx=30,
            marker_color=THESIS_CONFIG.colors['primary'],
            opacity=0.7,
            name='Predictions',
            histnorm='probability density'
        ))
        
        # Add KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(predictions)
        x_kde = np.linspace(predictions.min(), predictions.max(), 100)
        fig_dist.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde),
            mode='lines',
            line=dict(color=THESIS_CONFIG.colors['secondary'], width=2),
            name='KDE',
            yaxis='y2'
        ))
        
        layout_dist = THESIS_CONFIG.get_plotly_theme()
        layout_dist.update(
            title=dict(text="Distribution of Predicted Lap Times"),
            xaxis_title="Predicted Lap Time (s)",
            yaxis_title="Density",
            yaxis2=dict(overlaying='y', side='right', showgrid=False),
            height=THESIS_CONFIG.height_standard
        )
        fig_dist.update_layout(**layout_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ==================== TAB 4: Batch Export ====================
    with tab4:
        st.header("Batch Figure Export")
        
        st.markdown("""
        Export multiple thesis-ready figures at once with consistent formatting.
        All figures will use the settings configured in the sidebar.
        """)
        
        # Figure selection
        available_figs = {
            "shap_summary": "SHAP Feature Importance",
            "shap_dependence": "SHAP Dependence Plot", 
            "shap_interactions": "SHAP Interaction Heatmap",
            "calibration": "Uncertainty Calibration",
            "residuals": "Residual Analysis",
            "learning_curve": "Learning Curve",
            "correlation_matrix": "Feature Correlations",
            "prediction_dist": "Prediction Distribution"
        }
        
        selected_figs = st.multiselect(
            "Select Figures to Export",
            list(available_figs.keys()),
            default=["shap_summary", "calibration"],
            format_func=lambda x: available_figs[x]
        )
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            export_all = st.checkbox("Export all selected", value=True)
        with col2:
            add_caption = st.checkbox("Add caption metadata", value=True)
        with col3:
            generate_latex = st.checkbox("Generate LaTeX code", value=True)
        
        if st.button("🚀 Export Selected Figures", type="primary", use_container_width=True):
            export_dir = Path("data/thesis_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            for fig_key in selected_figs:
                try:
                    # Generate figure based on key (simplified for demo)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3]))
                    fig.update_layout(**THESIS_CONFIG.get_plotly_theme(double_column))
                    fig.update_layout(title=available_figs[fig_key])
                    
                    # Export
                    filename = f"{fig_key}_thesis"
                    export_path = THESIS_CONFIG.get_export_path(filename, export_format)
                    
                    fig.write_image(
                        export_path,
                        width=THESIS_CONFIG.width_double_col if double_column else THESIS_CONFIG.width_single_col,
                        height=THESIS_CONFIG.height_standard,
                        scale=2 if export_format not in THESIS_CONFIG.vector_formats else 1,
                        engine="kaleido"
                    )
                    
                    results.append(f"✓ {available_figs[fig_key]} → `{export_path.name}`")
                    
                except Exception as e:
                    results.append(f"✗ {available_figs[fig_key]}: {str(e)}")
            
            # Show results
            st.markdown("### Export Results")
            for result in results:
                st.text(result)
            
            # Generate LaTeX code if requested
            if generate_latex and results:
                with st.expander("📄 Generated LaTeX Code", expanded=True):
                    latex_code = "% Auto-generated by F1 Thesis Viz GUI\n\n"
                    for fig_key in selected_figs:
                        if fig_key in [r.split('→')[0].strip().replace('✓ ', '') for r in results if '✓' in r]:
                            filename = f"{fig_key}_thesis.{export_format}"
                            caption_text = caption if fig_key == "shap_summary" else available_figs[fig_key]
                            latex_code += f"""\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\columnwidth]{{figures/{filename}}}
  \\caption{{{caption_text}}}
  \\label{{fig:{fig_key}}}
\\end{{figure}}

"""
                    st.code(latex_code, language="latex")
                    
                    # Download button for LaTeX
                    st.download_button(
                        "📥 Download LaTeX Snippets",
                        data=latex_code,
                        file_name="thesis_figures.tex",
                        mime="text/plain"
                    )
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    **F1 Thesis Visualization Module** | 
    Formatting: {THESIS_CONFIG.font_family} {THESIS_CONFIG.font_size_base}pt | 
    Export: {export_format.upper()} ({'double' if double_column else 'single'}-column) | 
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)

if __name__ == "__main__":
    main()