"""
SHAP-based model interpretability visualizations for thesis
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional
import shap
from .export_config import THESIS_CONFIG

class SHAPAnalyzer:
    """Generate thesis-ready SHAP visualizations."""
    
    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit(self, X_sample: pd.DataFrame, max_samples: int = 100):
        """Initialize SHAP explainer with sampled data."""
        X_scaled = self.scaler.transform(X_sample.iloc[:max_samples])
        
        # Use appropriate explainer based on model type
        if hasattr(self.model, 'feature_importances_'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, X_scaled[:50])
        
        self.shap_values = self.explainer.shap_values(X_scaled)
        return self
    
    def create_summary_plot(self, title: str = "Feature Impact Analysis", 
                           max_features: int = 10) -> go.Figure:
        """Create thesis-ready SHAP summary bar plot."""
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # Multi-output models
            shap_abs = np.abs(self.shap_values[0]).mean(axis=0)
        else:
            shap_abs = np.abs(self.shap_values).mean(axis=0)
        
        # Select top features
        top_idx = np.argsort(shap_abs)[-max_features:][::-1]
        top_features = [self.feature_names[i] for i in top_idx]
        top_values = shap_abs[top_idx]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=top_values,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_values,
                colorscale=[[0, THESIS_CONFIG.colors['neutral']], 
                           [1, THESIS_CONFIG.colors['primary']]],
                line=dict(width=0),
                showscale=True,
                colorbar=dict(
                    title="|SHAP Value|",
                    titleside='right',
                    thickness=15,
                    len=0.5,
                    y=0.5,
                    yanchor='middle'
                )
            ),
            hovertemplate='<b>%{y}</b><br>Mean Impact: %{x:.3f}<extra></extra>'
        ))
        
        # Apply thesis theme
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text=title, font=dict(size=THESIS_CONFIG.font_size_title)),
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            height=THESIS_CONFIG.height_tall,
            margin=dict(l=150, r=20, t=50, b=30)  # Extra left margin for long feature names
        )
        fig.update_layout(**layout_config)
        
        return fig
    
    def create_dependence_plot(self, feature_name: str, 
                              color_feature: Optional[str] = None) -> go.Figure:
        """Create SHAP dependence plot with optional coloring."""
        
        # Find feature index
        feat_idx = self.feature_names.index(feature_name)
        
        # Extract SHAP values and feature values
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][:, feat_idx]
        else:
            shap_vals = self.shap_values[:, feat_idx]
        
        feat_vals = self.scaler.inverse_transform(
            self.explainer.data
        )[:, feat_idx] if hasattr(self.scaler, 'inverse_transform') else self.explainer.data[:, feat_idx]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Main scatter
        scatter_kwargs = dict(
            x=feat_vals, y=shap_vals,
            mode='markers',
            marker=dict(
                size=6,
                opacity=0.6,
                line=dict(width=0.5, color='rgba(0,0,0,0.1)')
            ),
            hovertemplate=f'{feature_name}: %{{x:.2f}}<br>SHAP: %{{y:.3f}}<extra></extra>'
        )
        
        if color_feature and color_feature in self.feature_names:
            # Add coloring by another feature
            color_idx = self.feature_names.index(color_feature)
            color_vals = self.explainer.data[:, color_idx]
            
            fig.add_trace(go.Scatter(
                **scatter_kwargs,
                marker=dict(
                    color=color_vals,
                    colorscale='RdYlBu',
                    colorbar=dict(title=color_feature),
                    showscale=True
                )
            ))
        else:
            fig.add_trace(go.Scatter(**scatter_kwargs, marker_color=THESIS_CONFIG.colors['primary']))
        
        # Add LOESS smooth line to show trend
        if len(feat_vals) > 20:
            from scipy import stats
            slope, intercept, r_val, _, _ = stats.linregress(feat_vals, shap_vals)
            x_line = np.array([feat_vals.min(), feat_vals.max()])
            y_line = intercept + slope * x_line
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=THESIS_CONFIG.colors['secondary'], width=2, dash='dash'),
                name=f'Trend (R²={r_val**2:.2f})',
                showlegend=True
            ))
        
        # Apply thesis theme
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text=f"SHAP Dependence: {feature_name}"),
            xaxis_title=feature_name,
            yaxis_title="SHAP Value (Impact on Prediction)",
            height=THESIS_CONFIG.height_standard
        )
        fig.update_layout(**layout_config)
        
        return fig
    
    def create_interaction_heatmap(self, top_n: int = 8) -> go.Figure:
        """Create SHAP interaction value heatmap for top feature pairs."""
        
        # Compute interaction values (sample for speed)
        if not hasattr(self.explainer, 'shap_interaction_values'):
            # Fallback for non-tree models
            return self._create_correlation_heatmap(top_n)
        
        # Sample data for computation
        X_sample = self.explainer.data[:100]
        interactions = self.explainer.shap_interaction_values(X_sample)
        
        # Select top features by main effect
        main_effects = np.abs(interactions).sum(axis=(0, 2)).mean(axis=0)
        top_idx = np.argsort(main_effects)[-top_n:][::-1]
        
        # Aggregate interaction matrix
        interaction_matrix = np.abs(
            interactions[:, top_idx, :][:, :, top_idx]
        ).mean(axis=0)
        
        # Prepare labels
        top_names = [self.feature_names[i] for i in top_idx]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=top_names,
            y=top_names,
            colorscale='RdYlBu',
            zmin=0,
            hovertemplate='%{y} × %{x}<br>Interaction: %{z:.3f}<extra></extra>',
            colorbar=dict(title="|SHAP Interaction|", thickness=15)
        ))
        
        # Apply thesis theme
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text="Top Feature Interactions (SHAP)"),
            xaxis_title="Feature",
            yaxis_title="Feature",
            height=THESIS_CONFIG.height_tall,
            margin=dict(l=120, r=20, t=50, b=100),  # Extra bottom for rotated labels
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
        fig.update_layout(**layout_config)
        
        return fig
    
    def _create_correlation_heatmap(self, top_n: int = 8) -> go.Figure:
        """Fallback: feature correlation heatmap if SHAP interactions unavailable."""
        from scipy.stats import pearsonr
        
        # Get feature correlations
        X_df = pd.DataFrame(self.explainer.data, columns=self.feature_names)
        corr_matrix = X_df.corr().abs()
        
        # Select top correlated pairs
        top_features = corr_matrix.mean().nlargest(top_n).index.tolist()
        subset = corr_matrix.loc[top_features, top_features]
        
        fig = go.Figure(data=go.Heatmap(
            z=subset.values,
            x=subset.columns,
            y=subset.index,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            hovertemplate='%{y} ↔ %{x}<br>|r|: %{z:.2f}<extra></extra>'
        ))
        
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text="Feature Correlation Matrix (Fallback)"),
            height=THESIS_CONFIG.height_tall
        )
        fig.update_layout(**layout_config)
        
        return fig