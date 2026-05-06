"""
Prediction uncertainty calibration visualizations for thesis
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve  # Fallback for sklearn < 1.5
from .export_config import THESIS_CONFIG

class CalibrationAnalyzer:
    """Generate thesis-ready calibration and uncertainty visualizations."""
    
    @staticmethod
    def create_calibration_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                               pred_std: Optional[np.ndarray] = None,
                               title: str = "Prediction Calibration",
                               n_bins: int = 10) -> go.Figure:
        """
        Create calibration plot comparing predicted vs. empirical coverage.
        
        Args:
            y_true: Actual values
            y_pred: Point predictions
            pred_std: Predicted standard deviations (for interval calibration)
            n_bins: Number of bins for calibration curve
        """
        
        fig = go.Figure()
        
        if pred_std is not None:
            # Interval calibration: check coverage at different confidence levels
            coverage_data = []
            
            for confidence in np.linspace(0.5, 0.99, n_bins):
                alpha = 1 - confidence
                z_score = 1.96 if confidence == 0.95 else np.abs(
                    np.percentile(np.random.normal(0, 1, 10000), alpha/2 * 100)
                )
                
                lower = y_pred - z_score * pred_std
                upper = y_pred + z_score * pred_std
                
                covered = ((y_true >= lower) & (y_true <= upper)).mean()
                coverage_data.append({
                    'nominal': confidence,
                    'empirical': covered,
                    'gap': covered - confidence
                })
            
            df = pd.DataFrame(coverage_data)
            
            # Plot empirical vs. nominal
            fig.add_trace(go.Scatter(
                x=df['nominal'], y=df['empirical'],
                mode='lines+markers',
                name='Empirical Coverage',
                line=dict(color=THESIS_CONFIG.colors['primary'], width=2.5),
                marker=dict(size=6, symbol='circle')
            ))
            
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0.5, 0.99], y=[0.5, 0.99],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color=THESIS_CONFIG.colors['neutral'], width=1.5, dash='dash'),
                showlegend=True
            ))
            
            # Add confidence band for sampling variability
            bin_size = len(y_true) // n_bins
            std_error = np.sqrt(0.95 * 0.05 / bin_size)  # Approximate
            
            fig.add_trace(go.Scatter(
                x=df['nominal'],
                y=df['nominal'] + 1.96 * std_error,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['nominal'],
                y=df['nominal'] - 1.96 * std_error,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0,114,178,0.1)',
                name='95% Sampling CI',
                showlegend=True
            ))
            
            # Calculate calibration metrics
            ece = np.mean(np.abs(df['empirical'] - df['nominal']))  # Expected Calibration Error
            mce = np.max(np.abs(df['empirical'] - df['nominal']))   # Maximum Calibration Error
            
            # Add annotation with metrics
            annotation_text = f"ECE: {ece:.3f}<br>MCE: {mce:.3f}"
            
        else:
            # Standard probability calibration (for classification-style outputs)
            # Bin predictions and compute empirical frequency
            from sklearn.calibration import calibration_curve
            
            prob_true, prob_pred = calibration_curve(
                (y_true <= y_pred).astype(int),  # Binary: was prediction an underestimate?
                np.clip((y_pred - y_true + 1) / 2, 0, 1),  # Pseudo-probability
                n_bins=n_bins
            )
            
            fig.add_trace(go.Scatter(
                x=prob_pred, y=prob_true,
                mode='lines+markers',
                name='Model Calibration',
                line=dict(color=THESIS_CONFIG.colors['primary'], width=2.5),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color=THESIS_CONFIG.colors['neutral'], width=1.5, dash='dash')
            ))
            
            annotation_text = f"Bins: {n_bins}"
        
        # Apply thesis theme
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text=title),
            xaxis_title="Nominal Confidence Level" if pred_std is not None else "Predicted Probability",
            yaxis_title="Empirical Coverage" if pred_std is not None else "Fraction of Positives",
            height=THESIS_CONFIG.height_standard,
            annotations=[
                dict(
                    x=0.98, y=0.02,
                    text=annotation_text,
                    xref="paper", yref="paper",
                    showarrow=False,
                    align="right",
                    font=dict(size=THESIS_CONFIG.annotation_font_size),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=THESIS_CONFIG.colors['grid'],
                    borderwidth=1
                )
            ]
        )
        fig.update_layout(**layout_config)
        
        return fig
    
    @staticmethod
    def create_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                                    feature_for_color: Optional[pd.Series] = None,
                                    title: str = "Residual Analysis") -> go.Figure:
        """Create residual plot with optional feature coloring and QQ plot inset."""
        
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        # Main residual scatter
        scatter_kwargs = dict(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.6,
                line=dict(width=0.3, color='rgba(0,0,0,0.1)')
            ),
            hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        )
        
        if feature_for_color is not None:
            # Color by feature value
            fig.add_trace(go.Scatter(
                **scatter_kwargs,
                marker=dict(
                    color=feature_for_color.values,
                    colorscale='RdYlBu',
                    colorbar=dict(title=feature_for_color.name),
                    showscale=True
                )
            ))
        else:
            fig.add_trace(go.Scatter(**scatter_kwargs, marker_color=THESIS_CONFIG.colors['primary']))
        
        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color=THESIS_CONFIG.colors['neutral'], 
                     line_width=1, opacity=0.5)
        
        # Add LOESS smooth to detect patterns
        if len(residuals) > 30:
            from scipy import stats
            slope, intercept, r_val, p_val, _ = stats.linregress(y_pred, residuals)
            x_line = np.array([y_pred.min(), y_pred.max()])
            y_line = intercept + slope * x_line
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=THESIS_CONFIG.colors['secondary'], width=2),
                name=f'Trend (R²={r_val**2:.3f})',
                showlegend=True
            ))
        
        # Add QQ plot as inset for normality check
        from scipy import stats
        qq_x = np.linspace(-3, 3, 100)
        qq_y = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
        
        # Standardize residuals for QQ
        std_resid = (residuals - residuals.mean()) / residuals.std()
        
        # Create inset axes
        fig.add_trace(go.Scatter(
            x=stats.norm.ppf(np.linspace(0.01, 0.99, len(std_resid))),
            y=np.sort(std_resid),
            mode='markers',
            marker=dict(size=3, color=THESIS_CONFIG.colors['accent'], opacity=0.7),
            name='QQ Plot',
            xaxis='x2', yaxis='y2',
            showlegend=True
        ))
        
        # QQ reference line
        fig.add_trace(go.Scatter(
            x=qq_x, y=qq_x,
            mode='lines',
            line=dict(color=THESIS_CONFIG.colors['neutral'], width=1, dash='dot'),
            xaxis='x2', yaxis='y2',
            showlegend=False
        ))
        
        # Apply thesis theme with inset configuration
        layout_config = THESIS_CONFIG.get_plotly_theme()
        layout_config.update(
            title=dict(text=title),
            xaxis_title="Predicted Value",
            yaxis_title="Residual (Actual − Predicted)",
            height=THESIS_CONFIG.height_standard,
            # Inset QQ plot configuration
            xaxis2=dict(
                domain=[0.65, 0.95], anchor='y2',
                title='Theoretical Quantiles',
                title_font=dict(size=8),
                tickfont=dict(size=7),
                gridcolor='rgba(0,0,0,0)',
                zeroline=False
            ),
            yaxis2=dict(
                domain=[0.65, 0.95], anchor='x2',
                title='Sample Quantiles',
                title_font=dict(size=8),
                tickfont=dict(size=7),
                gridcolor='rgba(0,0,0,0)',
                zeroline=False
            ),
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    text=f"MAE: {np.mean(np.abs(residuals)):.3f}<br>Std: {residuals.std():.3f}",
                    xref="paper", yref="paper",
                    showarrow=False,
                    align="left",
                    font=dict(size=THESIS_CONFIG.annotation_font_size),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=THESIS_CONFIG.colors['grid'],
                    borderwidth=1
                )
            ]
        )
        fig.update_layout(**layout_config)
        
        return fig