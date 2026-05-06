"""Plotly visualization utilities for F1 dashboard."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class F1VizUtils:
    """Collection of F1-specific visualization functions."""
    
    F1_COLORS = {
        'red': '#e10600', 'dark_red': '#a00000',
        'blue': '#00aaff', 'dark_blue': '#0066aa',
        'yellow': '#ffd700', 'silver': '#c0c0c0',
        'bronze': '#cd7f32', 'dark_bg': '#15151e',
        'card_bg': '#1e1e2a'
    }
    
    @staticmethod
    def create_position_distribution(simulation_results: pd.DataFrame, 
                                    top_n: int = 10) -> go.Figure:
        """Create violin plot of predicted position distributions."""
        fig = go.Figure()
        for driver in simulation_results['driver'].unique()[:top_n]:
            positions = simulation_results[
                (simulation_results['driver'] == driver) & 
                (simulation_results['finishing_position'].notna())
            ]['finishing_position']
            
            fig.add_trace(go.Violin(
                x=[driver]*len(positions), y=positions,
                name=driver, box_visible=True, meanline_visible=True,
                opacity=0.7, marker_color=F1VizUtils.F1_COLORS['red']
            ))
        
        fig.update_layout(
            title="Predicted Finishing Position Distribution",
            xaxis_title="Driver", yaxis_title="Finishing Position",
            yaxis_autorange='reversed',
            template="plotly_dark",
            plot_bgcolor=F1VizUtils.F1_COLORS['dark_bg'],
            paper_bgcolor=F1VizUtils.F1_COLORS['dark_bg'],
            height=500
        )
        return fig
    
    @staticmethod
    def create_podium_probability_chart(probabilities: dict) -> go.Figure:
        """Create horizontal bar chart of podium probabilities."""
        data = pd.DataFrame([
            {'driver': d, 'probability': p['podium_probability']*100}
            for d, p in probabilities.items()
        ]).sort_values('probability', ascending=False).head(12)
        
        fig = px.bar(
            data, x='probability', y='driver', orientation='h',
            color='probability', color_continuous_scale='YlOrRd',
            title="Podium Probability (%)"
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=F1VizUtils.F1_COLORS['dark_bg'],
            paper_bgcolor=F1VizUtils.F1_COLORS['dark_bg']
        )
        return fig
    
    @staticmethod
    def create_driver_comparison_radar(driver1_data: pd.Series, 
                                      driver2_data: pd.Series) -> go.Figure:
        """Create radar chart comparing two drivers across metrics."""
        metrics = ['Qualifying Pace', 'Race Pace', 'Consistency', 
                  'Tire Management', 'Wet Weather', 'Overtaking']
        
        fig = go.Figure()
        
        # Driver 1
        fig.add_trace(go.Scatterpolar(
            r=[driver1_data.get(m, 0.5) for m in metrics],
            theta=metrics, fill='toself', name=driver1_data.get('driver', 'Driver 1'),
            line_color=F1VizUtils.F1_COLORS['red']
        ))
        
        # Driver 2
        fig.add_trace(go.Scatterpolar(
            r=[driver2_data.get(m, 0.5) for m in metrics],
            theta=metrics, fill='toself', name=driver2_data.get('driver', 'Driver 2'),
            line_color=F1VizUtils.F1_COLORS['blue']
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, template="plotly_dark",
            plot_bgcolor=F1VizUtils.F1_COLORS['dark_bg'],
            paper_bgcolor=F1VizUtils.F1_COLORS['dark_bg']
        )
        return fig