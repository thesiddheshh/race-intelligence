"""
Thesis Figure Export Configuration
Academic-standard formatting for publication-ready visualizations
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ThesisExportConfig:
    """Centralized configuration for thesis-grade figure exports."""
    
    # Typography (academic standard)
    font_family: str = "Times New Roman"
    font_size_base: int = 11
    font_size_title: int = 14
    font_size_axis: int = 10
    
    # Dimensions (standard thesis column widths)
    width_single_col: int = 345    # pts (~4.8 inches)
    width_double_col: int = 710    # pts (~9.9 inches)
    height_standard: int = 250     # pts (~3.5 inches)
    height_tall: int = 400         # pts (~5.6 inches)
    dpi_print: int = 300           # For raster exports
    
    # Color scheme (colorblind-safe, print-friendly)
    colors: Dict[str, str] = None
    
    # Export settings
    vector_formats: tuple = ("pdf", "svg", "eps")
    raster_formats: tuple = ("png", "jpg")
    default_format: str = "pdf"
    
    # Statistical annotation style
    show_confidence_intervals: bool = True
    confidence_level: float = 0.95
    annotation_font_size: int = 9
    
    def __post_init__(self):
        if self.colors is None:
            # Okabe-Ito colorblind-safe palette + academic neutrals
            self.colors = {
                'primary': '#0072B2',    # Blue
                'secondary': '#D55E00',  # Vermilion  
                'accent': '#009E73',     # Bluish green
                'warning': '#CC79A7',    # Reddish purple
                'neutral': '#999999',    # Gray
                'background': '#FFFFFF',
                'grid': '#E6E6E6',
                'text': '#000000'
            }
    
    def get_plotly_theme(self, double_column: bool = False) -> dict:
        """Generate Plotly layout config matching thesis standards."""
        width = self.width_double_col if double_column else self.width_single_col
        
        return {
            "template": "plotly_white",  # Clean white background for print
            "font": dict(
                family=self.font_family,
                size=self.font_size_base,
                color=self.colors['text']
            ),
            "title": dict(
                font=dict(family=self.font_family, size=self.font_size_title),
                x=0.5, xanchor='center'
            ),
            "xaxis": dict(
                title_font=dict(family=self.font_family, size=self.font_size_axis),
                tickfont=dict(family=self.font_family, size=self.font_size_axis),
                gridcolor=self.colors['grid'],
                zeroline=True,
                zerolinecolor=self.colors['grid'],
                linewidth=1,
                linecolor=self.colors['neutral']
            ),
            "yaxis": dict(
                title_font=dict(family=self.font_family, size=self.font_size_axis),
                tickfont=dict(family=self.font_family, size=self.font_size_axis),
                gridcolor=self.colors['grid'],
                zeroline=True,
                zerolinecolor=self.colors['grid'],
                linewidth=1,
                linecolor=self.colors['neutral']
            ),
            "plot_bgcolor": self.colors['background'],
            "paper_bgcolor": self.colors['background'],
            "margin": dict(l=50, r=20, t=60, b=40),
            "height": self.height_standard,
            "width": width,
            "hovermode": "x unified",
            "showlegend": True,
            "legend": dict(
                font=dict(family=self.font_family, size=self.font_size_base - 1),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=self.colors['grid'],
                borderwidth=1
            )
        }
    
    def get_export_path(self, figure_name: str, format: Optional[str] = None) -> str:
        """Generate standardized export path."""
        from pathlib import Path
        fmt = format or self.default_format
        return f"data/thesis_exports/{figure_name}.{fmt}"

# Global instance
THESIS_CONFIG = ThesisExportConfig()