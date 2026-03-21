"""
VISUALIZATION MODULE FOR PREPROCESSING PIPELINE
================================================
Generates comprehensive visualizations for each preprocessing stage.
All plots saved to organized folders with descriptive names.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PreprocessingVisualizer:
    """Generate visualizations for all preprocessing stages."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def _create_folder(self, name: str) -> str:
        """Create and return folder path for visualization."""
        folder = os.path.join(self.viz_dir, name)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    def _save_plot(self, folder: str, filename: str):
        """Save current plot to folder."""
        filepath = os.path.join(folder, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    # ========================================================================
    # STAGE 1: QUALITY CHECK VISUALIZATIONS
    # ========================================================================
    
    def visualize_quality_check(self, report: Dict) -> str:
        """Visualize data quality metrics."""
        folder = self._create_folder("01_Quality_Check")
        
        # Plot 1: Complete vs Null Records
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        complete = report['complete_records']
        null_records = report['null_records']
        
        axes[0].bar(['Complete', 'With Nulls'], [complete, null_records], 
                    color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Data Completeness', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Records')
        for i, v in enumerate([complete, null_records]):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # Plot 2: Null values per column
        missing = report['missing_values']
        top_missing = dict(sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10])
        
        axes[1].barh(list(top_missing.keys()), list(top_missing.values()), color='#e67e22')
        axes[1].set_title('Top 10 Columns with Missing Values', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Missing Count')
        
        self._save_plot(folder, "01_data_completeness_overview")
        
        # Plot 3: Missing data percentage heatmap
        fig, ax = plt.subplots(figsize=(12, 4))
        missing_pct = {k: (v/report['total_rows']*100) for k, v in missing.items() 
                       if v > 0}
        missing_pct = dict(sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:15])
        
        ax.barh(list(missing_pct.keys()), list(missing_pct.values()), color='#3498db')
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Top 15 Columns - Missing Data Percentage', fontsize=12, fontweight='bold')
        for i, v in enumerate(missing_pct.values()):
            ax.text(v, i, f'{v:.1f}%', va='center')
        
        self._save_plot(folder, "02_missing_data_percentage")
        
        # Plot 4: Summary statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        stats_text = f"""
        QUALITY METRICS SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Rows: {report['total_rows']:,}
        Total Columns: {report['total_cols']}
        
        Complete Records (no nulls): {report['complete_records']:,} ({report['complete_records']/report['total_rows']*100:.1f}%)
        Records with Nulls: {report['null_records']:,} ({report['null_records']/report['total_rows']*100:.1f}%)
        
        Duplicate Rows: {report['duplicate_rows']}
        
        Columns with Missing Data: {sum(1 for v in missing.values() if v > 0)}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        self._save_plot(folder, "03_quality_summary_table")
        print(f"✓ Quality visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 2: TYPE DETECTION VISUALIZATIONS
    # ========================================================================
    
    def visualize_type_detection(self, types: Dict[str, str]) -> str:
        """Visualize detected data types distribution."""
        folder = self._create_folder("02_Type_Detection")
        
        # Count types
        type_counts = {}
        for dtype in types.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        # Plot 1: Type distribution pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(),
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Data Type Distribution', fontsize=12, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        self._save_plot(folder, "01_type_distribution_pie")
        
        # Plot 2: Type count bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(type_counts.keys(), type_counts.values(), color=colors)
        ax.set_title('Count of Each Data Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Columns')
        for i, (k, v) in enumerate(type_counts.items()):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=45)
        
        self._save_plot(folder, "02_type_count_bar")
        print(f"✓ Type detection visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 3: CLEANING VISUALIZATIONS
    # ========================================================================
    
    def visualize_cleaning(self, before_df: pd.DataFrame, after_df: pd.DataFrame) -> str:
        """Visualize cleaning impact."""
        folder = self._create_folder("03_Data_Cleaning")
        
        # Plot 1: Rows before/after
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(['Before', 'After'], [len(before_df), len(after_df)], 
                    color=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Rows Removed', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Rows')
        axes[0].text(0, len(before_df), str(len(before_df)), ha='center', va='bottom')
        axes[0].text(1, len(after_df), str(len(after_df)), ha='center', va='bottom')
        
        # Plot 2: Columns before/after
        axes[1].bar(['Before', 'After'], [len(before_df.columns), len(after_df.columns)],
                    color=['#e74c3c', '#2ecc71'])
        axes[1].set_title('Columns Removed', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Number of Columns')
        axes[1].text(0, len(before_df.columns), str(len(before_df.columns)), ha='center', va='bottom')
        axes[1].text(1, len(after_df.columns), str(len(after_df.columns)), ha='center', va='bottom')
        
        # Plot 3: Data quality improvement
        before_nulls = before_df.isnull().sum().sum()
        after_nulls = after_df.isnull().sum().sum()
        
        axes[2].bar(['Before', 'After'], [before_nulls, after_nulls],
                    color=['#e74c3c', '#2ecc71'])
        axes[2].set_title('Null Values Removed', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Number of Nulls')
        axes[2].text(0, before_nulls, str(before_nulls), ha='center', va='bottom')
        axes[2].text(1, after_nulls, str(after_nulls), ha='center', va='bottom')
        
        self._save_plot(folder, "01_cleaning_impact_overview")
        print(f"✓ Cleaning visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 4: AGGREGATION VISUALIZATIONS
    # ========================================================================
    
    def visualize_aggregation(self, df: pd.DataFrame, column: str, n_top: int = 10) -> str:
        """Visualize aggregation results."""
        folder = self._create_folder("04_Aggregation")
        
        value_counts = df[column].value_counts().head(n_top)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(value_counts.index, value_counts.values, color='#3498db')
        ax.set_xlabel('Count')
        ax.set_title(f'Top {n_top} Values in {column}', fontsize=12, fontweight='bold')
        
        for i, v in enumerate(value_counts.values):
            ax.text(v, i, f' {v}', va='center')
        
        self._save_plot(folder, f"01_aggregation_{column[:20]}")
        print(f"✓ Aggregation visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 6: OUTLIER DETECTION VISUALIZATIONS
    # ========================================================================
    
    def visualize_outliers(self, df: pd.DataFrame, column: str, outliers_df: pd.DataFrame) -> str:
        """Visualize outlier detection."""
        folder = self._create_folder("06_Outlier_Detection")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Box plot
        axes[0, 0].boxplot(df[column].dropna(), vert=True)
        axes[0, 0].set_title(f'Box Plot - {column}', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Value')
        
        # Histogram with outliers highlighted
        axes[0, 1].hist(df[column].dropna(), bins=50, alpha=0.7, color='#3498db', label='Normal')
        if len(outliers_df) > 0:
            axes[0, 1].hist(outliers_df[column].dropna(), bins=20, alpha=0.7, 
                           color='#e74c3c', label='Outliers')
        axes[0, 1].set_title(f'Distribution - {column}', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].set_ylabel('Frequency')
        
        # Outlier count
        outlier_pct = len(outliers_df) / len(df) * 100
        axes[1, 0].bar(['Normal', 'Outliers'], 
                       [len(df) - len(outliers_df), len(outliers_df)],
                       color=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Data Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].text(0, len(df) - len(outliers_df), str(len(df) - len(outliers_df)), 
                       ha='center', va='bottom')
        axes[1, 0].text(1, len(outliers_df), f'{len(outliers_df)}\n({outlier_pct:.1f}%)', 
                       ha='center', va='bottom')
        
        # Statistics
        stats_text = f"""
        OUTLIER STATISTICS
        ━━━━━━━━━━━━━━━━━━━━━━━━
        Total Records: {len(df):,}
        Normal: {len(df) - len(outliers_df):,}
        Outliers: {len(outliers_df)} ({outlier_pct:.2f}%)
        
        Mean: {df[column].mean():.2f}
        Median: {df[column].median():.2f}
        Std Dev: {df[column].std():.2f}
        Min: {df[column].min():.2f}
        Max: {df[column].max():.2f}
        """
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))
        
        self._save_plot(folder, f"01_outliers_{column[:20]}")
        print(f"✓ Outlier visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 7: CLASS IMBALANCE VISUALIZATIONS
    # ========================================================================
    
    def visualize_class_imbalance(self, y_original: pd.Series, 
                                  y_balanced: Optional[pd.Series] = None) -> str:
        """Visualize class imbalance before/after."""
        folder = self._create_folder("07_Class_Imbalance")
        
        fig, axes = plt.subplots(1, 2 if y_balanced is not None else 1, figsize=(14, 5))
        
        if y_balanced is None:
            axes = [axes]
        
        # Before
        dist_before = y_original.value_counts()
        axes[0].bar(dist_before.index, dist_before.values, color=['#e74c3c', '#2ecc71'][:len(dist_before)])
        axes[0].set_title('Class Distribution - Before SMOTE', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(dist_before.values):
            pct = v / len(y_original) * 100
            axes[0].text(i, v, f'{v}\n({pct:.1f}%)', ha='center', va='bottom')
        
        # After (if provided)
        if y_balanced is not None:
            dist_after = y_balanced.value_counts()
            axes[1].bar(dist_after.index, dist_after.values, color=['#e74c3c', '#2ecc71'][:len(dist_after)])
            axes[1].set_title('Class Distribution - After SMOTE', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Count')
            for i, v in enumerate(dist_after.values):
                pct = v / len(y_balanced) * 100
                axes[1].text(i, v, f'{v}\n({pct:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot(folder, "01_class_imbalance_before_after")
        print(f"✓ Class imbalance visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 11: PCA VISUALIZATIONS
    # ========================================================================
    
    def visualize_pca(self, pca_model: object, explained_variance: np.ndarray) -> str:
        """Visualize PCA results."""
        folder = self._create_folder("11_PCA_Dimensionality_Reduction")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variance explained
        cumsum = np.cumsum(explained_variance)
        axes[0].plot(range(1, len(explained_variance) + 1), explained_variance, 
                    'bo-', label='Individual')
        axes[0].plot(range(1, len(explained_variance) + 1), cumsum, 
                    'rs-', label='Cumulative')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Variance bar chart
        axes[1].bar(range(1, len(explained_variance) + 1), explained_variance, 
                   color='#3498db', alpha=0.7)
        axes[1].axhline(y=explained_variance.mean(), color='r', 
                       linestyle='--', label='Mean')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        axes[1].set_title('Individual Variance per Component', fontsize=12, fontweight='bold')
        axes[1].legend()
        
        self._save_plot(folder, "01_pca_variance_explained")
        print(f"✓ PCA visualizations saved to: {folder}")
        return folder
    
    # ========================================================================
    # STAGE 12: FINAL STATISTICS VISUALIZATIONS
    # ========================================================================
    
    def visualize_final_statistics(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """Visualize final dataset statistics."""
        folder = self._create_folder("12_ML_Preparation")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Train/Test split
        axes[0, 0].bar(['Train', 'Test'], [len(train_df), len(test_df)],
                      color=['#3498db', '#e67e22'])
        axes[0, 0].set_title('Train/Test Split', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].text(0, len(train_df), str(len(train_df)), ha='center', va='bottom')
        axes[0, 0].text(1, len(test_df), str(len(test_df)), ha='center', va='bottom')
        
        # Features
        axes[0, 1].text(0.5, 0.5, f'{len(train_df.columns)} Features',
                       fontsize=20, fontweight='bold', ha='center', va='center',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Total Features', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Data size
        train_size_mb = train_df.memory_usage(deep=True).sum() / 1024 / 1024
        test_size_mb = test_df.memory_usage(deep=True).sum() / 1024 / 1024
        
        axes[1, 0].bar(['Train', 'Test'], [train_size_mb, test_size_mb],
                      color=['#3498db', '#e67e22'])
        axes[1, 0].set_title('Memory Usage', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Size (MB)')
        axes[1, 0].text(0, train_size_mb, f'{train_size_mb:.2f} MB', ha='center', va='bottom')
        axes[1, 0].text(1, test_size_mb, f'{test_size_mb:.2f} MB', ha='center', va='bottom')
        
        # Summary
        summary_text = f"""
        FINAL DATASET SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━
        Train Samples: {len(train_df):,}
        Test Samples: {len(test_df):,}
        Total: {len(train_df) + len(test_df):,}
        
        Features: {len(train_df.columns)}
        
        Train Size: {train_size_mb:.2f} MB
        Test Size: {test_size_mb:.2f} MB
        
        Ready for ML Training ✓
        """
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round',
                       facecolor='lightgreen', alpha=0.5))
        
        self._save_plot(folder, "01_final_statistics_summary")
        print(f"✓ Final statistics visualizations saved to: {folder}")
        return folder


# Utility function
def create_summary_report(viz_dir: str) -> str:
    """Create HTML summary of all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Preprocessing Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
            h2 { color: #34495e; margin-top: 30px; }
            .stage { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
            img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>📊 Data Preprocessing Visualizations Report</h1>
        <p>Comprehensive visualizations of all preprocessing stages</p>
    </body>
    </html>
    """
    
    report_path = os.path.join(viz_dir, "index.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path