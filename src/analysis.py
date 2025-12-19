"""
ASA DataFest 2024: CourseKata Learning Analytics
================================================

This script analyzes student learning data from CourseKata's online statistics
education platform to identify:
- Psychological factors predicting student success
- "Stumbling block" chapters where students struggle
- Student behavioral profiles through clustering
- Network relationships between learning components

Author: DataFest 2024 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'Data/Random Sample of Data Files/'
OUTPUT_PATH = 'outputs/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def load_data():
    """Load all datasets from the data directory."""
    print("Loading data...")

    pulse = pd.read_csv(f'{DATA_PATH}checkpoints_pulse.csv')
    eoc = pd.read_csv(f'{DATA_PATH}checkpoints_eoc.csv')
    page_views = pd.read_csv(f'{DATA_PATH}page_views.csv')
    responses = pd.read_csv(f'{DATA_PATH}responses.csv', low_memory=False)
    media_views = pd.read_csv(f'{DATA_PATH}media_views.csv')

    print(f"  Pulse Checkpoints: {len(pulse):,} records")
    print(f"  EOC Assessments:   {len(eoc):,} records")
    print(f"  Page Views:        {len(page_views):,} records")
    print(f"  Responses:         {len(responses):,} records")
    print(f"  Media Views:       {len(media_views):,} records")

    return pulse, eoc, page_views, responses, media_views


def analyze_psychological_constructs(pulse, eoc):
    """Analyze relationship between psychological constructs and performance."""
    print("\nAnalyzing psychological constructs...")

    # Clean pulse data
    pulse_clean = pulse[pulse['response'] != 'NA'].copy()
    pulse_clean['response'] = pd.to_numeric(pulse_clean['response'], errors='coerce')
    pulse_clean = pulse_clean.dropna(subset=['response'])

    # Pivot and merge
    pulse_pivot = pulse_clean.pivot_table(
        index=['student_id', 'class_id', 'chapter_number'],
        columns='construct',
        values='response',
        aggfunc='mean'
    ).reset_index()

    merged = pulse_pivot.merge(
        eoc[['student_id', 'class_id', 'chapter_number', 'EOC', 'n_correct', 'n_possible', 'n_attempt']],
        on=['student_id', 'class_id', 'chapter_number'],
        how='inner'
    )

    # Calculate correlations
    constructs = ['Cost', 'Expectancy', 'Intrinsic Value', 'Utility Value']
    available_constructs = [c for c in constructs if c in merged.columns]

    if available_constructs:
        correlations = merged[available_constructs + ['EOC']].corr()['EOC'].drop('EOC')

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if x < 0 else '#27ae60' for x in correlations.values]
        bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor='black', linewidth=1.5)

        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Correlation with EOC Performance', fontsize=12)
        ax.set_title('Psychological Constructs vs Academic Performance', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, correlations.values):
            ax.text(val + 0.02 if val >= 0 else val - 0.08, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}construct_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Strongest predictor: {correlations.idxmax()} (r = {correlations.max():.3f})")
        return merged, correlations, available_constructs

    return merged, pd.Series(), []


def analyze_chapter_difficulty(eoc):
    """Identify stumbling block chapters."""
    print("\nAnalyzing chapter difficulty...")

    chapter_stats = eoc.groupby('chapter_number').agg({
        'EOC': ['mean', 'std', 'count'],
        'n_attempt': 'mean',
        'student_id': 'nunique'
    }).round(3)

    chapter_stats.columns = ['avg_score', 'score_std', 'observations', 'avg_attempts', 'unique_students']
    chapter_stats = chapter_stats.reset_index()

    # Calculate struggle index
    chapter_stats['struggle_index'] = (1 - chapter_stats['avg_score']) * np.log1p(chapter_stats['avg_attempts'])
    chapter_stats = chapter_stats.sort_values('chapter_number')

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average Score
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(chapter_stats['avg_score'])
    ax1.bar(chapter_stats['chapter_number'], chapter_stats['avg_score'], color=colors, edgecolor='black')
    ax1.axhline(y=chapter_stats['avg_score'].mean(), color='red', linestyle='--', label='Overall Average')
    ax1.set_xlabel('Chapter')
    ax1.set_ylabel('Average EOC Score')
    ax1.set_title('Performance by Chapter', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot 2: Attempts
    ax2 = axes[0, 1]
    ax2.bar(chapter_stats['chapter_number'], chapter_stats['avg_attempts'], color='#3498db', edgecolor='black')
    ax2.set_xlabel('Chapter')
    ax2.set_ylabel('Average Attempts')
    ax2.set_title('Effort Required by Chapter', fontweight='bold')

    # Plot 3: Struggle Index
    ax3 = axes[1, 0]
    colors = plt.cm.Reds(chapter_stats['struggle_index'] / chapter_stats['struggle_index'].max())
    ax3.bar(chapter_stats['chapter_number'], chapter_stats['struggle_index'], color=colors, edgecolor='black')
    ax3.set_xlabel('Chapter')
    ax3.set_ylabel('Struggle Index')
    ax3.set_title('Struggle Index (Low Score × High Attempts)', fontweight='bold')

    # Plot 4: Boxplot
    ax4 = axes[1, 1]
    eoc_box = eoc[eoc['chapter_number'] <= 12]
    eoc_box.boxplot(column='EOC', by='chapter_number', ax=ax4)
    ax4.set_xlabel('Chapter')
    ax4.set_ylabel('EOC Score')
    ax4.set_title('Score Distribution by Chapter', fontweight='bold')
    plt.suptitle('')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}chapter_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Identify stumbling blocks
    stumbling = chapter_stats.nlargest(3, 'struggle_index')
    print("  Stumbling blocks:")
    for _, row in stumbling.iterrows():
        print(f"    Chapter {int(row['chapter_number'])}: Score={row['avg_score']:.1%}, Attempts={row['avg_attempts']:.1f}")

    return chapter_stats, stumbling


def analyze_engagement(page_views, eoc):
    """Analyze student engagement patterns."""
    print("\nAnalyzing engagement patterns...")

    # Aggregate by student
    student_engagement = page_views.groupby('student_id').agg({
        'engaged': 'sum',
        'idle_brief': 'sum',
        'idle_long': 'sum',
        'off_page_brief': 'sum',
        'off_page_long': 'sum',
        'was_complete': 'mean',
        'chapter_number': 'nunique',
        'page': 'count'
    }).reset_index()

    student_engagement.columns = ['student_id', 'total_engaged_ms', 'total_idle_brief_ms',
                                   'total_idle_long_ms', 'total_off_brief_ms', 'total_off_long_ms',
                                   'completion_rate', 'chapters_accessed', 'page_views']

    # Convert to hours
    for col in ['total_engaged_ms', 'total_idle_brief_ms', 'total_idle_long_ms']:
        student_engagement[col.replace('_ms', '_hrs')] = student_engagement[col] / 3600000

    student_engagement['total_time_hrs'] = (student_engagement['total_engaged_ms'] +
                                             student_engagement['total_idle_brief_ms'] +
                                             student_engagement['total_idle_long_ms']) / 3600000
    student_engagement['engagement_ratio'] = student_engagement['total_engaged_ms'] / (
        student_engagement['total_engaged_ms'] + student_engagement['total_idle_brief_ms'] + 1)

    # Merge with performance
    student_performance = eoc.groupby('student_id').agg({
        'EOC': 'mean',
        'n_attempt': 'sum',
        'chapter_number': 'nunique'
    }).reset_index()
    student_performance.columns = ['student_id', 'avg_eoc', 'total_attempts', 'chapters_completed']

    engagement_performance = student_engagement.merge(student_performance, on='student_id', how='inner')

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    ax1.scatter(engagement_performance['total_engaged_hrs'],
                engagement_performance['avg_eoc'],
                alpha=0.5, c='#3498db', edgecolor='white')
    ax1.set_xlabel('Total Engaged Time (hours)')
    ax1.set_ylabel('Average EOC Score')
    ax1.set_title('Time on Task vs Performance', fontweight='bold')

    ax2 = axes[1]
    ax2.scatter(engagement_performance['engagement_ratio'],
                engagement_performance['avg_eoc'],
                alpha=0.5, c='#e74c3c', edgecolor='white')
    ax2.set_xlabel('Engagement Ratio')
    ax2.set_ylabel('Average EOC Score')
    ax2.set_title('Focus Quality vs Performance', fontweight='bold')

    ax3 = axes[2]
    ax3.scatter(engagement_performance['completion_rate'],
                engagement_performance['avg_eoc'],
                alpha=0.5, c='#27ae60', edgecolor='white')
    ax3.set_xlabel('Page Completion Rate')
    ax3.set_ylabel('Average EOC Score')
    ax3.set_title('Completion Rate vs Performance', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}engagement_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return engagement_performance


def cluster_students(engagement_performance):
    """Cluster students into behavioral profiles."""
    print("\nClustering students into behavioral profiles...")

    cluster_features = engagement_performance[[
        'engagement_ratio', 'completion_rate', 'avg_eoc', 'total_attempts'
    ]].dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_features)

    # K-means with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_features['cluster'] = kmeans.fit_predict(features_scaled)

    # Name clusters
    cluster_profiles = cluster_features.groupby('cluster').mean()
    cluster_names = {
        cluster_profiles['avg_eoc'].idxmax(): 'High Performers',
        cluster_profiles['avg_eoc'].idxmin(): 'Struggling Students',
    }
    remaining = set(range(4)) - set(cluster_names.keys())
    for c in remaining:
        if cluster_profiles.loc[c, 'engagement_ratio'] > cluster_profiles['engagement_ratio'].median():
            cluster_names[c] = 'Engaged Learners'
        else:
            cluster_names[c] = 'Passive Completers'

    cluster_features['profile'] = cluster_features['cluster'].map(cluster_names)

    # PCA visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    cluster_features['pca1'] = features_pca[:, 0]
    cluster_features['pca2'] = features_pca[:, 1]

    fig = px.scatter(
        cluster_features,
        x='pca1', y='pca2',
        color='profile',
        size='avg_eoc',
        hover_data=['engagement_ratio', 'completion_rate', 'avg_eoc'],
        title='Student Behavioral Clusters (PCA Projection)',
        labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(width=900, height=600)
    fig.write_html(f'{OUTPUT_PATH}student_clusters_interactive.html')

    print("  Student profiles identified:")
    for profile in cluster_features['profile'].unique():
        count = (cluster_features['profile'] == profile).sum()
        avg = cluster_features[cluster_features['profile'] == profile]['avg_eoc'].mean()
        print(f"    {profile}: {count} students, {avg:.1%} avg score")

    return cluster_features, cluster_names


def create_network_visualization(chapter_stats, merged, eoc, available_constructs):
    """Create force-directed network visualization."""
    print("\nCreating network visualization...")

    G = nx.Graph()

    construct_colors = {
        'Cost': '#e74c3c',
        'Expectancy': '#27ae60',
        'Intrinsic Value': '#3498db',
        'Utility Value': '#f39c12'
    }

    # Add chapter nodes
    for chapter in chapter_stats['chapter_number'].unique():
        row = chapter_stats[chapter_stats['chapter_number'] == chapter].iloc[0]
        G.add_node(
            f"Ch{int(chapter)}",
            node_type='chapter',
            score=row['avg_score'],
            struggle=row['struggle_index'],
            size=20 + row['struggle_index'] * 50
        )

    # Add construct nodes
    for construct in available_constructs:
        G.add_node(
            construct,
            node_type='construct',
            color=construct_colors.get(construct, '#9b59b6'),
            size=30
        )

    # Add chapter correlation edges
    chapter_corr = eoc.pivot_table(
        index='student_id',
        columns='chapter_number',
        values='EOC'
    ).corr()

    for i in chapter_corr.index:
        for j in chapter_corr.columns:
            if i < j and not pd.isna(chapter_corr.loc[i, j]):
                corr_val = chapter_corr.loc[i, j]
                if abs(corr_val) > 0.3:
                    G.add_edge(
                        f"Ch{int(i)}", f"Ch{int(j)}",
                        weight=abs(corr_val),
                        color='#95a5a6' if corr_val > 0 else '#e74c3c'
                    )

    # Add construct-chapter edges
    if len(merged) > 0:
        for chapter in merged['chapter_number'].unique():
            chapter_data = merged[merged['chapter_number'] == chapter]
            for construct in available_constructs:
                if construct in chapter_data.columns:
                    corr = chapter_data[construct].corr(chapter_data['EOC'])
                    if not pd.isna(corr) and abs(corr) > 0.2:
                        G.add_edge(
                            construct, f"Ch{int(chapter)}",
                            weight=abs(corr),
                            color=construct_colors.get(construct, '#9b59b6')
                        )

    # Create PyVis network
    net = Network(height='700px', width='100%', bgcolor='#ffffff', font_color='#333333')
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'chapter':
            score = data.get('score', 0.5)
            r = int(255 * (1 - score))
            g = int(255 * score)
            color = f'rgb({r}, {g}, 100)'
            size = data.get('size', 25)
            title = f"{node}\nAvg Score: {score:.1%}\nStruggle Index: {data.get('struggle', 0):.2f}"
        else:
            color = data.get('color', '#9b59b6')
            size = data.get('size', 30)
            title = f"{node}\n(Psychological Construct)"

        net.add_node(node, label=node, color=color, size=size, title=title)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0.5) * 3
        color = data.get('color', '#95a5a6')
        net.add_edge(u, v, value=weight, color=color)

    net.save_graph(f'{OUTPUT_PATH}learning_network.html')

    # Static visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    edges = G.edges(data=True)
    edge_colors = [d.get('color', '#cccccc') for _, _, d in edges]
    edge_widths = [d.get('weight', 0.5) * 3 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color=edge_colors, width=edge_widths, ax=ax)

    chapter_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'chapter']
    construct_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'construct']

    chapter_scores = [G.nodes[n].get('score', 0.5) for n in chapter_nodes]
    chapter_sizes = [G.nodes[n].get('size', 25) * 15 for n in chapter_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=chapter_nodes,
                            node_color=chapter_scores, cmap=plt.cm.RdYlGn,
                            node_size=chapter_sizes, ax=ax, vmin=0, vmax=1)

    construct_colors_list = [construct_colors.get(n, '#9b59b6') for n in construct_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=construct_nodes,
                            node_color=construct_colors_list, node_size=600,
                            node_shape='s', ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    ax.set_title('Learning Network: Chapters, Constructs, and Connections', fontsize=14, fontweight='bold')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label='Chapter Performance')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}learning_network_static.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def create_summary_dashboard(chapter_stats, correlations, engagement_performance, cluster_features):
    """Create summary dashboard visualization."""
    print("\nCreating summary dashboard...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Chapter Performance',
            'Psychological Constructs Impact',
            'Engagement vs Performance',
            'Student Profiles'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'pie'}]]
    )

    # Plot 1
    colors = ['#e74c3c' if x > chapter_stats['struggle_index'].quantile(0.75) else '#3498db'
              for x in chapter_stats['struggle_index']]
    fig.add_trace(
        go.Bar(x=chapter_stats['chapter_number'], y=chapter_stats['avg_score'], marker_color=colors),
        row=1, col=1
    )

    # Plot 2
    if len(correlations) > 0:
        colors2 = ['#e74c3c' if x < 0 else '#27ae60' for x in correlations.values]
        fig.add_trace(
            go.Bar(x=correlations.index, y=correlations.values, marker_color=colors2),
            row=1, col=2
        )

    # Plot 3
    fig.add_trace(
        go.Scatter(x=engagement_performance['engagement_ratio'],
                   y=engagement_performance['avg_eoc'],
                   mode='markers', marker=dict(size=8, opacity=0.5, color='#3498db')),
        row=2, col=1
    )

    # Plot 4
    profile_counts = cluster_features['profile'].value_counts()
    fig.add_trace(
        go.Pie(labels=profile_counts.index, values=profile_counts.values,
               marker_colors=['#27ae60', '#e74c3c', '#3498db', '#f39c12']),
        row=2, col=2
    )

    fig.update_layout(
        height=800, width=1200,
        title_text='DataFest 2024: CourseKata Learning Analytics Summary',
        showlegend=False
    )

    fig.write_html(f'{OUTPUT_PATH}summary_dashboard.html')
    try:
        fig.write_image(f'{OUTPUT_PATH}summary_dashboard.png', scale=2)
    except Exception as e:
        print(f"  Note: Could not save PNG (kaleido may not be installed): {e}")

    print("  Dashboard saved!")


def print_summary(eoc, correlations, stumbling, cluster_features):
    """Print summary of key findings."""
    print("\n" + "="*60)
    print("KEY FINDINGS: CourseKata Learning Analytics")
    print("="*60)

    print("\n[DATASET OVERVIEW]")
    print(f"   Unique students: {eoc['student_id'].nunique():,}")
    print(f"   Chapters analyzed: {eoc['chapter_number'].nunique()}")
    print(f"   Overall average EOC: {eoc['EOC'].mean():.1%}")

    print("\n[PSYCHOLOGICAL CONSTRUCTS]")
    if len(correlations) > 0:
        print(f"   Strongest predictor: {correlations.idxmax()} (r = {correlations.max():.3f})")

    print("\n[STUMBLING BLOCKS]")
    for _, row in stumbling.iterrows():
        print(f"   Chapter {int(row['chapter_number'])}: {row['avg_score']:.1%} avg score")

    print("\n[STUDENT PROFILES]")
    for profile in cluster_features['profile'].unique():
        count = (cluster_features['profile'] == profile).sum()
        avg = cluster_features[cluster_features['profile'] == profile]['avg_eoc'].mean()
        print(f"   {profile}: {count} students, {avg:.1%} avg")

    print("\n" + "="*60)
    print("All visualizations saved to outputs/ folder")
    print("="*60)


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("ASA DataFest 2024: CourseKata Learning Analytics")
    print("="*60)

    # Load data
    pulse, eoc, page_views, responses, media_views = load_data()

    # Analysis pipeline
    merged, correlations, available_constructs = analyze_psychological_constructs(pulse, eoc)
    chapter_stats, stumbling = analyze_chapter_difficulty(eoc)
    engagement_performance = analyze_engagement(page_views, eoc)
    cluster_features, cluster_names = cluster_students(engagement_performance)
    G = create_network_visualization(chapter_stats, merged, eoc, available_constructs)
    create_summary_dashboard(chapter_stats, correlations, engagement_performance, cluster_features)

    # Summary
    print_summary(eoc, correlations, stumbling, cluster_features)


if __name__ == "__main__":
    main()
