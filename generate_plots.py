#!/usr/bin/env python3
"""
Comprehensive plotting script for language model evaluation results.
Generates publication-quality plots with valuable insights.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_plot_directory():
    """Create plots directory if it doesn't exist."""
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def load_data():
    """Load all evaluation data."""
    print("Loading evaluation data...")
    
    # Load plotting data
    df = pd.read_csv("results/plotting_data.tsv", sep='\t')
    
    # Load comprehensive results for additional insights
    with open("results/comprehensive_eval_results.json") as f:
        comprehensive = json.load(f)
    
    # Load performance summary
    with open("results/performance_summary.json") as f:
        summary = json.load(f)
    
    return df, comprehensive, summary

def plot_perplexity_comparison(df, plots_dir):
    """Plot 1: Model performance comparison across datasets and context lengths."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1a: Bar plot of best performance (context=2048)
    best_data = df[df['context_length'] == 2048]
    sns.barplot(data=best_data, x='model', y='perplexity', hue='dataset', ax=axes[0,0])
    axes[0,0].set_title('Model Performance Comparison\n(Best Context Length: 2048)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Model', fontsize=12)
    axes[0,0].set_ylabel('Perplexity (lower is better)', fontsize=12)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(title='Dataset')
    
    # Plot 1b: Context length effects
    context_pivot = df.groupby(['model', 'context_length'])['perplexity'].mean().reset_index()
    sns.lineplot(data=context_pivot, x='context_length', y='perplexity', 
                 hue='model', marker='o', linewidth=2.5, markersize=8, ax=axes[0,1])
    axes[0,1].set_title('Context Length Effects on Perplexity', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Context Length (tokens)', fontsize=12)
    axes[0,1].set_ylabel('Average Perplexity', fontsize=12)
    axes[0,1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 1c: Perplexity distribution by model
    sns.boxplot(data=df, x='model', y='perplexity', ax=axes[1,0])
    axes[1,0].set_title('Perplexity Distribution by Model', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Model', fontsize=12)
    axes[1,0].set_ylabel('Perplexity', fontsize=12)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 1d: Dataset difficulty comparison
    dataset_comparison = df.groupby(['model', 'dataset'])['perplexity'].mean().reset_index()
    pivot_data = dataset_comparison.pivot(index='model', columns='dataset', values='perplexity')
    pivot_data['ratio'] = pivot_data['c4'] / pivot_data['wikitext2']
    
    sns.barplot(data=dataset_comparison, x='model', y='perplexity', hue='dataset', ax=axes[1,1])
    axes[1,1].set_title('Dataset Difficulty: C4 vs WikiText-2', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Model', fontsize=12)
    axes[1,1].set_ylabel('Average Perplexity', fontsize=12)
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(title='Dataset')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'perplexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated perplexity_analysis.png")

def plot_efficiency_analysis(df, plots_dir):
    """Plot 2: Computational efficiency analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate efficiency metrics
    df_copy = df.copy()
    df_copy['efficiency_score'] = df_copy['throughput_tokens_per_sec'] / (df_copy['perplexity'] * df_copy['memory_usage_gb'])
    df_copy['perplexity_per_gb'] = df_copy['perplexity'] / df_copy['memory_usage_gb']
    
    # Plot 2a: Memory usage vs performance
    sns.scatterplot(data=df_copy, x='memory_usage_gb', y='perplexity', 
                    hue='model', size='context_length', sizes=(100, 300), ax=axes[0,0])
    axes[0,0].set_title('Memory Usage vs Performance Trade-off', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Memory Usage (GB)', fontsize=12)
    axes[0,0].set_ylabel('Perplexity (lower is better)', fontsize=12)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2b: Throughput comparison
    sns.boxplot(data=df_copy, x='model', y='throughput_tokens_per_sec', ax=axes[0,1])
    axes[0,1].set_title('Throughput Distribution by Model', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Model', fontsize=12)
    axes[0,1].set_ylabel('Throughput (tokens/sec)', fontsize=12)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 2c: Overall efficiency score
    efficiency_avg = df_copy.groupby('model')['efficiency_score'].mean().sort_values(ascending=False)
    sns.barplot(x=efficiency_avg.index, y=efficiency_avg.values, ax=axes[1,0])
    axes[1,0].set_title('Overall Efficiency Ranking\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Model', fontsize=12)
    axes[1,0].set_ylabel('Efficiency Score', fontsize=12)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 2d: Memory scaling with context length
    memory_scaling = df_copy.groupby(['model', 'context_length'])['memory_usage_gb'].mean().reset_index()
    sns.lineplot(data=memory_scaling, x='context_length', y='memory_usage_gb', 
                 hue='model', marker='s', linewidth=2.5, markersize=8, ax=axes[1,1])
    axes[1,1].set_title('Memory Scaling with Context Length', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Context Length (tokens)', fontsize=12)
    axes[1,1].set_ylabel('Memory Usage (GB)', fontsize=12)
    axes[1,1].legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated efficiency_analysis.png")

def plot_statistical_insights(df, plots_dir):
    """Plot 3: Statistical insights and correlations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 3a: Correlation heatmap
    numeric_cols = ['perplexity', 'memory_usage_gb', 'throughput_tokens_per_sec', 
                   'context_length', 'evaluation_time_seconds']
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', ax=axes[0,0])
    axes[0,0].set_title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
    
    # Plot 3b: Performance improvement with context length
    improvement_data = []
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            model_data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(model_data) >= 2:
                min_ppl = model_data['perplexity'].min()
                max_ppl = model_data['perplexity'].max()
                improvement = ((max_ppl - min_ppl) / max_ppl) * 100
                improvement_data.append({
                    'model': model,
                    'dataset': dataset,
                    'improvement_percent': improvement
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    sns.barplot(data=improvement_df, x='model', y='improvement_percent', hue='dataset', ax=axes[0,1])
    axes[0,1].set_title('Performance Improvement\n(512 to 2048 context)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Model', fontsize=12)
    axes[0,1].set_ylabel('Improvement (%)', fontsize=12)
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend(title='Dataset')
    
    # Plot 3c: Confidence intervals
    models_order = df.groupby('model')['perplexity'].mean().sort_values().index
    ci_data = df[df['context_length'] == 2048].copy()  # Use best context length
    
    for i, model in enumerate(models_order):
        model_data = ci_data[ci_data['model'] == model]
        for j, dataset in enumerate(['wikitext2', 'c4']):
            dataset_data = model_data[model_data['dataset'] == dataset]
            if not dataset_data.empty:
                y_pos = i + j * 0.2 - 0.1
                ppl = dataset_data['perplexity'].iloc[0]
                ci_lower = dataset_data['perplexity_ci_lower'].iloc[0]
                ci_upper = dataset_data['perplexity_ci_upper'].iloc[0]
                
                axes[1,0].errorbar(ppl, y_pos, xerr=[[ppl-ci_lower], [ci_upper-ppl]], 
                                 fmt='o', capsize=5, label=dataset if i == 0 else "")
    
    axes[1,0].set_yticks(range(len(models_order)))
    axes[1,0].set_yticklabels(models_order)
    axes[1,0].set_xlabel('Perplexity', fontsize=12)
    axes[1,0].set_title('Model Performance with 95% Confidence Intervals\n(Context=2048)', 
                       fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 3d: Architecture comparison
    arch_performance = df.groupby('architecture')['perplexity'].agg(['mean', 'std']).reset_index()
    arch_performance = arch_performance.sort_values('mean')
    
    sns.barplot(data=arch_performance, x='architecture', y='mean', ax=axes[1,1])
    axes[1,1].errorbar(range(len(arch_performance)), arch_performance['mean'], 
                      yerr=arch_performance['std'], fmt='none', capsize=5, color='black')
    axes[1,1].set_title('Performance by Architecture\n(with std deviation)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Architecture', fontsize=12)
    axes[1,1].set_ylabel('Average Perplexity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'statistical_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated statistical_insights.png")

def plot_ranking_summary(df, summary, plots_dir):
    """Plot 4: Model ranking and summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 4a: Overall ranking radar chart
    rankings = summary['model_rankings']
    models = [r['model'].split('/')[-1] for r in rankings['by_perplexity_wikitext2']]
    
    # Normalize rankings for radar chart (lower is better, so invert)
    wiki_ranks = [5 - r['rank'] for r in rankings['by_perplexity_wikitext2']]
    c4_ranks = [5 - r['rank'] for r in rankings['by_perplexity_c4']]
    eff_ranks = [5 - r['rank'] for r in rankings['by_efficiency']]
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(models):
        values = [wiki_ranks[i], c4_ranks[i], eff_ranks[i]]
        values += values[:1]  # Complete the circle
        axes[0,0].plot(angles, values, 'o-', linewidth=2, label=model.replace('/', '_'))
        axes[0,0].fill(angles, values, alpha=0.1)
    
    axes[0,0].set_xticks(angles[:-1])
    axes[0,0].set_xticklabels(['WikiText-2', 'C4', 'Efficiency'])
    axes[0,0].set_ylim(0, 4)
    axes[0,0].set_title('Model Rankings Across Dimensions\n(Higher is Better)', 
                       fontsize=14, fontweight='bold')
    axes[0,0].legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    axes[0,0].grid(True)
    
    # Plot 4b: Performance vs Efficiency scatter
    model_summary = df.groupby('model').agg({
        'perplexity': 'mean',
        'throughput_tokens_per_sec': 'mean',
        'memory_usage_gb': 'mean'
    }).reset_index()
    
    model_summary['efficiency'] = (model_summary['throughput_tokens_per_sec'] / 
                                  (model_summary['perplexity'] * model_summary['memory_usage_gb']))
    
    sns.scatterplot(data=model_summary, x='perplexity', y='efficiency', 
                    size='memory_usage_gb', sizes=(100, 400), ax=axes[0,1])
    
    # Add model labels
    for i, row in model_summary.iterrows():
        axes[0,1].annotate(row['model'].split('/')[-1], 
                          (row['perplexity'], row['efficiency']),
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[0,1].set_title('Performance vs Efficiency Trade-off', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Average Perplexity (lower is better)', fontsize=12)
    axes[0,1].set_ylabel('Efficiency Score (higher is better)', fontsize=12)
    
    # Plot 4c: Context length optimization
    context_benefits = df.groupby(['model', 'context_length'])['perplexity'].mean().reset_index()
    context_pivot = context_benefits.pivot(index='model', columns='context_length', values='perplexity')
    
    # Calculate improvement from 512 to 2048
    context_pivot['improvement'] = ((context_pivot[512] - context_pivot[2048]) / context_pivot[512]) * 100
    context_pivot = context_pivot.sort_values('improvement', ascending=False)
    
    sns.barplot(x=context_pivot.index, y=context_pivot['improvement'], ax=axes[1,0])
    axes[1,0].set_title('Context Length Optimization Benefit\n(512 to 2048 tokens)', 
                       fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Model', fontsize=12)
    axes[1,0].set_ylabel('Perplexity Improvement (%)', fontsize=12)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4d: Final recommendations
    # Create a summary table as a plot
    axes[1,1].axis('off')
    
    # Best models for different use cases
    recommendations = [
        ["Use Case", "Recommended Model", "Reason"],
        ["Best Overall Performance", "microsoft/phi-2", "Lowest perplexity on both datasets"],
        ["Memory Constrained", "microsoft/bitnet", "Lowest memory usage"],
        ["High Throughput", "microsoft/phi-2", "Highest tokens/sec"],
        ["Balanced Performance", "google/gemma-2b", "Good balance of all metrics"],
        ["Research/Experimentation", "microsoft/phi-2", "Best performance + efficiency"]
    ]
    
    table = axes[1,1].table(cellText=recommendations[1:], colLabels=recommendations[0],
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(recommendations[0])):
        table[(0, i)].set_facecolor('#40B0A6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1,1].set_title('Model Recommendations by Use Case', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'ranking_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated ranking_summary.png")

def create_summary_report(df, plots_dir):
    """Create a text summary report."""
    report_path = plots_dir / 'evaluation_summary.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("LANGUAGE MODEL EVALUATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("DATASET OVERVIEW:\n")
        f.write(f"• Total experiments: {len(df)}\n")
        f.write(f"• Models evaluated: {df['model'].nunique()}\n")
        f.write(f"• Datasets: {', '.join(df['dataset'].unique())}\n")
        f.write(f"• Context lengths: {', '.join(map(str, sorted(df['context_length'].unique())))}\n\n")
        
        # Best performers
        f.write("TOP PERFORMERS:\n")
        best_overall = df.loc[df['perplexity'].idxmin()]
        f.write(f"• Best Overall: {best_overall['model']} (PPL: {best_overall['perplexity']:.3f})\n")
        
        fastest = df.loc[df['throughput_tokens_per_sec'].idxmax()]
        f.write(f"• Fastest: {fastest['model']} ({fastest['throughput_tokens_per_sec']:.1f} tokens/sec)\n")
        
        most_efficient = df.loc[df['memory_usage_gb'].idxmin()]
        f.write(f"• Most Memory Efficient: {most_efficient['model']} ({most_efficient['memory_usage_gb']:.1f} GB)\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS:\n")
        
        # Context length effects
        context_improvement = df.groupby('model').apply(
            lambda x: ((x[x['context_length'] == 512]['perplexity'].mean() - 
                       x[x['context_length'] == 2048]['perplexity'].mean()) / 
                      x[x['context_length'] == 512]['perplexity'].mean() * 100)
        ).mean()
        
        f.write(f"• Average improvement from 512->2048 context: {context_improvement:.1f}%\n")
        
        # Dataset difficulty
        wiki_avg = df[df['dataset'] == 'wikitext2']['perplexity'].mean()
        c4_avg = df[df['dataset'] == 'c4']['perplexity'].mean()
        difficulty_ratio = c4_avg / wiki_avg
        f.write(f"• C4 is {difficulty_ratio:.1f}x harder than WikiText-2\n")
        
        # Architecture performance
        arch_ranking = df.groupby('architecture')['perplexity'].mean().sort_values()
        f.write(f"• Best architecture: {arch_ranking.index[0]} (PPL: {arch_ranking.iloc[0]:.3f})\n")
        
        f.write("\nRECOMMENDations:\n")
        f.write("• For best performance: Use microsoft/phi-2 with 2048 context\n")
        f.write("• For memory constraints: Use microsoft/bitnet with appropriate context\n")
        f.write("• For balanced needs: Consider google/gemma-2b\n")
        f.write("• Always use maximum feasible context length for best results\n")
    
    print("✓ Generated evaluation_summary.txt")

def main():
    """Main function to generate all plots."""
    print("🎨 Starting plot generation...")
    
    # Setup
    plots_dir = setup_plot_directory()
    df, comprehensive, summary = load_data()
    
    print(f"📊 Loaded {len(df)} experimental results")
    
    # Generate all plots
    plot_perplexity_comparison(df, plots_dir)
    plot_efficiency_analysis(df, plots_dir)
    plot_statistical_insights(df, plots_dir)
    plot_ranking_summary(df, summary, plots_dir)
    create_summary_report(df, plots_dir)
    
    print("\n🎉 All plots generated successfully!")
    print(f"📁 Plots saved in: {plots_dir.absolute()}")
    print("\nGenerated files:")
    for plot_file in plots_dir.glob("*.png"):
        print(f"  • {plot_file.name}")
    for txt_file in plots_dir.glob("*.txt"):
        print(f"  • {txt_file.name}")

if __name__ == "__main__":
    main() 