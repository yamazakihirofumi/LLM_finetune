import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

data = pd.DataFrame({
    'Model': ['Qwen2.5-3B', 'Qwen2.5-3B', 'Qwen2.5-0.5B', 'Qwen2.5-0.5B', 'BERT', 'BERT'],
    'Stage': ['Before Fine-tuning', 'After Fine-tuning', 'Before Fine-tuning', 'After Fine-tuning', 'Before Fine-tuning', 'After Fine-tuning'],
    'Accuracy': [0.3753, 0.7210, 0.1490, 0.6532, 0.7304, 0.8232]
})

ax = sns.barplot(x='Model', y='Accuracy', hue='Stage', data=data, palette=['#FF1C00', '#FFAA00'])

for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{height:.1%}', ha='center', fontsize=10)

plt.title('Model Accuracy Before and After Fine-tuning', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.yticks([i/10 for i in range(0, 11)], [f'{i*10}%' for i in range(0, 11)])
plt.legend(title='Stage', loc='upper left')
plt.tight_layout()

plt.savefig('plot_result.jpg', dpi=300)
plt.close()