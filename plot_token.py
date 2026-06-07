import matplotlib.pyplot as plt

token_numbers = [2, 4, 8, 12, 16, 24, 32]
cider_scores = [0.417478632, 0.430286321, 0.499672098, 0.543517616, 0.596430341, 0.610138101, 0.615714933]
rouge_scores = [0.555704807, 0.566314541, 0.596516077, 0.614208582, 0.640489035, 0.635209985, 0.63672654]
bleu4_scores = [0.336333361, 0.343837502, 0.376427214, 0.400473677, 0.43634352, 0.440275209, 0.439592735]
meteor_scores = [0.268122657, 0.272645712, 0.289595705, 0.29554618, 0.30768557, 0.313547394, 0.309786296]

# Translated comment.
cider_scores = [x * 100 for x in cider_scores]
rouge_scores = [x * 100 for x in rouge_scores]
bleu4_scores = [x * 100 for x in bleu4_scores]
meteor_scores = [x * 100 for x in meteor_scores]

# Translated comment.
plt.figure(figsize=(10, 7), dpi=300)

# Translated comment.
plt.plot(token_numbers, cider_scores, marker='o', linestyle='-', label='CIDEr', color='tab:blue')
plt.plot(token_numbers, rouge_scores, marker='s', linestyle='--', label='ROUGE-L', color='tab:orange')
plt.plot(token_numbers, bleu4_scores, marker='^', linestyle='-.', label='BLEU-4', color='tab:green')
plt.plot(token_numbers, meteor_scores, marker='d', linestyle=':', label='METEOR', color='tab:red')

# Translated comment.
for x, y in zip(token_numbers, cider_scores):
    plt.text(x, y + 0.6, f'{y:.2f}', ha='center', va='bottom', fontsize=15, color='tab:blue')

for x, y in zip(token_numbers, rouge_scores):
    plt.text(x, y + 0.6, f'{y:.2f}', ha='center', va='bottom', fontsize=15, color='tab:orange')

for x, y in zip(token_numbers, bleu4_scores):
    plt.text(x, y + 0.6, f'{y:.2f}', ha='center', va='bottom', fontsize=15, color='tab:green')

for x, y in zip(token_numbers, meteor_scores):
    plt.text(x, y + 0.6, f'{y:.2f}', ha='center', va='bottom', fontsize=15, color='tab:red')

# Translated comment.
plt.xlabel('Number of Tokens', fontsize=16)
plt.ylabel('Metric Score (%)', fontsize=16)

# Translated comment.
plt.grid(True, linestyle='--', alpha=0.5)

# Translated comment.

plt.legend(loc='upper left', bbox_to_anchor=(0.72, 0.8),fontsize=16)
# Translated comment.
plt.xticks(token_numbers, fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(25, 68)  # Translated comment.
# Translated comment.
plt.tight_layout()

# Translated comment.
# plt.savefig('token_number_vs_metrics.png', dpi=300)

# Translated comment.
plt.show()