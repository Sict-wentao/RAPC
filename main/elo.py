import random

# 定义初始评分
INITIAL_RATING = 1500
K = 32

# 初始化选项及其评分
options = {f"Option {i+1}": INITIAL_RATING for i in range(1000)}

# 比较函数（示例）
def compare_options(option_a, option_b):
    # 可以使用模型生成的描述或其他比较方法
    # 这里使用随机比较结果作为示例
    return random.choice([option_a, option_b])

# 更新评分函数
def update_ratings(winner, loser):
    R_winner = options[winner]
    R_loser = options[loser]

    E_winner = 1 / (1 + 10 ** ((R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** ((R_winner - R_loser) / 400))

    options[winner] = R_winner + K * (1 - E_winner)
    options[loser] = R_loser + K * (0 - E_loser)

# 两两比较所有选项
for option_a in options:
    for option_b in options:
        if option_a != option_b:
            winner = compare_options(option_a, option_b)
            loser = option_b if winner == option_a else option_a
            update_ratings(winner, loser)

# 根据评分排序并选择前100个选项
top_100_options = sorted(options.items(), key=lambda x: x[1], reverse=True)[:100]

print("Top 100 options:")
for option, rating in top_100_options:
    print(f"{option}: {rating:.2f}")

