
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# 函数: 绘制学习曲线
def learning_curve_plot(save_path, csv_path, show=False, image_name="Learning Curve"):
    # 读取csv数据
    loss_data = pd.read_csv(csv_path)
    train_loss = loss_data['Train Loss']
    # 作图
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    ax.plot(train_loss, linewidth=1.5, label="Training", color='red')
    # 横轴、纵轴标题
    ax.set_xlabel("Iteration", fontsize=24)
    ax.set_ylabel("Loss Value", fontsize=24)
    # 修改坐标轴刻度的字体大小
    ax.tick_params(axis='both', which='major', labelsize=19)
    # 网格
    ax.grid(True, linestyle='--')
    # 标题
    plt.title('Learning Curve')
    # 自动调整子图的布局
    fig.tight_layout()
    # 保存与展示
    plt.savefig(os.path.join(save_path, f"{image_name}.png"), dpi=600)
    plt.savefig(os.path.join(save_path, f"{image_name}.svg"), format="svg", dpi=300)
    if show == True:
        plt.show()
    plt.close(fig)

# 绘制学习曲线
result_dir = "./result"
learning_curve_plot(save_path=result_dir,
                    csv_path=os.path.join(result_dir, "loss_data.csv"),
                    show=False, image_name="Learning Curve")
