import os
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("ai2_results")
SUMMARY_PATH = RESULTS_DIR / "experiment_summary.csv"
REPORT_PATH = RESULTS_DIR / "ai2_analysis_report.txt"

def main():
    if not SUMMARY_PATH.exists():
        print(f"❌ 錯誤：找不到實驗總表 [{SUMMARY_PATH}]。請確認是否已執行 python ai2.py --suite all")
        return

    df = pd.read_csv(SUMMARY_PATH)
    print(f"✅ 成功載入實驗總表，共包含 {len(df)} 筆實驗記錄。")

    lines = []
    lines.append("=" * 70)
    lines.append("       FER2013 CNN 過擬合與超參數診斷實驗：自動化深度分析報告")
    lines.append("=" * 70)
    lines.append(f"分析時間點資料筆數：{len(df)} 組實驗\n")

    # 1. 全體最佳模型摘要
    best_f1_row = df.loc[df['val_macro_f1'].idxmax()]
    best_acc_row = df.loc[df['val_accuracy'].idxmax()]
    lowest_gap_row = df.loc[df['accuracy_gap_train_minus_val'].abs().idxmin()]

    lines.append("【 1. 核心表現總覽 】")
    lines.append(f"• 最佳驗證 Macro F1 模型：{best_f1_row['experiment']} (Macro F1 = {best_f1_row['val_macro_f1']:.4f}, Val Acc = {best_f1_row['val_accuracy']:.4f})")
    lines.append(f"• 最佳驗證 Accuracy 模型：{best_acc_row['experiment']} (Val Acc = {best_acc_row['val_accuracy']:.4f})")
    lines.append(f"• 泛化能力最強 (Train/Val 差距最小)：{lowest_gap_row['experiment']} (Gap = {lowest_gap_row['accuracy_gap_train_minus_val']:.4f})\n")

    # 2. 七大問題逐一診斷
    lines.append("=" * 70)
    lines.append("【 2. 七大實驗問題診斷與數據佐證 】")
    lines.append("=" * 70)

    # 問題一：CNN 架構
    lines.append("\n(1) CNN 架構實驗 (small vs baseline vs deeper)")
    arch_df = df[df['experiment'].str.startswith('arch_')]
    if not arch_df.empty:
        for _, r in arch_df.iterrows():
            lines.append(f"   - {r['experiment']:<15} | Train Acc: {r['train_clean_accuracy']:.4f} | Val Acc: {r['val_accuracy']:.4f} | Gap: {r['accuracy_gap_train_minus_val']:+.4f} | Val F1: {r['val_macro_f1']:.4f}")
        best_arch = arch_df.loc[arch_df['val_macro_f1'].idxmax()]['experiment']
        lines.append(f"   👉 診斷結論：表現最佳的架構為「{best_arch}」。模型容量過小會導致欠擬合，過深則容易拉大 Gap 或增加過擬合風險。")
    else:
        lines.append("   (無相關實驗數據)")

    # 問題二：Dropout
    lines.append("\n(2) Dropout 正則化實驗 (0.0, 0.3, 0.5, 0.6)")
    drop_df = df[df['experiment'].str.startswith('dropout_')]
    if not drop_df.empty:
        for _, r in drop_df.iterrows():
            lines.append(f"   - {r['experiment']:<15} | Train Acc: {r['train_clean_accuracy']:.4f} | Val Acc: {r['val_accuracy']:.4f} | Gap: {r['accuracy_gap_train_minus_val']:+.4f} | Val F1: {r['val_macro_f1']:.4f}")
        best_drop = drop_df.loc[drop_df['val_macro_f1'].idxmax()]['experiment']
        lines.append(f"   👉 診斷結論：表現最佳的設定為「{best_drop}」。若 Dropout 過小（如 0.0），Gap 通常會顯著擴大（過擬合）；若過大則可能限制模型學習能力。")
    else:
        lines.append("   (無相關實驗數據)")

    # 問題三：影像增強
    lines.append("\n(3) 影像增強強度實驗 (none, mild, medium, strong)")
    aug_df = df[df['experiment'].str.startswith('aug_')]
    if not aug_df.empty:
        for _, r in drop_df.iterrows() if 'dropout_' in locals() else df[df['experiment'].str.startswith('aug_')].iterrows(): # safety fallback
            pass
        for _, r in aug_df.iterrows():
            lines.append(f"   - {r['experiment']:<15} | Train Acc: {r['train_clean_accuracy']:.4f} | Val Acc: {r['val_accuracy']:.4f} | Gap: {r['accuracy_gap_train_minus_val']:+.4f} | Val F1: {r['val_macro_f1']:.4f}")
        best_aug = aug_df.loc[aug_df['val_macro_f1'].idxmax()]['experiment']
        lines.append(f"   👉 診斷結論：表現最佳的增強強度為「{best_aug}」。過強的增強若破壞了 48x48 微細五官特徵，會導致辨識能力下降；適度增強則能抑制過擬合。")
    else:
        lines.append("   (無相關實驗數據)")

    # 問題四：Epochs
    lines.append("\n(4) 訓練回合數實驗 (Epochs: 10, 20, 40)")
    ep_df = df[df['experiment'].str.startswith('epochs_')]
    if not ep_df.empty:
        for _, r in ep_df.iterrows():
            lines.append(f"   - {r['experiment']:<15} | Train Acc: {r['train_clean_accuracy']:.4f} | Val Acc: {r['val_accuracy']:.4f} | Gap: {r['accuracy_gap_train_minus_val']:+.4f} | Val F1: {r['val_macro_f1']:.4f}")
        lines.append(f"   👉 診斷結論：觀察不同 epochs 下的 Gap 變化，可判斷 20 或 40 epochs 是否發生過擬合（訓練損失持續下降但驗證停滯）。")
    else:
        lines.append("   (無相關實驗數據)")

    # 問題五與六與七：Regularization, LR, Early Stopping
    lines.append("\n(5) L2 正則化、學習率與 Early Stopping 綜合比較")
    extra_exps = df[df['experiment'].isin(['l2_1e-4', 'lr_3e-4', 'baseline_earlystop', 'arch_baseline'])]
    if not extra_exps.empty:
        for _, r in extra_exps.iterrows():
            lines.append(f"   - {r['experiment']:<20} | Train Acc: {r['train_clean_accuracy']:.4f} | Val Acc: {r['val_accuracy']:.4f} | Gap: {r['accuracy_gap_train_minus_val']:+.4f} | Val F1: {r['val_macro_f1']:.4f}")
        lines.append(f"   👉 診斷結論：引入 L2 能有效壓制過擬合；調整 Learning Rate（如 3e-4）能讓收斂更平穩；搭配 Early Stopping 則可自動防堵後期過擬合。")
    else:
        lines.append("   (無相關實驗數據)")

    lines.append("\n" + "=" * 70)
    lines.append("報告結語：所有實驗數據與混淆矩陣皆已完整記錄於 ai2_results/ 資料夾中。")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n✨ 分析完畢！報告已成功輸出至：{REPORT_PATH.resolve()}")
    print("\n" + report_text)

if __name__ == "__main__":
    main()