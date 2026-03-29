"""
过滤器独立测评模块

评估过滤器的预测准确性，不依赖因子测评框架。

核心指标：
  - Recall: 真正发生事件的，过滤器抓到了多少
  - Precision: 被剔除的，真正发生了事件的比例
  - F1: Precision 和 Recall 的调和平均
  - 日均剔除数: 每天平均剔除多少只
  - 误杀率: 被剔除但未发生事件的占比
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


class FilterEvaluator:
    """
    过滤器独立测评

    将过滤器的 exclude_mask 与真实标签对比，计算分类指标。

    Args:
        exclude_mask: (N, T) bool, 过滤器输出，True=剔除
        labels: (N, T) bool, 真实标签，True=实际发生了 ST/退市
        dates: (T,) datetime64, 日期数组（用于按年分组）
        filter_name: str, 过滤器名称（用于报告显示）
    """

    def __init__(
        self,
        exclude_mask: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,
        filter_name: str = "",
    ):
        if exclude_mask.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: exclude_mask {exclude_mask.shape} vs labels {labels.shape}"
            )
        self.exclude_mask = exclude_mask.astype(bool)
        self.labels = labels.astype(bool)
        self.dates = np.asarray(dates)
        self.filter_name = filter_name

    def evaluate(self) -> Dict[str, Any]:
        """
        计算整体分类指标。

        Returns:
            dict: 包含 recall, precision, f1, daily_avg_excluded, false_positive_rate 等
        """
        return self._compute_metrics(self.exclude_mask, self.labels)

    def evaluate_by_year(self) -> pd.DataFrame:
        """
        按年度分组计算指标。

        Returns:
            DataFrame: index=年份, columns=各指标
        """
        dates_idx = pd.DatetimeIndex(self.dates)
        years = dates_idx.year
        unique_years = sorted(set(years))

        rows = []
        for year in unique_years:
            year_mask = (years == year)
            year_cols = np.where(year_mask)[0]
            if len(year_cols) == 0:
                continue

            exc_year = self.exclude_mask[:, year_cols]
            lab_year = self.labels[:, year_cols]
            metrics = self._compute_metrics(exc_year, lab_year)
            metrics["year"] = year
            rows.append(metrics)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("year")

    @staticmethod
    def _compute_metrics(
        exclude_mask: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算分类指标（展平为一维后计算）。

        定义：
          - TP: 被剔除且确实发生了事件
          - FP: 被剔除但未发生事件（误杀）
          - FN: 未被剔除但发生了事件（漏掉）
          - TN: 未被剔除且未发生事件
        """
        pred = exclude_mask.ravel()
        true = labels.ravel()

        tp = int(np.sum(pred & true))
        fp = int(np.sum(pred & ~true))
        fn = int(np.sum(~pred & true))
        tn = int(np.sum(~pred & ~true))

        total_positive = tp + fn  # 实际发生事件的总数
        total_predicted = tp + fp  # 被剔除的总数

        recall = tp / total_positive if total_positive > 0 else 0.0
        precision = tp / total_predicted if total_predicted > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        n_dates = exclude_mask.shape[1]
        daily_avg_excluded = exclude_mask.sum(axis=0).mean() if n_dates > 0 else 0.0

        false_positive_rate = fp / total_predicted if total_predicted > 0 else 0.0

        return {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_events": total_positive,
            "total_excluded": total_predicted,
            "daily_avg_excluded": round(float(daily_avg_excluded), 1),
            "false_positive_rate": round(false_positive_rate, 4),
        }

    def print_report(self):
        """打印测评报告到控制台"""
        metrics = self.evaluate()
        yearly = self.evaluate_by_year()

        lines = []
        lines.append("=" * 60)
        title = f"Filter Evaluation: {self.filter_name}" if self.filter_name else "Filter Evaluation"
        lines.append(title)
        lines.append("=" * 60)
        lines.append(f"  Data shape:        {self.exclude_mask.shape[0]} symbols x {self.exclude_mask.shape[1]} dates")
        lines.append(f"  Total events:      {metrics['total_events']}")
        lines.append(f"  Total excluded:    {metrics['total_excluded']}")
        lines.append("")
        lines.append(f"  Recall:            {metrics['recall']:.2%}")
        lines.append(f"  Precision:         {metrics['precision']:.2%}")
        lines.append(f"  F1:                {metrics['f1']:.2%}")
        lines.append(f"  False positive rate: {metrics['false_positive_rate']:.2%}")
        lines.append(f"  Daily avg excluded:  {metrics['daily_avg_excluded']:.1f}")
        lines.append("")
        lines.append(f"  Confusion matrix:")
        lines.append(f"    TP={metrics['tp']:>7d}  FP={metrics['fp']:>7d}")
        lines.append(f"    FN={metrics['fn']:>7d}  TN={metrics['tn']:>7d}")

        if not yearly.empty:
            lines.append("")
            lines.append("-" * 60)
            lines.append("By Year:")
            lines.append("-" * 60)
            lines.append(f"  {'Year':>6s}  {'Recall':>8s}  {'Precision':>10s}  {'F1':>6s}  {'Events':>7s}  {'Excluded':>8s}  {'DailyExcl':>9s}")
            for year, row in yearly.iterrows():
                lines.append(
                    f"  {year:>6d}  {row['recall']:>8.2%}  {row['precision']:>10.2%}  "
                    f"{row['f1']:>6.2%}  {row['total_events']:>7d}  "
                    f"{row['total_excluded']:>8d}  {row['daily_avg_excluded']:>9.1f}"
                )

        lines.append("=" * 60)
        print("\n".join(lines))
