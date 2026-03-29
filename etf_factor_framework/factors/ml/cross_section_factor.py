"""
ML截面因子实现

基于滚动LightGBM训练的截面排名因子，参考 etf_cross_ml Mode B 设计。

核心设计原则：
  - 以调仓频率（rebalance_freq 天）为步长滚动训练
  - 每次只使用历史数据（T_train 天训练 + T_val 天验证），天然无未来泄露
  - 标签为截面 CS-Rank（未来 n_forward 日收益率的百分位排名）
  - 数据泄露检测通过基类 detect_at_date() 实现：比较 calculate(完整数据) 与
    calculate(截断到目标日) 在目标日的截面预测，两次调用的 train/val 窗口完全相同

数据流：
  OHLCVData
    → _build_features_and_labels()  → long-format DataFrame
    → _crosssection_normalize()      → 截面标准化
    → _train_lgbm()                  → LightGBM Booster
    → predict()                      → FactorData (N × T)
"""

import multiprocessing as mp
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── fork 支持检测 ──────────────────────────────────────────────────────────────
_FORK_SUPPORTED: bool = sys.platform != 'win32'

# ── 并行 worker 共享的模块级全局变量（fork 后 COW 共享，不需要序列化大对象）─────
_g_df: Any = None
_g_feat_cols: Any = None
_g_all_dates: Any = None
_g_sym_to_i: Any = None
_g_date_to_j: Any = None
_g_n_forward: int = 21
_g_lgbm_params: Any = None
_g_lgbm_n_jobs: int = 1


def _cs_normalize(df: "pd.DataFrame", feat_cols: "List[str]") -> "pd.DataFrame":
    """截面 winsorize ±3σ + z-score（模块级，供 worker 调用）。"""
    df = df.copy()

    def _norm(s: "pd.Series") -> "pd.Series":
        mu, sigma = s.mean(), s.std()
        if sigma > 0:
            s = s.clip(mu - 3 * sigma, mu + 3 * sigma)
            return (s - s.mean()) / s.std()
        return s

    for col in feat_cols:
        df[col] = df.groupby('date')[col].transform(_norm)
    return df


def _train_window_worker(args: tuple):
    """
    单窗口训练 + 预测的 worker 函数（模块级，fork 子进程可直接调用）。

    使用模块级全局变量（由父进程 fork 前写入，COW 共享）：
      _g_df, _g_feat_cols, _g_all_dates, _g_sym_to_i, _g_date_to_j,
      _g_n_forward, _g_lgbm_params, _g_lgbm_n_jobs

    Args:
        args: (train_start_idx, val_start_idx, pred_idx, next_pred_idx)

    Returns:
        (sym_i_arr, date_j_arr, preds_arr) 或 None（数据不足 / 训练失败）
    """
    train_start_idx, val_start_idx, pred_idx, next_pred_idx = args

    df = _g_df
    feat_cols = _g_feat_cols
    all_dates = _g_all_dates
    sym_to_i = _g_sym_to_i
    date_to_j = _g_date_to_j
    n_forward = _g_n_forward
    lgbm_params = _g_lgbm_params
    n_jobs = _g_lgbm_n_jobs

    train_dates = all_dates[train_start_idx: val_start_idx]
    val_dates = all_dates[val_start_idx: pred_idx - n_forward]
    pred_date_range = all_dates[pred_idx: next_pred_idx]

    if len(train_dates) < 20 or len(val_dates) < 5:
        return None

    df_train = df[df['date'].isin(train_dates)].dropna(subset=['label'] + feat_cols)
    df_val = df[df['date'].isin(val_dates)].dropna(subset=['label'] + feat_cols)

    if len(df_train) < 20 or len(df_val) < 5:
        return None

    df_train = _cs_normalize(df_train, feat_cols)
    df_val = _cs_normalize(df_val, feat_cols)

    X_train = df_train[feat_cols].values.astype(np.float32)
    y_train = df_train['label'].values.astype(np.float32)
    X_val = df_val[feat_cols].values.astype(np.float32)
    y_val = df_val['label'].values.astype(np.float32)

    try:
        import lightgbm as lgb
    except ImportError:
        return None

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'seed': 42,
        'n_jobs': n_jobs,
        'num_leaves': lgbm_params['num_leaves'],
        'learning_rate': lgbm_params['learning_rate'],
        'feature_fraction': lgbm_params['feature_fraction'],
        'min_child_samples': lgbm_params['min_child_samples'],
        'lambda_l1': lgbm_params['lambda_l1'],
        'lambda_l2': lgbm_params['lambda_l2'],
    }
    n_estimators = lgbm_params.get('n_estimators', 300)
    early_stopping_rounds = lgbm_params.get('early_stopping_rounds', 30)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    try:
        model = lgb.train(
            lgb_params, train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )
    except Exception:
        return None

    df_pred = df[df['date'].isin(pred_date_range)].dropna(subset=feat_cols).copy()
    if df_pred.empty:
        return None

    df_pred = _cs_normalize(df_pred, feat_cols)
    df_pred['_pred'] = model.predict(df_pred[feat_cols].values.astype(np.float32))
    df_pred['_sym_i'] = df_pred['symbol'].map(sym_to_i)
    df_pred['_date_j'] = df_pred['date'].map(date_to_j)
    valid = df_pred.dropna(subset=['_sym_i', '_date_j'])
    if valid.empty:
        return None

    return (
        valid['_sym_i'].astype(int).values,
        valid['_date_j'].astype(int).values,
        valid['_pred'].values,
    )

try:
    from ..ohlcv_calculator import OHLCVFactorCalculator
    from ...core.ohlcv_data import OHLCVData
    from ...core.factor_data import FactorData
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from factors.ohlcv_calculator import OHLCVFactorCalculator
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData


class MLCrossSectionalFactor(OHLCVFactorCalculator):
    """
    基于滚动 LightGBM 训练的截面因子。

    每隔 rebalance_freq 天：
      1. 用最近 T_train 天历史数据训练 LightGBM（回归 CS-Rank）
      2. 用最近 T_val 天历史数据做验证集 early stopping
      3. 对当前截面所有标的预测因子分数

    特征（对每个 window W 计算）：
      - ret_W:      W 日动量（简单收益率）
      - vol_W:      日收益率的 W 日滚动标准差
      - volratio_W: 成交量 / W 日均量

    标签：
      - 未来 n_forward 日收益率的截面百分位排名（0=最差，1=最优）

    数据泄露检测：
      - 继承基类 detect_at_date()：比较 calculate(full) 与 calculate(truncated_to_t) 在 t 处的预测
      - 对同一 pred_idx=t，两次调用的 train/val 窗口完全相同，任何差异即为泄露

    Args:
        T_train_days:    训练窗口长度（交易日），默认 252（约 1 年）
        T_val_days:      验证窗口长度（交易日），默认 63（约 3 个月）
        n_forward:       标签预测窗口（交易日），默认 21（约 1 个月）
                         必须满足 n_forward < T_val_days
        rebalance_freq:  调仓间隔（交易日），默认 21
        feature_windows: 特征计算窗口列表，默认 [5, 10, 20, 60]
        lgbm_params:     LightGBM 参数覆盖字典
        verbose:         是否打印训练进度

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from etf_factor_framework.core.ohlcv_data import OHLCVData
        >>> from etf_factor_framework.factors.ml import MLCrossSectionalFactor
        >>>
        >>> # 构建测试数据（5 标的 × 500 天）
        >>> rng = np.random.default_rng(42)
        >>> N, T = 5, 500
        >>> symbols = np.array([f'ETF{i}' for i in range(N)])
        >>> dates = np.array(pd.date_range('2020-01-01', periods=T), dtype='datetime64[ns]')
        >>> close = (rng.standard_normal((N, T)).cumsum(axis=1) + 100).clip(1)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99, high=close * 1.02,
        ...     low=close * 0.98, close=close,
        ...     volume=np.abs(rng.standard_normal((N, T))) * 1e6 + 1e6,
        ...     symbols=symbols, dates=dates
        ... )
        >>> factor = MLCrossSectionalFactor(T_train_days=200, T_val_days=50, n_forward=10)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.shape)  # (5, 500)
    """

    DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 300,
        'early_stopping_rounds': 30,
    }

    def __init__(
        self,
        T_train_days: int = 252,
        T_val_days: int = 63,
        n_forward: int = 21,
        rebalance_freq: int = 21,
        feature_windows: Optional[List[int]] = None,
        lgbm_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        n_parallel_workers: int = 1,
        min_valid_ratio: float = 0.95,
    ):
        self._T_train = T_train_days
        self._T_val = T_val_days
        self._n_forward = n_forward
        self._rebalance_freq = rebalance_freq
        self._feature_windows = feature_windows or [5, 10, 20, 60]
        self._lgbm_params = {**self.DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self._verbose = verbose
        self._n_parallel_workers = n_parallel_workers
        self._min_valid_ratio = min_valid_ratio
        super().__init__()

    # ── 抽象属性实现 ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return (f"MLCrossSection"
                f"_train{self._T_train}"
                f"_val{self._T_val}"
                f"_n{self._n_forward}")

    @property
    def factor_type(self) -> str:
        return "MLCrossSectionalFactor"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            'T_train_days': self._T_train,
            'T_val_days': self._T_val,
            'n_forward': self._n_forward,
            'rebalance_freq': self._rebalance_freq,
            'feature_windows': self._feature_windows,
        }

    def get_params(self) -> Dict[str, Any]:
        return self.params

    # ── 参数验证 ──────────────────────────────────────────────────────────────

    def _validate_params(self):
        if self._T_train <= 0:
            raise ValueError(f"T_train_days must be positive, got {self._T_train}")
        if self._T_val <= 0:
            raise ValueError(f"T_val_days must be positive, got {self._T_val}")
        if self._n_forward <= 0:
            raise ValueError(f"n_forward must be positive, got {self._n_forward}")
        if self._n_forward >= self._T_val:
            warnings.warn(
                f"n_forward ({self._n_forward}) >= T_val_days ({self._T_val})，"
                f"验证集末端所有样本的 label 均为 NaN，实际有效验证样本极少。"
                f"建议 n_forward < T_val_days / 2。"
            )

    # ── 特征列名 ──────────────────────────────────────────────────────────────

    def _feat_cols(self) -> List[str]:
        """返回所有特征列名。"""
        cols = []
        for w in self._feature_windows:
            cols += [f'ret_{w}', f'vol_{w}', f'volratio_{w}']
        return cols

    # ── 特征 + 标签构建 ───────────────────────────────────────────────────────

    def _build_features_and_labels(
        self,
        ohlcv_data: OHLCVData,
        add_labels: bool = True,
    ) -> pd.DataFrame:
        """
        将 OHLCVData 转换为 long-format DataFrame，包含特征（以及可选的 CS-Rank 标签）。

        内部全程使用 numpy/pandas 向量化操作，不逐行循环。

        Returns:
            DataFrame，列为 [symbol, date, ret_W, vol_W, volratio_W, aux_close, (label)]
        """
        symbols = ohlcv_data.symbols          # (N,) ndarray
        dates_np = ohlcv_data.dates           # (T,) datetime64[ns]
        close = ohlcv_data.close              # (N, T) ndarray
        volume = ohlcv_data.volume            # (N, T) ndarray
        N, T = close.shape

        dates_pd = pd.DatetimeIndex(dates_np)

        # ── 日收益率 (N, T)，用于波动率特征 ──────────────────────────────────
        daily_ret = np.full((N, T), np.nan)
        daily_ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1

        # 转置为 (T, N) 供 pandas rolling 使用
        dr_T = pd.DataFrame(daily_ret.T, index=dates_pd)   # (T, N)
        vol_T = pd.DataFrame(volume.T, index=dates_pd)     # (T, N)

        # ── 计算各窗口特征 ────────────────────────────────────────────────────
        feat_arrays: Dict[str, np.ndarray] = {}

        for w in self._feature_windows:
            # ret_W: W 日动量
            mom = np.full((N, T), np.nan)
            if T > w:
                mom[:, w:] = close[:, w:] / close[:, :-w] - 1
            feat_arrays[f'ret_{w}'] = mom

            # vol_W: 日收益率的 W 日滚动标准差
            min_p = max(2, w // 2)
            vol_std = dr_T.rolling(w, min_periods=min_p).std().values.T  # (N, T)
            feat_arrays[f'vol_{w}'] = vol_std

            # volratio_W: 成交量 / W 日均量
            vol_ma = vol_T.rolling(w, min_periods=min_p).mean().values.T  # (N, T)
            with np.errstate(divide='ignore', invalid='ignore'):
                volratio = np.where(vol_ma > 0, volume / vol_ma, np.nan)
            feat_arrays[f'volratio_{w}'] = volratio

        # ── 标签：未来 n_forward 日 CS-Rank ──────────────────────────────────
        if add_labels:
            n = self._n_forward
            forward_ret = np.full((N, T), np.nan)
            if T > n:
                # position j 的标签依赖 close[j+n]，故只有 j < T-n 的位置有效
                forward_ret[:, :-n] = close[:, n:] / close[:, :-n] - 1

        # ── 展开为 long-format ────────────────────────────────────────────────
        # (N, T) → row-major flatten：[sym0_t0, sym0_t1, ..., sym0_tT-1, sym1_t0, ...]
        sym_idx = np.repeat(np.arange(N), T)   # (N*T,) 每个 symbol 重复 T 次
        date_idx = np.tile(np.arange(T), N)    # (N*T,) 日期索引循环 N 次

        data: Dict[str, Any] = {
            'symbol': symbols[sym_idx],
            'date': dates_pd[date_idx],
            'aux_close': close.flatten(),       # row-major flatten of (N, T)
        }
        for fname, farr in feat_arrays.items():
            data[fname] = farr.flatten()

        if add_labels:
            data['_future_ret'] = forward_ret.flatten()

        df = pd.DataFrame(data)

        if add_labels:
            # 截面百分位排名（每个 date 内部）
            df['label'] = df.groupby('date')['_future_ret'].transform(
                lambda x: x.rank(pct=True)
            )
            df.loc[df['_future_ret'].isna(), 'label'] = np.nan
            df.drop(columns=['_future_ret'], inplace=True)

        return df

    # ── 截面标准化 ────────────────────────────────────────────────────────────

    @staticmethod
    def _crosssection_normalize(
        df: pd.DataFrame,
        feat_cols: List[str],
    ) -> pd.DataFrame:
        """
        按 date 分组做截面标准化：winsorize ±3σ 后 z-score。

        每日内部独立计算，不跨训练/验证集，无泄露。
        """
        df = df.copy()

        def _norm(s: pd.Series) -> pd.Series:
            mu, sigma = s.mean(), s.std()
            if sigma > 0:
                s = s.clip(mu - 3 * sigma, mu + 3 * sigma)
                return (s - s.mean()) / s.std()
            return s

        for col in feat_cols:
            df[col] = df.groupby('date')[col].transform(_norm)

        return df

    # ── LightGBM 训练 ─────────────────────────────────────────────────────────

    def _train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """训练 LightGBM 回归模型（CS-Rank 目标），验证集 early stopping。"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm 未安装。请运行：pip install lightgbm"
            )

        n_jobs = max(1, (os.cpu_count() or 2) - 1)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': 42,
            'n_jobs': n_jobs,
            'num_leaves': self._lgbm_params['num_leaves'],
            'learning_rate': self._lgbm_params['learning_rate'],
            'feature_fraction': self._lgbm_params['feature_fraction'],
            'min_child_samples': self._lgbm_params['min_child_samples'],
            'lambda_l1': self._lgbm_params['lambda_l1'],
            'lambda_l2': self._lgbm_params['lambda_l2'],
        }

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.early_stopping(self._lgbm_params['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        return lgb.train(
            params,
            train_set,
            num_boost_round=self._lgbm_params['n_estimators'],
            valid_sets=[val_set],
            callbacks=callbacks,
        )

    # ── 单窗口数据准备 ────────────────────────────────────────────────────────

    def _prepare_window(
        self,
        df: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
        val_dates: pd.DatetimeIndex,
        feat_cols: List[str],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        为一个滚动窗口准备训练/验证数组。

        Returns:
            (X_train, y_train, X_val, y_val) 或 None（数据不足时）
        """
        df_train = df[df['date'].isin(train_dates)].dropna(subset=['label'] + feat_cols)
        df_val = df[df['date'].isin(val_dates)].dropna(subset=['label'] + feat_cols)

        if len(df_train) < 20 or len(df_val) < 5:
            return None

        df_train = self._crosssection_normalize(df_train, feat_cols)
        df_val = self._crosssection_normalize(df_val, feat_cols)

        return (
            df_train[feat_cols].values.astype(np.float32),
            df_train['label'].values.astype(np.float32),
            df_val[feat_cols].values.astype(np.float32),
            df_val['label'].values.astype(np.float32),
        )

    # ── 主接口：calculate ─────────────────────────────────────────────────────

    def calculate(self, fd) -> FactorData:
        """
        从 FundamentalData 中读取日期范围和数据库路径，自行加载 OHLCV，
        然后滚动训练 LightGBM 并预测截面因子值。

        接口与基本面因子对齐：calculator = factor_class(); factor_data = calculator.calculate(fd)

        滚动逻辑（以调仓日索引 pred_idx 为例）：
          训练集 = dates[pred_idx - T_train - T_val  :  pred_idx - T_val]
          验证集 = dates[pred_idx - T_val             :  pred_idx - n_forward]
          预测   = dates[pred_idx                     :  pred_idx + rebalance_freq]

        dates[0 : T_train + T_val] 的因子值为 NaN（无足够历史）。

        Args:
            fd: FundamentalData，用于获取 start_date / end_date / tushare_db 路径

        Returns:
            FactorData: shape (N, T)，NaN 表示无法预测的日期
        """
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
        from data.stock_data_loader import StockDataLoader

        start_date = fd.start_date.strftime('%Y-%m-%d')
        end_date   = fd.end_date.strftime('%Y-%m-%d')
        tushare_db = fd._tushare_db

        loader = StockDataLoader(tushare_db_path=tushare_db)
        ohlcv_data = loader.load_ohlcv(start_date, end_date, use_adjusted=True)
        loader.close()

        # 过滤有效数据比例不足的股票（新股/退市股）
        valid_ratio = (~np.isnan(ohlcv_data.close)).mean(axis=1)
        keep = valid_ratio >= self._min_valid_ratio
        ohlcv_data = OHLCVData(
            open=ohlcv_data.open[keep],   high=ohlcv_data.high[keep],
            low=ohlcv_data.low[keep],     close=ohlcv_data.close[keep],
            volume=ohlcv_data.volume[keep],
            symbols=ohlcv_data.symbols[keep], dates=ohlcv_data.dates,
        )

        symbols = ohlcv_data.symbols
        dates_np = ohlcv_data.dates
        N = ohlcv_data.n_assets
        T = ohlcv_data.n_periods

        feat_cols = self._feat_cols()
        factor_values = np.full((N, T), np.nan)

        if self._verbose:
            print(f"[{self.name}] 构建特征：{N} 标的 × {T} 日期")

        # 一次性构建完整 DataFrame
        df = self._build_features_and_labels(ohlcv_data, add_labels=True)
        all_dates = pd.DatetimeIndex(dates_np)

        # 快速查找映射
        sym_to_i: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        date_to_j: Dict[Any, int] = {d: j for j, d in enumerate(all_dates)}

        # 调仓日索引列表（最早从 T_train + T_val 开始）
        min_idx = self._T_train + self._T_val
        pred_indices = list(range(min_idx, T, self._rebalance_freq))

        if self._verbose:
            print(f"[{self.name}] 滚动训练：共 {len(pred_indices)} 个窗口")

        if self._n_parallel_workers > 1 and _FORK_SUPPORTED:
            # ── 并行路径：fork 前设置模块级全局，子进程 COW 共享大 DataFrame ──
            global _g_df, _g_feat_cols, _g_all_dates, _g_sym_to_i
            global _g_date_to_j, _g_n_forward, _g_lgbm_params, _g_lgbm_n_jobs

            n_workers = min(self._n_parallel_workers, len(pred_indices))
            _g_df = df
            _g_feat_cols = feat_cols
            _g_all_dates = all_dates
            _g_sym_to_i = sym_to_i
            _g_date_to_j = date_to_j
            _g_n_forward = self._n_forward
            _g_lgbm_params = self._lgbm_params
            # 每个 worker 的 lgbm 线程数：避免 CPU 过度订阅
            _g_lgbm_n_jobs = max(1, (os.cpu_count() or 2) // n_workers)

            window_args = []
            for pred_idx in pred_indices:
                val_start_idx = pred_idx - self._T_val
                train_start_idx = max(0, val_start_idx - self._T_train)
                next_pred_idx = min(pred_idx + self._rebalance_freq, T)
                window_args.append((train_start_idx, val_start_idx, pred_idx, next_pred_idx))

            if self._verbose:
                print(f"[{self.name}] 并行训练：{n_workers} 进程 × "
                      f"{len(window_args)} 窗口 "
                      f"(lgbm n_jobs/worker={_g_lgbm_n_jobs})")

            ctx = mp.get_context('fork')
            with ctx.Pool(n_workers) as pool:
                par_results = pool.map(_train_window_worker, window_args)

            for r in par_results:
                if r is not None:
                    sym_i_arr, date_j_arr, preds_arr = r
                    factor_values[sym_i_arr, date_j_arr] = preds_arr

        else:
            # ── 串行路径（默认 / Windows fallback）──────────────────────────
            for step, pred_idx in enumerate(pred_indices):
                val_start_idx = pred_idx - self._T_val
                train_start_idx = max(0, val_start_idx - self._T_train)

                train_dates = all_dates[train_start_idx: val_start_idx]
                val_dates = all_dates[val_start_idx: pred_idx - self._n_forward]

                if len(train_dates) < 20 or len(val_dates) < 5:
                    continue

                if self._verbose:
                    print(
                        f"  [{step+1}/{len(pred_indices)}] "
                        f"训练[{train_dates[0].date()} → {train_dates[-1].date()}] "
                        f"验证[{val_dates[0].date()} → {val_dates[-1].date()}] "
                        f"预测@{all_dates[pred_idx].date()}"
                    )

                window_data = self._prepare_window(df, train_dates, val_dates, feat_cols)
                if window_data is None:
                    continue

                X_train, y_train, X_val, y_val = window_data

                try:
                    model = self._train_lgbm(X_train, y_train, X_val, y_val)
                except Exception as e:
                    warnings.warn(f"[{self.name}] Step {step+1} 训练失败：{e}")
                    continue

                next_pred_idx = min(pred_idx + self._rebalance_freq, T)
                pred_date_range = all_dates[pred_idx:next_pred_idx]

                df_pred = df[df['date'].isin(pred_date_range)].dropna(subset=feat_cols).copy()
                if df_pred.empty:
                    continue

                df_pred = self._crosssection_normalize(df_pred, feat_cols)
                df_pred['_pred'] = model.predict(
                    df_pred[feat_cols].values.astype(np.float32)
                )

                df_pred['_sym_i'] = df_pred['symbol'].map(sym_to_i)
                df_pred['_date_j'] = df_pred['date'].map(date_to_j)
                valid = df_pred.dropna(subset=['_sym_i', '_date_j'])
                if not valid.empty:
                    factor_values[
                        valid['_sym_i'].astype(int).values,
                        valid['_date_j'].astype(int).values,
                    ] = valid['_pred'].values

        if self._verbose:
            valid_ratio = (~np.isnan(factor_values)).mean()
            print(f"[{self.name}] 完成，有效值比例：{valid_ratio:.2%}")

        return FactorData(
            values=factor_values,
            symbols=symbols,
            dates=dates_np,
            name=self.name,
            factor_type=self.factor_type,
            params=self.get_params(),
        )

    # ── 指定日期在线预测（生产推理接口）─────────────────────────────────────

    def predict_at_date(
        self,
        target_date,
        ohlcv_data: OHLCVData,
    ) -> np.ndarray:
        """
        仅使用 target_date 之前（含当日）的数据，预测该日截面因子值。

        用于生产推理：每次只用最新数据训练一次，返回当日截面因子分数。
        注意：数据泄露检测请使用基类 detect_at_date()，而非此方法。

        Args:
            target_date: 目标预测日期（str / datetime / Timestamp / datetime64 均可）
            ohlcv_data:  OHLCV 数据，可以包含 target_date 之后的数据（会被忽略）

        Returns:
            ndarray shape (N,) - 每个标的的因子分数，历史不足时返回全 NaN
        """
        target_dt = pd.Timestamp(target_date)
        dates_np = ohlcv_data.dates
        all_dates_pd = pd.DatetimeIndex(dates_np)

        # ── 只保留 ≤ target_date 的数据 ──────────────────────────────────────
        mask = all_dates_pd <= target_dt
        sub_idx = np.where(mask)[0]

        if len(sub_idx) < self._T_train + self._T_val:
            return np.full(ohlcv_data.n_assets, np.nan)

        sub_ohlcv = OHLCVData(
            open=ohlcv_data.open[:, sub_idx],
            high=ohlcv_data.high[:, sub_idx],
            low=ohlcv_data.low[:, sub_idx],
            close=ohlcv_data.close[:, sub_idx],
            volume=ohlcv_data.volume[:, sub_idx],
            symbols=ohlcv_data.symbols,
            dates=dates_np[sub_idx],
        )

        sub_T = len(sub_idx)
        feat_cols = self._feat_cols()

        # ── 构建子集特征 ──────────────────────────────────────────────────────
        df = self._build_features_and_labels(sub_ohlcv, add_labels=True)
        sub_dates_pd = pd.DatetimeIndex(sub_ohlcv.dates)

        # 训练窗口：以 sub_T-1（即 target_date）为调仓点
        pred_idx = sub_T - 1
        val_start_idx = max(0, pred_idx - self._T_val)
        train_start_idx = max(0, val_start_idx - self._T_train)

        # 与 calculate() 保持一致：截断 val 末端 n_forward 个日期，避免 early
        # stopping 使用依赖 target_date 当日收盘价的 label
        train_dates = sub_dates_pd[train_start_idx : val_start_idx]
        val_dates = sub_dates_pd[val_start_idx : pred_idx - self._n_forward]

        if len(train_dates) < 20 or len(val_dates) < 5:
            return np.full(ohlcv_data.n_assets, np.nan)

        window_data = self._prepare_window(df, train_dates, val_dates, feat_cols)
        if window_data is None:
            return np.full(ohlcv_data.n_assets, np.nan)

        X_train, y_train, X_val, y_val = window_data

        try:
            model = self._train_lgbm(X_train, y_train, X_val, y_val)
        except Exception as e:
            warnings.warn(f"[predict_at_date] 训练失败 @ {target_dt.date()}：{e}")
            return np.full(ohlcv_data.n_assets, np.nan)

        # ── 预测目标日期 ──────────────────────────────────────────────────────
        df_pred = df[df['date'] == target_dt].dropna(subset=feat_cols).copy()
        if df_pred.empty:
            return np.full(ohlcv_data.n_assets, np.nan)

        df_pred = self._crosssection_normalize(df_pred, feat_cols)
        preds = model.predict(df_pred[feat_cols].values.astype(np.float32))

        # 映射回原始 symbol 顺序
        sym_to_i: Dict[str, int] = {s: i for i, s in enumerate(ohlcv_data.symbols)}
        result = np.full(ohlcv_data.n_assets, np.nan)
        for k, (_, row) in enumerate(df_pred.iterrows()):
            idx = sym_to_i.get(row['symbol'])
            if idx is not None:
                result[idx] = preds[k]

        return result

    # ── 数据泄露检测 ─────────────────────────────────────────────────────────

    def detect_at_date(
        self,
        ohlcv_data: OHLCVData,
        test_date=None,
        tolerance: float = 1e-6,
        relative_tolerance: float = 1e-4,
        verbose: bool = False,
    ) -> dict:
        """
        通过比较 calculate(short_ohlcv) 与 calculate(full_ohlcv) 在 test_date
        处的截面预测来检测数据泄露。

        检测原理
        --------
        对于调仓点 pred_idx ≤ test_idx：
          - 训练窗口 = [pred_idx - T_train - T_val : pred_idx - T_val]
          - 验证窗口 = [pred_idx - T_val         : pred_idx - n_forward]
          两个窗口完全包含在 [0 : test_idx] 内。

        因此：
          short_run = calculate(ohlcv[:test_idx+1])           → test_idx 处的截面
          full_run  = calculate(ohlcv[:test_idx+n_forward+1]) → test_idx 处的截面

        两次调用中，负责填充 test_idx 处因子值的滚动窗口（pred_idx ≤ test_idx）
        所使用的 train/val 数据完全相同。若预测结果不同，说明存在未来数据泄露。

        full_ohlcv 多出的 n_forward 个时间步只影响 pred_idx > test_idx 的窗口，
        那些窗口填充的是 test_idx 之后的因子值，不参与本次比较。

        n_forward 泄露保证
        ------------------
        验证集截断至 pred_idx - n_forward，最后一个 val 日期的 label 使用
        close[pred_idx - 1]，在 short_ohlcv (test_idx+1 期) 中完全可得。
        full_ohlcv 多出的 n_forward 期不会改变 pred_idx ≤ test_idx 的 val label。

        Args:
            ohlcv_data:         完整 OHLCV 数据（需包含 test_date 之后至少 n_forward 期）
            test_date:          测试日期，默认取数据 3/4 处（需有足够训练历史）
            tolerance:          绝对差值容忍阈值
            relative_tolerance: 相对差值容忍阈值
            verbose:            是否打印进度信息

        Returns:
            dict: {
                'has_leakage': bool,
                'test_date': str,
                'short_periods': int,    # short_ohlcv 时间步数
                'full_periods': int,     # full_ohlcv 时间步数
                'mismatch_count': int,
                'mismatch_ratio': float,
                'max_abs_diff': float,
                'max_rel_diff': float,
                'mismatch_symbols': list[str],
            }
        """
        dates_np = ohlcv_data.dates
        T = len(dates_np)
        dates_pd = pd.DatetimeIndex(dates_np)

        # ── 确定测试索引 ──────────────────────────────────────────────────────
        if test_date is None:
            test_idx = T * 3 // 4
        else:
            test_dt_query = pd.Timestamp(test_date)
            test_idx = int(dates_pd.searchsorted(test_dt_query))
            test_idx = min(test_idx, T - 1)

        test_dt = dates_pd[test_idx]
        test_dt_str = str(test_dt.date())

        # short: 截至 test_idx（含）
        # full:  再延伸 n_forward 步，使"完整"数据在 pred_idx<=test_idx 的
        #        rolling 窗口上与 short 完全等价，同时为 pred_idx>test_idx 的
        #        窗口提供足够数据验证其他窗口的 label 不泄露到 test_idx 之前
        short_end = test_idx + 1
        full_end = min(short_end + self._n_forward, T)

        if full_end <= short_end:
            return {
                'has_leakage': False,
                'test_date': test_dt_str,
                'short_periods': short_end,
                'full_periods': full_end,
                'mismatch_count': 0,
                'mismatch_ratio': 0.0,
                'max_abs_diff': 0.0,
                'max_rel_diff': 0.0,
                'mismatch_symbols': [],
                'note': 'ohlcv_data has no data after test_date; cannot perform leakage test',
            }

        if verbose:
            print(f"[{self.name}] detect_at_date: test_date={test_dt_str}, "
                  f"short={short_end}, full={full_end}")

        def _make_ohlcv(end: int) -> OHLCVData:
            return OHLCVData(
                open=ohlcv_data.open[:, :end],
                high=ohlcv_data.high[:, :end],
                low=ohlcv_data.low[:, :end],
                close=ohlcv_data.close[:, :end],
                volume=ohlcv_data.volume[:, :end],
                symbols=ohlcv_data.symbols,
                dates=dates_np[:end],
            )

        short_ohlcv = _make_ohlcv(short_end)
        full_ohlcv = _make_ohlcv(full_end)

        # ── 分别运行 calculate ────────────────────────────────────────────────
        factor_short = self.calculate(short_ohlcv)
        factor_full = self.calculate(full_ohlcv)

        # ── 取 test_date 列的因子值 ───────────────────────────────────────────
        def _col_at(fd: 'FactorData') -> Optional[np.ndarray]:
            fd_dates_d = np.array(fd.dates, dtype='datetime64[D]')
            target_d = np.datetime64(test_dt_str, 'D')
            idx_arr = np.where(fd_dates_d == target_d)[0]
            return fd.values[:, idx_arr[0]] if len(idx_arr) > 0 else None

        vals_short = _col_at(factor_short)
        vals_full = _col_at(factor_full)

        if vals_short is None or vals_full is None:
            return {
                'has_leakage': False,
                'test_date': test_dt_str,
                'short_periods': short_end,
                'full_periods': full_end,
                'mismatch_count': 0,
                'mismatch_ratio': 0.0,
                'max_abs_diff': 0.0,
                'max_rel_diff': 0.0,
                'mismatch_symbols': [],
                'note': 'test_date not in factor output (insufficient history for rolling window)',
            }

        # ── 逐标的比较 ────────────────────────────────────────────────────────
        symbols = ohlcv_data.symbols
        N = len(symbols)
        mismatch_count = 0
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        mismatch_syms: List[str] = []
        valid_count = 0

        for i in range(N):
            s = float(vals_short[i])
            f = float(vals_full[i])
            if np.isnan(s) and np.isnan(f):
                continue
            valid_count += 1
            if np.isnan(s) or np.isnan(f):
                mismatch_count += 1
                mismatch_syms.append(str(symbols[i]))
                continue
            abs_diff = abs(s - f)
            rel_diff = abs_diff / (abs(s) + 1e-12)
            if abs_diff > tolerance and rel_diff > relative_tolerance:
                mismatch_count += 1
                mismatch_syms.append(str(symbols[i]))
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)

        has_leakage = mismatch_count > 0
        result = {
            'has_leakage': has_leakage,
            'test_date': test_dt_str,
            'short_periods': short_end,
            'full_periods': full_end,
            'mismatch_count': mismatch_count,
            'mismatch_ratio': mismatch_count / max(valid_count, 1),
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mismatch_symbols': mismatch_syms[:20],
        }

        if verbose:
            status = "泄露检出!" if has_leakage else "无泄露"
            print(f"  结果: {status} | mismatch={mismatch_count}/{valid_count} "
                  f"max_abs={max_abs_diff:.2e} max_rel={max_rel_diff:.2e}")

        return result
