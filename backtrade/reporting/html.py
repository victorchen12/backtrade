from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd


def _fig_html(fig: Any, *, include_js: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="inline" if include_js else False, config={"displaylogo": False, "responsive": True})


def _value(value: Any) -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return html.escape(str(value))


def _display_value(label: str, value: Any) -> str:
    if label == "净收益率" and value is not None:
        try:
            return f"{float(value) * 100:,.2f}%"
        except (TypeError, ValueError):
            return _value(value)
    return _value(value)


def _time_axis(frame: pd.DataFrame) -> tuple[list[Any], list[str], list[Any], list[str]]:
    if frame.empty:
        return [], [], [], []
    x_column = "plot_ts" if "plot_ts" in frame else "event_ts"
    actual_column = "actual_ts" if "actual_ts" in frame else "event_ts"
    axis_columns = [x_column, actual_column]
    if "trading_day" in frame:
        axis_columns.append("trading_day")
    valid = frame[axis_columns].dropna(subset=[x_column, actual_column]).drop_duplicates(x_column).sort_values(x_column)
    if valid.empty:
        return [], [], [], []
    valid = valid.reset_index(drop=True)
    all_actual = pd.to_datetime(valid[actual_column])
    if "trading_day" in valid and valid["trading_day"].notna().any():
        day_keys = valid["trading_day"].fillna(all_actual.dt.strftime("%Y-%m-%d")).astype(str).to_numpy()
    else:
        day_keys = all_actual.dt.strftime("%Y-%m-%d").to_numpy()
    day_starts = np.flatnonzero(np.r_[True, day_keys[1:] != day_keys[:-1]])
    selected = valid.iloc[day_starts]
    selected_actual = pd.to_datetime(selected[actual_column])
    if "trading_day" in selected and selected["trading_day"].notna().any():
        display_dates = pd.to_datetime(selected["trading_day"], errors="coerce").fillna(selected_actual)
    else:
        display_dates = selected_actual
    labels = [f"{value.month}/{value.day}" for value in display_dates]
    years = display_dates.dt.year.to_numpy()
    year_starts = np.flatnonzero(np.r_[True, years[1:] != years[:-1]])
    year_rows = selected.iloc[year_starts]
    return selected[x_column].tolist(), labels, year_rows[x_column].tolist(), [str(year) for year in display_dates.iloc[year_starts].dt.year]


def _set_time_axis(fig: Any, frame: pd.DataFrame) -> None:
    ticks, labels, year_ticks, year_labels = _time_axis(frame)
    if ticks:
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels)
    for tick, label in zip(year_ticks, year_labels):
        fig.add_annotation(x=tick, y=-0.2, xref="x", yref="paper", text=label, showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, color="#60717f"))
    if year_ticks:
        fig.update_layout(margin=dict(b=62))


def _x_column(frame: pd.DataFrame) -> str:
    return "plot_ts" if "plot_ts" in frame else "event_ts"


def _actual_customdata(frame: pd.DataFrame) -> list[str]:
    column = "actual_ts" if "actual_ts" in frame else "event_ts"
    return pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()


def _transaction_cash_points(accounts: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    x_column = "plot_ts" if "plot_ts" in snapshots or "plot_ts" in accounts else "event_ts"
    columns = [x_column, "actual_ts", "cash", "cumulative_net_pnl", "position_before", "position_after", "direction", "holding_state"]
    rows: list[dict[str, Any]] = []
    if not accounts.empty and "cash_after" in accounts:
        fills = accounts.sort_values(_x_column(accounts), kind="stable").copy()
        fills["cash"] = pd.to_numeric(fills["cash_after"], errors="coerce")
        fills = fills[fills["cash"].notna()].copy()
        if not fills.empty:
            net_pnl = fills["net_pnl"] if "net_pnl" in fills else pd.Series(0.0, index=fills.index)
            fills["cumulative_net_pnl"] = pd.to_numeric(net_pnl, errors="coerce").fillna(0.0).cumsum()
            before_source = fills["position_before"] if "position_before" in fills else pd.Series(0.0, index=fills.index)
            after_source = fills["position_after"] if "position_after" in fills else pd.Series(0.0, index=fills.index)
            before = pd.to_numeric(before_source, errors="coerce").fillna(0.0)
            after = pd.to_numeric(after_source, errors="coerce").fillna(0.0)
            fills["position_before"] = before
            fills["position_after"] = after
            fills["direction"] = after.where(after.ne(0), before)
            fills["holding_state"] = after
            rows.extend(fills[[x_column, "actual_ts", "cash", "cumulative_net_pnl", "position_before", "position_after", "direction", "holding_state"]].to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(x_column, kind="stable").reset_index(drop=True)


def _build_figures(data: dict, metrics: dict) -> list[str]:
    import plotly.graph_objects as go

    snapshots = data["snapshots"].copy()
    activity = data["activity"].copy()
    accounts = data.get("accounts", pd.DataFrame()).copy()
    hourly_market = data["hourly_market"]
    hourly_drawdown = data["hourly_drawdown"]
    market = data.get("market", pd.DataFrame())
    price_basis = str(data.get("price_basis", "raw"))
    price_label = "\u524d\u590d\u6743" if price_basis == "front_adjusted" else "\u539f\u59cb"
    manifest = data.get("manifest") or {}
    factor_name = str(manifest.get("factor_name") or "\u56e0\u5b50")
    strategy_config = ((manifest.get("config") or {}).get("strategy") or {})
    signal_mode = str(strategy_config.get("signal_mode") or "signed_factor")
    figures: list[str] = []

    if not snapshots.empty:
        x = _x_column(snapshots)
        customdata = _actual_customdata(snapshots)
        equity = go.Figure()
        equity.add_trace(go.Scatter(x=snapshots[x], y=snapshots["equity"], customdata=customdata, name="权益", mode="lines", hovertemplate="实际时间=%{customdata}<br>权益=%{y:.6f}<extra></extra>"))
        cash_points = _transaction_cash_points(accounts, snapshots)
        if not cash_points.empty:
            cash_x = _x_column(cash_points)
            cash_customdata = np.column_stack([
                _actual_customdata(cash_points),
                cash_points["cumulative_net_pnl"].map(lambda value: f"{value:.6f}").to_numpy(),
            ])
            equity.add_trace(go.Scatter(x=cash_points[cash_x], y=cash_points["cash"], customdata=cash_customdata, name="现金（成交点）", mode="lines+markers", line=dict(color="#00897b", width=1.4), marker=dict(color="#00897b", size=2.5), hovertemplate="实际时间=%{customdata[0]}<br>现金=%{y:.6f}<br>累计净 PnL=%{customdata[1]}<extra></extra>"))
            for value, label, color in [(-1, "空头成交", "#2e7d32"), (1, "多头成交", "#c62828")]:
                subset = cash_points[cash_points["direction"].eq(value)]
                if subset.empty:
                    continue
                subset_customdata = np.column_stack([
                    _actual_customdata(subset),
                    subset["cumulative_net_pnl"].map(lambda item: f"{item:.6f}").to_numpy(),
                ])
                equity.add_trace(go.Scatter(x=subset[cash_x], y=subset["cash"], customdata=subset_customdata, mode="markers", name=label, marker=dict(color=color, size=4, opacity=0.88), hovertemplate=f"实际时间=%{{customdata[0]}}<br>{label}<br>现金=%{{y:.6f}}<br>累计净 PnL=%{{customdata[1]}}<extra></extra>"))
                segment_x: list[Any] = []
                segment_y: list[float | None] = []
                for index in cash_points.index[cash_points["holding_state"].eq(value)]:
                    row_position = cash_points.index.get_loc(index)
                    if row_position >= len(cash_points) - 1:
                        continue
                    segment_x.extend([cash_points.iloc[row_position][cash_x], cash_points.iloc[row_position + 1][cash_x], None])
                    segment_y.extend([-1.0 if value < 0 else 1.0, -1.0 if value < 0 else 1.0, None])
                if segment_x:
                    equity.add_trace(go.Scatter(x=segment_x, y=segment_y, mode="lines", name="持有多头" if value > 0 else "持有空头", line=dict(color=color, width=3), opacity=0.86, hoverinfo="skip", connectgaps=False, yaxis="y3"))
        if not market.empty and {"plot_ts", "actual_ts", "front_adjusted_price"}.issubset(market.columns):
            raw_price = market[["plot_ts", "actual_ts", "front_adjusted_price"]].dropna().sort_values("plot_ts")
            if len(raw_price) > 80_000:
                raw_price = raw_price.iloc[::int(np.ceil(len(raw_price) / 80_000))]
            equity.add_trace(go.Scatter(x=raw_price["plot_ts"], y=raw_price["front_adjusted_price"], customdata=_actual_customdata(raw_price), name="实际价格（前复权）", mode="lines", yaxis="y2", line=dict(color="#78909c", width=1, dash="dash"), opacity=0.65, hovertemplate="实际时间=%{customdata}<br>前复权价格=%{y:.6f}<extra></extra>"))
        equity.update_layout(title="权益、现金、实际价格与持仓方向", xaxis_title="连续交易时间", yaxis=dict(title="金额", domain=[0.0, 0.82]), yaxis2=dict(title="前复权价格", overlaying="y", side="right", showgrid=False), yaxis3=dict(domain=[0.86, 1.0], range=[-1.2, 1.2], tickmode="array", tickvals=[-1, 0, 1], ticktext=["空", "平", "多"], showgrid=False, zeroline=False, title="持仓方向", anchor="x"), template="plotly_white", height=390, margin=dict(l=60, r=95, t=70, b=45))
        _set_time_axis(equity, market if not market.empty else snapshots)
        if price_basis != "front_adjusted":
            for trace in equity.data:
                if getattr(trace, "name", None):
                    trace.name = str(trace.name).replace("\u524d\u590d\u6743", price_label)
                if getattr(trace, "hovertemplate", None):
                    trace.hovertemplate = str(trace.hovertemplate).replace("\u524d\u590d\u6743", price_label)
            if getattr(equity.layout, "yaxis2", None) and equity.layout.yaxis2.title:
                equity.layout.yaxis2.title.text = f"{price_label}\u4ef7\u683c"
        equity.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.32, yanchor="top", font=dict(size=10)), margin=dict(l=60, r=95, t=70, b=120))
        figures.append(_fig_html(equity, include_js=True))

    if not hourly_market.empty:
        factor_price = go.Figure()
        if not market.empty and {"plot_ts", "actual_ts", "active_factor"}.issubset(market.columns):
            raw_factor = market[["plot_ts", "actual_ts", "active_factor"]].dropna().sort_values("plot_ts")
            if len(raw_factor) > 80_000:
                raw_factor = raw_factor.iloc[::int(np.ceil(len(raw_factor) / 80_000))]
            factor_price.add_trace(go.Scatter(x=raw_factor["plot_ts"], y=raw_factor["active_factor"], customdata=_actual_customdata(raw_factor), name=f"{factor_name} 因子", mode="lines", line=dict(color="#90a4ae", width=1), hovertemplate=f"实际时间=%{{customdata}}<br>{factor_name}=%{{y:.6f}}<extra></extra>"))
            if signal_mode == "ecdf_tail":
                short_threshold = float(strategy_config["short_threshold"])
                long_threshold = float(strategy_config["long_threshold"])
                factor_price.add_hline(y=short_threshold, line_dash="dot", line_color="#1976d2", annotation_text="short threshold", annotation_position="top left")
                factor_price.add_hline(y=long_threshold, line_dash="dot", line_color="#ef6c00", annotation_text="long threshold", annotation_position="bottom left")
            else:
                q10, q90 = raw_factor["active_factor"].quantile([0.10, 0.90]).tolist()
                factor_price.add_hline(y=float(q10), line_dash="dot", line_color="#1976d2", annotation_text="q10", annotation_position="top left")
                factor_price.add_hline(y=float(q90), line_dash="dot", line_color="#ef6c00", annotation_text="q90", annotation_position="bottom left")
        if signal_mode != "ecdf_tail" and "factor_q10" in hourly_market:
            factor_price.add_trace(go.Scatter(x=hourly_market["event_ts"], y=hourly_market["factor_q10"], customdata=_actual_customdata(hourly_market), name=f"{factor_name} q10（小时）", mode="lines", line=dict(color="#1976d2"), hovertemplate="实际时间=%{customdata}<br>q10=%{y:.6f}<extra></extra>"))
        if signal_mode != "ecdf_tail" and "factor_q90" in hourly_market:
            factor_price.add_trace(go.Scatter(x=hourly_market["event_ts"], y=hourly_market["factor_q90"], customdata=_actual_customdata(hourly_market), name=f"{factor_name} q90（小时）", mode="lines", line=dict(color="#ef6c00"), hovertemplate="实际时间=%{customdata}<br>q90=%{y:.6f}<extra></extra>"))
        if not market.empty and {"plot_ts", "actual_ts", "front_adjusted_price"}.issubset(market.columns):
            raw_price = market[["plot_ts", "actual_ts", "front_adjusted_price"]].dropna().sort_values("plot_ts")
            if len(raw_price) > 80_000:
                raw_price = raw_price.iloc[::int(np.ceil(len(raw_price) / 80_000))]
            factor_price.add_trace(go.Scatter(x=raw_price["plot_ts"], y=raw_price["front_adjusted_price"], customdata=_actual_customdata(raw_price), name="实际价格（前复权，tick）", mode="lines", yaxis="y2", line=dict(color="#263238", width=1), hovertemplate="实际时间=%{customdata}<br>前复权价格=%{y:.6f}<extra></extra>"))
        factor_price.update_layout(title=f"OOS {factor_name} 因子、实际价格与交易活动", xaxis_title="连续交易时间", yaxis=dict(title="因子值"), yaxis2=dict(title="前复权价格", overlaying="y", side="right"), template="plotly_white", height=420, margin=dict(l=55, r=65, t=55, b=45))
        _set_time_axis(factor_price, market if not market.empty else hourly_market)
        if price_basis != "front_adjusted":
            for trace in factor_price.data:
                if getattr(trace, "name", None):
                    trace.name = str(trace.name).replace("\u524d\u590d\u6743", price_label)
                if getattr(trace, "hovertemplate", None):
                    trace.hovertemplate = str(trace.hovertemplate).replace("\u524d\u590d\u6743", price_label)
            if getattr(factor_price.layout, "yaxis2", None) and factor_price.layout.yaxis2.title:
                factor_price.layout.yaxis2.title.text = f"{price_label}\u4ef7\u683c"
        figures.append(_fig_html(factor_price, include_js=False))

        diagnostics = go.Figure()
        if not hourly_drawdown.empty:
            diagnostics.add_trace(go.Bar(x=hourly_drawdown["event_ts"], y=hourly_drawdown["drawdown"], customdata=_actual_customdata(hourly_drawdown), name="小时回撤", marker_color="#c62828", yaxis="y", hovertemplate="实际时间=%{customdata}<br>回撤=%{y:.6f}<extra></extra>"))
        if "volume" in hourly_market:
            diagnostics.add_trace(go.Bar(x=hourly_market["event_ts"], y=hourly_market["volume"], customdata=_actual_customdata(hourly_market), name="小时成交量", marker_color="#2e7d32", opacity=0.55, yaxis="y2", hovertemplate="实际时间=%{customdata}<br>成交量=%{y:.6f}<extra></extra>"))
        diagnostics.update_layout(title="小时回撤与成交量诊断", barmode="overlay", xaxis_title="连续交易时间", yaxis=dict(title="回撤"), yaxis2=dict(title="成交量", overlaying="y", side="right"), template="plotly_white", height=320, margin=dict(l=55, r=65, t=55, b=45))
        _set_time_axis(diagnostics, hourly_market)
        figures.append(_fig_html(diagnostics, include_js=False))

    maker = data.get("maker_events")
    if maker is not None and not maker.empty:
        maker = maker.copy()
        before = pd.to_numeric(maker.get("queue_ahead_before", pd.Series(index=maker.index)), errors="coerce")
        after = pd.to_numeric(maker.get("queue_ahead_after", pd.Series(index=maker.index)), errors="coerce")
        evidence = maker[before.notna() | after.notna()].copy()
        if not evidence.empty:
            evidence["queue_before"] = before.loc[evidence.index].fillna(after.loc[evidence.index])
            evidence["queue_after"] = after.loc[evidence.index].fillna(evidence["queue_before"])
            x = _x_column(evidence)
            customdata = np.column_stack([
                _actual_customdata(evidence),
                evidence["event_type"].fillna("").astype(str),
                evidence["reason_code"].fillna("").astype(str),
                evidence["queue_before"].map(lambda value: f"{value:.2f}"),
                evidence["queue_after"].map(lambda value: f"{value:.2f}"),
            ])
            hover = "实际时间=%{customdata[0]}<br>事件=%{customdata[1]}<br>原因=%{customdata[2]}<br>队列前=%{customdata[3]}<br>队列后=%{customdata[4]}<extra></extra>"
            queue = go.Figure()
            transition_x: list[Any] = []
            transition_y: list[float | None] = []
            for row in evidence.itertuples(index=False):
                transition_x.extend([getattr(row, x), getattr(row, x), None])
                transition_y.extend([float(row.queue_before), float(row.queue_after), None])
            queue.add_trace(go.Scatter(x=transition_x, y=transition_y, mode="lines", name="队列变化", line=dict(color="#b0bec5", width=0.8), hoverinfo="skip"))
            queue.add_trace(go.Scatter(x=evidence[x], y=evidence["queue_before"], customdata=customdata, mode="markers", name="队列前", marker=dict(color="#607d8b", size=3, opacity=0.82), hovertemplate=hover))
            queue.add_trace(go.Scatter(x=evidence[x], y=evidence["queue_after"], customdata=customdata, mode="markers", name="队列后", marker=dict(color="#1565c0", size=3, opacity=0.88), hovertemplate=hover))
            event_labels = {"enqueue": ("进入队列", "#8e24aa", "triangle-up"), "progress": ("队列推进", "#00897b", "circle"), "fill": ("成交", "#c62828", "diamond"), "cancel": ("撤单", "#ef6c00", "x"), "rebaseline": ("重新基准", "#546e7a", "square")}
            for event_type, (label, color, symbol) in event_labels.items():
                subset = evidence[evidence["event_type"].astype(str).eq(event_type)]
                if subset.empty:
                    continue
                subset_data = customdata[evidence["event_type"].astype(str).eq(event_type).to_numpy()]
                queue.add_trace(go.Scatter(x=subset[x], y=subset["queue_after"], customdata=subset_data, mode="markers", name=label, marker=dict(color=color, size=5, symbol=symbol, opacity=0.9), hovertemplate=hover))
            queue.update_layout(title="Maker 队列变化（有效排队事件）", xaxis_title="连续交易时间", yaxis_title="前方队列数量（手）", template="plotly_white", height=280, margin=dict(l=55, r=25, t=55, b=45))
            _set_time_axis(queue, evidence)
            figures.append(_fig_html(queue, include_js=False))
    return figures


def render_report_html(data: dict, metrics: dict) -> str:
    manifest = data["manifest"]
    cards = [
        ("净 PnL", metrics.get("net_pnl")),
        ("净收益率", metrics.get("net_return")),
        ("最大回撤", metrics.get("max_drawdown")),
        ("总手续费", metrics.get("total_fee")),
        ("完整往返", metrics.get("round_trips")),
        ("因子有效率", metrics.get("factor_active_rate")),
        ("成交笔数", metrics.get("fill_count")),
        ("最终空仓", metrics.get("reconciliation", {}).get("final_position_flat")),
    ]
    card_html = "".join(f'<div class="kpi"><span>{html.escape(label)}</span><strong>{_display_value(label, value)}</strong></div>' for label, value in cards)
    figures = _build_figures(data, metrics)
    figure_html = "".join(f'<section class="chart">{figure}</section>' for figure in figures)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtrade {html.escape(str(manifest.get('factor_name', 'factor')))} 回测报告</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#f5f7fa; color:#263238; }}
body {{ margin:0; }} main {{ max-width:1440px; margin:0 auto; padding:28px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:20px; border-bottom:1px solid #d7dde3; padding-bottom:18px; }}
h1 {{ font-size:25px; margin:0; }} h2 {{ font-size:16px; margin:0 0 12px; }} p {{ margin:5px 0; color:#60717f; font-size:13px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:20px 0; }}
.kpi {{ background:#fff; border:1px solid #d7dde3; border-radius:6px; padding:13px 14px; min-height:54px; }} .kpi span {{ display:block; font-size:12px; color:#60717f; }} .kpi strong {{ display:block; margin-top:8px; font-size:19px; font-weight:650; }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .chart {{ background:#fff; border:1px solid #d7dde3; border-radius:6px; padding:4px; min-width:0; }}
@media (max-width:900px) {{ main {{ padding:14px; }} .charts {{ grid-template-columns:1fr; }} header {{ display:block; }} }}
</style></head><body><main>
<header><div><h1>Backtrade compact_v9 回测报告</h1><p>因子：{html.escape(str(manifest.get('factor_name', 'unknown')))} | 模式：{html.escape(str(manifest.get('match_mode', 'unknown')))}</p></div><p>产物版本：{html.escape(str(manifest.get('artifact_schema_version', 'unknown')))}</p></header>
<section class="kpis">{card_html}</section>
<section class="charts">{figure_html}</section>
</main></body></html>"""


__all__ = ["render_report_html"]
