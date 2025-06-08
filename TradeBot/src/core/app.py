import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import io
import sqlite3
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

# === Intégration du système de trading ===
from analysis.trading_analyzer import load_all_components_for_analysis, execute_backtest

# --- Dark mode toggle (Streamlit >= 1.10) ---
def set_dark_mode(enabled: bool):
    if enabled:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #18191A !important; color: #F5F6F7 !important; }
            .stDataFrame, .stTable { background-color: #23272F !important; }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #FFFFFF !important; color: #222 !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

# --- Chargement des résultats de backtest ---
@st.cache_data(show_spinner=True)
def load_backtest_results():
    try:
        components = load_all_components_for_analysis()
        backtest_result = execute_backtest(components)
        return backtest_result
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats de backtest : {e}")
        return None

backtest_result = load_backtest_results()

# --- Détection multi-stratégies ---
def is_multi_strategy(backtest_result):
    # Si le résultat contient 'performance_by_strategy_and_asset', on est en multi-stratégies
    return backtest_result and "performance_by_strategy_and_asset" in backtest_result

def get_strategies_and_assets(backtest_result):
    if is_multi_strategy(backtest_result):
        strategies = list(backtest_result["performance_by_strategy_and_asset"].keys())
        assets = set()
        for strat in strategies:
            assets.update(backtest_result["performance_by_strategy_and_asset"][strat].keys())
        return strategies, sorted(list(assets))
    elif backtest_result and "performance_by_asset" in backtest_result:
        return ["Hybride"], list(backtest_result["performance_by_asset"].keys())
    else:
        return [], []

# --- Sidebar: Filtres dynamiques ---
st.sidebar.header("Filtres avancés")

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark_mode = st.sidebar.toggle("Mode sombre", value=st.session_state["dark_mode"])
st.session_state["dark_mode"] = dark_mode
set_dark_mode(dark_mode)

strategies, assets = get_strategies_and_assets(backtest_result)

selected_strategy = st.sidebar.selectbox("Stratégie", strategies) if strategies else None
selected_asset = st.sidebar.selectbox("Actif", assets) if assets else None

# Filtres numériques
winrate_min = st.sidebar.slider("Winrate min (%)", 0, 100, 0, 1)
sharpe_min = st.sidebar.slider("Sharpe min", 0.0, 5.0, 0.0, 0.1)

# Période (si equity curve disponible)
period_days = st.sidebar.slider("Période (jours)", 7, 365, 90, 1)
end_date = datetime.now()
start_date = end_date - timedelta(days=period_days)

# --- Leaderboard multi-stratégies ---
def build_leaderboard_multi(backtest_result, winrate_min=0, sharpe_min=0):
    leaderboard = []
    if is_multi_strategy(backtest_result):
        for strat, assets_stats in backtest_result["performance_by_strategy_and_asset"].items():
            for asset, stats in assets_stats.items():
                if stats.get("winrate", 0)*100 >= winrate_min and stats.get("sharpe_ratio", 0) >= sharpe_min:
                    leaderboard.append({
                        "Stratégie": strat,
                        "Actif": asset,
                        "PnL": stats.get("total_pnl", 0),
                        "Drawdown": stats.get("max_drawdown", 0),
                        "Sharpe": stats.get("sharpe_ratio", 0),
                        "Winrate": stats.get("winrate", 0),
                        "Nb Trades": stats.get("num_trades", 0),
                    })
    elif backtest_result and "performance_by_asset" in backtest_result:
        for asset, stats in backtest_result["performance_by_asset"].items():
            if stats.get("winrate", 0)*100 >= winrate_min and stats.get("sharpe_ratio", 0) >= sharpe_min:
                leaderboard.append({
                    "Stratégie": "Hybride",
                    "Actif": asset,
                    "PnL": stats.get("total_pnl", 0),
                    "Drawdown": stats.get("max_drawdown", 0),
                    "Sharpe": stats.get("sharpe_ratio", 0),
                    "Winrate": stats.get("winrate", 0),
                    "Nb Trades": stats.get("num_trades", 0),
                })
    df = pd.DataFrame(leaderboard)
    
    # Handle empty DataFrame or missing columns
    if df.empty:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["Stratégie", "Actif", "PnL", "Drawdown", "Sharpe", "Winrate", "Nb Trades"])
    
    # Check if necessary columns exist before sorting
    sort_columns = []
    if "PnL" in df.columns:
        sort_columns.append("PnL")
    if "Sharpe" in df.columns:
        sort_columns.append("Sharpe")
    
    # Only sort if we have columns to sort by
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
    
    return df

# --- Export Excel, SQL, CSV ---
def export_csv_button(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Exporter CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
        use_container_width=True
    )

def export_excel_button(df, filename):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        label="Exporter Excel",
        data=output.getvalue(),
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

def export_sqlite_button(df, filename, table_name="data"):
    output = io.BytesIO()
    with sqlite3.connect(':memory:') as conn:
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        for line in conn.iterdump():
            output.write(f'{line}\n'.encode('utf-8'))
    st.download_button(
        label="Exporter SQL (SQLite dump)",
        data=output.getvalue(),
        file_name=filename,
        mime='application/sql',
        use_container_width=True
    )

# --- Export direct vers une vraie base SQL ---
def export_sqlalchemy_button(df, default_url, table_name):
    st.markdown("**Export direct vers une base SQL**")
    sql_url = st.text_input("URL de connexion SQLAlchemy", value=default_url, key=f"sql_url_{table_name}")
    if st.button(f"Exporter vers SQL ({table_name})", key=f"export_sql_{table_name}"):
        try:
            engine = create_engine(sql_url)
            df.to_sql(table_name, engine, index=False, if_exists='replace')
            st.success(f"Exporté avec succès dans la table '{table_name}' de {sql_url}")
        except Exception as e:
            st.error(f"Erreur export SQL : {e}")

# --- Pourcentage par stratégie/type de trade ---
def compute_trade_percentages(trades_df):
    if trades_df is None or trades_df.empty or 'side' not in trades_df:
        return {}
    total = len(trades_df)
    buy = len(trades_df[trades_df['side'].str.upper().str.contains('BUY')])
    sell = len(trades_df[trades_df['side'].str.upper().str.contains('SELL')])
    neutral = total - buy - sell
    return {
        'BUY': 100 * buy / total if total else 0,
        'SELL': 100 * sell / total if total else 0,
        'NEUTRE': 100 * neutral / total if total else 0,
        'TOTAL': total
    }

def compute_pnl_percentages(leaderboard_df):
    if leaderboard_df is None:
        return pd.DataFrame()
    if isinstance(leaderboard_df, dict):
        return pd.DataFrame()
    if leaderboard_df.empty or 'PnL' not in leaderboard_df.columns:
        return leaderboard_df
    total_pnl = leaderboard_df['PnL'].sum()
    leaderboard_df = leaderboard_df.copy()
    leaderboard_df['PnL_%'] = leaderboard_df['PnL'] / total_pnl * 100 if total_pnl != 0 else 0
    return leaderboard_df

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Leaderboard", "Décisions", "Performances", "Risques", "Backtesting"])

# --- Tab 1: Leaderboard ---
with tab1:
    st.header("Leaderboard Multi-Stratégies")
    leaderboard_df = build_leaderboard_multi(backtest_result, winrate_min, sharpe_min)
    leaderboard_df = compute_pnl_percentages(leaderboard_df)
    st.dataframe(leaderboard_df, use_container_width=True)
    col_csv, col_xlsx, col_sql, col_sqlalchemy = st.columns(4)
    with col_csv:
        export_csv_button(leaderboard_df, "leaderboard.csv")
    with col_xlsx:
        export_excel_button(leaderboard_df, "leaderboard.xlsx")
    with col_sql:
        export_sqlite_button(leaderboard_df, "leaderboard.sql")
    with col_sqlalchemy:
        export_sqlalchemy_button(leaderboard_df, "sqlite:///leaderboard.db", "leaderboard")
    # Pourcentage PnL par stratégie
    if 'Stratégie' in leaderboard_df.columns:
        st.subheader("% du PnL par stratégie")
        strat_pnl = leaderboard_df.groupby('Stratégie')['PnL'].sum()
        total_pnl = strat_pnl.sum()
        strat_pnl_pct = (strat_pnl / total_pnl * 100).round(2) if total_pnl != 0 else strat_pnl * 0
        st.bar_chart(strat_pnl_pct)

    # Equity curve custom avec tooltips ultra-personnalisés
    st.subheader("Courbe d'équité")
    chart_type = st.selectbox("Type de graphique", ["Ligne", "Aire", "Barres"])
    palette = px.colors.sequential.Blues if not dark_mode else px.colors.sequential.Plasma

    if selected_strategy and selected_asset:
        # Récupérer la courbe d'équité
        if is_multi_strategy(backtest_result):
            equity_curves = backtest_result.get("equity_curves", {}).get(selected_strategy, {})
            equity_curve = equity_curves.get(selected_asset)
        else:
            equity_curve = backtest_result.get("equity_curves", {}).get(selected_asset)
        if equity_curve is not None:
            equity_curve = equity_curve[(equity_curve.index >= pd.to_datetime(start_date)) & (equity_curve.index <= pd.to_datetime(end_date))]
            df_eq = pd.DataFrame({"Date": equity_curve.index, "Equity": equity_curve.values, "Stratégie": selected_strategy, "Actif": selected_asset})
            hovertemplate = (
                "<b>Date</b>: %{x|%Y-%m-%d %H:%M}<br>"
                "<b>Équité</b>: %{y:,.2f} €<br>"
                f"<b>Stratégie</b>: {selected_strategy}<br>"
                f"<b>Actif</b>: {selected_asset}"
            )
            if chart_type == "Ligne":
                fig = px.line(df_eq, x="Date", y="Equity", labels={"x": "Date", "y": "Équité (€)"}, color_discrete_sequence=palette)
            elif chart_type == "Aire":
                fig = px.area(df_eq, x="Date", y="Equity", labels={"x": "Date", "y": "Équité (€)"}, color_discrete_sequence=palette)
            else:
                fig = px.bar(df_eq, x="Date", y="Equity", labels={"x": "Date", "y": "Équité (€)"}, color_discrete_sequence=palette)
            fig.update_traces(hovertemplate=hovertemplate)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de courbe d'équité disponible pour cette combinaison.")
    else:
        st.info("Sélectionnez une stratégie et un actif pour afficher la courbe d'équité.")

# --- Tab 2: Décisions (Signaux de Trading) ---
with tab2:
    st.header("Décisions / Signaux de Trading")
    if selected_strategy and selected_asset:
        if is_multi_strategy(backtest_result):
            trades = backtest_result.get("trades_by_strategy_and_asset", {}).get(selected_strategy, {}).get(selected_asset, pd.DataFrame())
            price_series = backtest_result.get("price_series", {}).get(selected_strategy, {}).get(selected_asset)
        else:
            trades = backtest_result.get("trades_by_asset", {}).get(selected_asset, pd.DataFrame())
            price_series = backtest_result.get("price_series", {}).get(selected_asset)
        st.dataframe(trades, use_container_width=True)
        col_csv, col_xlsx, col_sql, col_sqlalchemy = st.columns(4)
        with col_csv:
            export_csv_button(trades, f"trades_{selected_strategy}_{selected_asset}.csv")
        with col_xlsx:
            export_excel_button(trades, f"trades_{selected_strategy}_{selected_asset}.xlsx")
        with col_sql:
            export_sqlite_button(trades, f"trades_{selected_strategy}_{selected_asset}.sql")
        with col_sqlalchemy:
            export_sqlalchemy_button(trades, "sqlite:///trades.db", f"trades_{selected_strategy}_{selected_asset}")
        # Pourcentage de trades par type
        trade_pct = compute_trade_percentages(trades)
        st.subheader("% de trades par type")
        st.write(trade_pct)
        # Affichage des signaux sur la courbe de prix (BUY, SELL, NEUTRE)
        if price_series is not None and not trades.empty:
            df_price = pd.DataFrame({"Date": price_series.index, "Prix": price_series.values})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_price["Date"], y=df_price["Prix"], mode="lines", name="Prix", line=dict(color='black', width=1)))
            # BUY
            buy_trades = trades[trades["side"].str.upper().str.contains("BUY")]
            fig.add_trace(go.Scatter(
                x=buy_trades["entry_time"],
                y=buy_trades["entry_price"],
                mode="markers",
                name="Entrées BUY",
                marker=dict(color='green', symbol='triangle-up', size=12),
                hovertemplate="<b>Entrée BUY</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Prix: %{y:,.2f} €<br>PnL: %{customdata[0]:,.2f} €<br>Setup: %{customdata[1]}<br>Proba: %{customdata[2]:.2f}<extra></extra>",
                customdata=np.stack([
                    buy_trades.get("profit_usd", [None]),
                    buy_trades.get("setup_quality", [None]),
                    buy_trades.get("xgb_proba", [None]) if "xgb_proba" in buy_trades else np.full(len(buy_trades), np.nan)
                ], axis=-1) if "profit_usd" in buy_trades and "setup_quality" in buy_trades else None
            ))
            # SELL
            sell_trades = trades[trades["side"].str.upper().str.contains("SELL")]
            fig.add_trace(go.Scatter(
                x=sell_trades["exit_time"],
                y=sell_trades["exit_price"],
                mode="markers",
                name="Sorties SELL",
                marker=dict(color='red', symbol='triangle-down', size=12),
                hovertemplate="<b>Sortie SELL</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Prix: %{y:,.2f} €<br>PnL: %{customdata[0]:,.2f} €<br>Setup: %{customdata[1]}<br>Proba: %{customdata[2]:.2f}<extra></extra>",
                customdata=np.stack([
                    sell_trades.get("profit_usd", [None]),
                    sell_trades.get("setup_quality", [None]),
                    sell_trades.get("xgb_proba", [None]) if "xgb_proba" in sell_trades else np.full(len(sell_trades), np.nan)
                ], axis=-1) if "profit_usd" in sell_trades and "setup_quality" in sell_trades else None
            ))
            # NEUTRE
            if 'side' in trades.columns:
                neutral_trades = trades[~trades['side'].str.upper().str.contains('BUY|SELL', na=False)]
                if not neutral_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=neutral_trades.get("entry_time", neutral_trades.index),
                        y=neutral_trades.get("entry_price", neutral_trades.get("exit_price", np.nan)),
                        mode="markers",
                        name="Signaux NEUTRE",
                        marker=dict(color='gray', symbol='circle', size=10),
                        hovertemplate="<b>Signal NEUTRE</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Prix: %{y:,.2f} €<br>Score: %{customdata[0]:.2f}<br>Setup: %{customdata[1]}<extra></extra>",
                        customdata=np.stack([
                            neutral_trades.get("confidence_score", [None]),
                            neutral_trades.get("setup_quality", [None])
                        ], axis=-1) if "confidence_score" in neutral_trades and "setup_quality" in neutral_trades else None
                    ))
            fig.update_layout(title="Prix & Signaux de Trading", xaxis_title="Date", yaxis_title="Prix", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun trade ou signal disponible pour cette combinaison.")
    else:
        st.info("Sélectionnez une stratégie et un actif.")

# --- Tab 3: Performances détaillées ---
with tab3:
    st.header("Performances détaillées")
    if selected_strategy and selected_asset:
        if is_multi_strategy(backtest_result):
            stats = backtest_result["performance_by_strategy_and_asset"][selected_strategy][selected_asset]
        else:
            stats = backtest_result["performance_by_asset"][selected_asset]
        st.json(stats)
    else:
        st.info("Sélectionnez une stratégie et un actif.")

# --- Tab 4: Risques ---
with tab4:
    st.header("Analyse des Risques")
    if selected_strategy and selected_asset:
        if is_multi_strategy(backtest_result):
            drawdown_curves = backtest_result.get("drawdown_curves", {}).get(selected_strategy, {})
            drawdown_curve = drawdown_curves.get(selected_asset)
        else:
            drawdown_curve = backtest_result.get("drawdown_curves", {}).get(selected_asset)
        if drawdown_curve is not None:
            drawdown_curve = drawdown_curve[(drawdown_curve.index >= pd.to_datetime(start_date)) & (drawdown_curve.index <= pd.to_datetime(end_date))]
            df_dd = pd.DataFrame({"Date": drawdown_curve.index, "Drawdown": drawdown_curve.values, "Stratégie": selected_strategy, "Actif": selected_asset})
            hovertemplate = (
                "<b>Date</b>: %{x|%Y-%m-%d %H:%M}<br>"
                "<b>Drawdown</b>: %{y:,.2f} €<br>"
                f"<b>Stratégie</b>: {selected_strategy}<br>"
                f"<b>Actif</b>: {selected_asset}"
            )
            fig = px.area(df_dd, x="Date", y="Drawdown", labels={"x": "Date", "y": "Drawdown (€)"}, title="Courbe de Drawdown", color_discrete_sequence=palette)
            fig.update_traces(hovertemplate=hovertemplate)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de courbe de drawdown disponible pour cette combinaison.")
    else:
        st.info("Sélectionnez une stratégie et un actif.")

# --- Tab 5: Backtesting (Raw Data) ---
with tab5:
    st.header("Données brutes de Backtest")
    st.write(backtest_result)

# --- Footer ---
st.markdown("© 2025 - Dashboard de Trading - Développé avec Streamlit")