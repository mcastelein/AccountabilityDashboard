from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import psycopg
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from psycopg.rows import dict_row

# Data now comes from Supabase (project "Side Projects", schema "accountability"), written
# live by the Accountability Bot. Money is computed once, in the database views
# accountability.exercise_money and accountability.wallets, so this app only displays it.
# The connection string lives in Streamlit secrets as database_url (read-only role).

st.set_page_config(page_title="Exercise Accountability Dashboard", layout="wide")
st.title("🏋️ Exercise Accountability Dashboard")

SYMBOL = {"EUR": "€", "USD": "$", "RMB": "¥"}


def week_label(monday: date) -> str:
    """'Oct W1 2026': the week belongs to the month of its Thursday."""
    thursday = monday + timedelta(days=3)
    return f"{thursday:%b} W{(thursday.day - 1) // 7 + 1} {thursday.year}"


def query(sql: str) -> pd.DataFrame:
    with psycopg.connect(st.secrets["database_url"], row_factory=dict_row, prepare_threshold=None) as conn:
        return pd.DataFrame(conn.execute(sql).fetchall())


@st.cache_data(ttl=300)
def load_weeks() -> pd.DataFrame:
    df = query("""
        select g.id as group_id, g.name as group_name, g.currency, m.display_name as person,
               w.week_start, w.done, w.target, w.completed_frac, w.skipped_frac, w.failed_frac,
               w.final, w.status, em.balance, em.counts
        from accountability.exercise_weeks w
        join accountability.groups g on g.id = w.group_id
        join accountability.members m on m.id = w.member_id
        left join accountability.exercise_money em
               on em.member_id = w.member_id and em.week_start = w.week_start
        where not m.pending
        order by w.week_start, m.display_name""")
    for col in ("done", "target", "completed_frac", "skipped_frac", "failed_frac", "balance"):
        df[col] = pd.to_numeric(df[col])
    df["Week"] = df["week_start"].map(week_label)
    return df


@st.cache_data(ttl=300)
def load_wallets() -> pd.DataFrame:
    df = query("""select group_id, display_name as person, exercise_balance, transaction_balance, wallet
                  from accountability.wallets""")
    for col in ("exercise_balance", "transaction_balance", "wallet"):
        df[col] = pd.to_numeric(df[col])
    return df


col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

santorini_colors = ["#ffffff", "#C2E9FB", "#6FC3DF", "#0F75BC", "#003F63"]
santorini_cmap = LinearSegmentedColormap.from_list("Santorini", santorini_colors)


def calculate_streak(person_df: pd.DataFrame) -> int:
    streak = 0
    for completed in reversed(person_df.sort_values("week_start")["completed_frac"].tolist()):
        if completed >= 1:
            streak += 1
        else:
            break
    return streak


def main():
    weeks = load_weeks()
    wallets = load_wallets()
    groups = weeks[["group_id", "group_name", "currency"]].drop_duplicates().sort_values("group_id")
    group_name = st.selectbox("Select Exercise Group:", groups["group_name"].tolist(), index=0)
    group = groups[groups["group_name"] == group_name].iloc[0]
    sym = SYMBOL.get(group["currency"], group["currency"] + " ")
    group_df = weeks[weeks["group_id"] == group["group_id"]].copy()
    final_df = group_df[group_df["final"]]

    # --- This week so far ---
    current = group_df[~group_df["final"]]
    if not current.empty:
        monday = current["week_start"].max()
        st.subheader(f"📅 This Week So Far ({week_label(monday)})")
        cur = current[current["week_start"] == monday].sort_values("completed_frac")
        labels = [(f"{d:g}/{t:g}" if pd.notna(t) else "") + (" (sick/skip)" if s == "sick/skip" else "")
                  for d, t, s in zip(cur["done"], cur["target"], cur["status"])]
        fig = go.Figure(go.Bar(
            x=cur["completed_frac"].tolist(), y=cur["person"].tolist(), orientation="h",
            marker=dict(color="#4A90E2"), text=labels, textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>"))
        fig.update_layout(xaxis=dict(range=[0, 1.15], tickformat=".0%", title=""), yaxis=dict(title=""),
                          plot_bgcolor="white", paper_bgcolor="white", height=max(250, 45 * len(cur)),
                          margin=dict(l=100, r=40, t=10, b=30))
        st.plotly_chart(fig, width="stretch")

    # --- Weekly Balance Section ---
    st.subheader(f"Weekly Balance Over Time by Person ({group_name})")
    money_df = final_df[final_df["counts"].fillna(False).astype(bool)]
    week_order = money_df[["week_start", "Week"]].drop_duplicates().sort_values("week_start")
    if week_order.empty:
        st.warning("⚠️ No finished weeks yet.")
    else:
        start = st.selectbox("Select starting week:", week_order["Week"].tolist(), index=0)
        start_date = week_order[week_order["Week"] == start]["week_start"].iloc[0]
        hide_inactive = st.checkbox("Hide inactive people (no data in latest week)", value=True)
        shown = money_df[money_df["week_start"] >= start_date]
        balance_table = shown.pivot_table(index="Week", columns="person", values="balance", aggfunc="first")
        ordered = week_order[week_order["week_start"] >= start_date]["Week"]
        balance_table = balance_table.reindex(ordered)
        if hide_inactive and not balance_table.empty:
            balance_table = balance_table[balance_table.columns[~balance_table.iloc[-1].isna()]]

        def recent_consecutive_streak(col):
            streak = 0
            for val in col.notna().to_numpy()[::-1]:
                if not val:
                    break
                streak += 1
            return streak

        sort_df = pd.DataFrame({"recent_streak": balance_table.apply(recent_consecutive_streak),
                                "count": balance_table.notna().sum()})
        balance_table = balance_table[sort_df.sort_values(["recent_streak", "count"], ascending=False).index]
        if balance_table.empty or balance_table.isna().all().all():
            st.warning("⚠️ No valid balance data available for the selected date range.")
        else:
            fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(balance_table))))
            sns.heatmap(balance_table, annot=True, fmt=".1f", cmap=santorini_cmap, linewidths=0.5,
                        linecolor="#d6eaf8", mask=balance_table.isna(),
                        cbar_kws={"label": f"Balance ({sym.strip()})", "shrink": 0.8}, ax=ax)
            ax.set_title(f"Weekly Balance Over Time by Person ({group_name})", fontsize=18, color="#003F63", pad=20)
            ax.set_ylabel("Week", fontsize=12)
            ax.set_xlabel("Person", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

    # --- Streak Leaderboard ---
    st.subheader("🔥 Streak Leaderboard 🔥")
    if not final_df.empty:
        latest = final_df["week_start"].max()
        active_people = final_df[final_df["week_start"] == latest]["person"].unique()
        streak_df = pd.DataFrame({
            "Person": active_people,
            "Streak": [calculate_streak(final_df[final_df["person"] == p]) for p in active_people],
        }).sort_values("Streak")
        streak_fig = go.Figure(go.Bar(
            x=streak_df["Streak"].to_list(), y=streak_df["Person"].to_list(), orientation="h",
            marker=dict(color="#4A90E2"), text=streak_df["Streak"], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Streak: %{x}<extra></extra>"))
        streak_fig.update_layout(
            xaxis=dict(title="Streak", tickfont=dict(color="#0A2342"), gridcolor="lightgray"),
            yaxis=dict(title="", tickfont=dict(color="#0A2342"), showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Helvetica, sans-serif", size=14, color="#0A2342"),
            margin=dict(l=100, r=40, t=0, b=40), height=400)
        st.plotly_chart(streak_fig, width="stretch")
    else:
        active_people = []

    # --- Participant Search ---
    st.subheader("🔍 Participant Search")
    people = sorted(group_df["person"].unique())
    selected = st.selectbox("Select a participant to view their completion history:", people, index=0)
    if selected:
        pdata = group_df[group_df["person"] == selected].sort_values("week_start")
        fig = go.Figure(go.Scatter(x=pdata["Week"].tolist(), y=pdata["completed_frac"].tolist(),
                                   mode="lines+markers", name="Completion"))
        fig.update_layout(title=f"Completion History for {selected} ({group_name})",
                          xaxis_title="Week", yaxis_title="Completion", yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig, width="stretch")

        pfinal = pdata[pdata["final"]]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Weeks", len(pfinal))
        c2.metric("100% Weeks", int((pfinal["completed_frac"] >= 1).sum()))
        c3.metric("Current Streak", f"{calculate_streak(pfinal) if selected in active_people else 0} weeks")
        c4.metric("Avg Completion", f"{pfinal['completed_frac'].mean():.1%}" if len(pfinal) else "n/a")

        st.write(f"Detailed History for {selected}")
        table = pdata[["Week", "done", "target", "completed_frac", "skipped_frac", "failed_frac", "status"]].copy()
        table.columns = ["Week", "Done", "Target", "Completed", "Skipped", "Failed", "Status"]
        for col in ("Completed", "Skipped", "Failed"):
            table[col] = table[col].apply(lambda x: f"{x:.0%}")
        for col in ("Done", "Target"):
            table[col] = table[col].apply(lambda x: f"{x:g}" if pd.notna(x) else "")
        table["Status"] = table["Status"].replace({"sheet": "from sheet"})
        st.dataframe(table, hide_index=True)

    # --- Total Balance ---
    st.subheader("🔷 Total Balance by Person 🔷")
    hide_no_balance = st.checkbox("Hide people with no balance", value=True)
    total = wallets[wallets["group_id"] == group["group_id"]][
        ["person", "exercise_balance", "transaction_balance", "wallet"]].copy()
    total.columns = ["Person", "Exercise Balance", "Transaction Balance", "Total Balance"]
    total = total.sort_values("Exercise Balance", ascending=False)
    if hide_no_balance:
        total = total[(total["Total Balance"] < -1) | (total["Total Balance"] > 1)]
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=list(total.columns), fill_color="rgb(0, 90, 156)",
                    font=dict(color="white", size=14), align="center", height=40),
        cells=dict(values=[[f"{x:.2f}" if col != "Person" else x for x in total[col]] for col in total.columns],
                   fill_color="rgb(240, 248, 255)", font=dict(color="rgb(20, 40, 80)", size=13),
                   align="center", height=30))])
    fig2.update_layout(width=800, height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig2, width="stretch")


if __name__ == "__main__":
    main()
