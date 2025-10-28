import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

# Set working directory if needed (optional)
# os.chdir(r'C:\Users\mpcas\Documents\AchieverzClub\python')

st.set_page_config(page_title="Exercise Accountability Dashboard", layout="wide")
st.title("🏋️ Exercise Accountability Dashboard")

# Load data from Google Sheets
@st.cache_data
def load_data():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1252I5iC7Dy3hMtQ0lDEVCxAf0B7jWFSoErll8rePOes/export?format=csv&gid=142314539'
    df = pd.read_csv(sheet_url, header=1, usecols=[0, 1, 2, 3, 4, 5, 6])
    df = df.rename(columns={'Week #':'Week','Exercise Group ID':'Group'})
    df['Fail Tax'] = np.nan
    df['Fail Money'] = np.nan
    return df



# Manual refresh button
col1, col2, col3 = st.columns([6, 1, 1])  # Adjust column widths
with col3:
    if st.button("🔄 Refresh Weekly Data"):
        st.cache_data.clear()

df = load_data()

@st.cache_data
def load_transactions():
    transactions_sheet_url = 'https://docs.google.com/spreadsheets/d/1252I5iC7Dy3hMtQ0lDEVCxAf0B7jWFSoErll8rePOes/export?format=csv&gid=422897383'
    transactions_df = pd.read_csv(transactions_sheet_url,usecols=[0, 1, 2, 3, 4, 5, 6])
    return transactions_df

# Helper function for balance calculation
def calculate_balance(df, group_id, month, week):
    one_week_glance = df[
        (df['Group'] == group_id) & 
        (df['Month'] == month) & 
        (df['Week'] == week)
    ].copy()
    one_week_glance['Balance'] = (
        (one_week_glance['Completed'] + one_week_glance['Skipped']) * 25.0
    ).astype('float64')
    if 'Fail Money' not in df.columns:
        df['Fail Money'] = np.nan
    fail_total = one_week_glance['Failed'].sum()
    completed_total = one_week_glance['Completed'].sum()
    for person in one_week_glance['Person'].unique():
        person_mask = one_week_glance['Person'] == person
        person_completed = one_week_glance.loc[person_mask, 'Completed'].sum()
        if completed_total == 0:
            fail_money = np.nan
        else:
            fail_money = fail_total * 25 * person_completed / completed_total
        one_week_glance.loc[person_mask, 'Fail_Money'] = fail_money
        df.loc[
            (df['Person'] == person) &
            (df['Month'] == month) &
            (df['Group'] == group_id) &
            (df['Week'] == week),
            'Fail Money'
        ] = fail_money
    one_week_glance['Balance'] = (
        (one_week_glance['Completed'] + one_week_glance['Skipped']) * 25.0 + 
        one_week_glance['Fail_Money']
    ).astype('float64')
    return one_week_glance

# Helper function for streak calculation
def calculate_streak(df, person, group_id):
    person_df = df[(df['Person'] == person) & (df['Group'] == group_id)]
    streak = 0
    for completed in reversed(person_df['Completed'].tolist()):
        if completed == 1:
            streak += 1
        else:
            break
    return streak

def main():
    df = load_data().copy()
    transactions_df = load_transactions().copy()
    group_options = sorted(df['Group'].dropna().unique())
    group_number = st.selectbox("Select Exercise Group:", group_options, index=0)

    # Drop January, February & March for group 1
    group_df = df[df['Group'] == group_number].copy()
    if group_number == 1:
        group_df = group_df[~group_df['Month'].isin(['January', 'February', 'March'])]

    # Month order
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    present_months = [m for m in month_order if m in df['Month'].unique()]
    group_df['Month'] = pd.Categorical(group_df['Month'], categories=present_months, ordered=True)








    # --- Weekly Balance Section ---
    st.subheader(f"Weekly Balance Over Time by Person (Group {group_number})")
    # Move month/week selectors here
    available_months = [m for m in month_order if m in group_df['Month'].unique()]
    start_month = st.selectbox("Select starting month:", available_months, index=0)
    available_weeks_in_month = sorted(group_df[group_df['Month'] == start_month]['Week'].unique())
    start_week = st.selectbox("Select starting week:", available_weeks_in_month, index=0)
    hide_inactive = st.checkbox("Hide inactive people (no data in latest week)", value=True)

        # Calculate balances for all weeks
    for group in df['Group'].unique():
        for month in df[df['Group'] == group]['Month'].unique():
            for week in df[(df['Month'] == month) & (df['Group'] == group)]['Week'].unique():
                one_week_result = calculate_balance(df, group, month, week)
                for idx, row in one_week_result.iterrows():
                    df.loc[
                        (df['Person'] == row['Person']) &
                        (df['Month'] == row['Month']) &
                        (df['Week'] == row['Week']) &
                        (df['Group'] == group),
                        'Balance'
                    ] = row['Balance']

    fail_tax_rate = 0
    df['Fail Tax'] = fail_tax_rate * df['Fail Money']
    if fail_tax_rate == 0 and 'Fail Tax' in df.columns:
        df.drop(columns=['Fail Tax'], inplace=True)

    group_df = df[df['Group'] == group_number].copy()
    if group_number == 1:
        group_df = group_df[~group_df['Month'].isin(['January', 'February', 'March'])]
    group_df['Month'] = pd.Categorical(group_df['Month'], categories=present_months, ordered=True)
    filtered_group_df = group_df[group_df.apply(
    lambda row: (
        month_order.index(row['Month']) > month_order.index(start_month)
        or (
            month_order.index(row['Month']) == month_order.index(start_month)
            and row['Week'] >= start_week
        )
    ), axis=1)]


    # Pivot for heatmap
    balance_table = filtered_group_df.pivot_table(
        index=['Month', 'Week'],
        columns='Person',
        values='Balance',
        aggfunc='first'
    )
    
    month_sorter = {m: i for i, m in enumerate(month_order)}
    existing = filtered_group_df[['Month', 'Week']].drop_duplicates()
    existing['MonthSort'] = existing['Month'].map(month_sorter)
    existing = existing.sort_values(by=['MonthSort', 'Week']).drop(columns='MonthSort')
    existing_index = pd.MultiIndex.from_frame(existing)
    balance_table = balance_table.loc[existing_index]
        # Remove inactive people if checkbox is checked
    if hide_inactive and not balance_table.empty:
        # Find the most recent (Month, Week) in the filtered data
        most_recent_idx = balance_table.index[-1]
        # Only keep people with non-NaN value in the most recent week
        active_people = balance_table.columns[~balance_table.loc[most_recent_idx].isna()]
        balance_table = balance_table[active_people]

    

    # Sort people by recent streak, count, last activity
    def recent_consecutive_streak(col):
        mask = col.notna().to_numpy()
        streak = 0
        for val in mask[::-1]:
            if val:
                streak += 1
            else:
                break
        return streak
    person_activity = balance_table.notna().sum()
    recent_streak = balance_table.apply(recent_consecutive_streak)
    last_active = balance_table.apply(lambda col: col.last_valid_index()[1] if col.last_valid_index() else -1)
    sort_df = pd.DataFrame({
        'recent_streak': recent_streak,
        'count': person_activity,
        'last': last_active
    })
    sort_order = sort_df.sort_values(
        by=['recent_streak', 'count', 'last'],
        ascending=[False, False, False]
    ).index.tolist()
    sorted_balance_table = balance_table[sort_order]

    santorini_colors = [
        "#ffffff",  # crisp white (buildings)
        "#C2E9FB",  # soft sky blue
        "#6FC3DF",  # mid ocean blue
        "#0F75BC",  # deep Aegean
        "#003F63"   # rich navy
    ]
    santorini_cmap = LinearSegmentedColormap.from_list("Santorini", santorini_colors)


    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        sorted_balance_table,
        annot=True,
        fmt=".1f",
        cmap=santorini_cmap,
        linewidths=0.5,
        linecolor='#d6eaf8',
        mask=sorted_balance_table.isna(),
        cbar_kws={'label': 'Balance (€)', 'shrink': 0.8},
        ax=ax
    )
    ax.set_title(f"Weekly Balance Over Time by Person (Group {group_number})", fontsize=18, color="#003F63", pad=20)
    ax.set_ylabel("Month, Week", fontsize=12)
    ax.set_xlabel("Person", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)











    # --- Streak Leaderboard Section (after filtering) ---
    st.subheader("🔥 Streak Leaderboard 🔥")
    # Find the most recent (Month, Week) in the filtered data
    if not filtered_group_df.empty:
        # Sort by Month and Week to get the most recent
        filtered_group_df = filtered_group_df.sort_values(['Month', 'Week'])
        most_recent_month = filtered_group_df['Month'].iloc[-1]
        most_recent_week = filtered_group_df['Week'].iloc[-1]
        active_people = filtered_group_df[
            (filtered_group_df['Month'] == most_recent_month) &
            (filtered_group_df['Week'] == most_recent_week)
        ]['Person'].unique()
    else:
        active_people = []
    # Calculate streaks for active people
    streak_list = [calculate_streak(df, person, group_number) for person in active_people]

    # Build streak_df once
    streak_df = pd.DataFrame({
        'Person': active_people,
        'Streak': streak_list
    })

    streak_df = streak_df.sort_values(by='Streak', ascending=True)

    # Now plot the same sorted DataFrame
    # santorini_blues = ['#0A2342', '#2E5984', '#4A90E2', '#A7C7E7', '#E6F0FA']
    santorini_blues = ['#4A90E2']
    bar_colors = santorini_blues * (len(streak_df) // len(santorini_blues) + 1)

    streak_fig = go.Figure(go.Bar(
        x=streak_df['Streak'].to_list(),
        y=streak_df['Person'].to_list(),
        orientation='h',
        marker=dict(color=bar_colors[:len(streak_df)]),
        text=streak_df['Streak'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Streak: %{x}<extra></extra>',
    ))

    streak_fig.update_layout(
        title=dict(
            # text='🔥 Streak Leaderboard 🔥',
            font=dict(size=20, color='#0A2342'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Streak',
            # titlefont=dict(color='#0A2342'),
            tickfont=dict(color='#0A2342'),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='',
            tickfont=dict(color='#0A2342'),
            showgrid=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family='Helvetica, sans-serif',
            size=14,
            color='#0A2342'
        ),
        margin=dict(l=100, r=40, t=0, b=40),
        height=400
    )


    st.plotly_chart(streak_fig, use_container_width=True)










    # --- Participant Search Section ---
    st.subheader("🔍 Participant Search")
    
    # Get all people in the selected group
    group_people = group_df['Person'].unique()
    selected_person = st.selectbox("Select a participant to view their completion history:", group_people, index=0)
    
    if selected_person:
        # Filter data for the selected person
        person_data = group_df[group_df['Person'] == selected_person].copy()
        
        # Sort by month and week for proper chronological order
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        person_data['Month'] = pd.Categorical(person_data['Month'], categories=month_order, ordered=True)
        person_data = person_data.sort_values(['Month', 'Week'])
        
        # Ensure Completed is float
        person_data['Completed'] = person_data['Completed'].astype(float)
        
        # Create x labels only for weeks where the participant has data
        x_labels = [f"{row['Month']} W{row['Week']}" for _, row in person_data.iterrows()]

        # Ensure x_labels are unique
        if len(set(x_labels)) != len(x_labels):
            st.warning("Duplicate week labels detected! This can cause plotting issues.")

        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_labels,
            y=person_data['Completed'].tolist(),  # ensure values not Series
            mode='lines+markers',
            name='Completion'
        ))

        fig.update_layout(
            title=f"Completion History for {selected_person} (Group {group_number})",
            xaxis_title='Week',
            yaxis_title='Completion',
            yaxis=dict(range=[0, 1.1])
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_weeks = len(person_data)
        completed_weeks = (person_data['Completed'] == 1).sum()
        current_streak = 0 if selected_person not in active_people else calculate_streak(df, selected_person, group_number)

        avg_completion = person_data['Completed'].mean()
        
        with col1:
            st.metric("Total Weeks", total_weeks)
        with col2:
            st.metric("100% Weeks", completed_weeks)
        with col3:
            st.metric("Current Streak", f"{current_streak} weeks")
        with col4:
            st.metric("Avg Completion", f"{avg_completion:.1%}")
        
        # Show detailed table
        st.write(f"Detailed History for {selected_person}")
        display_data = person_data[['Month', 'Week', 'Completed', 'Skipped', 'Failed']].copy()
        display_data['Completed'] = display_data['Completed'].apply(lambda x: f"{x:.0%}")
        display_data['Skipped'] = display_data['Skipped'].apply(lambda x: f"{x:.0%}")
        display_data['Failed'] = display_data['Failed'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(display_data)













    # Total balance table
    st.subheader("🔷 Total Balance by Person 🔷")
    hide_no_balance = st.checkbox("Hide people with no balance", value=True)

    total_balance = pd.DataFrame(columns=['Person','Exercise Balance','Transaction Balance','Total Balance'])
    group_transactions = transactions_df[transactions_df['Group']==group_number]
    i=0
    for dude in group_df['Person'].unique():
        person_balance = group_df[group_df['Person']==dude]['Balance']
        person_exercise_balance = person_balance.sum()-person_balance.count()*25
        person_trans_balance = group_transactions[group_transactions['Who']==dude]['Amount'].sum()
        total_balance.loc[i] = [dude,person_exercise_balance,person_trans_balance,person_exercise_balance+person_trans_balance]
        i=i+1
    total_balance = total_balance.sort_values(by=['Exercise Balance'],ascending=False)

    if hide_no_balance:
        total_balance = total_balance[
            (total_balance['Total Balance'] < -1) | (total_balance['Total Balance'] > 1)
        ]

    
    fig2 = go.Figure(data=[go.Table(
        header=dict(
            values=list(total_balance.columns),
            fill_color='rgb(0, 90, 156)',
            font=dict(color='white', size=14),
            align='center',
            height=40
        ),
        cells=dict(
            values=[
                [
                    f"{x:.2f}" if pd.api.types.is_numeric_dtype(total_balance[col]) else x
                    for x in total_balance[col]
                ]
                for col in total_balance.columns
            ],
            fill_color='rgb(240, 248, 255)',
            font=dict(color='rgb(20, 40, 80)', size=13),
            align='center',
            height=30
        )
    )])
    fig2.update_layout(
        width=800,
        height=600,
        title=dict(
            # text="",
            font=dict(size=20, color='rgb(0, 90, 156)'),
            x=0.5
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
