import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pdfkit
import plotly.graph_objects as go

os.chdir(r'C:\Users\mpcas\Documents\AchieverzClub\python')

def run_full_dashboard(group_number):

    sheet_url = 'https://docs.google.com/spreadsheets/d/1252I5iC7Dy3hMtQ0lDEVCxAf0B7jWFSoErll8rePOes/export?format=csv&gid=142314539'
    df = pd.read_csv(sheet_url, header=1, usecols=[0, 1, 2, 3, 4, 5, 6])

    # df = pd.read_csv('accountability_data.csv')
    # df['Completed'] = df['Completed'].str.rstrip('%').astype(float) / 100.0
    # df['Skipped'] = df['Skipped'].str.rstrip('%').astype(float) / 100.0
    # df['Failed'] = df['Failed'].str.rstrip('%').astype(float) / 100.0
    df = df.rename(columns={'Week #':'Week','Exercise Group ID':'Group'})
    # df['Fail Money'] = 0.0
    df['Fail Tax'] = np.nan
    df['Fail Money'] = np.nan

    # df['Balance'] = np.nan

    def calculate_balance(df, group_id, month, week):
        # Filter and copy the relevant week
        one_week_glance = df[
            (df['Group'] == group_id) & 
            (df['Month'] == month) & 
            (df['Week'] == week)
        ].copy()

        # Calculate balance
        one_week_glance['Balance'] = (
            (one_week_glance['Completed'] + one_week_glance['Skipped']) * 25.0
        ).astype('float64')

        # Ensure 'Fail Money' column exists
        if 'Fail Money' not in df.columns:
            df['Fail Money'] = np.nan

        # Distribute fail money
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


    fail_tax_rate = 0 # Eventually the goal will be to make this 10% or 25%
    df['Fail Tax'] = fail_tax_rate*df['Fail Money']  


    if fail_tax_rate==0:
        df.drop(columns=['Fail Tax'], inplace=True)

    group1 = df[df['Group']==group_number]
    # group2 = df[df['Group']==2]
    # group3 = df[df['Group']==3]
    # group4 = df[df['Group']==4]

    # Drop January, February & March
    if group_number==1:
        group1 = group1[~group1['Month'].isin(['January', 'February', 'March'])]

    # Define proper month order
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Convert Month column to a categorical type with the correct order
    # group3['Month'] = pd.Categorical(group3['Month'], categories=month_order, ordered=True)
    # Only use the months that actually appear in the data
    present_months = [m for m in month_order if m in df['Month'].unique()]
    group1['Month'] = pd.Categorical(group1['Month'], categories=present_months, ordered=True)


    # Create the pivot
    balance_table = group1.pivot_table(
        index=['Month', 'Week'],
        columns='Person',
        values='Balance',
        aggfunc='first'
        # fill_value=0
    )

    # Filter to only combinations that actually exist
    # Create sorting key for Month
    month_sorter = {m: i for i, m in enumerate(month_order)}
    existing = group1[['Month', 'Week']].drop_duplicates()
    existing['MonthSort'] = existing['Month'].map(month_sorter)

    # Now sort
    existing = existing.sort_values(by=['MonthSort', 'Week']).drop(columns='MonthSort')


    # existing = existing.sort_values(by=["Month", "Week"], key=lambda col: col.map({m: i for i, m in enumerate(month_order)} if col.name == "Month" else None))
    existing_index = pd.MultiIndex.from_frame(existing)

    # Reindex with only actual data combinations
    balance_table = balance_table.loc[existing_index]


    # Count non-NaN entries per person
    person_activity = balance_table.notna().sum()

    # Get last week each person was active (NaN otherwise)
    last_active = balance_table.apply(lambda col: col.last_valid_index()[1] if col.last_valid_index() else -1)

    # Get first week each person was active
    first_active = balance_table.apply(lambda col: col.first_valid_index()[1] if col.first_valid_index() else float('inf'))

    # Combine into a DataFrame
    sort_df = pd.DataFrame({
        'count': person_activity,
        'last': last_active,
        'first': first_active
    })

    # Custom sort: more activity → earlier join → later drop
    sort_order = sort_df.sort_values(by=['last','count' ,'first'], ascending=[True, False, False]).index.tolist()

    # Reorder the balance_table columns
    sorted_balance_table = balance_table[sort_order]

    # plt.figure(figsize=(12, 8))
    # sns.heatmap(
    #     sorted_balance_table,
    #     annot=True,
    #     fmt=".1f",
    #     cmap="YlGnBu",
    #     linewidths=0.5,
    #     linecolor='gray',
    #     mask=sorted_balance_table.isna(),
    #     cbar_kws={'label': 'Balance (€)'}
    # )
    # plt.title("Weekly Balance Over Time by Person (Group 1)")
    # plt.ylabel("Month, Week")
    # plt.xlabel("Person")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("group1_balance_heatmap.png", bbox_inches="tight")


    santorini_colors = [
        "#ffffff",  # crisp white (buildings)
        "#C2E9FB",  # soft sky blue
        "#6FC3DF",  # mid ocean blue
        "#0F75BC",  # deep Aegean
        "#003F63"   # rich navy
    ]

    santorini_cmap = LinearSegmentedColormap.from_list("Santorini", santorini_colors)

    # Plot heatmap with custom style
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        sorted_balance_table,
        annot=True,
        fmt=".1f",
        cmap=santorini_cmap,
        linewidths=0.5,
        linecolor='#d6eaf8',
        mask=sorted_balance_table.isna(),
        cbar_kws={'label': 'Balance (€)', 'shrink': 0.8}
    )

    # Set font and layout styling
    plt.title(f"Weekly Balance Over Time by Person (Group {group_number})", fontsize=18, color="#003F63", pad=20)
    plt.ylabel("Month, Week", fontsize=12)
    plt.xlabel("Person", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("group1_balance_heatmap_santorini.png", bbox_inches="tight")
    plt.show()

    total_balance = pd.DataFrame(columns=['Person','Exercise Balance','Transaction Balance','Total Balance'])

    transactions_sheet_url = 'https://docs.google.com/spreadsheets/d/1252I5iC7Dy3hMtQ0lDEVCxAf0B7jWFSoErll8rePOes/export?format=csv&gid=422897383'
    transactions_df = pd.read_csv(transactions_sheet_url,usecols=[0, 1, 2, 3, 4, 5, 6])
    transaction_balance = pd.DataFrame(columns=['Person',])
    group1_transactions = transactions_df[transactions_df['Group']==group_number]

    i=0
    for dude in group1['Person'].unique():
        person_balance = group1[group1['Person']==dude]['Balance']
        person_exercise_balance = person_balance.sum()-person_balance.count()*25
        person_trans_balance = group1_transactions[group1_transactions['Who']==dude]['Amount'].sum()
        total_balance.loc[i] = [dude,person_exercise_balance,person_trans_balance,person_exercise_balance+person_trans_balance]
        i=i+1

    total_balance = total_balance.sort_values(by=['Total Balance'],ascending=False)
    # total_balance



    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(total_balance.columns),
            fill_color='rgb(0, 90, 156)',  # deep Aegean blue
            font=dict(color='white', size=14),
            align='center',
            height=40
        ),
        cells=dict(
            values=[total_balance[col].round(2) for col in total_balance.columns],
            fill_color='rgb(240, 248, 255)',  # Santorini sky / whitewashed buildings
            font=dict(color='rgb(20, 40, 80)', size=13),
            align='center',
            height=30
        )
    )])

    fig.update_layout(
        width=800,
        height=600,
        title=dict(
            text="🔷 Total Balance by Person 🔷",
            font=dict(size=20, color='rgb(0, 90, 156)'),
            x=0.5
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    fig.show()

run_full_dashboard(1)