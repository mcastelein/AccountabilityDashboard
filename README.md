🏋️ Exercise Accountability Dashboard

A fully interactive Streamlit dashboard that helps groups stay accountable with their weekly exercise goals. Built for simplicity, speed, and insight.

![Dashboard Demo](dashboard_gif.gif)

🚀 Features:

📊 Weekly Balance Heatmap
- Visualize how each member is doing week by week
- Interactive filters: hide inactive people, select custom start dates
- Beautiful Santorini-style color palette for clean presentation

🔥 Streak Leaderboard
- Tracks current streaks based on consecutive 100% completed weeks
- Automatically filters to only include active members
- Horizontal bar chart with hover tooltips

🔎 Participant Drilldown
- Select any person to see a week-by-week line chart of their performance
- Shows total weeks, 100% completions, current streak, and average completion
- Cleanly formatted percentage history in table format

💰 Balance Summary Table
- Tracks financial balance for each member based on exercise and transactions
- Interactive toggle to hide people with no meaningful balance
- Displays all amounts with 2 decimal places

🧩 Tech Stack
- Streamlit – UI and dashboard framework
- Pandas & NumPy – data manipulation
- Plotly – interactive charts and tables
- Matplotlib + Seaborn – custom heatmap visualization
- Supabase Postgres (project Side Projects, schema `accountability`): source of truth, written live by the Accountability Bot. Set `database_url` in Streamlit secrets (read-only role).

🏃‍♂️‍➡️ How to Run
- It's deployed as a webapp, so in order to see if for yourself, all you have to do is click this link:
- https://exercise-accountability-dashboard.streamlit.app/
