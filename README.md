🏋️ Exercise Accountability Dashboard
A fully interactive Streamlit dashboard that helps groups stay accountable with their weekly exercise goals. Built for simplicity, speed, and insight.

<!-- Replace with your actual GIF path or link -->

🚀 Features
📊 Weekly Balance Heatmap
Visualize how each member is doing week by week

Interactive filters: hide inactive people, select custom start dates

Beautiful Santorini-style color palette for clean presentation

🔥 Streak Leaderboard
Tracks current streaks based on consecutive 100% completed weeks

Automatically filters to only include active members

Horizontal bar chart with hover tooltips

🔎 Participant Drilldown
Select any person to see a week-by-week line chart of their performance

Shows total weeks, 100% completions, current streak, and average completion

Cleanly formatted percentage history in table format

💰 Balance Summary Table
Tracks financial balance for each member based on exercise and transactions

Interactive toggle to hide people with no meaningful balance

Displays all amounts with 2 decimal places

🧩 Tech Stack
Streamlit – UI and dashboard framework

Pandas & NumPy – data manipulation

Plotly – interactive charts and tables

Matplotlib + Seaborn – custom heatmap visualization

Google Sheets – source of truth for exercise + transactions

🛠 Setup Instructions
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/exercise-dashboard.git
cd exercise-dashboard

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run exercise_dashboard_streamlit.py
📁 Folder Structure
bash
Copy
Edit
├── exercise_dashboard_streamlit.py  # Main dashboard file
├── requirements.txt
├── README.md
└── demo.gif  # Optional: a recording of dashboard usage
📎 Dependencies
Ensure you have the following in your requirements.txt:

txt
Copy
Edit
streamlit
pandas
numpy
matplotlib
seaborn
plotly
🧠 Tips
You can customize the Google Sheets source by editing the URLs inside load_data() and load_transactions().

For production deployments, consider hiding sensitive sheet URLs via secrets or environment variables.

📸 Screenshot
(Optional if you don't have a GIF)
