
from pathlib import Path
from datetime import datetime

import joblib
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GreenPlate AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "food_waste_model.pkl"
DATA_PATH = BASE_DIR / "cleaned_train.csv"
DB_PATH = BASE_DIR / "greenplate_history.db"

FEATURE_COLUMNS = [
    "meals_served",
    "kitchen_staff",
    "temperature_C",
    "humidity_percent",
    "day_of_week",
    "special_event",
    "past_waste_kg",
    "staff_experience",
    "waste_category"
]

def get_db_connection():
    """Open the local SQLite database used for persistent prediction history."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_db():
    """Create the prediction-history table if it does not exist."""
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                meals_served INTEGER NOT NULL,
                day TEXT NOT NULL,
                category TEXT NOT NULL,
                predicted_waste_kg REAL NOT NULL
            )
            """
        )


def save_prediction(timestamp, meals_served, day, category, predicted_waste):
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
            (created_at, meals_served, day, category, predicted_waste_kg)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, int(meals_served), day, category, float(predicted_waste))
        )


def read_prediction_history():
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT created_at AS "Date and Time",
                   meals_served AS "Meals Served",
                   day AS "Day",
                   category AS "Food Category",
                   ROUND(predicted_waste_kg, 2) AS "Predicted Waste (kg)"
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def clear_prediction_history():
    with get_db_connection() as connection:
        connection.execute("DELETE FROM prediction_history")


initialize_history_db()


DAY_OPTIONS = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


# --------------------------------------------------
# 2. LOAD MODEL AND DATASET
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "food_waste_kg",
        "meals_served",
        "waste_category"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    return df


try:
    model = load_model()
    data = load_dataset()
    app_ready = True
except Exception as error:
    model = None
    data = None
    app_ready = False


# --------------------------------------------------
# 3. CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F6F8F5;
        color: #26382F;
    }

    [data-testid="stSidebar"] {
        background-color: #173E32;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #173E32 !important;
    }

    p, label, .stMarkdown {
        color: #34483D;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 18px;
        border: 1px solid #DDE7DE;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #52665A !important;
    }

    [data-testid="stMetricValue"] {
        color: #173E32 !important;
    }

    .hero {
        background: linear-gradient(120deg, #E0EFE3, #F5F8F3);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #D5E5D7;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 16px;
        margin-bottom: 0;
    }

    .tip-card {
        background-color: #FFFFFF;
        border: 1px solid #DDE7DE;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    div.stButton > button {
        background-color: #245B43;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background-color: #347552;
        color: #FFFFFF !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #245B43;
        color: #FFFFFF !important;
        border-radius: 8px;
        min-height: 44px;
        border: none;
    }

    hr {
        border-color: #DDE7DE;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌿 GreenPlate AI")
    st.caption("Smart Food Waste Management")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Food Waste",
            "Prediction History",
            "Prevention Tips"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("Machine Learning Project")
    st.caption("Predict smarter. Waste less.")


# --------------------------------------------------
# 5. CHECK FILES
# --------------------------------------------------

if not app_ready:
    st.title("🌿 GreenPlate AI")
    st.error(
        "The app could not load the model or dataset."
    )
    st.markdown(
        "Check that these two files are in the same "
        "folder as app.py:"
    )
    st.code(
        "food_waste_model.pkl\n"
        "cleaned_train.csv"
    )
    st.info(
        "If the files exist, check the error details "
        "in the VS Code terminal."
    )
    st.stop()


# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🌿 GreenPlate AI</h1>
        <p>
            Predict smarter. Prepare better. Waste less.
            <br>
            An ML-based food waste prediction system
            for institutional canteens.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Canteen Overview")

    st.caption(
        "Statistics below describe your historical "
        "dataset, not live canteen measurements."
    )

    total_records = len(data)
    total_waste = data["food_waste_kg"].sum()
    average_waste = data["food_waste_kg"].mean()
    average_meals = data["meals_served"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        f"{total_records:,}"
    )
    c2.metric(
        "Recorded Waste",
        f"{total_waste:,.1f} kg"
    )
    c3.metric(
        "Average Waste",
        f"{average_waste:.2f} kg"
    )
    c4.metric(
        "Average Meals Served",
        f"{average_meals:.0f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Waste by Food Category")

        category_data = (
            data.groupby("waste_category", as_index=False)
            ["food_waste_kg"]
            .mean()
            .sort_values("food_waste_kg", ascending=False)
        )

        fig_category = px.bar(
            category_data,
            x="waste_category",
            y="food_waste_kg",
            color="food_waste_kg",
            color_continuous_scale=[
                "#C5E1CD",
                "#245B43"
            ],
            labels={
                "waste_category": "Food category",
                "food_waste_kg": "Average waste (kg)"
            }
        )

        fig_category.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_color="#26382F",
            showlegend=False
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    with right:
        st.subheader("Meals Served vs Waste")

        fig_scatter = px.scatter(
            data,
            x="meals_served",
            y="food_waste_kg",
            color_discrete_sequence=["#287A54"],
            opacity=0.7,
            labels={
                "meals_served": "Meals served",
                "food_waste_kg": "Food waste (kg)"
            }
        )

        fig_scatter.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_color="#26382F"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    # Additional visualizations built from the real dataset.
    st.divider()
    st.subheader("More Food-Waste Insights")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Average waste by weekday")
        if "day_of_week" in data.columns:
            weekday_map = {
                0: "Monday", 1: "Tuesday", 2: "Wednesday",
                3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
            }
            weekday_data = data.copy()
            weekday_data["Weekday"] = weekday_data["day_of_week"].map(weekday_map)
            weekday_data = (
                weekday_data.groupby("Weekday", as_index=False)["food_waste_kg"]
                .mean()
            )
            weekday_order = list(weekday_map.values())
            weekday_data["Weekday"] = pd.Categorical(
                weekday_data["Weekday"], categories=weekday_order, ordered=True
            )
            weekday_data = weekday_data.sort_values("Weekday")
            fig_weekday = px.line(
                weekday_data, x="Weekday", y="food_waste_kg",
                markers=True, line_shape="spline",
                labels={"Weekday": "Day of week", "food_waste_kg": "Average waste (kg)"}
            )
            fig_weekday.update_traces(line_color="#287A54", marker_size=9)
            fig_weekday.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font_color="#26382F", yaxis_gridcolor="#E4ECE5"
            )
            st.plotly_chart(fig_weekday, use_container_width=True)
        else:
            st.info("Weekday chart needs the day_of_week column.")

    with chart_right:
        st.markdown("#### Share of recorded waste by category")
        category_share = (
            data.groupby("waste_category", as_index=False)["food_waste_kg"]
            .sum()
            .sort_values("food_waste_kg", ascending=False)
        )
        fig_donut = px.pie(
            category_share, names="waste_category", values="food_waste_kg",
            hole=0.58, color_discrete_sequence=[
                "#245B43", "#4E8B69", "#8BB99A", "#C5E1CD", "#E4EFE6", "#A6C7B0"
            ],
            labels={"waste_category": "Food category", "food_waste_kg": "Recorded waste (kg)"}
        )
        fig_donut.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#26382F", legend_title_text="Food category"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("#### Food waste distribution")
    fig_hist = px.histogram(
        data, x="food_waste_kg", nbins=25,
        labels={"food_waste_kg": "Recorded food waste (kg)"},
        color_discrete_sequence=["#4E8B69"]
    )
    fig_hist.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#26382F", yaxis_title="Number of records",
        xaxis_gridcolor="#E4ECE5", yaxis_gridcolor="#E4ECE5"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "Charts summarize historical dataset records; they are not live measurements."
    )

    st.divider()

    st.subheader("How GreenPlate AI Works")

    step1, step2, step3 = st.columns(3)

    step1.markdown(
        "**01 · Enter details**\n\n"
        "Provide expected meals, staff and kitchen conditions."
    )

    step2.markdown(
        "**02 · Predict waste**\n\n"
        "Use the trained ML model to estimate food waste."
    )

    step3.markdown(
        "**03 · Prevent waste**\n\n"
        "Use the estimate to improve meal preparation."
    )


# --------------------------------------------------
# 7. FOOD WASTE PREDICTION
# --------------------------------------------------

elif page == "Predict Food Waste":

    st.title("Food Waste Prediction")

    st.write(
        "Enter the expected canteen conditions "
        "to estimate food waste in kilograms."
    )

    experience_options = [
        "beginner",
        "intermediate",
        "experienced"
    ]

    if "staff_experience" in data.columns:
        available_experience = (
            data["staff_experience"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )

        experience_options = sorted(
            set(experience_options + available_experience)
        )

    category_options = sorted(
        data["waste_category"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    with st.form("prediction_form"):

        st.subheader("Canteen Information")

        col1, col2 = st.columns(2)

        with col1:
            meals_served = st.number_input(
                "Expected meals served",
                min_value=1,
                max_value=10000,
                value=200,
                step=10
            )

            kitchen_staff = st.number_input(
                "Kitchen staff",
                min_value=1,
                max_value=500,
                value=10,
                step=1
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=28.0,
                step=0.5
            )

            humidity = st.slider(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )

        with col2:
            day_name = st.selectbox(
                "Day of week",
                list(DAY_OPTIONS.keys())
            )

            special_event = st.selectbox(
                "Is there a special event?",
                ["No", "Yes"]
            )

            past_waste = st.number_input(
                "Previous food waste (kg)",
                min_value=0.0,
                max_value=10000.0,
                value=40.0,
                step=1.0
            )

            staff_experience = st.selectbox(
                "Staff experience",
                experience_options
            )

            waste_category = st.selectbox(
                "Food waste category",
                category_options
            )

        submitted = st.form_submit_button(
            "Predict Food Waste",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "meals_served": meals_served,
            "kitchen_staff": kitchen_staff,
            "temperature_C": temperature,
            "humidity_percent": humidity,
            "day_of_week": DAY_OPTIONS[day_name],
            "special_event": (
                1 if special_event == "Yes" else 0
            ),
            "past_waste_kg": past_waste,
            "staff_experience": staff_experience,
            "waste_category": waste_category
        }])

        # Keep the input feature order consistent
        # with the features used during model training.
        input_data = input_data[FEATURE_COLUMNS]

        try:
            prediction = float(
                np.asarray(model.predict(input_data)).ravel()[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            # Food waste cannot be a negative quantity.
            prediction = max(0.0, prediction)

            st.session_state["latest_prediction"] = {
                "predicted_waste_kg": prediction,
                "meals_served": int(meals_served),
                "day": day_name,
                "category": waste_category,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }

            # Save to SQLite so the record remains after app restarts.
            save_prediction(
                timestamp=st.session_state["latest_prediction"]["timestamp"],
                meals_served=meals_served,
                day=day_name,
                category=waste_category,
                predicted_waste=prediction
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

    latest = st.session_state.get("latest_prediction")

    if latest is not None:
        st.divider()
        st.subheader("Your Prediction")

        result_col, detail_col = st.columns(2)

        result_col.metric(
            "Estimated Food Waste",
            f"{latest['predicted_waste_kg']:.2f} kg"
        )

        detail_col.metric(
            "Expected Meals",
            f"{latest['meals_served']:,}"
        )

        st.caption(
            "This is a prototype estimate from the trained "
            "model, not a verified real-world forecast."
        )

        if latest["predicted_waste_kg"] >= 50:
            st.warning(
                "The predicted waste is relatively high. "
                "Review meal demand, portion sizes and "
                "previous leftovers before preparing food."
            )
        else:
            st.success(
                "Review the estimate alongside actual "
                "leftovers and past meal demand."
            )

        st.info(
            "After the meal, weigh the actual leftovers "
            "to compare them with this prediction."
        )


# --------------------------------------------------
# 8. PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":

    st.title("Prediction History")
    st.write("Review and download predictions saved on this computer.")

    history_df = read_prediction_history()

    if not history_df.empty:
        total_predictions = len(history_df)
        average_prediction = history_df["Predicted Waste (kg)"].mean()
        highest_prediction = history_df["Predicted Waste (kg)"].max()

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predictions Recorded", f"{total_predictions:,}")
        metric2.metric("Average Predicted Waste", f"{average_prediction:.2f} kg")
        metric3.metric("Highest Prediction", f"{highest_prediction:.2f} kg")

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="greenplate_prediction_history.csv",
            mime="text/csv"
        )

        st.subheader("Prediction history trend")
        trend_df = history_df.copy()
        trend_df["Date and Time"] = pd.to_datetime(
            trend_df["Date and Time"], errors="coerce"
        )
        trend_df = trend_df.dropna(subset=["Date and Time"]).sort_values("Date and Time")
        if not trend_df.empty:
            fig_history = px.line(
                trend_df,
                x="Date and Time",
                y="Predicted Waste (kg)",
                markers=True,
                labels={
                    "Date and Time": "Prediction time",
                    "Predicted Waste (kg)": "Predicted waste (kg)"
                }
            )
            fig_history.update_traces(line_color="#287A54", marker_size=9)
            fig_history.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#26382F",
                yaxis_gridcolor="#E4ECE5"
            )
            st.plotly_chart(fig_history, use_container_width=True)

        if st.button("Clear Prediction History"):
            clear_prediction_history()
            st.success("Saved prediction history cleared.")
            st.rerun()
    else:
        st.info(
            "No predictions have been saved yet. Go to Predict Food Waste "
            "and submit the form to create your first record."
        )


# --------------------------------------------------
# 9. PREVENTION TIPS
# --------------------------------------------------

elif page == "Prevention Tips":

    st.title("Food Waste Prevention")
    st.write(
        "Simple actions institutional canteens can take "
        "to reduce avoidable food waste."
    )

    tips = [
        (
            "📊 Plan using demand",
            "Use historical meal counts and previous leftovers "
            "to estimate how much food to prepare."
        ),
        (
            "🥗 Prepare in batches",
            "Where practical and safe, prepare food in smaller "
            "batches and replenish based on actual demand."
        ),
        (
            "⚖️ Measure leftovers",
            "Record the weight of leftover food after each meal "
            "so future predictions can be evaluated."
        ),
        (
            "🍽️ Review portions",
            "Check portion sizes and offer suitable portion "
            "choices where the canteen allows it."
        ),
        (
            "📅 Compare weekdays",
            "Look for differences in attendance, menu choices "
            "and waste across days of the week."
        ),
        (
            "🌡️ Store food safely",
            "Follow food safety and temperature-control rules. "
            "Never reuse or donate food unless it is safe "
            "and permitted under applicable rules."
        )
    ]

    for title, description in tips:
        st.markdown(
            f"""
            <div class="tip-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Use Predictions Responsibly")

    st.write(
        "The model provides an estimate, not a guarantee. "
        "Compare predictions with actual measured waste "
        "and update the dataset as reliable new records "
        "become available."
    )

    st.caption(
        "GreenPlate AI · Machine Learning-Based Food Waste "
        "Prediction and Prevention System"
    )