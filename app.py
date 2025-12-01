import os
import glob
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import kagglehub

# Page configuration
st.set_page_config(
    page_title="Gaming Data Analysis",
    page_icon="🎮",
    layout="wide"
)

st.title("CS133 Final Project: Gaming Data Analysis")
st.markdown("---")

# Data loading functions
@st.cache_data
def load_platform_games(base_path, platform_name):
    platform_dir = os.path.join(base_path, platform_name)

    games_path = os.path.join(platform_dir, "games.csv")
    prices_path = os.path.join(platform_dir, "prices.csv")

    games = pd.read_csv(games_path)
    games["platform"] = platform_name

    # Normalize game ID column in 'games' DataFrame
    if "gameid" in games.columns:
        games = games.rename(columns={"gameid": "game_id"})
    elif "id" in games.columns:
        games = games.rename(columns={"id": "game_id"})

    # Attempt to load and merge prices
    if os.path.exists(prices_path):
        prices_df = pd.read_csv(prices_path)

        # Normalize game ID column in 'prices_df' DataFrame
        if "gameid" in prices_df.columns:
            prices_df = prices_df.rename(columns={"gameid": "game_id"})
        elif "id" in prices_df.columns:
            prices_df = prices_df.rename(columns={"id": "game_id"})

        latest_prices = prices_df
        if "date_acquired" in prices_df.columns:
            prices_df["date_acquired"] = pd.to_datetime(prices_df["date_acquired"])
            latest_prices = (
                prices_df.sort_values("date_acquired")
                      .groupby("game_id")
                      .tail(1)
            )

        # Identify the actual price column name - prioritize 'usd'
        price_column_name = None
        if 'usd' in latest_prices.columns:
            price_column_name = 'usd'
        elif 'eur' in latest_prices.columns:
            price_column_name = 'eur'
        elif 'gbp' in latest_prices.columns:
            price_column_name = 'gbp'
        # Fallback to general price names if no currency found
        for col in ['price', 'current_price', 'retail_price', 'final_price']:
            if col in latest_prices.columns:
                price_column_name = col
                break

        if price_column_name is not None:
            # Merge the identified price column
            if "game_id" in games.columns and "game_id" in latest_prices.columns:
                games = games.merge(
                    latest_prices[["game_id", price_column_name]],
                    on="game_id",
                    how="left"
                )
                # Rename the identified price column to 'prices'
                games = games.rename(columns={price_column_name: "prices"})
            else:
                games["prices"] = np.nan
        else:
            games["prices"] = np.nan
    else:
        games["prices"] = np.nan

    # Ensure 'prices' column exists in the final DataFrame
    if "prices" not in games.columns:
        games["prices"] = np.nan

    return games

@st.cache_data
def load_platform_players(base_path, platform_name):
    platform_dir = os.path.join(base_path, platform_name)
    players_path = os.path.join(platform_dir, "players.csv")
    if os.path.exists(players_path):
        players_df = pd.read_csv(players_path)
        players_df["platform"] = platform_name
        return players_df
    return pd.DataFrame()

@st.cache_data
def load_and_process_data():
    with st.spinner("Loading data from Kaggle..."):
        path = kagglehub.dataset_download(
            "artyomkruglov/gaming-profiles-2025-steam-playstation-xbox"
        )
    
    playstation_games = load_platform_games(path, "playstation")
    xbox_games = load_platform_games(path, "xbox")
    steam_games = load_platform_games(path, "steam")
    games_all = pd.concat([playstation_games, xbox_games, steam_games], ignore_index=True)
    
    df = games_all.copy()
    df = df.dropna(subset=["prices"])
    
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
    else:
        df["release_year"] = np.nan
    
    genre_col = "genre" if "genre" in df.columns else "genres"
    
    def clean_genre(s):
        if pd.isna(s):
            return "unknown"
        s = str(s)
        s = re.split(r"[,;/\|]", s)[0]
        s = re.sub(r"[\"'\[\]\\]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.lower()
        return s if s != "" else "unknown"
    
    if genre_col in df.columns:
        df[genre_col] = df[genre_col].apply(clean_genre)
    
    df["genres"] = df[genre_col].str.title()
    df["prices"] = df["prices"].astype(float)
    
    playstation_players = load_platform_players(path, "playstation")
    xbox_players = load_platform_players(path, "xbox")
    steam_players = load_platform_players(path, "steam")
    players_all = pd.concat([playstation_players, xbox_players, steam_players], ignore_index=True)
    
    if 'created' in players_all.columns:
        players_all['created'] = pd.to_datetime(players_all['created'], errors='coerce')
        players_all['creation_year'] = players_all['created'].dt.year
    
    return df, players_all, path

df, players_all, path = load_and_process_data()

# Q1: Price Distribution by Platform
st.header("Q1: How are prices distributed across platforms?")
st.markdown("**Plot:** Seaborn barplot of mean price by platform.")
st.markdown("**Implementation:** Use `df` filtered to rows with non-null `prices`, group by `platform`, and plot mean `prices` with a bar chart.")

price_by_platform = df.groupby("platform")["prices"].mean().reset_index()
fig_q1 = px.bar(
    price_by_platform,
    x="platform",
    y="prices",
    title="Game Price Distribution by Platform",
    labels={"prices": "Average Price (USD)", "platform": "Platform"},
    color="platform"
)
fig_q1.update_layout(showlegend=False)
st.plotly_chart(fig_q1, use_container_width=True)

st.markdown("**Takeaway:** Average prices differ by platform. PC/PlayStation tend to have higher average prices, while some other platforms (with more legacy/handheld style titles) skew lower.")
st.markdown("---")

# Q2: Game Releases Over Time
st.header("Q2: How did the number of releases evolve over time?")
st.markdown("**Plot:** Seaborn lineplot of yearly game counts by platform.")
st.markdown("**Implementation:** Parse `release_date` to `release_year` (when available), group by `release_year` and `platform`, and plot counts with lines.")

release_df = df.dropna(subset=["release_year"])
release_counts = (
    release_df.groupby(["release_year", "platform"])
              .size()
              .reset_index(name="num_games")
)

fig_q2 = px.line(
    release_counts,
    x="release_year",
    y="num_games",
    color="platform",
    markers=True,
    title="Number of Game Releases Over Time by Platform",
    labels={"num_games": "Number of Games", "release_year": "Release Year", "platform": "Platform"}
)
st.plotly_chart(fig_q2, use_container_width=True)

st.markdown("**Takeaway:** There are clear waves of growth post-2010 for PC and PlayStation. Xbox shows similar patterns with slightly different timing. Years with fewer releases often reflect missing or unparseable dates.")
st.markdown("---")

# Q3: Most Common Genres
st.header("Q3: What are the most common genres on each platform?")
st.markdown("**Plot:** Grouped bar chart of top 10 genres per platform.")
st.markdown("**Implementation:** Normalize the genre string in a column named `genre` or `genres` by casting to string, splitting on commas/semicolons, taking the first tag, and stripping brackets/quotes. Count by `platform` and genre, take the top 10 per platform, then plot a grouped bar chart.")

display_genre_col = "genres"
genre_counts = (
    df.groupby(["platform", display_genre_col])
      .size()
      .reset_index(name="count")
)

# Top 10 genres overall
top10_genres = (
    genre_counts.groupby(display_genre_col)["count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

top_genres = genre_counts[genre_counts[display_genre_col].isin(top10_genres)]

fig_q3_interactive = px.bar(
    top_genres,
    x="count",
    y=display_genre_col,
    color="platform",
    orientation="h",
    title="Top 10 Genres by Platform",
    labels={
        "count": "Number of Games",
        display_genre_col: "Genre",
        "platform": "Platform"
    },
    barmode="group",
    hover_data=["platform", "count"]
)

fig_q3_interactive.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    legend_title_text="Platform"
)
st.plotly_chart(fig_q3_interactive, use_container_width=True)

# Top 10 genres per platform
top_genres_per_plat = (
    genre_counts.sort_values("count", ascending=False)
    .groupby("platform")
    .head(10)
)

fig_q3_per_plat_interactive = px.bar(
    top_genres_per_plat,
    x="count",
    y=display_genre_col,
    color="platform",
    facet_col="platform",
    facet_col_wrap=3,
    orientation="h",
    title="Top 10 Genres per Platform",
    labels={
        "count": "Number of games",
        display_genre_col: "Genre",
        "platform": "Platform"
    }
)

fig_q3_per_plat_interactive.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    legend_title_text="Platform",
    margin=dict(l=60, r=20, t=60, b=40)
)

st.plotly_chart(fig_q3_per_plat_interactive, use_container_width=True)

st.markdown("**Takeaway:** Across platforms, Action, Adventure, and RPG are very prominent, but platform-specific flavor shows up (for example, more racing and sports representation on Xbox and PlayStation compared to some PC-heavy subgenres).")
st.markdown("---")

# Q4: Prices by Genre and Platform
st.header("Q4: How do prices vary by genre and platform?")
st.markdown("**Plot:** Horizontal bar chart of mean price by genre × platform, sorted by average price.")
st.markdown("**Implementation:** Group by `platform` and the cleaned genre column, compute mean `prices`, sort, and plot as a horizontal barplot.")

# Count how many games per platform+genre
counts = (
    df.groupby(["platform", display_genre_col])["prices"]
      .size()
      .reset_index(name="n_games")
)

avg_price_genre = (
    df.groupby(["platform", display_genre_col])["prices"]
      .mean()
      .reset_index()
      .merge(counts, on=["platform", display_genre_col])
)

# Filter out very rare combos (e.g., < 10 games)
min_games = 10
avg_price_genre_filtered = avg_price_genre[avg_price_genre["n_games"] >= min_games]

# Compute overall average price per genre (across platforms) for ordering
genre_price_order = (
    avg_price_genre_filtered.groupby(display_genre_col)["prices"]
    .mean()
    .sort_values(ascending=False)
    .index
)

fig_q4_interactive = px.bar(
    avg_price_genre_filtered,
    x="prices",
    y=display_genre_col,
    color="platform",
    orientation="h",
    title="Average Game Price by Genre and Platform",
    labels={
        "prices": "Average Price (USD)",
        display_genre_col: "Genre",
        "platform": "Platform",
        "n_games": "Number of Games"
    },
    hover_data=["platform", "prices", "n_games"],
    barmode="group"
)

fig_q4_interactive.update_layout(
    yaxis=dict(categoryorder="total ascending", categoryarray=list(genre_price_order)),
    legend_title_text="Platform",
    height=800
)
st.plotly_chart(fig_q4_interactive, use_container_width=True)

st.markdown("**Takeaway:** Certain genres like RPG and Simulation tend to have higher average prices, especially on PC/PlayStation. Smaller/party/indie-like genres are often lower. This backs up and refines the Q1 platform-level price differences.")
st.markdown("---")

# Q5: Account Creation Years
st.header("Q5: Which years have the most accounts created?")
st.markdown("**Plot:** Plotly interactive bar chart of account creation years.")
st.markdown("**Implementation:** Load each platform's `players.csv`. Convert `created` to datetime where present. Extract `creation_year` as `created.dt.year`. Count the number of accounts per year and use `plotly.express.bar` for an interactive bar chart.")

if 'created' in players_all.columns and 'creation_year' in players_all.columns:
    creation_year_data_plotly = players_all.dropna(subset=['creation_year'])
    
    creation_year_counts = creation_year_data_plotly['creation_year'].value_counts().sort_index().reset_index()
    creation_year_counts.columns = ['creation_year', 'count']
    
    fig_q5 = px.bar(
        creation_year_counts,
        x='creation_year',
        y='count',
        title='Interactive Distribution of Account Creation Years',
        labels={'creation_year': 'Year', 'count': 'Number of Accounts'}
    )
    
    fig_q5.update_layout(xaxis=dict(tickmode='linear'))
    
    st.plotly_chart(fig_q5, use_container_width=True)
    
    st.markdown("**Takeaway:** Account creation spikes line up with major platform cycles and big release eras. Interactivity (hover, zoom, export) helps pinpoint exact peaks and lets users explore the full time range.")
else:
    st.warning("The 'created' column was not found in the players data. Cannot plot account creation years.")

