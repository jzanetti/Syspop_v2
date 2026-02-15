import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
from pandas import read_parquet

df = read_parquet("output/results.parquet")

# Initialize the Dash app
app = Dash(__name__)

# Extract column names for dynamic generation
columns = df.columns.tolist()

# 1. Define the Layout
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
    children=[
        html.H2("Conditional Distribution Analysis"),
        # Target Column Selector
        html.Div(
            [
                html.Label(html.B("1. Select Target Column to Plot:")),
                dcc.Dropdown(
                    id="target-column",
                    options=[{"label": col, "value": col} for col in columns],
                    value=columns[0],
                    clearable=False,
                ),
            ],
            style={"width": "30%", "marginBottom": "30px"},
        ),
        html.Hr(),
        html.Label(html.B("2. Filter by ONE Condition (others will be disabled):")),
        # Dynamic generation of filter dropdowns
        html.Div(
            [
                html.Div(
                    [
                        html.Label(f"{col}:"),
                        dcc.Dropdown(
                            id=f"filter-{col}",
                            options=[
                                {"label": str(val), "value": val}
                                for val in df[col].dropna().unique()
                            ],
                            multi=True,
                            placeholder=f"Select {col}...",
                        ),
                    ],
                    id=f"filter-container-{col}",
                    style={
                        "width": "23%",
                        "display": "inline-block",
                        "margin": "10px 1% 10px 0",
                        "verticalAlign": "top",
                    },
                )
                for col in columns
            ]
        ),
        # Histogram Graph
        dcc.Graph(id="distribution-plot"),
    ],
)


# 2. Callback to hide the filter if it is the target column
@app.callback(
    [Output(f"filter-container-{col}", "style") for col in columns],
    [Input("target-column", "value")],
)
def toggle_filters(target_col):
    styles = []
    for col in columns:
        if col == target_col:
            styles.append({"display": "none"})
        else:
            styles.append(
                {
                    "width": "23%",
                    "display": "inline-block",
                    "margin": "10px 1% 10px 0",
                    "verticalAlign": "top",
                }
            )
    return styles


# --- NEW CALLBACK: Disable other dropdowns when one is in use ---
@app.callback(
    [Output(f"filter-{col}", "disabled") for col in columns],
    [Input(f"filter-{col}", "value") for col in columns],
)
def disable_other_filters(*filter_values):
    # Find if any filter currently has a selection
    active_index = None
    for i, val in enumerate(filter_values):
        if val and len(val) > 0:  # If it's not None and not an empty list
            active_index = i
            break

    disabled_states = []
    for i in range(len(columns)):
        if active_index is None:
            # Nothing selected anywhere? Enable everything.
            disabled_states.append(False)
        elif i == active_index:
            # This is the one currently being used. Keep it enabled.
            disabled_states.append(False)
        else:
            # This is not the active one. Disable it.
            disabled_states.append(True)

    return disabled_states


# 3. Callback for updating the histogram
callback_inputs = [Input("target-column", "value")] + [
    Input(f"filter-{col}", "value") for col in columns
]


@app.callback(Output("distribution-plot", "figure"), *callback_inputs)
def update_histogram(target_col, *filter_values):
    filtered_df = df.copy()
    grouping_col = None

    for col, selected_vals in zip(columns, filter_values):
        if col == target_col:
            continue

        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

            # Since only one column can be selected now, this will accurately
            # capture the one active column to use for grouping if multiple values are picked
            if len(selected_vals) > 1:
                grouping_col = col

    if filtered_df.empty:
        return px.histogram(title="No data matches the selected filters.")

    fig = px.histogram(
        filtered_df,
        x=target_col,
        color=grouping_col,
        barmode="group" if grouping_col else "relative",
        title=f"Distribution of {target_col} based on selected filters",
        text_auto=True,
    )

    fig.update_xaxes(categoryorder="category ascending")
    fig.update_layout(bargap=0.1)

    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
