import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
st.markdown("""
<style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #4472C4 !important;
        color: white !important;
    }
    .stMultiSelect [data-baseweb="tag"] [aria-label="close"] {
        color: white !important;
    }
    .stMultiSelect [data-baseweb="menu"] [aria-selected="true"] {
        background-color: #4472C4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Influencer Marketing Market Power Analyzer",
    page_icon="📊",
    layout="wide"
)

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Influencer Marketing Market Power Analyzer",
    page_icon="📊",
    layout="wide"
)

# ========== Title ==========
st.title("📊 Influencer Marketing Market Power Analyzer")
st.markdown("""
**Target Users**: Brand marketing managers, platform operation analysts  
**Core Function**: Evaluate KOL market power and optimize marketing investment strategies
""")

# ========== Sidebar: Model Parameter Controls ==========
st.sidebar.header("⚙️ Model Configuration")

# Stickiness Parameters
st.sidebar.subheader("User Stickiness Model")
stickiness_midpoint = st.sidebar.slider("Stickiness Midpoint", 0.05, 0.30, 0.10, 0.01)
stickiness_steepness = st.sidebar.slider("Steepness", 5, 50, 20, 1)
stickiness_max = st.sidebar.slider("Max Stickiness Value", 0.5, 1.0, 0.95, 0.05)

# ARPU Parameters
st.sidebar.subheader("ARPU Normalization")
arpu_benchmark = st.sidebar.slider("ARPU Benchmark", 5.0, 30.0, 10.0, 1.0)
arpu_max = st.sidebar.slider("Max Normalized Value", 0.5, 1.0, 0.95, 0.05)

# Demand Elasticity Parameters
st.sidebar.subheader("Demand Elasticity Model")
elasticity_base = st.sidebar.slider("Base Elasticity", 0.3, 1.5, 0.8, 0.1)
elasticity_decay = st.sidebar.slider("Decay Coefficient", 1.0, 5.0, 2.5, 0.5)
elasticity_floor = st.sidebar.slider("Min Elasticity Floor", 0.1, 0.5, 0.2, 0.05)

# Market Power Weights (Auto-normalizing)
st.sidebar.subheader("Market Power Weights")
w_attention = st.sidebar.slider("Attention Conversion Rate Weight", 0.0, 1.0, 0.35, 0.05)
w_stickiness = st.sidebar.slider("User Stickiness Weight", 0.0, 1.0, 0.35, 0.05)

total = w_attention + w_stickiness
if total >= 1.0:
    w_attention = w_attention / total * 0.99
    w_stickiness = w_stickiness / total * 0.99
w_arpu = 1 - w_attention - w_stickiness
st.sidebar.write(f"Normalized ARPU Weight: **{w_arpu:.3f}**")

# Market Structure Thresholds
st.sidebar.subheader("Market Structure Thresholds")
threshold_wta = st.sidebar.slider("Winner-Takes-All Threshold", 0.5, 0.9, 0.70, 0.05)
threshold_mc = st.sidebar.slider("Monopolistic Competition Threshold", 0.2, 0.6, 0.40, 0.05)

# ========== Dynamic MODEL_CONFIG ==========
MODEL_CONFIG = {
    "stickiness": {
        "midpoint": stickiness_midpoint,
        "steepness": stickiness_steepness,
        "max_value": stickiness_max
    },
    "arpu": {
        "benchmark": arpu_benchmark,
        "max_normalized": arpu_max
    },
    "demand_elasticity": {
        "base": elasticity_base,
        "decay": elasticity_decay,
        "min_floor": elasticity_floor
    },
    "market_power_weights": {
        "attention_conversion_rate": w_attention,
        "user_stickiness": w_stickiness,
        "normalized_arpu": w_arpu
    },
    "market_structure_thresholds": {
        "winner_takes_all": threshold_wta,
        "monopolistic_competition": threshold_mc
    }
}

# ========== Data Loading ==========
@st.cache_data
def load_data():
    df = pd.read_csv("data/influencer_marketing_roi_dataset.csv", encoding="utf-8-sig")
    df["fans"] = df["estimated_reach"]
    np.random.seed(42)
    df["interactions"] = df["engagements"] * np.random.uniform(0.98, 1.02, size=len(df))
    df["revenue"] = df["product_sales"] * np.random.uniform(0.97, 1.03, size=len(df))
    return df

try:
    df = load_data()
    st.sidebar.success(f"✅ Data loaded: {len(df)} campaigns")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# ========== Core Calculation Function ==========
def calculate_economic_indicators(fans, interactions, revenue):
    fans = float(fans)
    interactions = float(interactions)
    revenue = float(revenue)
    
    if fans <= 0:
        return None
    
    attention_conversion_rate = interactions / fans
    arpu_value = revenue / fans
    
    s = MODEL_CONFIG["stickiness"]
    exponent = -s["steepness"] * (attention_conversion_rate - s["midpoint"])
    user_stickiness = s["max_value"] / (1 + np.exp(exponent))
    
    a = MODEL_CONFIG["arpu"]
    normalized_arpu = min(arpu_value / a["benchmark"], a["max_normalized"])
    
    w = MODEL_CONFIG["market_power_weights"]
    market_power = (
        w["attention_conversion_rate"] * attention_conversion_rate +
        w["user_stickiness"] * user_stickiness +
        w["normalized_arpu"] * normalized_arpu
    )
    market_power = min(max(market_power, 0), 1)
    
    d = MODEL_CONFIG["demand_elasticity"]
    abs_elasticity = max(d["base"] * np.exp(-d["decay"] * user_stickiness), d["min_floor"])
    demand_elasticity = -abs_elasticity
    
    consumer_welfare = 1 - market_power
    
    t = MODEL_CONFIG["market_structure_thresholds"]
    if market_power >= t["winner_takes_all"]:
        market_structure = "Winner-Takes-All"
    elif market_power >= t["monopolistic_competition"]:
        market_structure = "Monopolistic Competition"
    else:
        market_structure = "Competitive Market"
    
    return {
        "fans": int(fans),
        "attention_conversion_rate": round(attention_conversion_rate, 4),
        "arpu": round(arpu_value, 4),
        "user_stickiness": round(user_stickiness, 4),
        "market_power": round(market_power, 4),
        "demand_elasticity": round(demand_elasticity, 4),
        "consumer_welfare": round(consumer_welfare, 4),
        "market_structure": market_structure
    }

# ========== Batch Computation ==========
@st.cache_data
def compute_all_indicators(_df, config):
    results = []
    for _, row in _df.iterrows():
        res = calculate_economic_indicators(row["fans"], row["interactions"], row["revenue"])
        if res:
            res["campaign_id"] = row["campaign_id"]
            results.append(res)
    return pd.DataFrame(results)

economic_df = compute_all_indicators(df, MODEL_CONFIG)

# ========== Page Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data Overview", 
    "🔬 Model Analysis", 
    "📊 Interactive Visualization", 
    "🔍 Single Case Diagnosis",
    "📊 Comparison Matrix"
])

# ========== TAB 1: Data Overview ==========
with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Campaigns", len(df))
    col2.metric("Avg Fans", f"{df['fans'].mean():,.0f}")
    col3.metric("Avg Interactions", f"{df['interactions'].mean():,.0f}")
    col4.metric("Avg Revenue", f"${df['revenue'].mean():,.2f}")
    
    st.subheader("Data Filtering")
    col_left, col_right = st.columns(2)
    
    with col_left:
        fan_range = st.slider(
            "Fans Range", 
            int(df["fans"].min()), 
            int(df["fans"].max()),
            (int(df["fans"].quantile(0.1)), int(df["fans"].quantile(0.9)))
        )
    
    with col_right:
        structure_filter = st.multiselect(
            "Market Structure", 
            options=economic_df["market_structure"].unique(),
            default=list(economic_df["market_structure"].unique())
        )
    
    filtered_df = economic_df[
        (economic_df["fans"] >= fan_range[0]) & 
        (economic_df["fans"] <= fan_range[1]) &
        (economic_df["market_structure"].isin(structure_filter))
    ]
    
    st.write(f"Filtered Data: **{len(filtered_df)}** records")
    st.dataframe(filtered_df.head(20), use_container_width=True)

# ========== TAB 2: Model Analysis ==========
with tab2:
    st.header("Model Output Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        wta_pct = (economic_df['market_structure'] == 'Winner-Takes-All').mean() * 100
        st.metric("Winner-Takes-All %", f"{wta_pct:.1f}%")
    with col2:
        st.metric("Avg Market Power", f"{economic_df['market_power'].mean():.3f}")
    with col3:
        st.metric("Avg Consumer Welfare", f"{economic_df['consumer_welfare'].mean():.3f}")
    
    st.subheader("Market Structure Distribution")
    structure_counts = economic_df["market_structure"].value_counts().reset_index()
    structure_counts.columns = ["Market Structure", "Count"]
    
    fig_pie = px.pie(
        structure_counts, 
        values="Count", 
        names="Market Structure", 
        hole=0.4, 
        color="Market Structure"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("Indicator Correlation Matrix")
    corr_vars = [
        "user_stickiness", 
        "market_power", 
        "demand_elasticity", 
        "consumer_welfare", 
        "arpu"
    ]
    corr_matrix = economic_df[corr_vars].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale="RdBu_r", 
        zmin=-1, 
        zmax=1
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ========== TAB 3: Interactive Visualization ==========
with tab3:
    st.header("Interactive Visualization")
    
    st.subheader("Custom Scatter Plot")
    col_x, col_y, col_color = st.columns(3)
    
    numeric_cols = [
        "fans", 
        "attention_conversion_rate", 
        "arpu", 
        "user_stickiness", 
        "market_power", 
        "demand_elasticity", 
        "consumer_welfare"
    ]
    
    with col_x:
        x_var = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index("market_power"))
    with col_y:
        y_var = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index("consumer_welfare"))
    with col_color:
        color_var = st.selectbox("Color By", ["market_structure"] + numeric_cols, index=0)
    
    fig_custom = px.scatter(
        filtered_df if 'filtered_df' in locals() else economic_df,
        x=x_var, 
        y=y_var, 
        color=color_var,
        hover_data=["campaign_id", "fans"],
        title=f"{x_var} vs {y_var}",
        opacity=0.7
    )
    st.plotly_chart(fig_custom, use_container_width=True)
    
    st.subheader("Preset Analysis Charts")
    
    chart_choice = st.selectbox("Select Chart", [
        "Market Power vs Consumer Welfare",
        "User Stickiness vs Demand Elasticity",
        "Audience Size (log) vs Market Power",
        "Market Power Distribution by Structure"
    ])
    
    if chart_choice == "Market Power vs Consumer Welfare":
        fig = px.scatter(
            economic_df, 
            x="market_power", 
            y="consumer_welfare",
            color="market_structure", 
            hover_data=["campaign_id"],
            title="Market Power vs Consumer Welfare",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_choice == "User Stickiness vs Demand Elasticity":
        fig = px.scatter(
            economic_df, 
            x="user_stickiness", 
            y="demand_elasticity",
            color="market_structure", 
            hover_data=["campaign_id"],
            title="User Stickiness vs Demand Elasticity",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_choice == "Audience Size (log) vs Market Power":
        economic_df["log_fans"] = np.log1p(economic_df["fans"])
        
        color_map = {
            "Winner-Takes-All": "#307EC7",
            "Monopolistic Competition": "#E15759",
            "Competitive Market": "#9DC3E6"
        }
        
        fig = px.scatter(
            economic_df, 
            x="log_fans", 
            y="market_power",
            color="market_structure",
            color_discrete_map=color_map,
            hover_data=["campaign_id", "fans"],
            title="Audience Size (log) vs Market Power (By Market Structure)",
            opacity=0.6,
            height=500
        )
        
        structures = economic_df["market_structure"].unique()
        
        for structure in structures:
            subset = economic_df[economic_df["market_structure"] == structure].copy()
            x = subset["log_fans"]
            y = subset["market_power"]
            
            if len(x) > 1:
                slope, intercept, _, _, std_err = stats.linregress(x, y)
                
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                y_line = np.clip(y_line, 0, 1)
                
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color=color_map.get(structure, "gray"), width=2),
                    name=f"{structure} trend",
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        fig.update_layout(
            xaxis_title="Log(Fans)",
            yaxis_title="Market Power",
            yaxis=dict(range=[0, 1]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_choice == "Market Power Distribution by Structure":
        fig = px.box(
            economic_df, 
            x="market_structure", 
            y="market_power",
            color="market_structure", 
            title="Market Power Distribution by Structure"
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== TAB 4: Single Case Diagnosis ==========
with tab4:
    st.header("🔍 Single Campaign Deep Diagnosis")
    
    campaign_id = st.selectbox(
        "Select Campaign ID", 
        options=sorted(economic_df["campaign_id"].unique())
    )
    
    selected = economic_df[economic_df["campaign_id"] == campaign_id].iloc[0]
    raw = df[df["campaign_id"] == campaign_id].iloc[0]
    
    st.subheader("Core Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Power", f"{selected['market_power']:.3f}")
    c2.metric("Market Structure", selected["market_structure"])
    c3.metric("Consumer Welfare", f"{selected['consumer_welfare']:.3f}")
    c4.metric("Demand Elasticity", f"{selected['demand_elasticity']:.3f}")
    
    st.subheader("Indicator Breakdown")
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.write("**Raw Data**")
        st.json({
            "Fans": f"{raw['fans']:,.0f}",
            "Interactions": f"{raw['interactions']:,.0f}",
            "Revenue": f"${raw['revenue']:,.2f}"
        })
        
        st.write("**Computed Metrics**")
        st.json({
            "Attention Conversion Rate": f"{selected['attention_conversion_rate']:.4f}",
            "ARPU": f"${selected['arpu']:.4f}",
            "User Stickiness": f"{selected['user_stickiness']:.4f}"
        })
    
    with detail_col2:
        attention_contrib = selected["attention_conversion_rate"] * MODEL_CONFIG["market_power_weights"]["attention_conversion_rate"]
        stickiness_contrib = selected["user_stickiness"] * MODEL_CONFIG["market_power_weights"]["user_stickiness"]
        arpu_contrib = min(selected["arpu"] / MODEL_CONFIG["arpu"]["benchmark"], 
                          MODEL_CONFIG["arpu"]["max_normalized"]) * MODEL_CONFIG["market_power_weights"]["normalized_arpu"]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Market Power",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=[
                "Attention Conversion<br>Contribution", 
                "User Stickiness<br>Contribution", 
                "ARPU<br>Contribution", 
                "Total Market<br>Power"
            ],
            y=[attention_contrib, stickiness_contrib, arpu_contrib, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig_waterfall.update_layout(
            title="Market Power Composition Analysis",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.subheader("Peer Comparison")
    same_structure = economic_df[
        economic_df["market_structure"] == selected["market_structure"]
    ]
    
    comparison_metrics = ["market_power", "consumer_welfare", "user_stickiness", "arpu"]
    comparison_data = []
    
    for metric in comparison_metrics:
        comparison_data.append({
            "Metric": metric,
            "This Campaign": selected[metric],
            "Peer Average": same_structure[metric].mean(),
            "Overall Average": economic_df[metric].mean()
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="This Campaign", x=comp_df["Metric"], y=comp_df["This Campaign"]))
    fig_comp.add_trace(go.Bar(name="Peer Average", x=comp_df["Metric"], y=comp_df["Peer Average"]))
    fig_comp.add_trace(go.Bar(name="Overall Average", x=comp_df["Metric"], y=comp_df["Overall Average"]))
    fig_comp.update_layout(
        barmode="group", 
        title="Metric Comparison", 
        height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ========== TAB 5: Comparison Matrix ==========
with tab5:
    st.header("📊 Multi-Campaign Comparison Matrix")
    st.markdown("Select multiple campaigns for side-by-side comparison and ranking analysis.")
    
   
    selected_campaigns = st.multiselect(
        "Select campaigns to compare (2-10 recommended):",
        options=sorted(economic_df["campaign_id"].unique()),
        default=sorted(economic_df["campaign_id"].unique())[:4],
        key="compare_select"
    )
    
    if len(selected_campaigns) < 2:
        st.warning("Please select at least 2 campaigns to compare.")
    else:
        compare_df = economic_df[economic_df["campaign_id"].isin(selected_campaigns)].copy()
        
       
        st.subheader("Comparison Dimensions")
        
        dimensions = st.multiselect(
            "Select metrics to compare:",
            options=["market_power", "consumer_welfare", "arpu", "user_stickiness", 
                    "attention_conversion_rate", "demand_elasticity", "fans"],
            default=["market_power", "consumer_welfare", "arpu"],
            key="compare_dims"
        )
        
        if dimensions:
            
            st.subheader("Performance Heatmap")
            
            heatmap_data = compare_df.set_index("campaign_id")[dimensions]
            
            
            heatmap_normalized = heatmap_data.copy()
            for col in dimensions:
                col_min = economic_df[col].min()
                col_max = economic_df[col].max()
                if col_max > col_min:
                    heatmap_normalized[col] = (heatmap_data[col] - col_min) / (col_max - col_min)
                else:
                    heatmap_normalized[col] = 0.5
            
            fig_heatmap = px.imshow(
                heatmap_normalized.T,
                labels=dict(x="Campaign", y="Metric", color="Normalized Score"),
                x=heatmap_normalized.index,
                y=heatmap_normalized.columns,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=True
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            
            st.subheader("Raw Values")
            st.dataframe(heatmap_data.round(3), use_container_width=True)
            
            
            st.subheader("Capability Radar")
            
            fig_radar = go.Figure()
            
            colors = px.colors.qualitative.Set2[:len(selected_campaigns)]
            
            for i, (_, row) in enumerate(compare_df.iterrows()):
                values = [row[d] for d in dimensions]
        
                normalized_values = []
                for d, v in zip(dimensions, values):
                    col_min = economic_df[d].min()
                    col_max = economic_df[d].max()
                    nv = (v - col_min) / (col_max - col_min) if col_max > col_min else 0.5
                    normalized_values.append(nv)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values + [normalized_values[0]],
                    theta=dimensions + [dimensions[0]],
                    fill='toself',
                    name=row["campaign_id"],
                    line_color=colors[i]
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Capability Comparison"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.subheader("Ranking Analysis")
            
            rankings = pd.DataFrame({"campaign_id": compare_df["campaign_id"]})
            
            for dim in dimensions:
                rankings[f"{dim}_rank"] = compare_df[dim].rank(ascending=False).astype(int)
            
            rankings["average_rank"] = rankings[[c for c in rankings.columns if "_rank" in c]].mean(axis=1)
            rankings = rankings.sort_values("average_rank")
            
            st.dataframe(rankings, use_container_width=True, hide_index=True)
            
            
            best = rankings.iloc[0]
            st.success(f"🏆 **Best Overall: {best['campaign_id']}** (Average Rank: {best['average_rank']:.1f})")
            
           
            st.subheader("Category Winners")
            winner_cols = st.columns(len(dimensions))
            for i, dim in enumerate(dimensions):
                winner_id = compare_df.loc[compare_df[dim].idxmax(), "campaign_id"]
                winner_val = compare_df[dim].max()
                with winner_cols[i]:
                    st.metric(f"Best {dim}", winner_id, f"{winner_val:.3f}")
            
            
            st.subheader("Gap Analysis")
            
            gap_metric = st.selectbox("Select metric for gap analysis:", dimensions, key="gap_metric")
            
            gap_data = compare_df[["campaign_id", gap_metric]].sort_values(gap_metric, ascending=False)
            best_val = gap_data[gap_metric].max()
            gap_data["gap_to_best"] = best_val - gap_data[gap_metric]
            gap_data["gap_pct"] = (gap_data["gap_to_best"] / best_val * 100).round(1)
            
            fig_gap = px.bar(
                gap_data,
                x="campaign_id",
                y="gap_to_best",
                color="campaign_id",
                title=f"Gap to Best ({gap_metric})",
                labels={"gap_to_best": f"Gap to {gap_data.iloc[0]['campaign_id']}"}
            )
            st.plotly_chart(fig_gap, use_container_width=True)
            
           
            st.subheader("Export Comparison")
            
            export_df = heatmap_data.copy()
            export_df["average_rank"] = rankings.set_index("campaign_id")["average_rank"]
            
            csv = export_df.to_csv()
            st.download_button(
                "📥 Download Comparison (CSV)",
                csv,
                "campaign_comparison.csv",
                "text/csv",
                use_container_width=True
            )
