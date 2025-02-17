


# # Example data
# # S = 1208                                # Current stock price
# # K = 1210                                # Strike price
# # T = 7/365                               # Time to expiration (1 week converted to years)
# # r = 2.05051/100                            # Risk-free interest rate (US Real Interest Rate)
# # implied_volatility_value = 0.57         # Implied volatility (converted from percentage)
# # historic_volatility_value = 0.47        # Historic volatility (converted from percentage)

# # S = underlying price ($$$ per share)
# # K = strike price ($$$ per share)
# # σ(sigma) = volatility (% p.a.)
# # r = continuously compounded risk-free interest rate (% p.a.)
# # q = continuously compounded dividend yield (% p.a.)
# # t = time to expiration (% of year)



from numpy import exp, sqrt, log
from scipy.stats import norm
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def _calculate_d1_d2(self):
        # Calculate d1 and d2 used in the Black-Scholes formula
        d1 = (np.log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    # Call Option Greeks and Price

    def _call_delta(self, d1):
        # Delta is the first derivative of option price with respect to underlying price
        return norm.cdf(d1)

    def _call_gamma(self, d1):
        # Gamma is the second derivative of option price with respect to underlying price
        return norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))

    def _call_theta(self, d1, d2):
        # Theta is the first derivative of option price with respect to time to expiration
        theta = -(self.current_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_maturity)) - self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        return theta / 365

    def _call_vega(self, d1):
        # Vega is the first derivative of option price with respect to volatility σ(Sigma)
        return self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity) / 100

    def _call_rho(self, d2):
        # Rho is the first derivative of option price with respect to interest rate
        return self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)

    # Put Option Greeks and Price

    def _put_delta(self, d1):
        # Delta for put option
        return norm.cdf(d1) - 1

    def _put_gamma(self, d1):
        # Gamma for put option
        return norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))

    def _put_theta(self, d1, d2):
        # Theta for put option
        theta = -(self.current_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_maturity)) + self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        return theta / 365

    def _put_vega(self, d1):
        # Vega for put option
        return self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity) / 100

    def _put_rho(self, d2):
        # Rho for put option
        return -self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

    # Call and Put Prices

    def _call_price(self, d1, d2):
        # Calculate call option price
        return self.current_price * norm.cdf(d1) - (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))

    def _put_price(self, d1, d2):
        # Calculate put option price
        return (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) - self.current_price * norm.cdf(-d1)

    def calculateEverything(self):
        # Run Black-Scholes calculations and print results
        d1, d2 = self._calculate_d1_d2()

        call_delta = self._call_delta(d1)
        call_gamma = self._call_gamma(d1)
        call_theta = self._call_theta(d1, d2)
        call_vega = self._call_vega(d1)
        call_rho = self._call_rho(d2)
        call_price = self._call_price(d1, d2)

        put_delta = self._put_delta(d1)
        put_gamma = self._put_gamma(d1)
        put_theta = self._put_theta(d1, d2)
        put_vega = self._put_vega(d1)
        put_rho = self._put_rho(d2)
        put_price = self._put_price(d1, d2)

        results = {
            'Call Delta': call_delta,
            'Call Gamma': call_gamma,
            'Call Theta': call_theta,
            'Call Vega': call_vega,
            'Call Rho': call_rho,
            'Call Price': call_price,
            'Put Delta': put_delta,
            'Put Gamma': put_gamma,
            'Put Theta': put_theta,
            'Put Vega': put_vega,
            'Put Rho': put_rho,
            'Put Price': put_price
        }

        return results
    def calculate_prices(self):
        d1, d2 = self._calculate_d1_d2()
        call_price = self._call_price(d1, d2)
        put_price = self._put_price(d1, d2)
        return call_price, put_price

    
    def calculate_call_greeks(self):
        d1, d2 = self._calculate_d1_d2()
        greeks = {
            'Call Delta': round(self._call_delta(d1), 2),
            'Call Gamma': round(self._call_gamma(d1), 2),
            'Call Theta': round(self._call_theta(d1, d2), 2),
            'Call Vega': round(self._call_vega(d1), 2),
            'Call Rho': round(self._call_rho(d2), 2)
        }
        return greeks

    def calculate_put_greeks(self):
        d1, d2 = self._calculate_d1_d2()
        greeks = {
            'Put Delta': round(self._put_delta(d1), 2),
            'Put Gamma': round(self._put_gamma(d1), 2),
            'Put Theta': round(self._put_theta(d1, d2), 2),
            'Put Vega': round(self._put_vega(d1), 2),
            'Put Rho': round(self._put_rho(d2), 2)
        }
        return greeks

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            call_prices[i, j], put_prices[i, j] = bs_temp.calculate_prices()
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL Price Heatmap')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT Price Heatmap')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

def plot_call_greeks_correlation(bs_model, spot_range, vol_range):
    greeks_list = []

    for vol in vol_range:
        for spot in spot_range:
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            greeks = bs_temp.calculate_call_greeks()
            greeks['Spot Price'] = spot
            greeks['Volatility'] = vol
            greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    correlation_matrix = greeks_df.corr()

    # Plotting the correlation matrix
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title('Correlation Between Greeks and Other Parameters')
    
    return fig_corr

def plot_put_greeks_correlation(bs_model, spot_range, vol_range):
    greeks_list = []

    for vol in vol_range:
        for spot in spot_range:
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            greeks = bs_temp.calculate_put_greeks()
            greeks['Spot Price'] = spot
            greeks['Volatility'] = vol
            greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    correlation_matrix = greeks_df.corr()

    # Plotting the correlation matrix
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title('Correlation Between Greeks and Other Parameters')
    
    return fig_corr

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/aidan-ruvins/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Aidan Ruvins`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Generate Heatmap')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

call_greeks = bs_model.calculate_call_greeks()
put_greeks = bs_model.calculate_put_greeks()
with col1:
    st.subheader("Greeks for Call Options")
    st.write(call_greeks)
with col2:
    st.subheader("Greeks for Put Options")
    st.write(put_greeks)

st.subheader("Explanation of Greeks")
st.write("Delta: The first derivative of option price with respect to underlying price.")
st.write("Gamma: The second derivative of option price with respect to underlying price.")
st.write("Theta: The first derivative of option price with respect to time to expiration.")
st.write("Vega: The first derivative of option price with respect to volatility σ(Sigma).")
st.write("Rho: The first derivative of option price with respect to interest rate.")


st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)


with col1:
    st.subheader("Correlation Matrix of Call Greeks")
    fig_corr = plot_call_greeks_correlation(bs_model, spot_range, vol_range)
    st.pyplot(fig_corr)
with col2:
    st.subheader("Correlation Matrix of Put Greeks")
    fig_corr = plot_put_greeks_correlation(bs_model, spot_range, vol_range)
    st.pyplot(fig_corr)