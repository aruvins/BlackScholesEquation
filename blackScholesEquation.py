from scipy.stats import norm
import pandas as pd
import numpy as np
import yfinance as yf

aapl= yf.Ticker("AAPL")
# aapl.options # list of dates 
DF_calls = aapl.option_chain().calls
DF_puts = aapl.option_chain().puts

print(DF_calls)
print("\n")
print(DF_puts)

# Example data
S = 1208                                # Current stock price
K = 1210                                # Strike price
T = 7/365                               # Time to expiration (1 week converted to years)
r = 2.05051/100                            # Risk-free interest rate (US Real Interest Rate)
implied_volatility_value = 0.57         # Implied volatility (converted from percentage)
historic_volatility_value = 0.47        # Historic volatility (converted from percentage)

# S = underlying price ($$$ per share)
# K = strike price ($$$ per share)
# σ(sigma) = volatility (% p.a.)
# r = continuously compounded risk-free interest rate (% p.a.)
# q = continuously compounded dividend yield (% p.a.)
# t = time to expiration (% of year)



# Delta is the first derivative of option price with respect to underlying price S
def call_delta(S,K,T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

# Gamma is the second derivative of option price with respect to underlying price S
def call_gamma(S,K,T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Theta is the first derivative of option price with respect to time to expiration t
def call_theta(S,K,T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta / 365

# Vega is the first derivative of option price with respect to volatility σ(Sigma)
def call_vega(S,K,T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega / 100

# Rho is the first derivative of option price with respect to interest rate r
def call_rho(S,K,T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return rho


def volatility(S,K,T, r, sigma):
    call_delta_value = call_delta(S,K,T, r, sigma)
    call_gamma_value = call_gamma(S,K,T, r, sigma)
    call_theta_value = call_theta(S,K,T, r, sigma)
    call_vega_value = call_vega(S,K,T, r, sigma)
    call_rho_value = call_rho(S,K,T, r, sigma)

    print(f"Call Delta: {call_delta_value:.2f}")
    print(f"Call Gamma: {call_gamma_value:.2f}")
    print(f"Call Theta: {call_theta_value:.2f}")
    print(f"Call Vega: {call_vega_value:.2f}")
    print(f"Call Rho: {call_rho_value:.2f}")


volatility(S, K, T, r, implied_volatility_value)