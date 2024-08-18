


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

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def run(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (np.log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        # GREEKS
        # Delta is the first derivative of option price with respect to underlying price
        def call_delta(current_price, strike, time_to_maturity, interest_rate, volatility):
            delta = norm.cdf(d1)
            return delta

        # Gamma is the second derivative of option price with respect to underlying price
        def call_gamma(current_price, strike, time_to_maturity, interest_rate, volatility):
            gamma = norm.pdf(d1) / (current_price * volatility * np.sqrt(time_to_maturity))
            return gamma

        # Theta is the first derivative of option price with respect to time to expiration
        def call_theta(current_price, strike, time_to_maturity, interest_rate, volatility):
            theta = -(current_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_maturity)) - interest_rate * strike * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
            return theta / 365

        # Vega is the first derivative of option price with respect to volatility σ(Sigma)
        def call_vega(current_price, strike, time_to_maturity, interest_rate, volatility):
            vega = current_price * norm.pdf(d1) * np.sqrt(time_to_maturity)
            return vega / 100

        # Rho is the first derivative of option price with respect to interest rate
        def call_rho(current_price, strike, time_to_maturity, interest_rate, volatility):
            rho = strike * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
            return rho
        
        def put_delta(current_price, strike, time_to_maturity, interest_rate, volatility):
            delta = norm.cdf(d1) - 1
            return delta

        def put_gamma(current_price, strike, time_to_maturity, interest_rate, volatility):
            gamma = norm.pdf(d1) / (current_price * volatility * np.sqrt(time_to_maturity))
            return gamma

        def put_theta(current_price, strike, time_to_maturity, interest_rate, volatility):
            theta = -(current_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_maturity)) + interest_rate * strike * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)
            return theta / 365

        def put_vega(current_price, strike, time_to_maturity, interest_rate, volatility):
            vega = current_price * norm.pdf(d1) * np.sqrt(time_to_maturity)
            return vega / 100

        def put_rho(current_price, strike, time_to_maturity, interest_rate, volatility):
            rho = -strike * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)
            return rho
        
        def call_price(current_price, strike, time_to_maturity, interest_rate, volatility):
            cPrice = current_price * norm.cdf(d1) - (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2))
            return cPrice
            
        def put_price(current_price, strike, time_to_maturity, interest_rate, volatility):
            pPrice = (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)) - current_price * norm.cdf(-d1)
            return pPrice

        def sigma_calls(current_price,strike, time_to_maturity, interest_rate, volatility):
            call_delta_value = call_delta(current_price, strike, time_to_maturity, interest_rate, volatility)
            call_gamma_value = call_gamma(current_price, strike, time_to_maturity, interest_rate, volatility)
            call_theta_value = call_theta(current_price, strike, time_to_maturity, interest_rate, volatility)
            call_vega_value = call_vega(current_price, strike, time_to_maturity, interest_rate, volatility)
            call_rho_value = call_rho(current_price, strike, time_to_maturity, interest_rate, volatility)
            call_price_value = call_price(current_price, strike, time_to_maturity, interest_rate, volatility)


            print(f"Call Delta: {call_delta_value:.2f}")
            print(f"Call Gamma: {call_gamma_value:.2f}")
            print(f"Call Theta: {call_theta_value:.2f}")
            print(f"Call Vega: {call_vega_value:.2f}")
            print(f"Call Rho: {call_rho_value:.2f}")
            print(f"Call Price: ${call_price_value:.2f}")

        def sigma_puts(current_price, strike, time_to_maturity, interest_rate, volatility):
            put_delta_value = put_delta(current_price, strike, time_to_maturity, interest_rate, volatility)
            put_gamma_value = put_gamma(current_price, strike, time_to_maturity, interest_rate, volatility)
            put_theta_value = put_theta(current_price, strike, time_to_maturity, interest_rate, volatility)
            put_vega_value = put_vega(current_price, strike, time_to_maturity, interest_rate, volatility)
            put_rho_value = put_rho(current_price, strike, time_to_maturity, interest_rate, volatility)
            put_price_value = put_price(current_price, strike, time_to_maturity, interest_rate, volatility)

            print(f"Put Delta: {put_delta_value:.2f}")
            print(f"Put Gamma: {put_gamma_value:.2f}")
            print(f"Put Theta: {put_theta_value:.2f}")
            print(f"Put Vega: {put_vega_value:.2f}")
            print(f"Put Rho: {put_rho_value:.2f}")
            print(f"Put Price: ${put_price_value:.2f}")

        sigma_puts(current_price, strike, time_to_maturity, interest_rate, volatility)   
        print()
        print()
        sigma_calls(current_price, strike, time_to_maturity, interest_rate, volatility)  


if __name__ == "__main__":
    # ticker= yf.Ticker("AAPL")
    # # aapl.options # list of dates 
    # DF_calls = ticker.option_chain().calls
    # DF_puts = ticker.option_chain().puts

    # print(DF_calls)
    # print("\n")
    # print(DF_puts)
    
    
    time_to_maturity = 7/365
    strike = 1210
    current_price = 1208
    volatility = 0.57
    interest_rate = 2.13/100

    # Black Scholes
    BS = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate)
    print(BS.run())