import numpy as np
import scipy
import pandas as pd

## Pipeline
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output

## Data set
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

## Filter/factor/classifier
from quantopian.pipeline.filters.morningstar import Q1500US

def make_pipeline():
    # Factors
    mkt_cap = morningstar.valuation.market_cap.latest
    
    # Filters
    base_universe = Q1500US()
    high_mkt_cap = (mkt_cap > 1e8) & mkt_cap.top(500)

    return Pipeline(
      columns={
        'Market Cap': mkt_cap
      },
      screen = base_universe & high_mkt_cap
    )

def initialize(context):
    context.lookback = 300
    context.return_window = 60
    context.long_leverage = 0.5
    context.short_leverage = -0.5
    
    # We will rebalance once at the start of each month at market open. 
    schedule_function(rebalance,
                      date_rules.month_start(),
                      time_rules.market_open())
    
    # After our pipeline is attached, it will be run once
    # each day of the simulation. Attaching our pipeline will
    # produce a new output dataframe each day. 
    attach_pipeline(make_pipeline(), 'pipeline')
    
def before_trading_start(context, data):
    # Get the security list for the current date in the simulation. 
    context.output = pipeline_output('pipeline')
    context.security_list = context.output.index

    # Get historical prices (a pandas DataFrame indexed by date, with
    # assets as columns)
    hist_prices = data.history(context.security_list,
                               'price',
                               context.lookback,
                               '1d')
    
    # Drop the assets with a missing price and calculate logarithmic prices.
    prices = np.log(hist_prices).dropna(axis=1)
    
    # Calculate log returns
    R = (prices - prices.shift(context.return_window)).dropna()
    R = R[np.isfinite(R[R.columns])].fillna(0)

    # Subtract the cross-sectional average out of each data point on each day.
    ranks = (R.T - R.T.mean()).T.mean()
    
    # Take the top and botton percentiles for the long and short baskets
    lower, upper = ranks.quantile([.05, .95])
    context.shorts = ranks[ranks <= lower].index.tolist()
    context.longs = ranks[ranks >= upper].index.tolist()
    
    # Calculate weights
    context.long_weight, context.short_weight = compute_weights(context)

def compute_weights(context):
    # Compute even target weights for our long positions and short positions.
    long_weight = context.long_leverage / len(context.longs)
    short_weight = context.short_leverage / len(context.shorts)

    return long_weight, short_weight

def rebalance(context, data):
    # Exit positions that are no longer wanted
    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)

    # Enter new positions or rebalance old ones
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, context.long_weight)

    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, context.short_weight)
