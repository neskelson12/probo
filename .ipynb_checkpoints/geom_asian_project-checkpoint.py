from probo.marketdata import MarketData
from probo.payoff import ExoticPayoff, geometricAsianCallPayoff
from probo.engine import MonteCarloEngine, NaiveMonteCarloPricer, PathwiseNaiveMonteCarloPricer, ControlVariatePricer
from probo.facade import OptionFacade
import time

## Set up the market data
spot = 100.0
rate = 0.06
volatility = 0.20
dividend = 0.03
thedata = MarketData(rate, spot, volatility, dividend)

## Set up the option
expiry = 1
strike = 100.0
the_simple_mc_call = ExoticPayoff(expiry, strike, geometricAsianCallPayoff)
the_geom_mc_call = ExoticPayoff(expiry, strike, geometricAsianCallPayoff)

## Set up Naive Monte Carlo
nreps = 10000
steps = 10
# pricer = NaiveMonteCarloPricer
pricer = PathwiseNaiveMonteCarloPricer
mcengine = MonteCarloEngine(nreps, steps, pricer)
pricer2 = ControlVariatePricer
mcengine2 =  MonteCarloEngine(nreps, steps, pricer2)

## Calculate the price
option1 = OptionFacade(the_simple_mc_call, mcengine, thedata)
time_start1 = time.clock()
price1 = option1.price()
time_elapsed1 = (time.clock() - time_start1)
# print("The call price via Simple Monte Carlo is: {0:.4f}. Standard Error is: {1:.4f} Elapsed Time: {2:.4f} seconds.".format(*price1, time_elapsed1))

option2 = OptionFacade(the_geom_mc_call, mcengine2, thedata)
time_start2 = time.clock()
price2 = option2.price()
time_elapsed2 = (time.clock() - time_start2)
# print("The call price via Geometric Asian Control Variate Monte Carlo is: {0:.4f}. Standard Error is: {1:.4f}. Elapsed Time: {2:.4f} seconds.".format(*price2, time_elapsed2))

print('| Name Of Option                | Call Option Price | Standard Error     | Relative Computation Time |')
print('| ----------------------------- | ----------------- | ------------------ | ------------------------- |')
print('| Simple Monte Carlo            | {0:.4f}            | {1:.4f}             | {2:.4f}                    |'.format(*price1, time_elapsed1))
print('| Geometric Asian Monte Carlo   | {0:.4f}            | {1:.4f}             | {2:.4f}                    |'.format(*price2, time_elapsed2))