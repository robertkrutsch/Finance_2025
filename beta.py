import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def visualize_beta(stock_prices, market_prices, fn="beta_visualization.png"):
    stock_prices = np.array(stock_prices)
    market_prices = np.array(market_prices)
    
    # Calculate daily returns
    stock_returns = (stock_prices[1:] - stock_prices[:-1]) / stock_prices[:-1]
    market_returns = (market_prices[1:] - market_prices[:-1]) / market_prices[:-1]
    
    # Linear regression for beta
    beta, alpha = np.polyfit(market_returns, stock_returns, 1)
    predicted = alpha + beta * market_returns
    residuals = stock_returns - predicted
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((stock_returns - np.mean(stock_returns))**2)
    r_squared = 1 - ss_res/ss_tot
    
    # Plot scatter and regression
    plt.figure(figsize=(10, 6))
    plt.scatter(market_returns, stock_returns, alpha=0.6, label='Data points')
    x_vals = np.linspace(min(market_returns), max(market_returns), 100)
    plt.plot(x_vals, alpha + beta * x_vals, color='red', label=f'Beta={beta:.2f}, R²={r_squared:.2f}')
    
    # Shade ±1 std of residuals
    std_res = np.std(residuals)
    plt.fill_between(x_vals, alpha + beta*x_vals - std_res, alpha + beta*x_vals + std_res,
                     color='red', alpha=0.2, label='±1 Std of residuals')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Market Returns")
    plt.ylabel("Stock Returns")
    plt.title("Stock vs Market Returns with Beta Regression")
    plt.legend()
    plt.grid(True)
    plt.savefig(fn)
    #plt.show()
    
    # Histogram of residuals
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    #plt.show()
    
    print(f"Beta: {beta:.4f}, R²: {r_squared:.4f}, Residual Std Dev: {std_res:.4f}")
    return beta, r_squared, std_res

def calculate_wacc(equity_weight, debt_weight, cost_of_equity, cost_of_debt, tax_rate):
    """
    Computes the Weighted Average Cost of Capital (WACC).

    Args:
    equity_weight (float): Weight of equity in capital structure (E/V).
    debt_weight (float): Weight of debt in capital structure (D/V).
    cost_of_equity (float): Cost of equity as a decimal (e.g. 0.08 for 8%).
    cost_of_debt (float): Pre-tax cost of debt as a decimal.
    tax_rate (float): Corporate tax rate as a decimal.

    Returns:
    float: WACC as a decimal.
    """
    if abs((equity_weight + debt_weight) - 1.0) > 1e-6:
        raise ValueError("Equity weight and debt weight must sum to 1.")

    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    return wacc

def re_lever_beta(beta_unlevered, debt_weight, equity_weight, tax_rate):
    """
    Re-levers beta using Hamada equation.
    """
    d_to_e = debt_weight / equity_weight if equity_weight > 0 else 0
    return beta_unlevered * (1 + (1 - tax_rate) * d_to_e)


market_prices2 = [23938.27,
24032.43,
23943.19,
24036.75,
22591.7,
22305.01,
22544.9,
21632.16,
19836.37,
19644.1,
19091.28,
19386.12,
18950.61,
18479.02,
18243.04,
18592.72,
17904.92,
18503.65,
17740.54,
16812.85,
16775.62,
16252.53,
14847.26,
15354.62,
15950.86,
16464.81,
16183.82,
15712.76,
15948.27,
15649,
15352.46,
15192.29,
13946.64,
14571.86,
13254.22,
11965.34,
12817.61,
13543.2,
12829.91,
14426.72,
13938.14,
14310.83,
14299.05,
15527.12,
15857.3,
15162.34,
15712.96,
15229.21,
15824.24,
15568.88,
15573.49,
15436.66,
15117.19,
15027.25,
13773.15,
13430.33,
13652.7,
13273.73,
11639.25,
12714.02,
12926.3,
    
]

market_prices1 = [ 23902.21,
24065.47,
23909.61,
23997.48,
22496.98,
22163.49,
22551.43,
21732.05,
19909.14,
19626.45,
19077.54,
19324.93,
18906.92,
18508.65,
18235.45,
18497.94,
17932.17,
18492.49,
17678.19,
16903.76,
16751.64,
16215.43,
14829.62,
15386.58,
15947.08,
16446.83,
16147.90,
15664.02,
15922.38,
15628.84,
15365.14,
15128.27,
13923.59,
14397.04,
13253.74,
12114.36,
12834.96,
13484.05,
12783.77,
14388.35,
14097.88,
14414.75,
14461.02,
15471.20,
15884.86,
15100.13,
15688.77,
15260.69,
15835.09,
15544.39,
15531.04,
15421.13,
15135.91,
15008.34,
13786.29,
13432.87,
13718.78,
13291.16,
11556.48,
12760.73,
12945.38]


stock_prices2 = [55.6,
59.2,
58.8,
55.4,
49.6,
52.6,
45.4,
47.4,
41.3,
41,
41.6,
43.3,
44.5,
45.7,
44.5,
47,
43.6,
45.3,
41.4,
42.3,
44.6,
44.9,
42.5,
38.1,
38,
40.6,
37.6,
38.8,
41.6,
39.1,
36.9,
37.25,
36.75,
38.65,
36.6,
37.55,
40.15,
44.6,
42.4,
42.35,
43.3,
48.25,
47,
49.1,
54.1,
51.6,
64.6,
67.1,
72.3,
75.3,
78.5,
73.6,
73.2,
64.6,
61.8,
68.4,
59.4,
59.8,
61.4,
62.8,
61.6,
]

stock_prices1 = [55.6,
57.6,
57.4,
55.2,
48.7,
51.8,
45.2,
46.2,
41.2,
41.1,
41.9,
42.6,
43.3,
44.3,
45.1,
47,
43.6,
45.1,
40.9,
41.7,
43.7,
44.7,
41.1,
38.1,
37.7,
40.2,
37.7,
38.7,
41.5,
38.2,
36.7,
37.45,
36.3,
38.85,
36.15,
37.5,
40.6,
44.75,
41.75,
45.7,
44.05,
49.25,
47.1,
48.95,
54.2,
51,
65,
65.6,
71.3,
74.9,
77.9,
73.9,
73.4,
64.4,
62.2,
67,
58.2,
59.6,
61.2,
62.6,
61.8]


try:
    beta, r_squared, std_res = visualize_beta(stock_prices1, market_prices1,fn="frankfurt.png")
    print(f"The calculated beta frankfurt is : {beta:.2f}")

    cost_equity = 2.71/100 + beta * 4.33/100
    print(f"The cost of equity frankfurt is : {cost_equity*100:.2f} %")


    beta, r_squared, std_res = visualize_beta(stock_prices2, market_prices2,fn="ariva.png")
    print(f"The calculated beta ariva is: {beta:.2f}")
    cost_equity = 2.71/100 + beta * 4.33/100
    print(f"The cost of equity ariva is : {cost_equity*100:.2f} %")
except ValueError as e:
    print(f"Error: {e}")

stock_diff = (np.array(stock_prices2) - np.array(stock_prices1))/np.array(stock_prices2) 
market_diff = (np.array(market_prices2) - np.array(market_prices1))/np.array(market_prices2)

plt.figure(figsize=(12, 8))

# Stock price differences subplot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
plt.plot(stock_diff, label='Stock Prices Difference', marker='o', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylabel('Stock Price Difference')
plt.title('Stock Prices Difference Over Time')
plt.legend()
plt.grid(True)

# Market price differences subplot
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
plt.plot(market_diff, label='Market Prices Difference', marker='x', color='green')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Time Index')
plt.ylabel('Market Price Difference')
plt.title('Market Prices Difference Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust spacing between plots
plt.savefig("price_differences_subplot.png")
#plt.show()


# Example usage:
equity_weight = 0.737 # 73.7%
debt_weight = 0.263 # 26.3%
cost_of_equity = 0.0556 # 5.56%
cost_of_debt = 0.0473 # 4.73%
tax_rate = 0.289 # 28.9%

wacc_value = calculate_wacc(equity_weight, debt_weight, cost_of_equity, cost_of_debt, tax_rate)
print(f"WACC: {wacc_value * 100:.2f}%")

# Sensitivity analysis: vary debt ratio from 0% to 60%
debt_ratios = np.linspace(0, 0.6, 50)
wacc_values = []
for d in debt_ratios:
    e = 1 - d
    wacc_values.append(calculate_wacc(e, d, cost_of_equity, cost_of_debt, tax_rate))

# Calculate 20% increase in debt ratio
debt_ratio_increased = min(debt_weight * 1.2, 0.99) # cap at <100%
equity_ratio_increased = 1 - debt_ratio_increased
wacc_after_increase = calculate_wacc(equity_ratio_increased, debt_ratio_increased, cost_of_equity, cost_of_debt, tax_rate)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(debt_ratios, np.array(wacc_values) * 100, label='WACC vs Debt Ratio')
plt.axvline(debt_weight, color='blue', linestyle='--', label=f'Current Debt Ratio ({debt_weight:.2f})')
plt.scatter(debt_ratio_increased, wacc_after_increase * 100, color='red', zorder=5, label=f'20% Higher Debt Ratio\nWACC={wacc_after_increase*100:.2f}%')
plt.xlabel('Debt Ratio (D/V)')
plt.ylabel('WACC (%)')
plt.title('WACC Sensitivity to Debt Ratio')
plt.legend()
plt.grid(True)
plt.savefig("wacc_debt_ratio.png")
#plt.show()




# Inputs
rf = 0.0271 # risk-free rate 2.5%
erp = 0.0433 # equity risk premium 3.4%
beta_levered = 0.66

# Capital structure
current_debt_weight = 0.263
current_equity_weight = 1 - current_debt_weight
tax_rate = 0.289

# === INPUTS ===
market_cap = 1_110_000_000 # Market capitalization (EUR)
debt_book = 395_581_000 # Current debt (EUR)
EBIT = 194_000_000 # Operating profit (EUR)
base_cost_of_debt = 0.0473# Observed pre-tax cost of debt (4.73%)

# === HELPER FUNCTIONS ===
def spread_from_icr(icr):
    if icr >= 12.5:
        return 0.0075
    elif icr >= 9.5:
        return 0.01
    elif icr >= 7.5:
        return 0.015
    elif icr >= 4.5:
        return 2/100
    elif icr >= 3.5:
        return 2.25/100
    elif icr >= 3:
        return 3.5/100
    elif icr >= 2.5:
        return 4.75/100
    elif icr >= 2:
        return 6.5/100
    elif icr >= 1.5:
        return 8/100
    elif icr >= 1.25:
        return 10/100
    elif icr >= 0.8:
        return 11.5/100
    elif icr >= 0.5:
        return 12.7/100
    else:
        return 14/100
    
    

# === STEP 1: UNLEVER BETA ===
V = market_cap + debt_book
current_d = debt_book / V
current_e = 1 - current_d
current_d_to_e = current_d / current_e
beta_unlevered = beta_levered / (1 + (1 - tax_rate) * current_d_to_e)
print(f"Unlevered Beta: {beta_unlevered:.3f}")

# === STEP 2: SENSITIVITY ANALYSIS ===
debt_ratios = np.linspace(0, 0.7, 200)
wacc_values = []
rd_values = []
re_values = []
icr_values = []

for d in debt_ratios:
    D = d * V
    E = V - D

    Rd = base_cost_of_debt
    
    for tmp in range(50):
        interest = Rd * D
        icr = EBIT / interest
        spread = spread_from_icr(icr)
        Rd_new = rf + spread
        if abs(Rd_new - Rd) < 1e-8:
            Rd = Rd_new
            break
        Rd = Rd_new
    print(tmp)

    d_to_e = (d / (1 - d)) if (1 - d) > 0 else 1e6
    beta_re = beta_unlevered * (1 + (1 - tax_rate) * d_to_e)
    Re = rf + beta_re * erp

    WACC = (1 - d) * Re + d * Rd * (1 - tax_rate)

    wacc_values.append(WACC)
    rd_values.append(Rd)
    re_values.append(Re)
    icr_values.append(icr)

wacc_values = np.array(wacc_values)
rd_values = np.array(rd_values)
re_values = np.array(re_values)
icr_values = np.array(icr_values)

# === STEP 3: FIND OPTIMUM ===
opt_idx = np.argmin(wacc_values)
opt_d = debt_ratios[opt_idx]
opt_wacc = wacc_values[opt_idx]

print(f"Optimal Debt Ratio: {opt_d*100:.1f}%")
print(f"Minimum WACC: {opt_wacc*100:.2f}%")
print(f"Cost of Debt at Optimal: {rd_values[opt_idx]*100:.2f}%")
print(f"Cost of Equity at Optimal: {re_values[opt_idx]*100:.2f}%")
print(f"ICR at Optimal: {icr_values[opt_idx]:.2f}x")

# === PLOTS WITH ANNOTATIONS ===
plt.figure(figsize=(9, 6))
plt.plot(debt_ratios * 100, wacc_values * 100, label='WACC')
plt.scatter(opt_d * 100, opt_wacc * 100, color='red', zorder=5)
plt.annotate(f'Optimal\n{opt_d*100:.1f}%, {opt_wacc*100:.2f}%',
xy=(opt_d*100, opt_wacc*100), xytext=(opt_d*100+3, opt_wacc*100+0.2),
arrowprops=dict(arrowstyle='->'))
plt.axvline(current_d * 100, color='blue', linestyle='--')
plt.annotate(f'Current\n{current_d*100:.1f}%', xy=(current_d*100, opt_wacc*100),
xytext=(current_d*50+3, opt_wacc*90+0.5),
arrowprops=dict(arrowstyle='->'))
plt.xlabel('Debt Ratio (D/V %)')
plt.ylabel('WACC (%)')
plt.title('WACC vs Debt Ratio with ICR-based Rd')
plt.grid(True)
plt.legend()
plt.savefig("wacc_optimal_debt_ratio.png")
plt.show()

plt.figure(figsize=(9, 6))
plt.plot(debt_ratios * 100, rd_values * 100, label='Cost of Debt (Rd)')
plt.plot(debt_ratios * 100, re_values * 100, label='Cost of Equity (Re)')
plt.scatter(opt_d * 100, rd_values[opt_idx] * 100, color='red')
plt.annotate(f'Optimal Rd\n{rd_values[opt_idx]*100:.2f}%',
xy=(opt_d*100, rd_values[opt_idx]*100),
xytext=(opt_d*100+3, rd_values[opt_idx]*100+1),
arrowprops=dict(arrowstyle='->'))
plt.axvline(current_d * 100, color='blue', linestyle='--')
plt.xlabel('Debt Ratio (D/V %)')
plt.ylabel('Cost (%)')
plt.title('Cost of Debt and Equity vs Debt Ratio')
plt.grid(True)
plt.legend()
plt.savefig("cost_of_debt_equity_vs_debt_ratio.png")
plt.show()

plt.figure(figsize=(9, 6))
plt.plot(debt_ratios * 100, icr_values, label='ICR (EBIT / Interest)')
plt.scatter(opt_d * 100, icr_values[opt_idx], color='red')
plt.annotate(f'Optimal ICR\n{icr_values[opt_idx]:.2f}x',
xy=(opt_d*100, icr_values[opt_idx]),
xytext=(opt_d*100+3, icr_values[opt_idx]+1),
arrowprops=dict(arrowstyle='->'))
plt.axvline(current_d * 100, color='blue', linestyle='--')
plt.xlabel('Debt Ratio (D/V %)')
plt.ylabel('Interest Coverage (x)')
plt.title('Interest Coverage vs Debt Ratio')
plt.grid(True)
plt.legend()
plt.savefig("icr_vs_debt_ratio.png")
plt.show()