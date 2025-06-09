import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio_simulator import AssetUniverse, Constraints, PortfolioGenerator

st.set_page_config(page_title="Portfolio Simulator", layout="wide")

st.title("Portfolio Simulation Tool")
st.markdown("""
This tool performs portfolio optimization and Monte Carlo simulations based on user-defined asset constraints.
Upload your asset universe CSV file to get started. [Asset Name, Asset Class, Sector, Region, Expected Return, Standard Deviation]
""")

# File uploader
uploaded_file = st.file_uploader("Upload your asset universe CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Create AssetUniverse instance
        asset_universe = AssetUniverse(uploaded_file)
        
        # Display asset universe summary in an expander
        with st.expander("Asset Universe Summary", expanded=True):
            st.write("Total number of assets:", len(asset_universe.data))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Asset Classes:")
                st.write(asset_universe.data['Asset Class'].value_counts())
            
            with col2:
                st.write("Sectors:")
                st.write(asset_universe.data['Sector'].value_counts())
            
            with col3:
                st.write("Regions:")
                st.write(asset_universe.data['Region'].value_counts())
            
            st.write("Return Statistics:")
            st.write(f"Average Expected Return: {asset_universe.data['Expected Return'].mean():.2%}")
            st.write(f"Average Standard Deviation: {asset_universe.data['Standard Deviation'].mean():.2%}")

        # Simulation parameters
        st.header("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_portfolios = st.number_input("Number of Portfolios", min_value=100, max_value=1000, value=500, step=100)
            num_simulations = st.number_input("Number of Monte Carlo Simulations per Portfolio", min_value=1000, max_value=20000, value=10000, step=1000)
            risk_free_rate = st.number_input("Risk-free Rate", min_value=0.0, max_value=0.2, value=0.04, step=0.01, format="%.3f")
            utility_lambda = st.number_input("Utility Lambda (Risk Aversion)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

        # Create constraints based on unique values in the data
        unique_asset_classes = asset_universe.data['Asset Class'].unique()
        unique_sectors = asset_universe.data['Sector'].unique()
        unique_regions = asset_universe.data['Region'].unique()

        st.header("Portfolio Constraints")

        # Asset Class Constraints
        st.subheader("Asset Class Constraints")
        asset_class_ranges = {}
        for asset_class in unique_asset_classes:
            col1, col2 = st.columns(2)
            with col1:
                min_weight = st.number_input(f"Min {asset_class}", 0.0, 1.0, 0.0, 0.1, format="%.2f", key=f"min_{asset_class}")
            with col2:
                max_weight = st.number_input(f"Max {asset_class}", min_weight, 1.0, 1.0, 0.1, format="%.2f", key=f"max_{asset_class}")
            asset_class_ranges[asset_class] = (min_weight, max_weight)

        # Sector Constraints
        st.subheader("Sector Constraints")
        sector_ranges = {}
        for sector in unique_sectors:
            col1, col2 = st.columns(2)
            with col1:
                min_weight = st.number_input(f"Min {sector}", 0.0, 1.0, 0.0, 0.1, format="%.2f", key=f"min_{sector}")
            with col2:
                max_weight = st.number_input(f"Max {sector}", min_weight, 1.0, 1.0, 0.1, format="%.2f", key=f"max_{sector}")
            sector_ranges[sector] = (min_weight, max_weight)

        # Region Constraints
        st.subheader("Region Constraints")
        region_ranges = {}
        for region in unique_regions:
            col1, col2 = st.columns(2)
            with col1:
                min_weight = st.number_input(f"Min {region}", 0.0, 1.0, 0.0, 0.1, format="%.2f", key=f"min_{region}")
            with col2:
                max_weight = st.number_input(f"Max {region}", min_weight, 1.0, 1.0, 0.1, format="%.2f", key=f"max_{region}")
            region_ranges[region] = (min_weight, max_weight)

        # Run simulation button
        if st.button("Run Simulation"):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create constraints
            constraints = Constraints(
                asset_class_ranges=asset_class_ranges,
                sector_ranges=sector_ranges,
                region_ranges=region_ranges,
                num_portfolios=num_portfolios,
                num_simulations=num_simulations,
                risk_free_rate=risk_free_rate,
                utility_lambda=utility_lambda
            )

            # Create portfolio generator
            generator = PortfolioGenerator(asset_universe, constraints)

            # Initialize arrays to store results
            all_weights = []
            all_returns = []
            all_stdevs = []
            all_sharpe_ratios = []
            all_utility_scores = []

            # Run portfolio simulations
            for i in range(num_portfolios):
                # Generate portfolio weights
                weights = generator.generate_portfolio()
                all_weights.append(weights)
                
                # Get expected returns and standard deviations for the portfolio
                exp_returns = asset_universe.data['Expected Return'].values
                exp_stdevs = asset_universe.data['Standard Deviation'].values
                
                # Calculate portfolio's expected return and standard deviation
                portfolio_exp_return = np.sum(weights * exp_returns)
                portfolio_exp_stdev = np.sqrt(np.sum(weights**2 * exp_stdevs**2))
                
                # Convert to monthly parameters
                monthly_exp_return = portfolio_exp_return / 12
                monthly_exp_stdev = portfolio_exp_stdev / np.sqrt(12)
                
                # Run Monte Carlo simulation for this portfolio
                simulated_annual_returns = []
                
                for _ in range(num_simulations):
                    # Generate 12 monthly returns
                    monthly_returns = np.random.normal(monthly_exp_return, monthly_exp_stdev, 12)
                    
                    # Calculate annual return through compounding
                    annual_return = np.prod(1 + monthly_returns) - 1
                    simulated_annual_returns.append(annual_return)
                
                # Convert to numpy array for calculations
                simulated_annual_returns = np.array(simulated_annual_returns)
                
                # Calculate median return and standard deviation of returns
                median_return = np.median(simulated_annual_returns)
                realized_stdev = np.std(simulated_annual_returns)
                
                # Calculate Sharpe ratio and utility score
                sharpe_ratio = (median_return - risk_free_rate) / realized_stdev if realized_stdev > 0 else 0
                utility_score = median_return - (utility_lambda / 2) * realized_stdev**2
                
                # Store results
                all_returns.append(median_return)
                all_stdevs.append(realized_stdev)
                all_sharpe_ratios.append(sharpe_ratio)
                all_utility_scores.append(utility_score)
                
                # Update progress
                if (i + 1) % 10 == 0 or i == num_portfolios - 1:
                    progress = (i + 1) / num_portfolios
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + 1}/{num_portfolios} portfolios")

            # Convert lists to numpy arrays
            all_returns = np.array(all_returns)
            all_stdevs = np.array(all_stdevs)
            all_sharpe_ratios = np.array(all_sharpe_ratios)
            all_utility_scores = np.array(all_utility_scores)
            all_weights = np.array(all_weights)

            # Find optimal portfolios
            max_sharpe_idx = np.argmax(all_sharpe_ratios)
            max_utility_idx = np.argmax(all_utility_scores)

            # Create visualization
            fig = plt.figure(figsize=(12, 8))
            
            # Plot all portfolios
            plt.scatter(all_stdevs, all_returns, alpha=0.4, color='blue', label='Simulated Portfolios', s=30)
            
            # Plot efficient frontier
            # Sort by standard deviation for smooth curve
            sort_idx = np.argsort(all_stdevs)
            sorted_stdevs = all_stdevs[sort_idx]
            sorted_returns = all_returns[sort_idx]
            
            # Find maximum return for each unique standard deviation
            unique_stdevs = np.unique(sorted_stdevs)
            max_returns = [np.max(all_returns[all_stdevs == std]) for std in unique_stdevs]
            
            # Fit a polynomial to create smooth efficient frontier
            z = np.polyfit(unique_stdevs, max_returns, 4)
            p = np.poly1d(z)
            x_frontier = np.linspace(min(unique_stdevs), max(unique_stdevs), 100)
            y_frontier = p(x_frontier)
            
            # Plot the efficient frontier
            plt.plot(x_frontier, y_frontier, 'r--', linewidth=2, label='Efficient Frontier')
            
            # Highlight optimal portfolios
            plt.scatter(all_stdevs[max_sharpe_idx], all_returns[max_sharpe_idx],
                      color='green', s=200, label='Max Sharpe Ratio', marker='*')
            plt.scatter(all_stdevs[max_utility_idx], all_returns[max_utility_idx],
                      color='orange', s=200, label='Max Utility', marker='*')

            plt.xlabel('Standard Deviation (Risk)')
            plt.ylabel('Expected Return')
            plt.title('Portfolio Optimization Results')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Display the plot
            st.pyplot(fig)

            # Display optimal portfolio details
            st.subheader("Optimal Portfolios")
            
            col1, col2 = st.columns(2)
            
            def display_portfolio_details(col, portfolio_type, idx, weights):
                with col:
                    st.write(f"{portfolio_type} Portfolio:")
                    st.write(f"Return: {all_returns[idx]:.3%}")
                    st.write(f"Standard Deviation: {all_stdevs[idx]:.3%}")
                    st.write(f"Sharpe Ratio: {all_sharpe_ratios[idx]:.3f}")
                    st.write(f"Utility Score: {all_utility_scores[idx]:.3f}")
                    
                    # Create pie chart
                    fig = plt.figure(figsize=(10, 8))
                    significant_idx = weights > 0.01  # Show only weights > 1%
                    significant_weights = weights[significant_idx]
                    significant_names = asset_universe.data['Asset Name'].values[significant_idx]
                    
                    # Sort by weight for better visualization
                    sort_idx = np.argsort(significant_weights)[::-1]
                    sorted_weights = significant_weights[sort_idx]
                    sorted_names = significant_names[sort_idx]
                    
                    plt.pie(sorted_weights,
                           labels=[f"{name}\n{w:.1%}" for name, w in zip(sorted_names, sorted_weights)],
                           autopct='%1.1f%%',
                           startangle=90)
                    plt.title(f"{portfolio_type} Portfolio Composition")
                    st.pyplot(fig)
                    
                    # Display full portfolio composition table
                    st.write("Full Portfolio Composition:")
                    portfolio_df = pd.DataFrame({
                        'Asset': asset_universe.data['Asset Name'],
                        'Weight': weights,
                        'Asset Class': asset_universe.data['Asset Class'],
                        'Sector': asset_universe.data['Sector'],
                        'Region': asset_universe.data['Region']
                    })
                    portfolio_df = portfolio_df[portfolio_df['Weight'] > 0].sort_values('Weight', ascending=False)
                    portfolio_df['Weight'] = portfolio_df['Weight'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(portfolio_df)

            # Display details for both optimal portfolios
            display_portfolio_details(col1, "Maximum Sharpe Ratio", max_sharpe_idx, all_weights[max_sharpe_idx])
            display_portfolio_details(col2, "Maximum Utility", max_utility_idx, all_weights[max_utility_idx])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload your asset universe CSV file to begin the simulation.") 