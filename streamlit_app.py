import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from portfolio_simulator import AssetUniverse, Constraints, PortfolioSimulation

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
            num_simulations = st.number_input("Number of Monte Carlo Simulations", min_value=1000, max_value=20000, value=10000, step=1000)
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
            # Create a placeholder for the progress bar
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

            # Run simulation
            simulation = PortfolioSimulation(asset_universe, constraints)
            
            # Create a custom run method to track progress
            results = []
            for i in range(num_portfolios):
                weights = simulation.generator.generate_portfolio()
                portfolio_metrics = simulation.analyzer.calculate_portfolio_metrics(weights)
                
                returns, stds = simulation.simulator.run_simulation(
                    portfolio_metrics.return_value,
                    portfolio_metrics.std_dev
                )
                
                # Calculate median values from simulations
                median_return = np.median(returns)
                median_std = np.median(stds)
                
                # Calculate final metrics using median values
                final_result = simulation.analyzer.calculate_final_metrics(
                    median_return, 
                    median_std,
                    portfolio_metrics.weights
                )
                results.append(final_result)
                
                # Update progress every 10 portfolios
                if (i + 1) % 10 == 0 or i == num_portfolios - 1:
                    progress = (i + 1) / num_portfolios
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + 1}/{num_portfolios} portfolios")

            # Clear the status text after completion
            status_text.text("Simulation completed! Generating visualizations...")

            # Create plots
            fig = plt.figure(figsize=(12, 8))
            
            # Extract results
            returns = np.array([r.return_value for r in results])
            stds = np.array([r.std_dev for r in results])
            sharpe_ratios = np.array([r.sharpe_ratio for r in results])
            utility_scores = np.array([r.utility_score for r in results])

            # Find optimal portfolios
            max_sharpe_idx = np.argmax(sharpe_ratios)
            max_utility_idx = np.argmax(utility_scores)

            # Sort portfolios by standard deviation for better visualization
            sort_idx = np.argsort(stds)
            stds = stds[sort_idx]
            returns = returns[sort_idx]

            # Plot all portfolios with reduced alpha for better visualization
            plt.scatter(stds, returns, alpha=0.4, color='blue', label='Simulated Portfolios', s=30)

            # Calculate and plot the efficient frontier
            # Use a more sophisticated approach to find the efficient frontier
            unique_stds = np.unique(stds)
            max_returns = np.array([returns[stds == std].max() for std in unique_stds])
            
            # Smooth the efficient frontier using a polynomial fit
            z = np.polyfit(unique_stds, max_returns, 4)  # Using 4th degree polynomial
            p = np.poly1d(z)
            x_frontier = np.linspace(min(unique_stds), max(unique_stds), 100)
            y_frontier = p(x_frontier)
            
            # Plot the smoothed efficient frontier
            plt.plot(x_frontier, y_frontier, 'r--', linewidth=2, label='Efficient Frontier')

            # Highlight optimal portfolios with larger markers
            plt.scatter(stds[max_sharpe_idx], returns[max_sharpe_idx],
                      color='green', s=200, label='Max Sharpe Ratio', marker='*')
            plt.scatter(stds[max_utility_idx], returns[max_utility_idx],
                      color='orange', s=200, label='Max Utility', marker='*')

            plt.xlabel('Standard Deviation')
            plt.ylabel('Expected Return')
            plt.title('Portfolio Optimization Results')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Display the plot
            st.pyplot(fig)

            # Display optimal portfolio details
            st.subheader("Optimal Portfolios")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Maximum Sharpe Ratio Portfolio:")
                st.write(f"Return: {returns[max_sharpe_idx]:.3%}")
                st.write(f"Standard Deviation: {stds[max_sharpe_idx]:.3%}")
                st.write(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.3f}")
                st.write(f"Utility Score: {utility_scores[max_sharpe_idx]:.3f}")
                
                # Create pie chart for max Sharpe portfolio
                fig_sharpe = plt.figure(figsize=(10, 8))
                weights = results[max_sharpe_idx].weights
                asset_names = asset_universe.data['Asset Name'].values
                significant_idx = weights > 0.01
                sorted_idx = np.argsort(weights[significant_idx])[::-1]
                sorted_weights = weights[significant_idx][sorted_idx]
                sorted_names = asset_names[significant_idx][sorted_idx]
                plt.pie(sorted_weights,
                       labels=[f"{name}\n{w:.1%}" for name, w in zip(sorted_names, sorted_weights)],
                       autopct='%1.1f%%',
                       startangle=90)
                plt.title("Max Sharpe Portfolio Composition")
                st.pyplot(fig_sharpe)
            
            with col2:
                st.write("Maximum Utility Portfolio:")
                st.write(f"Return: {returns[max_utility_idx]:.3%}")
                st.write(f"Standard Deviation: {stds[max_utility_idx]:.3%}")
                st.write(f"Sharpe Ratio: {sharpe_ratios[max_utility_idx]:.3f}")
                st.write(f"Utility Score: {utility_scores[max_utility_idx]:.3f}")
                
                # Create pie chart for max utility portfolio
                fig_utility = plt.figure(figsize=(10, 8))
                weights = results[max_utility_idx].weights
                significant_idx = weights > 0.01
                sorted_idx = np.argsort(weights[significant_idx])[::-1]
                sorted_weights = weights[significant_idx][sorted_idx]
                sorted_names = asset_names[significant_idx][sorted_idx]
                plt.pie(sorted_weights,
                       labels=[f"{name}\n{w:.1%}" for name, w in zip(sorted_names, sorted_weights)],
                       autopct='%1.1f%%',
                       startangle=90)
                plt.title("Max Utility Portfolio Composition")
                st.pyplot(fig_utility)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload your asset universe CSV file to begin the simulation.") 