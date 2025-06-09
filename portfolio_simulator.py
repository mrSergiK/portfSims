import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpldatacursor import datacursor
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

class PortfolioResult(NamedTuple):
    return_value: float
    std_dev: float
    weights: np.ndarray  # Add weights to store portfolio composition

class SimulationResult(NamedTuple):
    return_value: float
    std_dev: float
    sharpe_ratio: float
    utility_score: float
    weights: np.ndarray  # Add weights to store portfolio composition

@dataclass
class Constraints:
    asset_class_ranges: Dict[str, Tuple[float, float]]
    sector_ranges: Dict[str, Tuple[float, float]]
    region_ranges: Dict[str, Tuple[float, float]]
    num_portfolios: int
    num_simulations: int
    risk_free_rate: float
    utility_lambda: float

class AssetUniverse:
    def __init__(self, file_path: str):
        """Initialize asset universe from CSV file"""
        self.data = pd.read_csv(file_path)
        self.validate_data()
        self._convert_numeric_columns()
        self._filter_negative_returns()
        self._print_summary()
        
    def validate_data(self):
        """Validate that the data has all required columns"""
        required_columns = [
            'Asset Name', 'Asset Class', 'Sector', 'Region',
            'Expected Return', 'Standard Deviation'
        ]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
    def _convert_numeric_columns(self):
        """Convert numeric columns to float type"""
        numeric_columns = ['Expected Return', 'Standard Deviation']
        for col in numeric_columns:
            try:
                # Convert string values to numeric, handling various formats
                def convert_value(val):
                    if isinstance(val, (int, float)):
                        return float(val)
                    # Remove any spaces and % signs
                    val = str(val).strip().rstrip('%')
                    # Handle parentheses notation for negative numbers
                    if val.startswith('(') and val.endswith(')'):
                        val = '-' + val[1:-1]
                    return float(val) / 100.0

                self.data[col] = self.data[col].apply(convert_value)
            except ValueError as e:
                raise ValueError(f"Error converting {col} to numeric values. Please ensure all values are numbers, percentages, or in parentheses for negative values. Error: {e}")
                
    def _filter_negative_returns(self):
        """Remove assets with negative expected returns"""
        positive_returns = self.data['Expected Return'] >= 0
        self.data = self.data[positive_returns].reset_index(drop=True)
        
    def _print_summary(self):
        """Print summary of the asset universe"""
        print("\nAsset Universe Summary:")
        print(f"Total number of assets: {len(self.data)}")
        print("\nAsset Classes:")
        print(self.data['Asset Class'].value_counts())
        print("\nSectors:")
        print(self.data['Sector'].value_counts())
        print("\nRegions:")
        print(self.data['Region'].value_counts())
        print("\nReturn Statistics:")
        print(f"Average Expected Return: {self.data['Expected Return'].mean():.2%}")
        print(f"Average Standard Deviation: {self.data['Standard Deviation'].mean():.2%}")
        print("\n")

class PortfolioGenerator:
    def __init__(self, asset_universe: AssetUniverse, constraints: Constraints):
        self.asset_universe = asset_universe
        self.constraints = constraints
        self.max_attempts = 10000
        
    def generate_portfolio(self) -> np.ndarray:
        """Generate a single portfolio that satisfies all constraints"""
        data = self.asset_universe.data
        n_assets = len(data)
        
        for attempt in range(self.max_attempts):
            try:
                # Start with sparse random weights (many will be 0)
                weights = np.zeros(n_assets)
                # Randomly select ~20-40% of assets to have non-zero weights initially
                active_assets = np.random.choice(
                    n_assets,
                    size=np.random.randint(n_assets // 5, n_assets // 2),
                    replace=False
                )
                weights[active_assets] = np.random.random(len(active_assets))
                weights = weights / np.sum(weights)
                
                # Handle constraints in sequence: Asset Class -> Sector -> Region
                # Asset Class constraints
                class_adjustments_needed = True
                max_class_attempts = 50
                class_attempt = 0
                
                while class_adjustments_needed and class_attempt < max_class_attempts:
                    class_adjustments_needed = False
                    for asset_class, (min_weight, max_weight) in self.constraints.asset_class_ranges.items():
                        indices = data['Asset Class'] == asset_class
                        current_weight = weights[indices].sum()
                        
                        if current_weight < min_weight:
                            # If below minimum, increase weights
                            if weights[indices].sum() == 0:
                                # If no assets selected, randomly select some
                                class_assets = np.where(indices)[0]
                                active_assets = np.random.choice(
                                    class_assets,
                                    size=max(1, len(class_assets) // 3),
                                    replace=False
                                )
                                weights[active_assets] = min_weight / len(active_assets)
                            else:
                                # Scale up existing weights
                                scale = min_weight / current_weight
                                weights[indices] *= scale
                            class_adjustments_needed = True
                        elif current_weight > max_weight:
                            # If above maximum, decrease weights
                            scale = max_weight / current_weight
                            weights[indices] *= scale
                            class_adjustments_needed = True
                    
                    # Normalize
                    if class_adjustments_needed:
                        weights = weights / np.sum(weights)
                    class_attempt += 1
                
                if class_attempt >= max_class_attempts:
                    continue
                
                # Sector constraints
                sector_adjustments_needed = True
                max_sector_attempts = 50
                sector_attempt = 0
                
                while sector_adjustments_needed and sector_attempt < max_sector_attempts:
                    sector_adjustments_needed = False
                    for sector, (min_weight, max_weight) in self.constraints.sector_ranges.items():
                        indices = data['Sector'] == sector
                        current_weight = weights[indices].sum()
                        
                        if current_weight < min_weight:
                            if weights[indices].sum() == 0:
                                sector_assets = np.where(indices)[0]
                                active_assets = np.random.choice(
                                    sector_assets,
                                    size=max(1, len(sector_assets) // 3),
                                    replace=False
                                )
                                weights[active_assets] = min_weight / len(active_assets)
                            else:
                                scale = min_weight / current_weight
                                weights[indices] *= scale
                            sector_adjustments_needed = True
                        elif current_weight > max_weight:
                            scale = max_weight / current_weight
                            weights[indices] *= scale
                            sector_adjustments_needed = True
                    
                    if sector_adjustments_needed:
                        weights = weights / np.sum(weights)
                    sector_attempt += 1
                
                if sector_attempt >= max_sector_attempts:
                    continue
                
                # Region constraints
                region_adjustments_needed = True
                max_region_attempts = 50
                region_attempt = 0
                
                while region_adjustments_needed and region_attempt < max_region_attempts:
                    region_adjustments_needed = False
                    for region, (min_weight, max_weight) in self.constraints.region_ranges.items():
                        indices = data['Region'] == region
                        current_weight = weights[indices].sum()
                        
                        if current_weight < min_weight:
                            if weights[indices].sum() == 0:
                                region_assets = np.where(indices)[0]
                                active_assets = np.random.choice(
                                    region_assets,
                                    size=max(1, len(region_assets) // 3),
                                    replace=False
                                )
                                weights[active_assets] = min_weight / len(active_assets)
                            else:
                                scale = min_weight / current_weight
                                weights[indices] *= scale
                            region_adjustments_needed = True
                        elif current_weight > max_weight:
                            scale = max_weight / current_weight
                            weights[indices] *= scale
                            region_adjustments_needed = True
                    
                    if region_adjustments_needed:
                        weights = weights / np.sum(weights)
                    region_attempt += 1
                
                if region_attempt >= max_region_attempts:
                    continue
                
                # Final normalization
                weights = weights / np.sum(weights)
                
                # Verify all constraints
                if self._check_constraints(weights):
                    return weights
                    
            except Exception as e:
                continue
                
        raise RuntimeError("Failed to generate valid portfolio. Constraints may be too restrictive.")
    
    def _check_constraints(self, weights: np.ndarray) -> bool:
        """Check if portfolio weights satisfy all constraints"""
        data = self.asset_universe.data
        
        # Check asset class constraints
        for asset_class, (min_weight, max_weight) in self.constraints.asset_class_ranges.items():
            class_weight = weights[data['Asset Class'] == asset_class].sum()
            if not (min_weight <= class_weight <= max_weight):
                if class_weight < min_weight:
                    print(f"Failed: {asset_class} weight {class_weight:.3f} below minimum {min_weight}")
                else:
                    print(f"Failed: {asset_class} weight {class_weight:.3f} above maximum {max_weight}")
                return False
        
        # Check sector constraints
        for sector, (min_weight, max_weight) in self.constraints.sector_ranges.items():
            sector_weight = weights[data['Sector'] == sector].sum()
            if not (min_weight <= sector_weight <= max_weight):
                if sector_weight < min_weight:
                    print(f"Failed: {sector} sector weight {sector_weight:.3f} below minimum {min_weight}")
                else:
                    print(f"Failed: {sector} sector weight {sector_weight:.3f} above maximum {max_weight}")
                return False
        
        # Check region constraints
        for region, (min_weight, max_weight) in self.constraints.region_ranges.items():
            region_weight = weights[data['Region'] == region].sum()
            if not (min_weight <= region_weight <= max_weight):
                if region_weight < min_weight:
                    print(f"Failed: {region} region weight {region_weight:.3f} below minimum {min_weight}")
                else:
                    print(f"Failed: {region} region weight {region_weight:.3f} above maximum {max_weight}")
                return False
        
        return True

class PortfolioAnalyzer:
    def __init__(self, asset_universe: AssetUniverse, risk_free_rate: float, utility_lambda: float):
        self.asset_universe = asset_universe
        self.risk_free_rate = risk_free_rate
        self.utility_lambda = utility_lambda
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> PortfolioResult:
        """Calculate expected return and standard deviation for a portfolio"""
        exp_returns = self.asset_universe.data['Expected Return'].values
        std_devs = self.asset_universe.data['Standard Deviation'].values
        
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_std = np.sqrt(np.sum(weights**2 * std_devs**2))  # Simplified, assuming no correlations
        
        return PortfolioResult(
            return_value=portfolio_return,
            std_dev=portfolio_std,
            weights=weights
        )
    
    def calculate_final_metrics(self, return_value: float, std_dev: float, weights: np.ndarray) -> SimulationResult:
        """Calculate Sharpe ratio and utility score for final simulation results"""
        sharpe_ratio = (return_value - self.risk_free_rate) / std_dev if std_dev > 0 else 0
        # Standard utility formula: U = E(R) - (λ/2) * σ²
        utility_score = return_value - (self.utility_lambda / 2) * std_dev**2
        
        return SimulationResult(
            return_value=return_value,
            std_dev=std_dev,
            sharpe_ratio=sharpe_ratio,
            utility_score=utility_score,
            weights=weights
        )

class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
    
    def run_simulation(self, exp_return: float, std_dev: float, months: int = 12) -> Tuple[List[float], List[float]]:
        """Run Monte Carlo simulation for given portfolio metrics"""
        monthly_return = exp_return / 12
        monthly_std = std_dev / np.sqrt(12)
        
        returns = []
        stds = []
        
        for _ in range(self.num_simulations):
            monthly_returns = np.random.normal(monthly_return, monthly_std, months)
            cumulative_return = np.prod(1 + monthly_returns) - 1
            period_std = np.std(monthly_returns) * np.sqrt(12)
            
            returns.append(cumulative_return)
            stds.append(period_std)
        
        return returns, stds

class PortfolioSimulation:
    def __init__(self, asset_universe: AssetUniverse, constraints: Constraints):
        self.asset_universe = asset_universe
        self.constraints = constraints
        self.generator = PortfolioGenerator(asset_universe, constraints)
        self.analyzer = PortfolioAnalyzer(
            asset_universe,
            risk_free_rate=constraints.risk_free_rate,
            utility_lambda=constraints.utility_lambda
        )
        self.simulator = MonteCarloSimulator(constraints.num_simulations)
        
    def run(self) -> List[SimulationResult]:
        """Run the complete simulation process"""
        simulation_results = []
        
        for i in range(self.constraints.num_portfolios):
            weights = self.generator.generate_portfolio()
            portfolio_metrics = self.analyzer.calculate_portfolio_metrics(weights)
            
            returns, stds = self.simulator.run_simulation(
                portfolio_metrics.return_value,
                portfolio_metrics.std_dev
            )
            
            # Calculate median values from simulations
            median_return = np.median(returns)
            median_std = np.median(stds)
            
            # Calculate final metrics using median values
            final_result = self.analyzer.calculate_final_metrics(
                median_return, 
                median_std,
                portfolio_metrics.weights
            )
            simulation_results.append(final_result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{self.constraints.num_portfolios} portfolios")
        
        return simulation_results
    
    def plot_results(self, results: List[SimulationResult]):
        """Plot the simulation results with interactive hover and click functionality"""
        returns = [r.return_value for r in results]
        stds = [r.std_dev for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        utility_scores = [r.utility_score for r in results]
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_utility_idx = np.argmax(utility_scores)
        
        # Main scatter plot with efficient frontier
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all portfolios
        scatter = ax.scatter(stds, returns, alpha=0.6, color='blue', label='Simulated Portfolios', s=50)
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            idx = ind["ind"][0]
            text = (f'Return: {returns[idx]:.3f}\n'
                   f'Std Dev: {stds[idx]:.3f}\n'
                   f'Sharpe: {sharpe_ratios[idx]:.3f}\n'
                   f'Utility: {utility_scores[idx]:.3f}')
            annot.set_text(text)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        def onclick(event):
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    # Create pie chart for clicked portfolio
                    plt.figure(figsize=(10, 8))
                    self._create_pie_chart(
                        results[idx].weights,
                        'Portfolio Composition',
                        f'Return: {returns[idx]:.3f}, Std Dev: {stds[idx]:.3f}\n' +
                        f'Sharpe: {sharpe_ratios[idx]:.3f}, Utility: {utility_scores[idx]:.3f}'
                    )
                    plt.show()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_press_event", onclick)
        
        # Highlight optimal portfolios
        if max_sharpe_idx == max_utility_idx:
            ax.scatter(stds[max_sharpe_idx], returns[max_sharpe_idx], 
                      color='red', s=200, label='Max Sharpe & Utility', marker='*')
        else:
            ax.scatter(stds[max_sharpe_idx], returns[max_sharpe_idx], 
                      color='green', s=200, label='Max Sharpe Ratio', marker='*')
            ax.scatter(stds[max_utility_idx], returns[max_utility_idx], 
                      color='orange', s=200, label='Max Utility', marker='*')
        
        # Fit and plot the efficient frontier curve
        points = np.array([(std, ret) for std, ret in zip(stds, returns)])
        points = points[points[:, 0].argsort()]
        frontier_points = []
        max_return = float('-inf')
        for std, ret in points:
            if ret > max_return:
                frontier_points.append((std, ret))
                max_return = ret
        frontier_points = np.array(frontier_points)
        z = np.polyfit(frontier_points[:, 0], frontier_points[:, 1], 3)
        p = np.poly1d(z)
        x_frontier = np.linspace(frontier_points[0, 0], frontier_points[-1, 0], 100)
        y_frontier = p(x_frontier)
        ax.plot(x_frontier, y_frontier, 'r--', linewidth=2, label='Efficient Frontier')
        
        # Set axis limits with small padding
        padding = 0.05
        x_range = max(stds) - min(stds)
        y_range = max(returns) - min(returns)
        ax.set_xlim(min(stds) - x_range * padding, max(stds) + x_range * padding)
        ax.set_ylim(min(returns) - y_range * padding, max(returns) + y_range * padding)
        
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Expected Return')
        ax.set_title('Monte Carlo Simulation Results')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    def _create_pie_chart(self, weights: np.ndarray, title: str, subtitle: str):
        """Create a pie chart for portfolio composition"""
        asset_names = self.asset_universe.data['Asset Name'].values
        # Filter out tiny weights for better visualization
        significant_idx = weights > 0.01  # Show only weights > 1%
        
        # Sort weights by size for better visualization
        sorted_idx = np.argsort(weights[significant_idx])[::-1]
        sorted_weights = weights[significant_idx][sorted_idx]
        sorted_names = asset_names[significant_idx][sorted_idx]
        
        plt.pie(sorted_weights, 
               labels=[f"{name}\n{w:.1%}" for name, w in zip(sorted_names, sorted_weights)],
               autopct='%1.1f%%',
               startangle=90)
        plt.title(f"{title}\n\n{subtitle}")
        plt.axis('equal')

def main():
    # Example usage with customizable risk-free rate and utility lambda
    constraints = Constraints(
        asset_class_ranges={
            'ETF': (0.5, 0.7),           # Primary allocation to ETFs
            'Single Stock': (0.2, 0.4),   # Secondary allocation to stocks
            'Alternative': (0.0, 0.1)     # Small allocation to alternatives
        },
        sector_ranges={
            'Broad': (0.2, 0.6),         # Primary allocation to broad market
            'Technology': (0.0, 0.25),     # Reduced tech maximum
            'Finance': (0.0, 0.15),
            'Consumer Discretionary': (0.0, 0.15),
            'Healthcare': (0.0, 0.15),
            'Consumer Staples': (0.0, 0.15),
            'Energy': (0.0, 0.15),
            'Alternative': (0.0, 0.15),
            'Communication Services': (0.0, 0.15),
            'Industrials': (0.0, 0.15),
            'Materials': (0.0, 0.15),
            'Dividend': (0.0, 0.15),
            'Utilities': (0.0, 0.15)
        },
        region_ranges={
            'US': (0.4, 0.7),            # Primary US allocation
            'Europe': (0.1, 0.3),        # Secondary European allocation
            'Global': (0.1, 0.3),        # Secondary global allocation
            'Asia': (0.0, 0.15),         # Small allocations to other regions
            'EM': (0.0, 0.15)
        },
        num_portfolios=500,
        num_simulations=10000,
        risk_free_rate=0.04,
        utility_lambda=4.0
    )
    
    try:
        asset_universe = AssetUniverse('asset_universe.csv')
        simulation = PortfolioSimulation(asset_universe, constraints)
        results = simulation.run()
        simulation.plot_results(results)
    except FileNotFoundError:
        print("Please create an asset_universe.csv file with the required columns")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 