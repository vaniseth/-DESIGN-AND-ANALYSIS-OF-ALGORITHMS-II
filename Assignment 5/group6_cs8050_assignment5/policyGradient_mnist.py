"""
Policy Gradient (REINFORCE) Implementation on MNIST Dataset
Reinforcement Learning approach to digit classification
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

class PolicyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities (policy)
    """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network for baseline (reduce variance)
    """
    def __init__(self, input_dim=784, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class REINFORCEAgent:
    """
    Policy Gradient (REINFORCE) agent with baseline for MNIST classification
    """
    
    def __init__(self, learning_rate=0.0005, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.results = {}
        
        print(f"Using device: {self.device}")
        print(f"Policy network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"Value network parameters: {sum(p.numel() for p in self.value_net.parameters()):,}")
        
    def select_action(self, state):
        """
        Select action based on policy network
        Returns: action, log_prob, entropy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), m.entropy()
    
    def get_value(self, state):
        """Get value estimate from value network"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.value_net(state)
    
    def update_policy(self, states, actions, returns, log_probs, entropies):
        """
        Update policy using REINFORCE with baseline
        
        Time Complexity: O(n * m)
        - n: number of samples in batch
        - m: number of parameters in network
        
        Space Complexity: O(n + m)
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Get value estimates (baseline)
        values = self.value_net(states_tensor).squeeze()
        
        # Calculate advantages
        advantages = returns_tensor - values.detach()
        
        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        # Add entropy bonus for exploration
        entropy_loss = -self.entropy_coef * torch.stack(entropies).mean()
        
        total_policy_loss = torch.stack(policy_loss).mean() + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Value loss
        value_loss = F.mse_loss(values, returns_tensor)
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        return total_policy_loss.item(), value_loss.item()


class REINFORCEAnalyzer:
    """
    REINFORCE analyzer for MNIST dataset
    """
    
    def __init__(self):
        self.results = {}
        
    def load_mnist_data(self, n_samples=10000):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype('float32')
        y = mnist.target.astype(int)
        
        # Normalize data to [0, 1]
        X = X / 255.0
        
        # Use subset for faster computation
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {X.shape[1]}")
        print(f"Data range: [{X.min():.3f}, {X.max():.3f}]\n")
        
        return X_train, X_test, y_train, y_test
    
    def train_agent(self, agent, X_train, y_train, episodes=100, batch_size=64):
        """
        Train REINFORCE agent
        """
        print("=" * 70)
        print("Training REINFORCE Agent")
        print("=" * 70)
        
        n_samples = len(X_train)
        episode_rewards = []
        episode_losses = []
        episode_value_losses = []
        episode_accuracies = []
        training_times = []
        episode_entropies = []
        
        for episode in range(episodes):
            start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_reward = 0
            correct = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                batch_states = []
                batch_actions = []
                batch_log_probs = []
                batch_rewards = []
                batch_entropies = []
                
                for state, true_label in zip(batch_X, batch_y):
                    # Select action
                    action, log_prob, entropy = agent.select_action(state)
                    
                    # Better reward shaping: +1 for correct, -0.1 for wrong
                    reward = 1.0 if action == true_label else -0.1
                    
                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(reward)
                    batch_entropies.append(entropy)
                    
                    if action == true_label:
                        correct += 1
                        total_reward += 1.0
                    else:
                        total_reward -= 0.1
                
                # Calculate discounted returns
                returns = []
                R = 0
                for r in reversed(batch_rewards):
                    R = r + agent.gamma * R
                    returns.insert(0, R)
                
                # Normalize returns
                returns = np.array(returns)
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
                # Update policy
                policy_loss, value_loss = agent.update_policy(
                    batch_states, batch_actions, returns, 
                    batch_log_probs, batch_entropies
                )
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += torch.stack(batch_entropies).mean().item()
                num_batches += 1
            
            episode_time = time.time() - start_time
            avg_reward = total_reward / n_samples
            accuracy = correct / n_samples
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_entropy = total_entropy / num_batches
            
            episode_rewards.append(avg_reward)
            episode_accuracies.append(accuracy)
            episode_losses.append(avg_policy_loss)
            episode_value_losses.append(avg_value_loss)
            training_times.append(episode_time)
            episode_entropies.append(avg_entropy)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{episodes} | "
                      f"Accuracy: {accuracy:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Loss: {avg_policy_loss:.4f} | "
                      f"Time: {episode_time:.2f}s")
        
        self.results['training'] = {
            'episode_rewards': episode_rewards,
            'episode_accuracies': episode_accuracies,
            'episode_losses': episode_losses,
            'episode_value_losses': episode_value_losses,
            'training_times': training_times,
            'episode_entropies': episode_entropies,
            'episodes': list(range(1, episodes + 1))
        }
        
        print("\n✓ Training complete!\n")
        return episode_rewards, episode_accuracies
    
    def evaluate_agent(self, agent, X_test, y_test):
        """Evaluate trained agent"""
        print("=" * 70)
        print("Evaluating REINFORCE Agent")
        print("=" * 70)
        
        agent.policy.eval()
        correct = 0
        predictions = []
        action_probs = []
        
        with torch.no_grad():
            for state, true_label in zip(X_test, y_test):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs, dim=1).item()
                predictions.append(action)
                action_probs.append(probs.cpu().numpy()[0])
                
                if action == true_label:
                    correct += 1
        
        accuracy = correct / len(y_test)
        
        print(f"✓ Test Accuracy: {accuracy:.4f} ({correct}/{len(y_test)} correct)")
        
        # Calculate per-class accuracy
        class_correct = [0] * 10
        class_total = [0] * 10
        
        for pred, true in zip(predictions, y_test):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1
        
        class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                           for i in range(10)]
        
        print("\nPer-Class Accuracy:")
        for digit in range(10):
            print(f"  Digit {digit}: {class_accuracies[digit]:.4f} "
                  f"({class_correct[digit]}/{class_total[digit]})")
        
        self.results['evaluation'] = {
            'test_accuracy': accuracy,
            'predictions': predictions,
            'action_probs': action_probs,
            'class_accuracies': class_accuracies,
            'class_correct': class_correct,
            'class_total': class_total
        }
        
        print("\n✓ Evaluation complete!\n")
        return accuracy, predictions
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("=" * 70)
        print("Generating Visualizations")
        print("=" * 70)
        
        # Create output directory
        os.makedirs('images', exist_ok=True)
        
        fig = plt.figure(figsize=(18, 10))
        
        # Plot 1: Training Accuracy over Episodes
        if 'training' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results['training']
            ax1.plot(data['episodes'], data['episode_accuracies'], 
                    linewidth=2, color='#2E86AB', marker='o', markersize=3)
            ax1.set_xlabel('Episode', fontsize=11)
            ax1.set_ylabel('Training Accuracy', fontsize=11)
            ax1.set_title('Learning Curve\n(Accuracy over Episodes)', 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Add final accuracy annotation
            final_acc = data['episode_accuracies'][-1]
            ax1.axhline(y=final_acc, color='red', linestyle='--', alpha=0.5, 
                       label=f'Final: {final_acc:.3f}')
            ax1.legend()
        
        # Plot 2: Policy Loss over Episodes
        if 'training' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results['training']
            ax2.plot(data['episodes'], data['episode_losses'], 
                    linewidth=2, color='#E63946', marker='s', markersize=3)
            ax2.set_xlabel('Episode', fontsize=11)
            ax2.set_ylabel('Policy Loss', fontsize=11)
            ax2.set_title('Policy Loss Curve\n(Lower is Better)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Reward per Episode
        if 'training' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results['training']
            ax3.plot(data['episodes'], data['episode_rewards'], 
                    linewidth=2, color='#06A77D', marker='^', markersize=3)
            ax3.set_xlabel('Episode', fontsize=11)
            ax3.set_ylabel('Average Reward', fontsize=11)
            ax3.set_title('Reward Progress\n(Higher is Better)', 
                         fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Per-Class Accuracy
        if 'evaluation' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results['evaluation']
            digits = list(range(10))
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            bars = ax4.bar(digits, data['class_accuracies'], 
                          color=colors, alpha=0.8, edgecolor='black')
            ax4.set_xlabel('Digit Class', fontsize=11)
            ax4.set_ylabel('Accuracy', fontsize=11)
            ax4.set_title('Per-Class Test Accuracy', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.set_xticks(digits)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, acc in zip(bars, data['class_accuracies']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 5: Training Time per Episode
        if 'training' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            data = self.results['training']
            ax5.plot(data['episodes'], data['training_times'], 
                    linewidth=2, color='#9D4EDD', marker='d', markersize=3)
            ax5.set_xlabel('Episode', fontsize=11)
            ax5.set_ylabel('Training Time (seconds)', fontsize=11)
            ax5.set_title('Time Complexity Analysis\nO(n × m)', 
                         fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(data['training_times'])
            ax5.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5, 
                       label=f'Avg: {avg_time:.2f}s')
            ax5.legend()
        
        # Plot 6: Entropy over Episodes (Exploration metric)
        if 'training' in self.results:
            ax6 = plt.subplot(2, 3, 6)
            data = self.results['training']
            ax6.plot(data['episodes'], data['episode_entropies'], 
                    linewidth=2, color='#F18F01', marker='*', markersize=3)
            ax6.set_xlabel('Episode', fontsize=11)
            ax6.set_ylabel('Policy Entropy', fontsize=11)
            ax6.set_title('Exploration Metric\n(Higher = More Exploration)', 
                         fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add annotation about entropy
            ax6.text(0.05, 0.05, 'High entropy = exploring\nLow entropy = exploiting', 
                    transform=ax6.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('images/reinforce_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Analysis saved as 'images/reinforce_analysis.png'")
        
        print("✓ All visualizations complete!\n")
        
        return fig


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("POLICY GRADIENT (REINFORCE) ON MNIST DATASET")
    print("=" * 70)
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize analyzer
    analyzer = REINFORCEAnalyzer()
    
    # Load MNIST data
    X_train, X_test, y_train, y_test = analyzer.load_mnist_data(n_samples=10000)
    
    # Initialize REINFORCE agent
    agent = REINFORCEAgent(learning_rate=0.0005, gamma=0.99, entropy_coef=0.01)
    analyzer.agent = agent
    
    # Train agent (more episodes for better convergence)
    analyzer.train_agent(agent, X_train, y_train, episodes=100, batch_size=64)
    
    # Evaluate agent
    analyzer.results['y_test'] = y_test
    accuracy, predictions = analyzer.evaluate_agent(agent, X_test, y_test)
    
    # Generate visualizations
    analyzer.visualize_results()
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print("\nOutput files:")
    print("  • images/reinforce_analysis.png - Complete analysis dashboard")
    print()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()