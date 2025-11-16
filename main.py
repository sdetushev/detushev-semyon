"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(PolicyNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.feature_network = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, 3)
        
        with torch.no_grad():
            self.fc_out.bias.data = torch.tensor([2.0, -1.0, -1.0])
    
    def forward(self, x):
        features = self.feature_network(x)
        logits = self.fc_out(features)
        logits[:, 0] += 1.0
        return torch.softmax(logits, dim=-1)

class ContextualBanditPolicy:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.q_networks = {}
        self.policy_network = None
        self.label_encoders = {}
        self.feature_cols = None
        self.action_stats = {}
        self.qt_transformers = {}
        
    def create_features(self, df):
        df_processed = df.copy()
        
        categorical_cols = ['zip_code', 'channel', 'history_segment']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str)).astype(np.float32)
            else:
                df_processed[col] = df_processed[col].astype(str)
                mask = df_processed[col].isin(self.label_encoders[col].classes_)
                df_processed.loc[mask, col] = self.label_encoders[col].transform(df_processed.loc[mask, col])
                df_processed.loc[~mask, col] = 0
                df_processed[col] = df_processed[col].astype(np.float32)
        
        numeric_cols = ['recency', 'history']
        for col in numeric_cols:
            if col not in self.qt_transformers:
                self.qt_transformers[col] = QuantileTransformer(output_distribution='normal', random_state=42)
                original_values = df_processed[col].values.reshape(-1, 1)
                df_processed[col] = self.qt_transformers[col].fit_transform(original_values).flatten()
            else:
                original_values = df_processed[col].values.reshape(-1, 1)
                df_processed[col] = self.qt_transformers[col].transform(original_values).flatten()
        
        df_processed['total_interest'] = df_processed['mens'] + df_processed['womens']
        df_processed['log_history'] = np.log1p(df_processed['history'].clip(lower=0) + 1)
        df_processed['recency_inv'] = 1 / (df_processed['recency'] + 0.1)
        df_processed['mens_affinity'] = df_processed['mens'] * (1 - df_processed['womens'])
        df_processed['womens_affinity'] = df_processed['womens'] * (1 - df_processed['mens'])
        df_processed['preference_strength'] = np.abs(df_processed['mens'] - df_processed['womens'])
        df_processed['is_high_value'] = (df_processed['history'] > df_processed['history'].median()).astype(np.float32)
        df_processed['is_active'] = (df_processed['recency'] <= 3).astype(np.float32)
        
        self.feature_cols = [
            'recency', 'history', 'mens', 'womens', 'newbie', 
            'zip_code', 'channel', 'history_segment',
            'total_interest', 'log_history', 'recency_inv',
            'mens_affinity', 'womens_affinity', 'preference_strength',
            'is_high_value', 'is_active'
        ]
        
        for col in self.feature_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(np.float32)
        
        return df_processed
    
    def compute_action_stats(self, train_processed):
        action_stats = train_processed.groupby('segment').agg({
            'visit': ['count', 'mean', 'sum']
        }).round(4)
        action_stats.columns = ['count', 'conversion_rate', 'total_visits']
        
        action_map = {'Mens E-Mail': 0, 'Womens E-Mail': 1, 'No E-Mail': 2}
        for action_name, action_idx in action_map.items():
            if action_name in action_stats.index:
                self.action_stats[action_idx] = action_stats.loc[action_name, 'conversion_rate']
        
        return action_stats
    
    def fit_q_networks(self, train_processed):
        X = train_processed[self.feature_cols].values.astype(np.float32)
        action_map = {'Mens E-Mail': 0, 'Womens E-Mail': 1, 'No E-Mail': 2}
        train_processed['action'] = train_processed['segment'].map(action_map)
        
        self.compute_action_stats(train_processed)
        input_dim = len(self.feature_cols)
        
        for action in [0, 1, 2]:
            action_mask = train_processed['action'] == action
            X_action = X[action_mask]
            y_action = train_processed.loc[action_mask, 'visit'].values.astype(np.float32)
            
            if len(X_action) > 100:
                q_network = QNetwork(input_dim).to(self.device)
                optimizer = optim.Adam(q_network.parameters(), lr=0.001, weight_decay=0.01)
                criterion = nn.BCELoss()
                
                dataset = TensorDataset(torch.from_numpy(X_action).float(), torch.from_numpy(y_action).float().unsqueeze(1))
                dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
                
                q_network.train()
                for epoch in range(30):
                    for batch_X, batch_y in dataloader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        optimizer.zero_grad()
                        predictions = q_network(batch_X)
                        loss = criterion(predictions, batch_y)
                        loss.backward()
                        optimizer.step()
                
                self.q_networks[action] = q_network
                q_network.eval()
    
    def doubly_robust_estimation(self, train_processed):
        X = train_processed[self.feature_cols].values.astype(np.float32)
        action_map = {'Mens E-Mail': 0, 'Womens E-Mail': 1, 'No E-Mail': 2}
        actions = train_processed['segment'].map(action_map).values
        rewards = train_processed['visit'].values.astype(np.float32)
        
        n_samples = len(train_processed)
        dr_values = np.zeros((n_samples, 3), dtype=np.float32)
        propensity = 1.0 / 3.0
        
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            
            for action in [0, 1, 2]:
                if action in self.q_networks:
                    direct_preds = self.q_networks[action](X_tensor).cpu().numpy().flatten()
                else:
                    direct_preds = np.full(n_samples, self.action_stats.get(action, 0.15), dtype=np.float32)
                
                ips_weights = (actions == action).astype(np.float32) / propensity
                ips_weights = np.clip(ips_weights, 0, 3.0)
                ips_correction = ips_weights * (rewards - direct_preds)
                dr_values[:, action] = direct_preds + ips_correction
        
        return dr_values
    
    def fit_policy_network(self, train_processed, dr_values):
        X = train_processed[self.feature_cols].values.astype(np.float32)
        X_policy = np.column_stack([X, dr_values]).astype(np.float32)
        optimal_actions = np.argmax(dr_values, axis=1).astype(np.int64)
        
        dataset = TensorDataset(torch.from_numpy(X_policy).float(), torch.from_numpy(optimal_actions).long())
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        input_dim = X_policy.shape[1]
        self.policy_network = PolicyNetwork(input_dim).to(self.device)
        optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001, weight_decay=0.1)
        criterion = nn.CrossEntropyLoss()
        
        self.policy_network.train()
        best_score = -float('inf')
        best_model_state = None
        
        for epoch in range(100):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.policy_network(batch_X)
                loss = criterion(outputs, batch_y)
                
                probs = torch.softmax(outputs, dim=1)
                target_probs = torch.tensor([0.8, 0.1, 0.1]).repeat(probs.shape[0], 1).to(self.device)
                kl_penalty = nn.functional.kl_div(torch.log(probs + 1e-8), target_probs, reduction='batchmean')
                total_loss = loss + 2.0 * kl_penalty
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if epoch % 20 == 0:
                with torch.no_grad():
                    train_probs = self.policy_network(torch.from_numpy(X_policy).float().to(self.device)).cpu().numpy()
                    snips = self.snips_score(train_probs, optimal_actions, train_processed['visit'].values)
                
                if snips > best_score:
                    best_score = snips
                    best_model_state = self.policy_network.state_dict().copy()
        
        if best_model_state is not None:
            self.policy_network.load_state_dict(best_model_state)
        self.policy_network.eval()
    
    def snips_score(self, pi, a, r, mu=1/3):
        with torch.no_grad():
            pi_tensor = torch.tensor(pi, dtype=torch.float32)
            a_tensor = torch.tensor(a, dtype=torch.long)
            r_tensor = torch.tensor(r, dtype=torch.float32)
            
            w = pi_tensor[torch.arange(len(a_tensor)), a_tensor] / mu
            num = (w * r_tensor).sum()
            den = w.sum().clamp_min(1e-12)
            return (num / den).item()
    
    def fit(self, train_df):
        train_processed = self.create_features(train_df)
        self.fit_q_networks(train_processed)
        dr_values = self.doubly_robust_estimation(train_processed)
        self.fit_policy_network(train_processed, dr_values)
    
    def predict_policy(self, test_df, exploration=0.02):
        test_processed = self.create_features(test_df)
        X_test = test_processed[self.feature_cols].values.astype(np.float32)
        
        q_values = np.zeros((len(X_test), 3), dtype=np.float32)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test).float().to(self.device)
            for action in [0, 1, 2]:
                if action in self.q_networks:
                    q_values[:, action] = self.q_networks[action](X_tensor).cpu().numpy().flatten()
                else:
                    q_values[:, action] = self.action_stats.get(action, 0.15)
        
        X_test_policy = np.column_stack([X_test, q_values]).astype(np.float32)
        
        if self.policy_network is not None:
            with torch.no_grad():
                X_policy_tensor = torch.from_numpy(X_test_policy).float().to(self.device)
                policy_probs = self.policy_network(X_policy_tensor).cpu().numpy()
            policy_probs = (1 - exploration) * policy_probs + exploration / 3
        else:
            exp_q = np.exp(q_values * 2.0)
            policy_probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        
        mens_affinity_mask = test_processed['mens_affinity'] == 1
        womens_affinity_mask = test_processed['womens_affinity'] == 1
        high_value_mask = test_processed['is_high_value'] == 1
        
        policy_probs[mens_affinity_mask, 0] *= 1.4
        policy_probs[mens_affinity_mask, 2] *= 0.6
        policy_probs[womens_affinity_mask, 1] *= 1.4
        policy_probs[womens_affinity_mask, 2] *= 0.6
        policy_probs[high_value_mask, 2] *= 0.7
        
        policy_probs[:, 2] = np.clip(policy_probs[:, 2], 0.01, 0.15)
        row_sums = policy_probs.sum(axis=1)
        policy_probs = policy_probs / row_sums[:, np.newaxis]
        
        return policy_probs


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission
    test_df = pd.read_csv('data/test.csv')
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'p_mens_email': predictions[:, 0],
        'p_womens_email': predictions[:, 1],
        'p_no_email': predictions[:, 2]
    })
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка данных
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print("Размер train данных:", train_df.shape)
    print("Размер test данных:", test_df.shape)
    
    # Обучение финальной модели
    final_policy = ContextualBanditPolicy()
    final_policy.fit(train_df)
    
    # Создание предсказаний
    predictions = final_policy.predict_policy(test_df)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()