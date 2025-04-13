import sqlite3
import json
from typing import Dict, Any, Optional
import os

class ExperimentDB:
    def __init__(self, db_path: str = "./database/experiments.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create experiments table
        c.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encoder_name TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                train_loss REAL,
                val_loss REAL,
                test_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def experiment_exists(self, encoder_name: str, hyperparameters: Dict[str, Any]) -> bool:
        """Check if an experiment with the same parameters exists"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        hyper_json = json.dumps(hyperparameters, sort_keys=True)
        c.execute('''
            SELECT id FROM experiments 
            WHERE encoder_name = ? AND hyperparameters = ?
        ''', (encoder_name, hyper_json))
        
        result = c.fetchone()
        conn.close()
        
        return result is not None
    
    def save_experiment(self, encoder_name: str, hyperparameters: Dict[str, Any], 
                       train_loss: float, val_loss: Optional[float] = None, test_loss: Optional[float] = None) -> int:
        """Save experiment results to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        hyper_json = json.dumps(hyperparameters, sort_keys=True)
        c.execute('''
            INSERT INTO experiments (encoder_name, hyperparameters, train_loss, val_loss, test_loss)
            VALUES (?, ?, ?, ?, ?)
        ''', (encoder_name, hyper_json, train_loss, val_loss, test_loss))
        
        experiment_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return experiment_id
    
    def get_experiment_results(self, encoder_name: str, hyperparameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment configuration"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        hyper_json = json.dumps(hyperparameters, sort_keys=True)
        c.execute('''
            SELECT * FROM experiments 
            WHERE encoder_name = ? AND hyperparameters = ?
        ''', (encoder_name, hyper_json))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'encoder_name': result[1],
                'hyperparameters': json.loads(result[2]),
                'train_loss': result[3],
                'val_loss': result[4],
                'created_at': result[5]
            }
        return None 