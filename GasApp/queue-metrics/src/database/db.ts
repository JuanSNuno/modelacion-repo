import * as SQLite from 'expo-sqlite';

export const db = SQLite.openDatabaseSync('queuemetrics.db');

export const initDb = () => {
  db.execSync(`
    CREATE TABLE IF NOT EXISTS studies (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      context_description TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS queue_models (
      id TEXT PRIMARY KEY,
      study_id TEXT,
      type TEXT,
      servers_count INTEGER,
      lambda_calculated REAL,
      mu_calculated REAL,
      result_L REAL,
      result_Lq REAL,
      result_W REAL,
      result_Wq REAL,
      result_P0 REAL,
      result_Rho REAL,
      FOREIGN KEY(study_id) REFERENCES studies(id) ON DELETE CASCADE
    );
  `);
};
