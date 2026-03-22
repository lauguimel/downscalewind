"""
simulation_registry.py — SQLite database tracking all CFD simulations.

Stores parameters, mesh stats, convergence metrics, probe data, and file paths
for every run. Enables cross-run comparison and parameter optimization.

Usage
-----
    from simulation_registry import SimulationRegistry

    db = SimulationRegistry("data/simulations.db")

    # Register a new run
    run_id = db.register_run(
        label="res62 inletOutlet",
        case_dir="data/cases/poc_mesh_convergence/case_res62",
        params={"fine_cell_size": 62.5, "coriolis": True, "transport_T": True, ...},
    )

    # Store mesh stats
    db.update_mesh(run_id, n_cells=183402, max_non_ortho=46.4, mesh_time_s=72)

    # Store convergence metrics
    db.update_metrics(run_id, metrics_dict)

    # Store probe data (CSV string)
    db.store_probe_data(run_id, probe_csv_string)

    # Query
    df = db.to_dataframe()  # all runs as pandas DataFrame
    db.get_probe_data(run_id)  # returns CSV string
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "simulations.db"


class SimulationRegistry:
    """SQLite-backed registry of CFD simulation runs."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                label       TEXT NOT NULL,
                case_dir    TEXT,
                output_dir  TEXT,
                study_name  TEXT,

                -- Parameters (JSON blob for flexibility)
                params      TEXT,

                -- Mesh stats
                n_cells         INTEGER,
                max_non_ortho   REAL,
                max_skewness    REAL,
                mesh_time_s     REAL,

                -- Solver stats
                solver_name     TEXT,
                n_iterations    INTEGER,
                wall_time_s     REAL,
                cpu_time_s      REAL,

                -- Convergence metrics
                final_residual_Ux   REAL,
                final_speed_avg     REAL,
                final_speed_max     REAL,

                -- ERA5 comparison metrics
                rmse_bl     REAL,
                rmse_mid    REAL,
                rmse_upper  REAL,
                bias_bl     REAL,
                bias_mid    REAL,
                bias_upper  REAL,
                speedup_100m REAL,

                -- Full metrics JSON (everything from evaluate_case)
                metrics     TEXT,

                -- Status
                status      TEXT DEFAULT 'created'
            );

            CREATE TABLE IF NOT EXISTS probe_data (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL,
                probe_name  TEXT NOT NULL,
                z_agl       REAL NOT NULL,
                speed       REAL,
                ux          REAL,
                uy          REAL,
                w           REAL,
                k           REAL,
                epsilon     REAL,
                T           REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_probe_run ON probe_data(run_id);
            CREATE INDEX IF NOT EXISTS idx_runs_study ON runs(study_name);
        """)
        self.conn.commit()

    def register_run(
        self,
        label: str,
        case_dir: str | Path,
        params: dict,
        study_name: str | None = None,
    ) -> int:
        """Register a new simulation run. Returns run_id."""
        case_dir = str(case_dir)
        if study_name is None:
            study_name = Path(case_dir).parent.name

        cur = self.conn.execute(
            """INSERT INTO runs (created_at, label, case_dir, study_name, params,
                                 solver_name, status)
               VALUES (?, ?, ?, ?, ?, ?, 'created')""",
            (
                datetime.now().isoformat(),
                label,
                case_dir,
                study_name,
                json.dumps(params),
                params.get("solver", "simpleFoam"),
            ),
        )
        self.conn.commit()
        run_id = cur.lastrowid
        logger.info("Registered run #%d: %s", run_id, label)
        return run_id

    def update_mesh(
        self, run_id: int,
        n_cells: int, max_non_ortho: float,
        max_skewness: float = 0, mesh_time_s: float = 0,
    ):
        """Update mesh statistics for a run."""
        self.conn.execute(
            """UPDATE runs SET n_cells=?, max_non_ortho=?, max_skewness=?,
                              mesh_time_s=?, status='meshed'
               WHERE id=?""",
            (n_cells, max_non_ortho, max_skewness, mesh_time_s, run_id),
        )
        self.conn.commit()

    def update_metrics(self, run_id: int, metrics: dict):
        """Update convergence and ERA5 comparison metrics."""
        self.conn.execute(
            """UPDATE runs SET
                n_iterations=?, wall_time_s=?, cpu_time_s=?,
                final_residual_Ux=?, final_speed_avg=?, final_speed_max=?,
                rmse_bl=?, rmse_mid=?, rmse_upper=?,
                bias_bl=?, bias_mid=?, bias_upper=?,
                speedup_100m=?,
                metrics=?, output_dir=?, status='evaluated'
               WHERE id=?""",
            (
                metrics.get("n_iterations"),
                metrics.get("wall_time_s"),
                metrics.get("cpu_time_s"),
                metrics.get("final_residual_Ux"),
                metrics.get("final_speed_avg"),
                metrics.get("final_speed_max"),
                metrics.get("rmse_bl"),
                metrics.get("rmse_mid"),
                metrics.get("rmse_upper"),
                metrics.get("bias_bl"),
                metrics.get("bias_mid"),
                metrics.get("bias_upper"),
                metrics.get("speedup_100m"),
                json.dumps(metrics),
                metrics.get("output_dir", ""),
                run_id,
            ),
        )
        self.conn.commit()

    def store_probe_data(self, run_id: int, probes: dict):
        """Store vertical profile data for all probes.

        probes: dict from extract_profiles() — {name: {z_agl, speed, ux, uy, w, k, ...}}
        """
        # Clear existing probe data for this run
        self.conn.execute("DELETE FROM probe_data WHERE run_id=?", (run_id,))

        rows = []
        for name, prof in probes.items():
            z_agl = prof["z_agl"]
            for i in range(len(z_agl)):
                rows.append((
                    run_id, name, float(z_agl[i]),
                    float(prof["speed"][i]),
                    float(prof["ux"][i]),
                    float(prof["uy"][i]),
                    float(prof["w"][i]),
                    float(prof["k"][i]),
                    float(prof.get("epsilon", [0]*len(z_agl))[i]),
                    float(prof["T"][i]) if "T" in prof else None,
                ))

        self.conn.executemany(
            """INSERT INTO probe_data
               (run_id, probe_name, z_agl, speed, ux, uy, w, k, epsilon, T)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        logger.info("Stored %d probe measurements for run #%d", len(rows), run_id)

    def update_status(self, run_id: int, status: str):
        self.conn.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
        self.conn.commit()

    def to_dataframe(self, study_name: str | None = None):
        """Export all runs as a pandas DataFrame."""
        import pandas as pd
        query = "SELECT * FROM runs"
        params = ()
        if study_name:
            query += " WHERE study_name=?"
            params = (study_name,)
        query += " ORDER BY created_at"
        return pd.read_sql_query(query, self.conn, params=params)

    def get_probe_data(self, run_id: int):
        """Get probe data as a pandas DataFrame."""
        import pandas as pd
        return pd.read_sql_query(
            "SELECT * FROM probe_data WHERE run_id=? ORDER BY probe_name, z_agl",
            self.conn, params=(run_id,),
        )

    def get_probe_comparison(self, run_ids: list[int], probe_name: str = "centre"):
        """Compare probe data across multiple runs."""
        import pandas as pd
        placeholders = ",".join("?" * len(run_ids))
        query = f"""
            SELECT p.*, r.label, r.n_cells
            FROM probe_data p
            JOIN runs r ON p.run_id = r.id
            WHERE p.run_id IN ({placeholders}) AND p.probe_name = ?
            ORDER BY r.n_cells, p.z_agl
        """
        return pd.read_sql_query(query, self.conn, params=(*run_ids, probe_name))

    def close(self):
        self.conn.close()
