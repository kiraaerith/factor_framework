"""
Factor Evaluation Dashboard - Flask Web App

Usage:
    cd etf_factor_framework/web
    python app.py
    # Open http://localhost:5001
"""

import json
import os
import re
import sqlite3

from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Locate DB: {project_root}/factor_eval_result/factor_eval.db
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "factor_eval_result", "factor_eval.db")
STATUS_JSON = os.path.join(
    PROJECT_ROOT, "agent_project", "Fundamental_Factors", "factor_eval_status.json"
)


def get_db():
    """Open read-only SQLite connection."""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_factor_status() -> dict:
    """Load full factor status data from JSON."""
    if not os.path.isfile(STATUS_JSON):
        return {}
    with open(STATUS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("factors", {})


def load_factor_categories() -> dict:
    """Load factor_id -> category mapping from status JSON."""
    return {
        fid: fdata.get("category", "")
        for fid, fdata in load_factor_status().items()
    }


def extract_factor_id(expression_name: str) -> str:
    """Extract base factor ID from expression_name.

    e.g. 'EP_TTM_raw' -> 'EP_TTM', 'COMP_GROWTH_size' -> 'COMP_GROWTH'
    """
    for suffix in ("_raw", "_size", "_industry"):
        if expression_name.endswith(suffix):
            return expression_name[: -len(suffix)]
    return expression_name


def rate_factor(rank_icir: float) -> str:
    if rank_icir is None:
        return "C"
    if abs(rank_icir) >= 1.0:
        return "A"
    if abs(rank_icir) >= 0.5:
        return "B"
    return "C"


# SQL subquery to deduplicate: keep only the latest record per combo
DEDUP_SQL = """
    SELECT * FROM factor_evaluation_results
    WHERE id IN (
        SELECT MAX(id) FROM factor_evaluation_results
        GROUP BY expression_name, neutralization_method, top_k, rebalance_freq
    )
"""


# ── Routes ──────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/factor_index")
def api_factor_index():
    """Return all factors from status JSON with their metadata and status."""
    factors_raw = load_factor_status()
    factors = []
    for fid, fd in factors_raw.items():
        er = fd.get("eval_result") or {}
        factors.append(
            {
                "factor_id": fid,
                "factor_name": fd.get("factor_name", ""),
                "category": fd.get("category", ""),
                "sub_category": fd.get("sub_category", ""),
                "direction": fd.get("direction", ""),
                "formula_brief": fd.get("formula_brief", ""),
                "status": fd.get("status", ""),
                "last_step": fd.get("last_step_completed", ""),
                "last_updated": fd.get("last_updated", ""),
                "spec_path": fd.get("spec_path", ""),
                "code_path": er.get("factor_code_path", ""),
                "best_rank_icir": er.get("best_rank_ic_ir"),
                "best_sharpe": er.get("best_sharpe"),
                "best_neutralization": er.get("best_neutralization", ""),
                "failure_info": fd.get("failure_info"),
            }
        )
    return jsonify({"factors": factors})


@app.route("/api/summary")
def api_summary():
    """Return per-factor summary (best combo by rank_icir)."""
    categories = load_factor_categories()
    db = get_db()
    cur = db.cursor()

    cur.execute(
        f"""
        SELECT id, expression_name, neutralization_method, top_k, rebalance_freq,
               sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir,
               other_metrics
        FROM ({DEDUP_SQL})
        ORDER BY expression_name, rank_icir DESC
        """
    )
    rows = cur.fetchall()

    # Check which factors have return curves
    cur.execute(
        "SELECT DISTINCT expression_name FROM factor_return_curves"
    )
    curve_expressions = {r[0] for r in cur.fetchall()}

    # Load direction per factor_type from expression_params
    cur.execute(
        "SELECT DISTINCT factor_type, expression_params FROM factor_evaluation_results WHERE expression_params IS NOT NULL"
    )
    direction_map = {}
    for r in cur.fetchall():
        try:
            d = json.loads(r[1]).get("direction")
            if d is not None:
                direction_map[r[0]] = d
        except (json.JSONDecodeError, TypeError):
            pass
    db.close()

    # Group by base factor_id, pick best sharpe
    best = {}  # factor_id -> best row dict
    for row in rows:
        expr = row["expression_name"]
        fid = extract_factor_id(expr)
        sharpe = row["sharpe"] or float("-inf")

        if fid not in best or sharpe > (best[fid]["best_sharpe"] or float("-inf")):
            # Parse annualized_return from other_metrics
            ann_ret = None
            om = row["other_metrics"]
            if om:
                try:
                    om_data = json.loads(om)
                    ann_ret = om_data.get("returns_metrics", {}).get(
                        "annualized_return"
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass

            best[fid] = {
                "factor_id": fid,
                "category": categories.get(fid, ""),
                "direction": direction_map.get(fid),
                "best_rank_icir": row["rank_icir"],
                "best_icir": row["icir"],
                "best_rank_ic": row["rank_ic"],
                "best_ic": row["ic"],
                "best_sharpe": row["sharpe"],
                "best_max_drawdown": row["max_drawdown"],
                "best_annualized_return": ann_ret,
                "best_neutralization": row["neutralization_method"],
                "best_top_k": row["top_k"],
                "best_rebalance_freq": row["rebalance_freq"],
                "best_combo": (
                    f"{row['neutralization_method']} / "
                    f"top{row['top_k']} / "
                    f"freq{row['rebalance_freq']}"
                ),
                "rating": rate_factor(row["rank_icir"]),
                "has_curve": expr in curve_expressions,
            }

    # Sort by sharpe desc
    factors = sorted(
        best.values(), key=lambda x: x["best_sharpe"] or float("-inf"), reverse=True
    )
    return jsonify({"factors": factors})


@app.route("/api/details/<factor_id>")
def api_details(factor_id):
    """Return all combos for a factor."""
    db = get_db()
    cur = db.cursor()

    # Exact match: {factor_id}_raw, {factor_id}_size, {factor_id}_industry, or {factor_id}
    expr_names = [f"{factor_id}_raw", f"{factor_id}_size", f"{factor_id}_industry", factor_id]
    placeholders = ",".join("?" * len(expr_names))
    cur.execute(
        f"""
        SELECT id, expression_name, neutralization_method, top_k, rebalance_freq,
               sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir,
               other_metrics
        FROM ({DEDUP_SQL})
        WHERE expression_name IN ({placeholders})
        ORDER BY sharpe DESC
        """,
        expr_names,
    )
    rows = cur.fetchall()

    # Check curves
    cur.execute(
        f"SELECT DISTINCT expression_name FROM factor_return_curves WHERE expression_name IN ({placeholders})",
        expr_names,
    )
    curve_expressions = {r[0] for r in cur.fetchall()}
    db.close()

    results = []
    for row in rows:
        ann_ret = None
        om = row["other_metrics"]
        if om:
            try:
                om_data = json.loads(om)
                ann_ret = om_data.get("returns_metrics", {}).get("annualized_return")
            except (json.JSONDecodeError, AttributeError):
                pass

        results.append(
            {
                "id": row["id"],
                "expression_name": row["expression_name"],
                "neutralization": row["neutralization_method"],
                "top_k": row["top_k"],
                "rebalance_freq": row["rebalance_freq"],
                "sharpe": row["sharpe"],
                "max_drawdown": row["max_drawdown"],
                "calmar": row["calmar"],
                "ic": row["ic"],
                "icir": row["icir"],
                "rank_ic": row["rank_ic"],
                "rank_icir": row["rank_icir"],
                "annualized_return": ann_ret,
                "has_curve": row["expression_name"] in curve_expressions,
            }
        )

    # Extract direction from expression_params (separate query since DEDUP_SQL doesn't include it)
    direction = None
    if results:
        db2 = get_db()
        cur2 = db2.cursor()
        cur2.execute(
            "SELECT expression_params FROM factor_evaluation_results WHERE expression_name IN ({}) LIMIT 1".format(placeholders),
            expr_names,
        )
        param_row = cur2.fetchone()
        db2.close()
        if param_row and param_row[0]:
            try:
                direction = json.loads(param_row[0]).get("direction")
            except (json.JSONDecodeError, TypeError):
                pass

    return jsonify({"factor_id": factor_id, "direction": direction, "results": results})


@app.route("/api/curve/<factor_id>")
def api_curve(factor_id):
    """Return cumulative return curves for all combos of a factor."""
    db = get_db()
    cur = db.cursor()

    expr_names = [f"{factor_id}_raw", f"{factor_id}_size", f"{factor_id}_industry", factor_id]
    placeholders = ",".join("?" * len(expr_names))
    cur.execute(
        f"""
        SELECT rc.expression_name, rc.neutralization_method, rc.top_k,
               rc.rebalance_freq, rc.dates, rc.daily_returns,
               er.rank_icir, er.sharpe
        FROM factor_return_curves rc
        INNER JOIN ({DEDUP_SQL}) er ON rc.result_id = er.id
        WHERE rc.expression_name IN ({placeholders})
        ORDER BY er.rank_icir DESC
        """,
        expr_names,
    )
    rows = cur.fetchall()
    db.close()

    combos = []
    for row in rows:
        dates_raw = row["dates"]
        returns_raw = row["daily_returns"]

        try:
            dates = json.loads(dates_raw)
            daily_returns = json.loads(returns_raw)
        except (json.JSONDecodeError, TypeError):
            continue

        # Compute cumulative returns: (1+r).cumprod() - 1
        cum = []
        acc = 1.0
        for r in daily_returns:
            acc *= 1.0 + r
            cum.append(round(acc - 1.0, 6))

        combos.append(
            {
                "label": (
                    f"{row['neutralization_method']} / "
                    f"top{row['top_k']} / "
                    f"freq{row['rebalance_freq']}"
                ),
                "neutralization": row["neutralization_method"],
                "top_k": row["top_k"],
                "rebalance_freq": row["rebalance_freq"],
                "rank_icir": row["rank_icir"],
                "sharpe": row["sharpe"],
                "dates": dates,
                "cumulative_returns": cum,
            }
        )

    return jsonify({"factor_id": factor_id, "combos": combos})


@app.route("/api/file/<factor_id>/<file_type>")
def api_file(factor_id, file_type):
    """Return content of a factor's spec or code file.

    file_type: 'spec' or 'code'
    """
    if file_type not in ("spec", "code"):
        return jsonify({"error": "file_type must be 'spec' or 'code'"}), 400

    factors = load_factor_status()
    fdata = factors.get(factor_id)
    if not fdata:
        return jsonify({"error": f"Factor {factor_id} not found in status"}), 404

    if file_type == "spec":
        rel_path = fdata.get("spec_path", "")
    else:
        # code path is in eval_result
        rel_path = fdata.get("eval_result", {}).get("factor_code_path", "")

    if not rel_path:
        return jsonify({"error": f"No {file_type} path for {factor_id}"}), 404

    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    # Security: ensure path is under PROJECT_ROOT
    abs_path = os.path.realpath(abs_path)
    if not abs_path.startswith(os.path.realpath(PROJECT_ROOT)):
        return jsonify({"error": "Invalid path"}), 403

    if not os.path.isfile(abs_path):
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    return jsonify({
        "factor_id": factor_id,
        "file_type": file_type,
        "path": rel_path,
        "content": content,
    })


if __name__ == "__main__":
    if not os.path.isfile(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        exit(1)
    print(f"DB: {DB_PATH}")
    print(f"Open http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=True)
