"""Print compact SQLite registry counts for a PoC automation run."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    tables = [row["name"] for row in con.execute("select name from sqlite_master where type='table' order by name")]
    print("tables", ",".join(tables))
    for table in ("tuning_candidates", "case_runs", "evaluation_results", "agent_trial_observations"):
        if table in tables:
            count = con.execute(f"select count(*) from {table}").fetchone()[0]
            print(table, count)

    if "tuning_candidates" in tables:
        print("candidate_status")
        for row in con.execute("select status, count(*) as n from tuning_candidates group by status order by status"):
            print(f"  {row['status']}: {row['n']}")

    if "case_runs" in tables:
        print("case_run_status")
        for row in con.execute("select status, count(*) as n from case_runs group by status order by status"):
            print(f"  {row['status']}: {row['n']}")

    if "evaluation_results" in tables and "case_runs" in tables:
        case_run_columns = {row["name"] for row in con.execute("pragma table_info(case_runs)")}
        if "split" in case_run_columns:
            print("total_score_by_split")
            for row in con.execute(
                """
                select cr.split, count(*) as cases, round(avg(er.score), 4) as total_score
                from case_runs cr
                join evaluation_results er on er.run_id = cr.run_id and er.evaluator_name='total_score'
                group by cr.split
                order by cr.split
                """
            ):
                print(f"  {row['split']}: cases={row['cases']} avg_total={row['total_score']}")
        else:
            row = con.execute(
                """
                select count(*) as cases, round(avg(er.score), 4) as total_score
                from case_runs cr
                join evaluation_results er on er.run_id = cr.run_id and er.evaluator_name='total_score'
                """
            ).fetchone()
            print(f"total_score_overall cases={row['cases']} avg_total={row['total_score']}")

    if "agent_trial_observations" in tables:
        print("trial_status")
        for row in con.execute(
            "select status, count(*) as n from agent_trial_observations group by status order by status"
        ):
            print(f"  {row['status']}: {row['n']}")
        row = con.execute(
            """
            select
              min(json_extract(summary_json, '$.case_count')) as min_cases,
              max(json_extract(summary_json, '$.case_count')) as max_cases,
              round(avg(json_extract(summary_json, '$.case_count')), 2) as avg_cases
            from agent_trial_observations
            """
        ).fetchone()
        print(f"trial_case_count min={row['min_cases']} max={row['max_cases']} avg={row['avg_cases']}")

    if "tuning_effects" in tables:
        print("top_effects")
        effect_columns = {row["name"] for row in con.execute("pragma table_info(tuning_effects)")}
        print("  columns=" + ",".join(sorted(effect_columns)))
        order_column = None
        for candidate_column in ("delta_vs_baseline", "delta", "score_delta", "total_delta", "total_score_delta"):
            if candidate_column in effect_columns:
                order_column = candidate_column
                break
        if order_column is None:
            return 0
        for row in con.execute(
            f"""
            select *
            from tuning_effects
            order by {order_column} desc
            limit 10
            """
        ):
            values = dict(row)
            rendered = ", ".join(f"{key}={values[key]}" for key in values.keys() if key in {
                "tuning_id",
                "split",
                "case_id",
                "effect_label",
                "delta",
                "delta_vs_baseline",
                "total_score_delta",
                "regression",
                "regression_count",
                "positive_count",
                "negative_count",
            })
            print(f"  {rendered}")

    if "promotion_decisions" in tables:
        print("promotion_decisions")
        for row in con.execute(
            "select decision, count(*) as n from promotion_decisions group by decision order by decision"
        ):
            print(f"  {row['decision']}: {row['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
