# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import sqlite3
import os

def force_cleanup():
    db_path = "logs/optuna/tuning.db"
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    print(">>> Opening database for deep cleaning...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # 1. Identify trials with value 0.0 (The ones that mess up the reports)
        cur.execute("SELECT trial_id FROM trial_values WHERE value = 0.0")
        broken_ids = [row[0] for row in cur.fetchall()]
        
        # 2. Also identify trials in RUNNING or FAIL state
        cur.execute("SELECT trial_id FROM trials WHERE state != 'COMPLETE'")
        broken_ids.extend([row[0] for row in cur.fetchall()])
        
        broken_ids = list(set(broken_ids)) # Unique IDs

        if not broken_ids:
            print("  No broken trials found. DB is already clean.")
            return

        print("  Found {0} broken/incomplete trials. Purging from all tables...".format(len(broken_ids)))
        
        id_str = ",".join(map(str, broken_ids))
        tables = [
            "trial_params", 
            "trial_values", 
            "trial_user_attributes", 
            "trial_system_attributes", 
            "trials"
        ]
        
        for table in tables:
            cur.execute("DELETE FROM {0} WHERE trial_id IN ({1})".format(table, id_str))
            print("    Deleted from {0}".format(table))

        conn.commit()
        print("\n>>> [Success] Physical cleanup complete! {0} trials removed.".format(len(broken_ids)))
        
        print("  Shrinking database file...")
        conn.execute("VACUUM")
        
    except Exception as e:
        print("  [Error] During SQL cleanup: {0}".format(e))
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    force_cleanup()
