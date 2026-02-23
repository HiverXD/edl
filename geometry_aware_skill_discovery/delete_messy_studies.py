# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import sqlite3
import os

def delete_studies():
    db_path = "logs/optuna/tuning.db"
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    print(">>> Opening database for Study Purge...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute("SELECT study_id, study_name FROM studies")
        all_studies = cur.fetchall()
        
        to_delete_ids = []
        for s_id, s_name in all_studies:
            if s_name.startswith("peak"):
                print("  Marking for deletion: {0}".format(s_name))
                to_delete_ids.append(s_id)
        
        if not to_delete_ids:
            print("  No messy studies found to delete.")
            return

        id_str = ",".join(map(str, to_delete_ids))

        cur.execute("SELECT trial_id FROM trials WHERE study_id IN ({0})".format(id_str))
        trial_ids = [row[0] for row in cur.fetchall()]
        
        if trial_ids:
            t_id_str = ",".join(map(str, trial_ids))
            cur.execute("DELETE FROM trial_params WHERE trial_id IN ({0})".format(t_id_str))
            cur.execute("DELETE FROM trial_values WHERE trial_id IN ({0})".format(t_id_str))
            cur.execute("DELETE FROM trial_user_attributes WHERE trial_id IN ({0})".format(t_id_str))
            cur.execute("DELETE FROM trial_system_attributes WHERE trial_id IN ({0})".format(t_id_str))
            cur.execute("DELETE FROM trials WHERE trial_id IN ({0})".format(t_id_str))
            print("    Deleted {0} trials.".format(len(trial_ids)))

        cur.execute("DELETE FROM study_user_attributes WHERE study_id IN ({0})".format(id_str))
        cur.execute("DELETE FROM study_system_attributes WHERE study_id IN ({0})".format(id_str))
        cur.execute("DELETE FROM studies WHERE study_id IN ({0})".format(id_str))
        print("    Deleted {0} studies.".format(len(to_delete_ids)))

        conn.commit()
        print("\n>>> [Success] Purge complete.")
        conn.execute("VACUUM")
        
    except Exception as e:
        print("  [Error]: {0}".format(e))
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    delete_studies()
