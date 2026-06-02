import pandas as pd
import os

# =====================================================
# 🔧 File Paths
# =====================================================
disagreements_file = "gsm8k_gemini_counterfactual_disagreements.csv"
counterfactuals_file = "gsm8k_counterfactuals.csv"

output_file = "gsm8k_counterfactuals.csv"  # output file

# =====================================================
# 📂 Load Data
# =====================================================
dis_df = pd.read_csv(disagreements_file)
cf_df = pd.read_csv(counterfactuals_file)

print(f"✅ Loaded {len(dis_df)} disagreements and {len(cf_df)} counterfactual examples.\n")

# If an existing output file exists, load it (resume safely)
if os.path.exists(output_file):
    print(f"🔁 Found existing output file: {output_file} — loading it for resume.\n")
    cf_df = pd.read_csv(output_file)
else:
    print("🆕 No existing output file found — starting fresh.\n")

# =====================================================
# 🔁 Iterate through disagreements
# =====================================================
total = len(dis_df)
for idx, row in dis_df.iterrows():
    qid = int(row["id"])
    version = row["version"]

    # 🔢 Progress counter
    done = idx + 1
    remaining = total - done
    print(f"\n🔢 Progress: {done}/{total} done — {remaining} remaining\n")

    # Find corresponding question/answer columns
    question_col = f"{version}_question"
    answer_col = f"{version}_answer"

    # Get the matching row from the counterfactual dataset
    cf_row_idx = cf_df.index[cf_df["id"] == qid]
    if len(cf_row_idx) == 0:
        print(f"❌ No matching ID {qid} found — skipping.\n")
        continue

    cf_row_idx = cf_row_idx[0]
    question_text = cf_df.loc[cf_row_idx, question_col]
    old_answer = cf_df.loc[cf_row_idx, answer_col]
    gemini_answer = row["gemini_answer"]

    print("=" * 90)
    print(f"🆔 ID: {qid}")
    print(f"📖 Version: {version}")
    print(f"❓ Question: {question_text}\n")
    print(f"📘 GSM8K Answer (current): {old_answer}")
    print(f"🤖 Gemini Answer (suggested): {gemini_answer}")
    print("=" * 90)

    # Ask user whether to replace
    user_input = input("🔄 Replace GSM8K answer with Gemini answer? (yes/no): ").strip().lower()

    if user_input == "yes":
        cf_df.loc[cf_row_idx, answer_col] = gemini_answer
        cf_df.to_csv(output_file, index=False)  # ✅ instant save after each change
        print(f"✅ Replaced and saved {answer_col} for ID {qid} with: {gemini_answer}\n")
    else:
        print(f"⏩ Skipped ID {qid}\n")

print("🎉 All done! Every accepted replacement has been saved immediately.")
print(f"💾 Final file: {output_file}")
