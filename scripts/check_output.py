import csv

rows = list(csv.DictReader(open('team_submission.csv', 'r', encoding='utf-8')))

cv_count = sum(1 for r in rows if 'computer vision' in r['reasoning'].lower() or 'cv engineer' in r['reasoning'].lower())
print(f"CV/Vision engineers in top 100: {cv_count}")
print(f"Score range: {rows[0]['score']} to {rows[-1]['score']}")

print("\nCV engineers found:")
for r in rows:
    if 'computer vision' in r['reasoning'].lower():
        print(f"  Rank {r['rank']}: score={r['score']} | {r['reasoning'][:120]}")

consulting = ['tcs', 'wipro', 'infosys', 'accenture', 'cognizant', 'capgemini', 'deloitte', 'hcl']
print("\nConsulting firm current employers in top 100:")
for r in rows:
    for firm in consulting:
        if firm in r['reasoning'].lower():
            print(f"  Rank {r['rank']}: {r['reasoning'][:100]}")
            break
