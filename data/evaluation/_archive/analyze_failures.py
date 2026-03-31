import csv

failed_convs = ['conv_006', 'conv_007', 'conv_011', 'conv_013', 'conv_018', 'conv_019', 'conv_024', 'conv_026', 'conv_048', 'conv_049', 'conv_050', 'conv_052']

failed_turns = []

# Read CSV with comma delimiter
with open('turn_results_routed.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['conversation_id'] in failed_convs and row['tool_correct'] == 'False':
            failed_turns.append(row)

print("Found %d failed turns\n" % len(failed_turns))

# Analyze by root cause
root_causes = {}

for turn in failed_turns:
    conv = turn['conversation_id']
    turn_num = turn['turn']
    user_msg = turn['user']
    expected_intent = turn['expected_intent']
    entity_resolved = turn['entity_resolved']
    tool_precision = turn['tool_precision']
    scope = turn['expected_scope']
    location = turn['expected_location']
    pattern = turn['pattern']
    difficulty = turn['difficulty']
    
    # Determine root cause
    if entity_resolved == 'False':
        if location.strip() == '':
            cause = "Entity Resolution: Location missing"
        else:
            cause = "Entity Resolution: Other entity failed"
    elif float(tool_precision) == 0.0:
        if 'comparison' in expected_intent or scope == 'city' and 'location_comparison' in expected_intent:
            cause = "Tool Mismatch: Comparison/Multi-location query failed"
        elif 'weekly' in user_msg.lower() or 'tuần' in user_msg or 'period' in expected_intent:
            cause = "Tool Mismatch: Long-term forecast tool unavailable"
        elif 'uv' in user_msg.lower() or 'expert' in expected_intent:
            cause = "Tool Mismatch: Specialized weather parameter tool missing"
        elif 'hourly_forecast' == expected_intent or 'daily_rhythm' in turn['tools_called']:
            cause = "Tool Mismatch: Hourly/time-of-day forecast tool unavailable"
        else:
            cause = "Tool Mismatch: Tool not matching intent"
    else:
        cause = "Tool Match: Partial success (%.1f precision)" % float(tool_precision)
    
    if cause not in root_causes:
        root_causes[cause] = []
    root_causes[cause].append({
        'conv': conv,
        'turn': turn_num,
        'pattern': pattern,
        'difficulty': difficulty,
        'user': user_msg[:50],
        'intent': expected_intent,
        'entity': entity_resolved
    })

# Print summary
print("=" * 90)
print("ROOT CAUSE ANALYSIS")
print("=" * 90)

for cause in sorted(root_causes.keys(), key=lambda x: -len(root_causes[x])):
    items = root_causes[cause]
    print("\n%s (%d failures)" % (cause, len(items)))
    print("-" * 90)
    
    for item in items:
        print("  %s Turn %s [%s/%s] - Intent: %s" % (item['conv'], item['turn'], item['pattern'], item['difficulty'], item['intent']))

print("\n" + "=" * 90)
print("SCENARIO BREAKDOWN (OLD vs NEW)")
print("=" * 90)

old_count = len([t for t in failed_turns if int(t['conversation_id'].split('_')[1]) <= 42])
new_count = len([t for t in failed_turns if int(t['conversation_id'].split('_')[1]) >= 43])

print("\nOLD Scenarios (conv_001-042): %d failures" % old_count)
print("NEW Scenarios (conv_043-060): %d failures" % new_count)

