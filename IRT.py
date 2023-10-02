import numpy as np
from girth import grm_mml_eap
import pandas as pd

# load annotated data
df = pd.read_csv('IRT_annotation.csv')

# map model_id to (model_name, setting)
one = "gpt4"
two = "chatgpt"
three = "gpt3"
four = "bard"
five = "vicuna"
six = "alpaca"
de_entity = {
    1: (two, "best_golden"), 2: (five, "best_golden"), 3: (six, "best_golden"), 4: (three, "best_golden"), 5: (one, "best_golden"), 6: (four, "best_golden"),
    7: (three, "all_baseline"), 8: (five, "all_baseline"), 9: (two, "all_baseline"), 10: (four, "all_baseline"), 11: (six, "all_baseline"), 12: (one, "all_baseline"),
    13: (three, "all_golden"), 14: (four, "all_golden"), 15: (one, "all_golden"), 16: (two, "all_golden"), 17: (five, "all_golden"), 18: (six, "all_golden"),
    19: (three, "best_baseline"), 20: (four, "best_baseline"), 21: (five, "best_baseline"), 22: (one, "best_baseline"), 23: (six, "best_baseline"), 24: (two, "best_baseline"),
    25: (two, "top2_baseline"), 26: (five, "top2_baseline"), 27: (one, "top2_baseline"), 28: (three, "top2_baseline"), 29: (four, "top2_baseline"), 30: (six, "top2_baseline"),
    31: (one, "top2_golden"), 32: (two, "top2_golden"), 33: (five, "top2_golden"), 34: (three,"top2_golden" ), 35: (four, "top2_golden"), 36: (six, "top2_golden"),
}
mapping = {one: 0, two: 1, three: 2, four: 3, five: 4, six: 5}
print([one, two, three, four, five, six])

topics = df['topic'].unique().tolist()
assert len(topics) == 31
stances = df['stance'].unique().tolist()
assert len(stances) == 2
# distinguish different settings (don't distinguish between topics/stances)
for setting in ["best_golden", "all_baseline", "all_golden", "best_baseline", "top2_baseline", "top2_golden"]:
    sub_df = df[df['setting'] == setting]
    assert len(sub_df) == 62*6

    # map each topic&stance input to list of scores (for six models)
    # row index: [one, two, three, four, five, six]
    responses = [] # (62, 6)
    for topic in topics:
        for stance in stances:
            group_df = sub_df[(sub_df['topic'] == topic) & (sub_df['stance'] == stance)]
            assert len(group_df) == 6
            resp = [-1, -1, -1, -1, -1, -1]
            for i in range(len(group_df)):
                model_id = group_df.iloc[i]['model_id']
                model_name, s = de_entity[model_id]
                assert s == setting
                resp[mapping[model_name]] = group_df.iloc[i]['ranking']
            responses.append(resp)
    assert len(responses) == 62
    responses = np.array(responses)
    print(f"------------{setting}-----------------")
    ltr_result = grm_mml_eap(responses)
    print(ltr_result['Ability'])
    # break